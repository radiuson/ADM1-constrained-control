#!/usr/bin/env python3
"""Oracle stage of the independent-disturbance evaluation (optional in the prereg).

Every conventional configuration on the 30 test scenarios gives a test
envelope that is selected on test (an upper bound for the conventional side,
not a deployable tuning).  Learned runs are compared with it as in the
manuscript: D = methane - E(rate), completed-step rate, non-dominated envelope.

    python3 scripts/independent_eval_oracle.py <outdir> [reported_envelope.csv]
"""
import csv
import json
import os
import statistics as st
import sys

import numpy as np


def load_json(path):
    """Read path, or path + '.gz' where the record is stored compressed."""
    import gzip
    if os.path.exists(path):
        return json.load(open(path))
    return json.load(gzip.open(path + '.gz', 'rt'))

ON = {'CPO', 'PCPO', 'CUP', 'PPOLag', 'TRPOLag', 'TRPOPID', 'CPPOPID', 'FOCOPS', 'OnCRPO', 'P3O'}


def pooled(per, ids):
    n = sum(per[i]['steps'] for i in ids)
    k = sum(per[i]['viol'] for i in ids)
    return 100 * k / n, sum(per[i]['ch4_sum'] for i in ids) / n, n


def envelope(pts):
    E, best = [], -1.0
    for v, c, name in sorted(pts, key=lambda p: (p[0], -p[1])):
        if c > best:
            if E and E[-1][0] == v:
                E[-1] = (v, c, name)
            else:
                E.append((v, c, name))
            best = c
    return E


def main():
    d = sys.argv[1]
    rep_env = sys.argv[2] if len(sys.argv) > 2 else None
    S = json.load(open(os.path.join(d, 'scenarios.json')))
    tid = [s['id'] for s in S['test']]
    vid = [s['id'] for s in S['validation']]
    A = load_json(os.path.join(d, 'test_conventional_all.json'))
    V = {r['name']: r for r in load_json(os.path.join(d, 'val_conventional.json'))}
    full = 60 * len(tid)
    pts, done = [], set()
    for r in A:
        v, c, n = pooled(r['per_scenario'], tid)
        pts.append((v, c, r['name']))   # completed-step rate, as the manuscript's envelope
        if n == full:
            done.add(r['name'])
    print(f'{len(A)} conventional configurations; {len(done)} complete all {len(tid)} test scenarios')
    Ec = envelope([p for p in pts if p[2] in done])
    print(f'envelope from complete configurations only: {len(Ec)} points, '
          + ', '.join(f'E({b})={np.interp(b, [e[0] for e in Ec], [e[1] for e in Ec]):.0f}' for b in (0, 1, 5)))
    E = envelope(pts)
    print(f'envelope points from configurations with a terminated scenario: {sum(e[2] not in done for e in E)}')
    ex, ey = np.array([e[0] for e in E]), np.array([e[1] for e in E])
    print(f'test envelope: {len(E)} points, {ey[0]:.0f} at {ex[0]:.2f} % ... {ey[-1]:.0f} at {ex[-1]:.2f} %')
    for b in (0, 0.5, 1, 2, 3, 5, 10):
        print(f'  E({b:>4} %) = {np.interp(b, ex, ey):7.1f}')
    zero = [p for p in pts if p[0] == 0 and p[2] in done]
    print(f'zero violations on all test scenarios: {len(zero)} configurations, best {max(p[1] for p in zero):.0f}'
          if zero else 'no conventional configuration has zero violations on test')
    # validation -> test shift for every configuration complete on both
    dv = []
    for r in A:
        if r['name'] in V:
            v1, c1, n1 = pooled(V[r['name']]['per_scenario'], vid)
            v2, c2, n2 = pooled(r['per_scenario'], tid)
            if n1 == 600 and n2 == full:
                dv.append((v2 - v1, c2 - c1))
    print(f'validation -> test, {len(dv)} configurations complete on both: rate change median '
          f'{st.median(x for x, _ in dv):+.2f} pp, methane change median {st.median(y for _, y in dv):+.0f}')
    R = json.load(open(os.path.join(d, 'test_rl.json')))
    if rep_env:
        Er = sorted((float(r['vfa_violation_rate']), float(r['methane_mean'])) for r in csv.DictReader(open(rep_env)))
        rx, ry = zip(*Er)
    rows = []
    for r in R:
        v, c, n = pooled(r['per_scenario'], tid)
        row = {'algorithm': r['algorithm'], 'seed': r['seed'], 'cost_limit': r['cost_limit'],
               'G1': r['G1'], 'G2': r['G2'], 'class': 'on' if r['algorithm'] in ON else 'off',
               'test_rate': v, 'test_methane': c, 'test_completed': n,
               'D_test': c - float(np.interp(v, ex, ey)), 'above_test_envelope': int(c > np.interp(v, ex, ey))}
        if rep_env:
            row['D_reported'] = r['reported_methane'] - float(np.interp(r['reported_viol'], rx, ry))
        rows.append(row)
    with open(os.path.join(d, 'oracle_D.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(d, 'oracle_envelope.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['test_rate', 'test_methane', 'name'])
        w.writerows(E)
    for g in ('G1', 'G2'):
        print(f'\n{g}')
        for cl in ('on', 'off'):
            H = [x for x in rows if x[g] and x['class'] == cl]
            s = f'  {cl:>3} n={len(H):3d}  D_test median {st.median(x["D_test"] for x in H):6.0f}' \
                f'  above {sum(x["above_test_envelope"] for x in H):3d}'
            if rep_env:
                s += f'  D_reported median {st.median(x["D_reported"] for x in H):6.0f}'
            print(s)
        for a in sorted({x['algorithm'] for x in rows if x[g]}):
            H = [x for x in rows if x[g] and x['algorithm'] == a]
            s = f'    {a:<8} n={len(H):2d}  D_test {st.median(x["D_test"] for x in H):6.0f}' \
                f'  above {sum(x["above_test_envelope"] for x in H):2d}'
            if rep_env:
                s += f'  D_reported {st.median(x["D_reported"] for x in H):6.0f}'
            print(s)


if __name__ == '__main__':
    main()
