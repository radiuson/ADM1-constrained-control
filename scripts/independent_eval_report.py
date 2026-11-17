#!/usr/bin/env python3
"""Summarise the independent-disturbance evaluation (rules: analysis/independent_eval_prereg.md).

    python3 scripts/independent_eval_report.py <outdir>
"""
import csv
import json
import os
import random
import statistics as st
import sys

STEPS = 60
ON = {'CPO', 'PCPO', 'CUP', 'PPOLag', 'TRPOLag', 'TRPOPID', 'CPPOPID', 'FOCOPS', 'OnCRPO', 'P3O'}


def pool(per, ids):
    n = sum(per[i]['steps'] for i in ids)
    k = sum(per[i]['viol'] for i in ids)
    c = sum(per[i]['ch4_sum'] for i in ids)
    full = STEPS * len(ids)
    return {'viol_rate': 100 * k / n, 'fa_rate': 100 * (k + full - n) / full,
            'methane': c / n,
            'zero_scen': sum(per[i]['viol'] == 0 and per[i]['steps'] == STEPS for i in ids),
            'terminated_scen': sum(per[i]['steps'] < STEPS for i in ids)}


def fam_boot(rows, key, B=4000, seed=1):
    """Median over runs, resampling families then runs within family."""
    fams = {}
    for r in rows:
        fams.setdefault(r['algorithm'], []).append(r[key])
    rng = random.Random(seed)
    names = list(fams)
    out = []
    for _ in range(B):
        v = []
        for f in (rng.choice(names) for _ in names):
            v += [rng.choice(fams[f]) for _ in fams[f]]
        out.append(st.median(v))
    out.sort()
    return st.median([x for f in fams.values() for x in f]), out[int(.025 * B)], out[int(.975 * B)]


def main():
    d = sys.argv[1]
    S = json.load(open(os.path.join(d, 'scenarios.json')))
    tid = [s['id'] for s in S['test']]
    sel = json.load(open(os.path.join(d, 'selected.json')))
    TC = {r['name']: r for r in json.load(open(os.path.join(d, 'test_conventional_selected.json')))}
    R = json.load(open(os.path.join(d, 'test_rl.json')))
    out = {'n_test': len(tid), 'conventional': {}, 'G1': {}, 'G2': {}}
    print(f'test scenarios: {len(tid)}')
    print('\nconventional, selected on validation:')
    conv = {}
    for b, v in sel.items():
        if not v:
            print(f'  {b:>2} %: none feasible on validation')
            continue
        p = pool(TC[v['name']]['per_scenario'], tid)
        conv[b] = p
        out['conventional'][b] = {'name': v['name'], 'validation': v['validation'], 'test': p}
        print(f"  {b:>2} %: {v['name']:<40} val {v['validation']['methane']:.0f} / fa {v['validation']['fa_rate']:.2f}"
              f"   test {p['methane']:.0f} / fa {p['fa_rate']:.2f} / zero-scen {p['zero_scen']}/{len(tid)}"
              f" / meets budget {p['fa_rate'] <= float(b) + 1e-9}")
    rows = []
    for r in R:
        p = pool(r['per_scenario'], tid)
        rows.append({**{k: r[k] for k in ('algorithm', 'seed', 'cost_limit', 'G1', 'G2',
                                          'reported_methane', 'reported_viol')},
                     'class': 'on' if r['algorithm'] in ON else 'off', **p})
    with open(os.path.join(d, 'test_rl_runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for g, ref, budget in (('G1', '0', 0.0), ('G2', '1', 1.0)):
        G = [r for r in rows if r[g]]
        c = conv.get(ref)
        print(f'\n{g}: {len(G)} runs ' + ('(zero violations in the reported evaluation)' if g == 'G1'
                                          else '(all 14 families at d = 0.6)'))
        res = {}
        for cl in ('on', 'off', 'all'):
            H = [r for r in G if cl == 'all' or r['class'] == cl]
            if not H:
                continue
            zero = sum(r['zero_scen'] == len(tid) for r in H)
            meet = sum(r['fa_rate'] <= budget + 1e-9 for r in H)
            term = sum(r['terminated_scen'] > 0 for r in H)
            m, lo, hi = fam_boot(H, 'methane')
            fa_m = st.median(r['fa_rate'] for r in H)
            above = sum(c is not None and r['methane'] > c['methane'] and r['fa_rate'] <= c['fa_rate'] + 1e-9
                        for r in H)
            res[cl] = {'runs': len(H), 'zero_on_all_test': zero, 'meet_budget': meet,
                       'any_termination': term, 'median_methane': m, 'ci': [lo, hi],
                       'median_fa_rate': fa_m, 'dominate_selected_conventional': above,
                       'median_zero_scen': st.median(r['zero_scen'] for r in H)}
            print(f'  {cl:>3}: n={len(H):3d}  zero viol on all {len(tid)} test scen {zero:3d}'
                  f'  meet {budget:g} % {meet:3d}  any termination {term:3d}'
                  f'  median methane {m:.0f} [{lo:.0f}, {hi:.0f}]  median fa {fa_m:.2f}'
                  f'  median zero-scen {res[cl]["median_zero_scen"]}'
                  f'  >conv & <=rate {above}')
        if c:
            print(f'  conventional selected for {ref} %: methane {c["methane"]:.0f}, fa {c["fa_rate"]:.2f}, '
                  f'zero-scen {c["zero_scen"]}/{len(tid)}')
        fams = sorted({r['algorithm'] for r in G})
        for fa in fams:
            H = [r for r in G if r['algorithm'] == fa]
            print(f'    {fa:<8} n={len(H):2d}  methane med {st.median(r["methane"] for r in H):6.0f}'
                  f'  fa med {st.median(r["fa_rate"] for r in H):5.2f}'
                  f'  zero-all {sum(r["zero_scen"] == len(tid) for r in H):2d}'
                  f'  meet {sum(r["fa_rate"] <= budget + 1e-9 for r in H):2d}'
                  f'  reported methane med {st.median(r["reported_methane"] for r in H):6.0f}')
        out[g] = res
    json.dump(out, open(os.path.join(d, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
