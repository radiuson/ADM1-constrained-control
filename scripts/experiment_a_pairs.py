#!/usr/bin/env python3
"""Experiment A per pair: dD = dM - [E(v_aligned) - E(v_original)], and the
budget comparison on the manuscript's convention (best conventional tuning
meeting the budget, no interpolation).

    python3 scripts/export_figure_data.py <figdir> experiment_A
    python3 scripts/experiment_a_pairs.py experiment_A <figdir> [out.csv]
"""
import csv, json, os, sys, random, statistics as st, numpy as np, collections
S, FD = sys.argv[1], sys.argv[2]
env = sorted((float(x['vfa_violation_rate']), float(x['methane_mean']))
             for x in csv.DictReader(open(f'{FD}/envelope_full.csv')))
ev, em = np.array([a for a, _ in env]), np.array([b for _, b in env])
conv = [(float(r['vfa_violation_rate']), float(r['methane_mean']))
        for r in csv.DictReader(open(f'{FD}/conventional_runs.csv')) if r['in_reported_set'] == '1']
E = lambda v: float(np.interp(v, ev, em)) if v <= ev[-1] else None
B = {0.6: 1, 1.8: 3, 3.0: 5, 6.0: 10}
CB = {c: max(m for v, m in conv if v <= b) for c, b in B.items()}
load = lambda p: {(x['algo'], x['cost_limit'], x['seed']): x for x in json.load(open(p))}
A, C = load(f'{S}/methane_only/scored.json'), load(f'{S}/reward_control/scored.json')
keys = sorted(set(A) & set(C))
fa = lambda x: 100 * (x['viol'] / 100 * x['steps'] + (420 - x['steps'])) / 420
rows = []
for k in keys:
    a, o = A[k], C[k]
    Ea, Eo = E(a['viol']), E(o['viol'])
    ok = Ea is not None and Eo is not None
    rows.append(dict(pair_id=f'{k[0]}_c{k[1]}_s{k[2]}', algorithm=k[0], cost_limit=k[1], seed=k[2],
        methane_original=o['ch4'], methane_aligned=a['ch4'], dM=a['ch4'] - o['ch4'],
        viol_original=o['viol'], viol_aligned=a['viol'], dv=a['viol'] - o['viol'],
        E_original=Eo, E_aligned=Ea, dE=(Ea - Eo) if ok else None,
        D_original=None if Eo is None else o['ch4'] - Eo, D_aligned=None if Ea is None else a['ch4'] - Ea,
        dD=(a['ch4'] - o['ch4'] - (Ea - Eo)) if ok else None, D_pair_valid=int(ok),
        budget_pct=B[k[1]], in_budget_original=int(fa(o) <= B[k[1]]), in_budget_aligned=int(fa(a) <= B[k[1]])))
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(FD, 'experiment_A_pairs.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'{len(rows)} pairs -> {out}')
rng = random.Random(1)
def boot(by, n=4000):
    seeds = list(by); m = []
    for _ in range(n):
        v = [x for s in (rng.choice(seeds) for _ in seeds) for x in by[s]]
        m.append(np.mean(v))
    return np.percentile(m, [2.5, 97.5])
for alg in sorted({r['algorithm'] for r in rows}):
    R = [r for r in rows if r['algorithm'] == alg]
    ns = len({r['seed'] for r in R})
    print(f'\n{alg}  ({len(R)} pairs, {ns} seeds)')
    for key in ('dM', 'dv', 'dE', 'dD'):
        by = collections.defaultdict(list)
        for r in R:
            if r[key] is not None: by[r['seed']].append(r[key])
        v = [x for l in by.values() for x in l]; lo, hi = boot(by)
        print(f'  {key}: n={len(v):2d} mean {np.mean(v):+8.2f} median {np.median(v):+8.2f}  seed-block [{lo:+.2f}, {hi:+.2f}]')
    print(f'  in budget: original {sum(r["in_budget_original"] for r in R)}/{len(R)}, aligned {sum(r["in_budget_aligned"] for r in R)}/{len(R)}')
    for c in B:
        rr = [r for r in R if r['cost_limit'] == c]
        io = [r['methane_original'] for r in rr if r['in_budget_original']]
        ia = [r['methane_aligned'] for r in rr if r['in_budget_aligned']]
        f_ = lambda l: f'{max(l) - CB[c]:+5.0f} (best of {len(l)}/{len(rr)} in budget)' if l else ' none in budget'
        print(f'   budget {B[c]:>2} %: conventional {CB[c]:.0f}   original {f_(io)}   aligned {f_(ia)}')
