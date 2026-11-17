#!/usr/bin/env python3
"""Medians, intervals and rank correlations of the independent-disturbance
evaluation, as reported in the manuscript.

    python3 scripts/independent_eval_intervals.py [results_independent]

Groups: G1 = constrained runs with zero violations in the original
evaluation, G2 = all constrained runs at the tightest cost limit (d = 0.6),
each split by class.  D is the difference from the envelope built on the
test scenarios (oracle) and, for comparison, on the original scenarios.
Intervals resample families and then runs within each drawn family.
"""
import csv
import os
import statistics as st
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BOOT, SEED = 4000, 0


def boot_median(rows, key, rng):
    fams = sorted({r['algorithm'] for r in rows})
    by = {f: np.array([float(r[key]) for r in rows if r['algorithm'] == f]) for f in fams}
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(fams, size=len(fams), replace=True)
        vals = np.concatenate([rng.choice(by[f], size=len(by[f]), replace=True) for f in pick])
        out[b] = np.median(vals)
    return np.percentile(out, [2.5, 97.5])


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results_independent')
    R = list(csv.DictReader(open(os.path.join(d, 'test_runs_learned.csv'))))
    rng = np.random.default_rng(SEED)
    for g in ('G1', 'G2'):
        for cl in ('on-policy', 'off-policy'):
            H = [r for r in R if r[g] == '1' and r['class'] == cl]
            for key in ('D_test_oracle', 'D_reported'):
                lo, hi = boot_median(H, key, rng)
                print(f'{g} {cl:10s} n={len(H):3d} {key:14s} median {st.median(float(r[key]) for r in H):6.0f} '
                      f'[{lo:5.0f},{hi:5.0f}]')
            above = sum(float(r['D_test_oracle']) > 0 for r in H)
            zero = sum(r['test_zero_violation_scenarios'] == '30' for r in H)
            print(f'   above the test envelope {above}/{len(H)}; zero violations on all 30 test scenarios {zero}')
    H = [r for r in R if r['G2'] == '1']
    fams = sorted({r['algorithm'] for r in H})
    rep = [st.median(float(r['D_reported']) for r in H if r['algorithm'] == f) for f in fams]
    tst = [st.median(float(r['D_test_oracle']) for r in H if r['algorithm'] == f) for f in fams]
    print(f'G2 family-median rank correlation, original against test: {spearman(rep, tst):.2f}')


if __name__ == '__main__':
    main()
