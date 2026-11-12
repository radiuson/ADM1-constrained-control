"""Observed Lagrange multipliers against P3O's exactness threshold.

P3O's Theorem 1 requires kappa >= ||lambda-bar||_inf, the infinity norm of
the optimal multiplier.  lambda-bar is not computable here, but the
Lagrangian families solve the same constrained problem on the same
environment and cost limits, and their multiplier trajectories estimate it.
This reports the peak and final multiplier per family and cost limit
against the library's kappa = 20.
"""
import csv
import glob
import json
import os

KAPPA = 20.0
FAMILIES = ('PPOLag', 'TRPOLag', 'CPPOPID', 'TRPOPID', 'FOCOPS')
COL = 'Metrics/LagrangeMultiplier'


def cost_limit(run_dir):
    cfg = os.path.join(run_dir, 'config.json')
    if not os.path.exists(cfg):
        return None
    with open(cfg) as fh:
        d = json.load(fh)
    for key in ('lagrange_cfgs', 'algo_cfgs'):
        sub = d.get(key) or {}
        if 'cost_limit' in sub:
            return float(sub['cost_limit'])
    return None


def series(path):
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            raw = (row.get(COL) or '').strip()
            if raw:
                try:
                    out.append(float(raw))
                except ValueError:
                    pass
    return out


def main():
    print(f'{"family":10} {"d":>5} {"runs":>5} {"peak":>9} {"final":>9}'
          f'  {"peak > kappa":>12}')
    overall = []
    for fam in FAMILIES:
        by_limit = {}
        for run in sorted(glob.glob(f'models_cmdp/{fam}-{{ADM1-v0}}/seed-*')):
            prog = os.path.join(run, 'progress.csv')
            if not os.path.exists(prog):
                continue
            d = cost_limit(run)
            s = series(prog)
            if d is None or not s:
                continue
            by_limit.setdefault(d, []).append((max(s), s[-1]))
        for d in sorted(by_limit):
            rows = by_limit[d]
            peak = max(r[0] for r in rows)
            fin = max(r[1] for r in rows)
            overall.append(peak)
            flag = 'yes' if peak > KAPPA else 'no'
            print(f'{fam:10} {d:5.1f} {len(rows):5d} {peak:9.2f} '
                  f'{fin:9.2f}  {flag:>12}')
    if overall:
        print()
        print(f'largest multiplier observed anywhere: {max(overall):.2f}')
        print(f'library kappa: {KAPPA:.0f}')
        n_above = sum(1 for p in overall if p > KAPPA)
        print(f'(family, limit) cells whose peak exceeds kappa: '
              f'{n_above} of {len(overall)}')


if __name__ == '__main__':
    main()
