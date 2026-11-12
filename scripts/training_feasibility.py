#!/usr/bin/env python
"""Training-side constraint feasibility, per constrained family.

Table~\\ref{tab:cmdp} measures the realised rate on the fixed-horizon
evaluation.  What the optimiser reads instead is the episode cost
accumulated to the terminating step, logged by the library as
Metrics/EpCost against Metrics/EpLen.  This reports, per family and cost
limit, the mean of that cost over the final epochs of each run and its
difference from the limit, so training feasibility can be read separately
from evaluation tracking.

Usage:  python scripts/training_feasibility.py [--tail N]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics as st
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'models_cmdp')
# The four limits of the main design; the directory name carries the seed
# alone, so the limit is read from the run's own config.
LIMITS = (0.6, 1.8, 3.0, 6.0)


def cost_limit(run_dir: str) -> float | None:
    """The cost limit this run was trained at, from its saved config."""
    cfg = os.path.join(run_dir, 'config.json')
    if not os.path.exists(cfg):
        return None
    import json
    with open(cfg) as fh:
        d = json.load(fh)
    for key in ('lagrange_cfgs', 'algo_cfgs'):
        sub = d.get(key) or {}
        if 'cost_limit' in sub:
            return float(sub['cost_limit'])
    return None


def final_cost(path: str, tail: int) -> tuple[float, float] | None:
    """Mean EpCost and EpLen over the last `tail` logged epochs."""
    costs, lens = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            c, l = row.get('Metrics/EpCost'), row.get('Metrics/EpLen')
            if c in (None, '') or l in (None, ''):
                continue
            try:
                costs.append(float(c)); lens.append(float(l))
            except ValueError:
                pass
    if not costs:
        return None
    return st.mean(costs[-tail:]), st.mean(lens[-tail:])


def design_runs() -> set[str]:
    """Run directories of the 560 constrained runs of the reported design.

    models_cmdp also holds runs outside that design (additional seeds,
    repeated launches), so the reported statistics are restricted to the
    runs the per-run record lists as the main design.
    """
    rec = os.path.join(os.environ.get('FIGURE_DATA', os.path.join(os.path.dirname(ROOT), 'figures')),
                       'rl_original_runs.csv')
    with open(rec) as fh:
        return {os.path.normpath(os.path.join(os.path.dirname(ROOT), r['model_path']))
                for r in csv.DictReader(fh)
                if r['dataset'] == 'main' and r['formulation'] == 'constrained'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tail', type=int, default=10,
                    help='epochs averaged at the end of each run')
    args = ap.parse_args()

    by = defaultdict(list)
    keep = design_runs()
    for fam in sorted(os.listdir(ROOT)):
        name = re.sub(r'-\{.*\}$', '', fam)
        fam_dir = os.path.join(ROOT, fam)
        if not os.path.isdir(fam_dir):
            continue
        for run in sorted(os.listdir(fam_dir)):
            rd = os.path.join(fam_dir, run)
            p = os.path.join(rd, 'progress.csv')
            if not os.path.exists(p) or os.path.normpath(rd) not in keep:
                continue
            d = cost_limit(rd)
            if d is None or round(d, 4) not in LIMITS:
                continue
            got = final_cost(p, args.tail)
            if got:
                by[(name, round(d, 4))].append(got)

    print(f'training-side feasibility, mean over the last {args.tail} epochs '
          f'of each run')
    print(f'{"family":10}{"d":>6}{"runs":>6}{"E[Jc_train]":>13}'
          f'{"-d":>8}{"EpLen":>8}')
    print('-' * 51)
    fam_delta = defaultdict(list)
    for (fam, d) in sorted(by):
        rows = by[(fam, d)]
        mc = st.mean(c for c, _ in rows)
        ml = st.mean(l for _, l in rows)
        print(f'{fam:10}{d:6.1f}{len(rows):6d}{mc:13.2f}{mc - d:+8.2f}{ml:8.1f}')
        fam_delta[fam].append(mc - d)
    print()
    print(f'{"family":10}{"mean E[Jc_train]-d over the four limits":>42}')
    for fam in sorted(fam_delta):
        v = fam_delta[fam]
        print(f'{fam:10}{st.mean(v):+42.2f}')


if __name__ == '__main__':
    main()
