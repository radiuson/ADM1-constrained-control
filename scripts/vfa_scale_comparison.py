#!/usr/bin/env python
"""Simulated VFA against the plant's recorded VFA, and the mapped thresholds.

Section "Scope of the plant anchoring" states how far simulated VFA sits
below the recorded series and maps the model's 300 mg/L soft limit onto the
plant scale with that ratio.  This script fixes the protocol for both:

* Plant loading.  Scenario plant_load, feed strength 1.0, each of five feed
  rates across the actuator range held for SETTLE_DAYS one-day steps from the
  scenario's initial state; the reading is the VFA on the last day, in
  mg HAc-eq/L.  The ratio is that value over the median of every recorded
  digester-day VFA in the raw laboratory file.
* Upper end of the loading axis.  Scenario peak_load at the maximum feed
  rate, at strength 1.0 and 1.3.  The hard limits stop an episode at
  1500 mg/L, so termination is switched off here and the run is held for
  UPPER_DAYS days; the reading is the last-day VFA.
* Mapped thresholds.  300 mg/L divided by each end of the plant-loading
  ratio, and the share of recorded digester-day observations above each.

    python scripts/vfa_scale_comparison.py
"""
from __future__ import annotations

import csv
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.adm1_gym_env_std import ADM1Env_Std

PLANT_CSV = os.path.join(os.environ.get('MUSCATINE_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'muscatine')), 'LAB', 'LABS-raw.csv')
SETTLE_DAYS = 120         # same hold as the upper end of the loading axis
UPPER_DAYS = 120
FEEDS = (41, 70, 100, 130, 159)
SOFT_MG = 300.0
KGCOD_TO_MG_HAC = 1000.0 * 60.0 / 64.0


def plant_vfa(path: str) -> list[float]:
    out = []
    with open(path, encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            for key in ('Dig1-VFA_mgL', 'Dig2-VFA_mgL'):
                raw = (row.get(key) or '').strip()
                if raw:
                    try:
                        out.append(float(raw))
                    except ValueError:
                        pass
    return out


def run(scenario: str, q: float, mult: float, days: int,
        no_termination: bool = False) -> list[float]:
    env = ADM1Env_Std(scenario_name=scenario, enable_disturbances=False,
                      step_size=1.0)
    if no_termination:
        env.VFA_HARD_KGCOD_M3 = float('inf')
        env.NH3_HARD_KMOL_M3 = float('inf')
        env.PH_HARD_LO, env.PH_HARD_HI = -float('inf'), float('inf')
    env.reset(seed=7)
    act = np.array([q, mult], dtype=np.float32)
    trace = []
    for _ in range(days):
        _, _, term, trunc, _ = env.step(act)
        trace.append(env._calculate_total_vfa(env.current_state)
                     * KGCOD_TO_MG_HAC)
        if term:
            break
    return trace


def main() -> None:
    rec = sorted(plant_vfa(PLANT_CSV))
    n = len(rec)
    med = st.median(rec)
    print(f'plant VFA: {n} digester-day observations, median {med:.1f}, '
          f'p25 {rec[n // 4]:.0f}, p75 {rec[3 * n // 4]:.0f}, '
          f'max {rec[-1]:.0f} mg/L')
    print()
    print(f'plant_load, strength 1.0, {SETTLE_DAYS} days at constant feed')
    ratios = []
    for q in FEEDS:
        v = run('plant_load', q, 1.0, SETTLE_DAYS)[-1]
        ratios.append(v / med)
        print(f'  q={q:3d}  VFA {v:7.1f} mg/L   ratio {v / med:.3f}')
    lo, hi = min(ratios), max(ratios)
    print(f'  ratio range {lo:.3f}-{hi:.3f}')
    print()
    print(f'peak_load, q={FEEDS[-1]}, {UPPER_DAYS} days, termination off')
    for mult in (1.0, 1.3):
        tr = run('peak_load', FEEDS[-1], mult, UPPER_DAYS, no_termination=True)
        print(f'  strength {mult:.1f}: day30 {tr[29]:7.0f}  day60 {tr[59]:7.0f}'
              f'  last {tr[-1]:7.0f}  max {max(tr):7.0f} mg/L')
    print()
    print('mapped thresholds (300 mg/L / ratio) and plant exceedance')
    for r in (hi, lo):
        t = SOFT_MG / r
        k = sum(1 for v in rec if v > t)
        print(f'  ratio {r:.3f} -> {t:6.0f} mg/L   {k:4d}/{n} '
              f'= {100.0 * k / n:.1f}%')


if __name__ == '__main__':
    main()
