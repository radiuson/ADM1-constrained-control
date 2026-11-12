#!/usr/bin/env python
"""Simulated buffering capacity against the plant's recorded alkalinity.

The plant logs a titration alkalinity per digester in mg CaCO3/L.  The
simulator's Nordmann titration reports TAC in the same unit, so the two are
compared directly, at the loading the plant itself runs and across the
loading axis.  Feed rate is held at the scenario's own value, so the
comparison is of the model's buffering at a given load and not of a
controller.

Usage:  ADM1_PAPER_DATA=... python scripts/alkalinity_comparison.py [plant_csv]
"""
from __future__ import annotations

import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.adm1_gym_env_std import ADM1Env_Std, STD_SCENARIOS

PLANT_CSV = os.path.join(os.environ.get('MUSCATINE_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'muscatine')), 'LAB', 'LABS-raw.csv')
SETTLE_STEPS = 30          # one-day control steps held at constant feed before reading
FEEDS = (41, 70, 100, 130, 159)   # across the actuator range, m3/d
EVAL_SCENARIOS = [s for s in STD_SCENARIOS if s != 'acidified_recovery']


def plant_alkalinity(path: str) -> list[float]:
    """Every recorded digester-day alkalinity, both digesters, as measured."""
    out = []
    with open(path, encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            for key in ('Dig1-alk_mgL', 'Dig2-alk_mgL'):
                raw = (row.get(key) or '').strip()
                if not raw:
                    continue
                try:
                    out.append(float(raw))
                except ValueError:
                    pass
    return out


def simulated(scenario: str, q_ad: float = 100.0) -> dict:
    """TAC, FOS/TAC and VFA after settling at a mid feed rate.

    Scenario loading is set by the influent multiplier, so the feed rate is
    held at one value across the set and the reading differs by loading
    alone.
    """
    import numpy as np
    env = ADM1Env_Std(scenario_name=scenario, enable_disturbances=False,
                      step_size=1.0)
    env.reset(seed=7)
    act = np.array([q_ad, 1.0], dtype=np.float32)
    for _ in range(SETTLE_STEPS):
        _, _, term, trunc, _ = env.step(act)
        if term or trunc:
            break
    s = env.current_state
    fos, tac = env._simulated_nordmann(s)
    return {
        'scenario': scenario,
        'tac': tac,
        'fos_tac': fos / max(tac, 1e-9),
        'vfa': env._calculate_total_vfa(s),
        'pH': s.get('pH', float('nan')),
    }


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else PLANT_CSV
    rec = plant_alkalinity(path)
    rec.sort()
    n = len(rec)
    print(f'plant alkalinity: {n} digester-day observations, mg CaCO3/L')
    print(f'  median {st.median(rec):.0f}   p5 {rec[n // 20]:.0f} '
          f'  p95 {rec[-n // 20]:.0f}   min {rec[0]:.0f}   max {rec[-1]:.0f}')
    print()
    print(f'{"scenario":16} {"TAC":>9} {"FOS/TAC":>9} {"VFA":>8} {"pH":>6}'
          f'  {"TAC/plant median":>17}')
    med = st.median(rec)
    sims = []
    for sc in EVAL_SCENARIOS:
        for q in FEEDS:
            r = simulated(sc, q)
            r['q_ad'] = q
            sims.append(r)
            print(f'{r["scenario"]:16}{q:4d} {r["tac"]:9.0f} {r["fos_tac"]:9.3f} '
                  f'{r["vfa"]:8.3f} {r["pH"]:6.2f}  {r["tac"] / med:17.2f}')
    print()
    tacs = sorted(r['tac'] for r in sims)
    print(f'simulated TAC across the scenario set: median {st.median(tacs):.0f}, '
          f'range {tacs[0]:.0f}-{tacs[-1]:.0f} mg CaCO3/L')
    print(f'ratio of simulated median to plant median: '
          f'{st.median(tacs) / med:.2f}')
    pl = [r['tac'] for r in sims if r['scenario'] == 'plant_load']
    print(f'plant_load across the feed range: TAC {min(pl):.0f}-{max(pl):.0f}, '
          f'ratio {min(pl) / med:.2f}-{max(pl) / med:.2f}')
    print(f'simulated FOS/TAC median over the set: '
          f'{st.median(r["fos_tac"] for r in sims):.3f}')


if __name__ == '__main__':
    main()
