#!/usr/bin/env python3
"""Simulate the seven scenarios under fixed reference actions, for the
scenario-profile figure.  Two action settings bound what a controller faces:
the maximum corner (the probe the text describes) and the plant's own
mid-range setting.  Input to plot_env_figures.py.

    python3 scripts/scenario_traces.py
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'figures', 'scenario_traces.json')
sys.path.insert(0, ROOT)
from env.adm1_gym_env_std import ADM1Env_Std as E

SC = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
      'fog_surge', 'elevated_start']
MG = 1000 / 1.0667
Q_LO, Q_HI = 41.0, 159.0


def trace(scen, q, mult, steps=60):
    env = E(scen, obs_mode='scada', step_size=1.0)
    env.reset(seed=7)
    vfa, ch4, ph = [], [], []
    for _ in range(steps):
        _, _, te, tr, _ = env.step(np.array([q, mult], dtype=np.float32))
        st = env.solver.state
        vfa.append(sum(st[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac')) * MG)
        ch4.append(env.solver.q_ch4)
        ph.append(-np.log10(max(st['S_H_ion'], 1e-14)))
        if te or tr:
            break
    return {'vfa': vfa, 'ch4': ch4, 'ph': ph, 'steps': len(vfa)}


out = {}
for s in SC:
    out[s] = {
        'max': trace(s, Q_HI, 1.3),
        'mid': trace(s, (Q_LO + Q_HI) / 2, 1.0),
    }
    print(s, 'max peak VFA %.0f' % max(out[s]['max']['vfa']),
          'steps', out[s]['max']['steps'],
          '| mid peak %.0f' % max(out[s]['mid']['vfa']), flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, 'w'))
print('wrote', OUT)
