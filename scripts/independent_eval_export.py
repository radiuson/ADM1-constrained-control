#!/usr/bin/env python3
"""Flat tables of the independent-disturbance evaluation.

    python3 scripts/independent_eval_export.py <outdir> <reported_envelope.csv>

Writes into <outdir>:
  test_runs_learned.csv     one row per learned run (195): reported and test metrics, D against the
                            oracle test envelope and against the five validation-selected tunings
  test_runs_conventional_selected.csv   the five validation-selected tunings, validation and test
  test_per_scenario.csv     long format: controller x test scenario (steps, violating steps, methane)
  g1_onpolicy_46.csv        the 46 on-policy runs with zero violations in the reported evaluation
"""
import csv
import json
import os
import sys

import numpy as np

ON = {'CPO', 'PCPO', 'CUP', 'PPOLag', 'TRPOLag', 'TRPOPID', 'CPPOPID', 'FOCOPS', 'OnCRPO', 'P3O'}
STEPS = 60


def pool(per, ids):
    n = sum(per[i]['steps'] for i in ids)
    k = sum(per[i]['viol'] for i in ids)
    full = STEPS * len(ids)
    return {'completed_steps': n, 'violating_steps': k,
            'rate_completed': 100 * k / n, 'rate_failure_adjusted': 100 * (k + full - n) / full,
            'methane_mean': sum(per[i]['ch4_sum'] for i in ids) / n,
            'zero_violation_scenarios': sum(per[i]['viol'] == 0 and per[i]['steps'] == STEPS for i in ids),
            'terminated_scenarios': sum(per[i]['steps'] < STEPS for i in ids)}


def w(path, rows):
    with open(path, 'w', newline='') as f:
        x = csv.DictWriter(f, fieldnames=list(rows[0]))
        x.writeheader()
        x.writerows(rows)


def main():
    d, rep = sys.argv[1], sys.argv[2]
    S = json.load(open(os.path.join(d, 'scenarios.json')))
    tid, vid = [s['id'] for s in S['test']], [s['id'] for s in S['validation']]
    sel = json.load(open(os.path.join(d, 'selected.json')))
    TC = {r['name']: r for r in json.load(open(os.path.join(d, 'test_conventional_selected.json')))}
    R = json.load(open(os.path.join(d, 'test_rl.json')))
    O = sorted((float(r['test_rate']), float(r['test_methane']))
               for r in csv.DictReader(open(os.path.join(d, 'oracle_envelope.csv'))))
    ox, oy = zip(*O)
    Er = sorted((float(r['vfa_violation_rate']), float(r['methane_mean'])) for r in csv.DictReader(open(rep)))
    rx, ry = zip(*Er)
    conv = []
    for b, v in sel.items():
        t = pool(TC[v['name']]['per_scenario'], tid)
        conv.append({'budget_pct': b, 'name': v['name'], 'kind': v['kind'],
                     'val_rate': v['validation']['fa_rate'], 'val_methane': v['validation']['methane'],
                     **{'test_' + k: x for k, x in t.items()},
                     'test_meets_budget': int(t['rate_failure_adjusted'] <= float(b) + 1e-9)})
    w(os.path.join(d, 'test_runs_conventional_selected.csv'), conv)
    cx = [c['test_rate_completed'] for c in sorted(conv, key=lambda c: c['test_rate_completed'])]
    cy = [c['test_methane_mean'] for c in sorted(conv, key=lambda c: c['test_rate_completed'])]
    rows, long = [], []
    for r in R:
        t = pool(r['per_scenario'], tid)
        v = t['rate_completed']
        rows.append({'algorithm': r['algorithm'], 'class': 'on-policy' if r['algorithm'] in ON else 'off-policy',
                     'seed': r['seed'], 'cost_limit': r['cost_limit'], 'G1': int(r['G1']), 'G2': int(r['G2']),
                     'reported_rate': r['reported_viol'], 'reported_methane': r['reported_methane'],
                     'D_reported': r['reported_methane'] - float(np.interp(r['reported_viol'], rx, ry)),
                     **{'test_' + k: x for k, x in t.items()},
                     'D_test_oracle': t['methane_mean'] - float(np.interp(v, ox, oy)),
                     'D_test_selected': t['methane_mean'] - float(np.interp(v, cx, cy)),
                     'test_rate_below_lowest_selected': int(v < cx[0]),
                     'test_meets_1pct': int(t['rate_failure_adjusted'] <= 1 + 1e-9)})
        for i in tid:
            p = r['per_scenario'][i]
            long.append({'controller': f"{r['algorithm']}-seed{r['seed']}-d{r['cost_limit']}", 'type': 'learned', 'scenario': i,
                         'steps': p['steps'], 'violating_steps': p['viol'], 'methane_mean': p['ch4_sum'] / p['steps']})
    for c in conv:
        for i in tid:
            p = TC[c['name']]['per_scenario'][i]
            long.append({'controller': c['name'], 'type': f"conventional_selected_{c['budget_pct']}pct",
                         'scenario': i, 'steps': p['steps'], 'violating_steps': p['viol'],
                         'methane_mean': p['ch4_sum'] / p['steps']})
    w(os.path.join(d, 'test_runs_learned.csv'), rows)
    w(os.path.join(d, 'test_per_scenario.csv'), long)
    w(os.path.join(d, 'g1_onpolicy_46.csv'), [x for x in rows if x['G1'] and x['class'] == 'on-policy'])
    print(len(rows), 'learned,', len(conv), 'conventional,', len(long), 'per-scenario rows,',
          sum(x['G1'] and x['class'] == 'on-policy' for x in rows), 'G1 on-policy')


if __name__ == '__main__':
    main()
