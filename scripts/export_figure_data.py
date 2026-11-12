#!/usr/bin/env python3
"""Export the per-run tables behind the main figures as CSV.

One row per evaluation; nothing is aggregated, so the figure author can
recompute every summary and interval.  The loaders and filters are the ones
build_tables.py applies, imported from it, so the exported sets are the sets
the manuscript's tables are computed from.

    ADM1_PAPER_DATA=paper_data python3 scripts/export_figure_data.py <outdir> [expA_dir]
"""
from __future__ import annotations

import csv
import glob
import json
import os
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_tables as BT  # noqa: E402

T = BT.EPISODE_STEPS
S = 7
FULL = S * T
FAMILY = {'dual': 'dual-loop PI', 'pi': 'single-loop PI',
          'const': 'constant feed', 'rule': 'rule-based'}
BUDGET = {0.6: 1.0, 1.8: 3.0, 3.0: 5.0, 6.0: 10.0}
WEIGHT = {'lw0p5': 0.5, 'lw1': 1.0, 'lw2': 2.0, 'lw5': 5.0}


def write(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in fields})
    print(f'{os.path.basename(path):28s} {len(rows)} rows')


def step_candidates(v):
    """Completed-step counts n <= 420 that make the stored rate k/n exact."""
    return [n for n in range(1, FULL + 1)
            if round(v * n / 100) >= 1 and abs(v * n / 100 - round(v * n / 100)) < 1e-6]


def envelope_of(B):
    pts = sorted((b['viol'], b['ch4'], b['_id']) for b in B)
    # non-dominated points only (the highest flow at each rate), as in
    # build_tables.envelope
    E, best = [], -1.0
    for v, c, i in pts:
        if c > best:
            if E and E[-1][0] == v:
                E[-1] = (v, c, i)
            else:
                E.append((v, c, i))
            best = c
    return E


# --------------------------------------------------------------- conventional
def conventional():
    rows, B = [], []
    files = sorted(glob.glob(f'{BT.DATA}/evbase/*.json')) + \
        sorted(glob.glob(f'{BT.DUAL}/*.json'))
    for f in files:
        b = json.load(open(f))
        eid = ('dual/' if f.startswith(BT.DUAL) else 'evbase/') + \
            os.path.basename(f)[:-5]
        kind = b['kind']
        mult = BT._mult(b)
        # the configuration identity: name plus whatever gains the record holds
        if kind == 'pi' and 'Kc' in b:
            cid = f"pi_sp{re.search(r'sp([0-9.]+)', b['name']).group(1)}" \
                  f"_Kc{b['Kc']:g}_ti{b['tau_i']:g}_m{mult:g}"
        elif kind == 'pi':
            cid = b['name'] + '#' + eid   # earliest records store no gains
        else:
            cid = b['name']
        if 'steps' in b:
            n, src = b['steps'], 'stored'
        elif b['viol'] == 0:
            # a hard-band breach is also a soft violation, so a run with no
            # violation cannot have terminated
            n, src = FULL, 'inferred_zero_violation'
        else:
            c = step_candidates(b['viol'])
            n, src = (c[0], 'inferred_unique') if len(c) == 1 else (None, 'ambiguous')
        cand = '' if src != 'ambiguous' else ';'.join(map(str, step_candidates(b['viol'])))
        k = None if n is None else int(round(b['viol'] * n / 100))
        r = {'evaluation_id': eid, 'config_id': cid, 'family': FAMILY[kind],
             'name': b['name'], 'feed_multiplier': mult,
             'Kc': b.get('Kc'), 'tau_i': b.get('tau_i'),
             'gains_recovered': b.get('gains_recovered'),
             'methane_mean': b['ch4'], 'vfa_violation_rate': b['viol'],
             'completed_steps': n, 'violating_steps': k,
             'completed_steps_source': src, 'completed_steps_candidates': cand,
             'terminated_early': None if n is None else int(n < FULL),
             'static_reference': int(kind != 'dual'),
             'in_reported_set': int(mult <= BT.MULT_MAX)}
        rows.append(r)
        if r['in_reported_set']:
            B.append(dict(b, _id=eid))
    # evaluations of the same configuration
    grp = Counter(r['config_id'] for r in rows if r['in_reported_set'])
    for r in rows:
        r['evaluations_of_config'] = grp.get(r['config_id'], 1)
    Ef = envelope_of(B)
    Es = envelope_of([b for b in B if b['kind'] != 'dual'])
    on_f = {i for *_, i in Ef}
    on_s = {i for *_, i in Es}
    for r in rows:
        r['on_full_envelope'] = int(r['evaluation_id'] in on_f)
        r['on_static_envelope'] = int(r['evaluation_id'] in on_s)
    return rows, B, Ef, Es


# --------------------------------------------------------------- learned runs
def counts(ps):
    n = sum(v['steps'] for v in ps.values())
    k = int(round(sum(v['steps'] * v['viol'] / 100.0 for v in ps.values())))
    return n, k


def absorbing(n, k):
    return 100.0 * (k + (FULL - n)) / FULL


def rl_rows():
    rows = []
    # reward-penalty: the main design is exactly build_tables' filter
    main_ids = set()
    R = defaultdict(list)
    for f in glob.glob(f'{BT.DATA}/evres/*.json'):
        d = json.load(open(f))
        d['_f'] = 'evres/' + os.path.basename(f)[:-5]
        if d.get('steps') != 150000 or '_h5_' in d.get('model', ''):
            continue
        if d.get('w') not in BT.WEIGHTS:
            continue
        a = (d.get('algo') or 'sac').lower()
        want = BT.CORRECTED.get(a)
        if want and not d.get('model', '').startswith(want):
            continue
        R[a].append(d)
    for a in R:
        seen = {}
        for d in sorted(R[a], key=lambda x: x.get('model', '')):
            seen.setdefault((d.get('w'), d.get('seed')), d)
        main_ids |= {d['_f'] for d in seen.values()}
    label = {k: lab for lab, k, _ in BT.REWARD_ROWS}

    def why_not(d):
        a = (d.get('algo') or 'sac').lower()
        if d.get('steps') != 150000:
            return f"training_steps_{d.get('steps')}"
        if '_h5_' in d.get('model', ''):
            return 'h5_variant'
        if d.get('w') not in BT.WEIGHTS:
            return 'weight_outside_grid'
        want = BT.CORRECTED.get(a)
        if want and not d.get('model', '').startswith(want):
            return 'superseded_settings'
        return 'duplicate_cell'

    for sub, tag in (('evres', None), ('evres_grad', 'gradient_step_check'),
                     ('evres_hp', 'hyperparameter_check')):
        for f in sorted(glob.glob(f'{BT.DATA}/{sub}/*.json')):
            d = json.load(open(f))
            fid = f'{sub}/' + os.path.basename(f)[:-5]
            n, k = counts(d['per_scenario'])
            a = (d.get('algo') or 'sac').lower()
            if sub == 'evres':
                ds = 'main' if fid in main_ids else 'extra:' + why_not(dict(d, _f=fid))
            else:
                ds = 'extra:' + tag
            w = d.get('w')
            rows.append({
                'run_id': fid, 'algorithm': label.get(a, a),
                'formulation': 'reward-penalty', 'seed': d['seed'],
                'penalty_weight': WEIGHT.get(w, w), 'cost_limit': None,
                'nominal_budget_pct': None, 'training_steps': d.get('steps'),
                'methane_mean': d['ch4'], 'vfa_violation_rate': d['viol'],
                'failure_adjusted_rate': absorbing(n, k),
                'completed_steps': n, 'violating_steps': k,
                'terminated_early': int(n < FULL), 'dataset': ds,
                'model_path': d.get('model')})
    for f in sorted(glob.glob(f'{BT.DATA}/evcmdp/*.json')):
        d = json.load(open(f))
        n, k = counts(d['per_scenario'])
        c = d['cost_limit']
        rows.append({
            'run_id': 'evcmdp/' + os.path.basename(f)[:-5],
            'algorithm': d['algo'], 'formulation': 'constrained',
            'seed': d['seed'], 'penalty_weight': None, 'cost_limit': c,
            'nominal_budget_pct': BUDGET.get(c), 'training_steps': 150000,
            'methane_mean': d['ch4'], 'vfa_violation_rate': d['viol'],
            'failure_adjusted_rate': absorbing(n, k),
            'completed_steps': n, 'violating_steps': k,
            'terminated_early': int(n < FULL),
            'dataset': 'main' if d['algo'] != 'NaturalPG' else 'extra:not_in_design',
            'model_path': d.get('run_dir')})
    return rows


# ---------------------------------------------------------------- downsample
def downsample(B, rl, out, draws=200, size=330, seed=20261110):
    """Reference-set size against composition, as in Section 6.2."""
    rng = random.Random(seed)
    runs = defaultdict(list)
    for r in rl:
        if r['dataset'] == 'main' and r['algorithm'] in ('PPO', 'TRPO'):
            runs[r['algorithm']].append(r)
    pools = {'full_set': B, 'dual_loop_only': [b for b in B if b['kind'] == 'dual']}
    rows, ids = [], []
    for ref, pool in pools.items():
        for i in range(draws):
            pick = rng.sample(pool, size)
            E = envelope_of(pick)
            xs = [e[0] for e in E]
            ys = [e[1] for e in E]
            ids.append({'draw_id': f'{ref}-{i:03d}', 'reference_type': ref,
                        'evaluation_ids': [b['_id'] for b in pick]})
            for a, R in runs.items():
                D = [r['methane_mean'] - float(np.interp(r['vfa_violation_rate'], xs, ys))
                     for r in R if xs[0] <= r['vfa_violation_rate'] <= xs[-1]]
                rows.append({'draw_id': f'{ref}-{i:03d}', 'reference_type': ref,
                             'sample_size': size, 'algorithm': a,
                             'scored_runs': len(D), 'median_D': st.median(D)})
    json.dump(ids, open(os.path.join(out, 'downsample_draw_ids.json'), 'w'))
    return rows


# ---------------------------------------------------------------- experiment A
def expA(d):
    """Every planned (variant, algo, limit, seed) cell, with its status."""
    algos, lims = ('CPO', 'PPOLag', 'SACLag', 'TD3Lag'), (0.6, 1.8, 3.0, 6.0)
    # three paired seeds for every family; CPO and PPOLag later extended to ten
    seeds_of = {'CPO': range(1, 11), 'PPOLag': range(1, 11), 'SACLag': (1, 2, 3), 'TD3Lag': (1, 2, 3)}
    arms = {'aligned': os.path.join(d, 'methane_only'),
            'original': os.path.join(d, 'reward_control')}
    scored = {}
    for var, p in arms.items():
        f = os.path.join(p, 'scored.json')
        for r in (json.load(open(f)) if os.path.exists(f) else []):
            scored[(var, r['algo'], round(r['cost_limit'], 2), r['seed'])] = r
    rows = []
    for var, p in arms.items():
        for a in algos:
            for c in lims:
                for s in seeds_of[a]:
                    r = scored.get((var, a, c, s))
                    row = {'run_id': f'{var}/{a}_c{c}_s{s}', 'reward_variant': var,
                           'pair_id': f'{a}_c{c}_s{s}', 'algorithm': a,
                           'formulation': 'constrained', 'seed': s,
                           'penalty_weight': None, 'cost_limit': c,
                           'nominal_budget_pct': BUDGET[c], 'training_steps': 150000}
                    if r:
                        ps = r['per_scenario']
                        n = sum(v['steps'] for v in ps.values())
                        k = sum(v['violating'] for v in ps.values())
                        row.update(run_status='completed', methane_mean=r['ch4'],
                                   vfa_violation_rate=r['viol'],
                                   failure_adjusted_rate=absorbing(n, k),
                                   completed_steps=n, violating_steps=k,
                                   terminated_early=int(n < FULL),
                                   model_path=os.path.relpath(r['run_dir'], d))
                    else:
                        row['run_status'] = 'not_finished'
                    rows.append(row)
    # runs that shared a log directory with another run at launch: their
    # checkpoints and configs overwrote one another, so they are kept here as
    # discarded and their cells are re-run
    for f in sorted(glob.glob(os.path.join(arms['aligned'], '_collided', '*_meta.json'))):
        m = json.load(open(f))
        c = float(m['cost_limit'])
        rows.append({'run_id': 'aligned/_collided/' + os.path.basename(f)[:-10],
                     'reward_variant': 'aligned',
                     'pair_id': f"{m['algo']}_c{c}_s{m['seed']}",
                     'run_status': 'discarded_log_collision',
                     'algorithm': m['algo'], 'formulation': 'constrained',
                     'seed': m['seed'], 'cost_limit': c,
                     'nominal_budget_pct': BUDGET.get(c), 'training_steps': 150000})
    # runs that failed at start-up on a CUDA out-of-memory error and were
    # launched again in the same cell
    for f in sorted(glob.glob(os.path.join(arms['original'], '_crashed', '*_c*_s*.log'))):
        a, c, s = os.path.basename(f)[:-4].split('_')
        c, s = float(c[1:]), int(s[1:])
        rows.append({'run_id': 'original/_crashed/' + os.path.basename(f)[:-4],
                     'reward_variant': 'original', 'pair_id': f'{a}_c{c}_s{s}',
                     'run_status': 'discarded_startup_crash', 'algorithm': a,
                     'formulation': 'constrained', 'seed': s, 'cost_limit': c,
                     'nominal_budget_pct': BUDGET.get(c), 'training_steps': 150000})
    return rows


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    conv, B, Ef, Es = conventional()
    write(os.path.join(out, 'conventional_runs.csv'), conv,
          ['evaluation_id', 'config_id', 'family', 'name', 'feed_multiplier',
           'Kc', 'tau_i', 'gains_recovered', 'methane_mean', 'vfa_violation_rate',
           'completed_steps', 'violating_steps', 'completed_steps_source',
           'completed_steps_candidates', 'terminated_early', 'static_reference',
           'in_reported_set', 'evaluations_of_config', 'on_full_envelope',
           'on_static_envelope'])
    # The running maximum keeps every evaluation that raises the best flow so
    # far, including several at the same violation rate (441 of the 558 sit at
    # zero).  Only the highest at each rate is non-dominated; interpolating
    # through the lower ones pulls the reference down to the left of a stack.
    for nm, E in (('envelope_full.csv', Ef), ('envelope_static.csv', Es)):
        top = {}
        for v, c, i in E:
            top[v] = max(top.get(v, -1.0), c)
        write(os.path.join(out, nm),
              [{'vfa_violation_rate': v, 'methane_mean': c, 'evaluation_id': i,
                'non_dominated': int(c == top[v])} for v, c, i in E],
              ['vfa_violation_rate', 'methane_mean', 'evaluation_id', 'non_dominated'])
    rl = rl_rows()
    f_rl = ['run_id', 'algorithm', 'formulation', 'seed', 'penalty_weight',
            'cost_limit', 'nominal_budget_pct', 'training_steps', 'methane_mean',
            'vfa_violation_rate', 'failure_adjusted_rate', 'completed_steps',
            'violating_steps', 'terminated_early', 'dataset', 'model_path']
    write(os.path.join(out, 'rl_original_runs.csv'), rl, f_rl)
    write(os.path.join(out, 'downsample_draws.csv'), downsample(B, rl, out),
          ['draw_id', 'reference_type', 'sample_size', 'algorithm',
           'scored_runs', 'median_D'])
    if len(sys.argv) > 2:
        # D against the full envelope at the completed-step rate, empty beyond
        # the envelope's support, as for the reported runs
        ex, ey = [e[0] for e in Ef], [e[1] for e in Ef]
        A = expA(sys.argv[2])
        for r in A:
            v = r.get('vfa_violation_rate')
            r['D'] = (r['methane_mean'] - float(np.interp(v, ex, ey))
                      if v is not None and v <= ex[-1] else None)
        write(os.path.join(out, 'experiment_A_runs.csv'), A,
              ['run_id', 'reward_variant', 'pair_id', 'run_status'] + f_rl[1:-2]
              + ['D', 'model_path'])


if __name__ == '__main__':
    main()
