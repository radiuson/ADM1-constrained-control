#!/usr/bin/env python3
"""Evaluation on disturbance profiles that neither training nor tuning saw.

Rules were fixed before any result; see results_independent/PREREGISTRATION.md.
Stages (each writes JSON into <outdir>; later stages read earlier ones):

  gen      draw 200 candidate scenarios (seed 20261111)
  screen   keep operable ones in draw order: 10 validation, 30 test
  val      every re-instantiable conventional configuration on validation
  select   best configuration per budget on validation
  testconv selected configurations on test
  testrl   learned groups G1 (zero violations in the reported evaluation)
           and G2 (all 14 families at d = 0.6) on test
  oracle   (optional) every conventional configuration on test

    python3 scripts/independent_eval.py <stage> <outdir> [workers]

The testrl stage needs the trained constrained policies, which are not
distributed; its output is stored in results_independent/test_rl.json.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def load_json(path):
    """Read path, or path + '.gz' where the record is stored compressed."""
    import gzip
    if os.path.exists(path):
        return json.load(open(path))
    return json.load(gzip.open(path + '.gz', 'rt'))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

MG = 1000 / 1.0667
VFA_SOFT = 0.320
STEPS, SEED = 60, 7
Q_LO, Q_HI = 41.0, 159.0
BUDGETS = (0, 1, 3, 5, 10)
# Written by scripts/export_figure_data.py.
FD = os.environ.get('FIGDATA', os.path.join(ROOT, 'figdata'))
SPIKE_VARS = {'lipid': ['X_li', 'S_fa'], 'carbohydrate': ['X_ch'],
              'particulate': ['X_ch', 'X_pr', 'X_li', 'X_xc']}


# ------------------------------------------------------------- environment
def make_env(spec, reward_config=None):
    from env.adm1_gym_env_std import ADM1Env_Std

    class HeldOut(ADM1Env_Std):
        def __init__(self):
            super().__init__('nominal', reward_config=reward_config,
                             obs_mode='scada', step_size=1.0)
            cfg = {'name': spec['id'], 'duration_days': 60,
                   'step_size_days': 0.01041667,
                   'initial_state': spec['init'],
                   'influent': {'type': 'default', 'noise_level': 0.0,
                                'multiplier': spec['mult']},
                   'temperature': 35.0, 'disturbances': []}
            if spec['event']:
                e = spec['event']
                cfg['disturbances'] = [{'time_days': float(e['start']),
                                        'type': 'influent_spike',
                                        'magnitude': e['mag'],
                                        'duration_hours': 24.0 * e['dur'],
                                        'affected_vars': SPIKE_VARS[e['kind']]}]
            self.scenario_manager.scenarios[spec['id']] = cfg
            self.scenario_name = spec['id']
            self.scenario_config = self.scenario_manager.load_scenario(spec['id'])
            self.scenario_duration = 60
            self.max_steps = 60
            self._offset = int(spec['start_day'])

        def _build_influent_dict(self, step=0):
            return super()._build_influent_dict(step + self._offset)

    return HeldOut()


def _vfa(st):
    return sum(st[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac'))


def run_conventional(make_ctrl, spec):
    env = make_env(spec)
    env.reset(seed=SEED)
    ctrl = make_ctrl()
    ctrl.reset()
    q, m = Q_HI, 1.0
    n = k = 0
    ch4 = 0.0
    for _ in range(STEPS):
        st = env.solver.state
        out = ctrl.act(_vfa(st) * MG, st['S_IC'],
                       -np.log10(max(st['S_H_ion'], 1e-14)),
                       env.solver.q_ch4, q, m)
        q, m = out if isinstance(out, tuple) else (out, m)
        _, _, te, tr, _ = env.step(np.array([q, m], dtype=np.float32))
        n += 1
        k += int(_vfa(env.solver.state) > VFA_SOFT)
        ch4 += env.solver.q_ch4
        if te or tr:
            break
    return {'steps': n, 'viol': k, 'ch4_sum': ch4}


# ------------------------------------------------------------- controllers
def conventional_specs():
    """(name, kind, params) for every re-instantiable configuration."""
    rows = [r for r in csv.DictReader(open(os.path.join(FD, 'conventional_runs.csv')))
            if r['in_reported_set'] == '1']
    specs, seen = [], set()
    for r in rows:
        nm = r['name']
        if nm in seen:
            continue
        if r['family'] == 'dual-loop PI':
            p = dict(re.findall(r'_(sp|Kc|ti|Km|b)([0-9.]+)', nm))
            specs.append((nm, 'dual', {k: float(v) for k, v in p.items()}))
        elif r['family'] == 'constant feed':
            q = float(re.search(r'q([0-9.]+)', nm).group(1))
            specs.append((nm, 'const', {'q': q, 'mult': float(r['feed_multiplier'] or 1.3)}))
        elif r['family'] == 'rule-based':
            lvl = float(re.search(r'rule_([0-9.]+)', nm).group(1))
            specs.append((nm, 'rule', {'lvl': lvl, 'mult': float(r['feed_multiplier'] or 1.3)}))
        elif r['family'] == 'single-loop PI' and r['gains_recovered'] == 'True':
            sp = float(re.search(r'sp([0-9.]+)', nm).group(1))
            nm2 = f'{nm}_Kc{float(r["Kc"]):g}_ti{float(r["tau_i"]):g}'
            if nm2 in seen:
                continue
            seen.add(nm2)
            specs.append((nm2, 'pi', {'sp': sp, 'Kc': float(r['Kc']), 'ti': float(r['tau_i']),
                                      'mult': float(r['feed_multiplier'] or 1.3)}))
            continue
        else:
            continue
        seen.add(nm)
    from threshold_sensitivity import PI_SP, PI_GAINS, PI_MULT
    for sp in PI_SP:
        for kc, ti in PI_GAINS:
            for m in PI_MULT:
                nm = f'pi_sp{sp:g}_m{m:g}_Kc{kc:g}_ti{ti:g}'
                if nm not in seen:
                    seen.add(nm)
                    specs.append((nm, 'pi', {'sp': sp, 'Kc': kc, 'ti': ti, 'mult': m}))
    return specs


def factory(kind, p):
    from training.baselines import DualLoopPI, PIFeed, ConstantFeed, RuleBased
    if kind == 'dual':
        return lambda: DualLoopPI(p['sp'], Q_LO, Q_HI, Kc=p['Kc'], tau_i=p['ti'],
                                  Km=p['Km'], band=p['b'])
    if kind == 'pi':
        return lambda: PIFeed(p['sp'], Q_LO, Q_HI, Kc=p['Kc'], tau_i=p['ti'], mult=p['mult'])
    if kind == 'const':
        return lambda: ConstantFeed(p['q'], mult=p['mult'])
    return lambda: RuleBased(p['lvl'], Q_LO, Q_HI, mult=p['mult'])


def conv_job(args):
    (name, kind, p), scen = args
    return {'name': name, 'kind': kind,
            'per_scenario': {s['id']: run_conventional(factory(kind, p), s) for s in scen}}


def rl_job(args):
    run_dir, meta, scen = args
    import torch
    torch.set_num_threads(1)
    from omnisafe.models.actor import ActorBuilder
    from omnisafe.common.normalizer import Normalizer
    from env.adm1_gym_env_std import ADM1Env_Std as E
    from env.normalized_wrapper import NormalizedADM1Env
    from training.omnisafe_env import CMDP_REWARD
    d = os.path.join(ROOT, run_dir)
    cfg = json.load(open(os.path.join(d, 'config.json')))
    ck = sorted(glob.glob(os.path.join(d, 'torch_save', 'epoch-*.pt')),
                key=lambda q: int(q.split('epoch-')[1].split('.pt')[0]))
    params = torch.load(ck[-1], map_location='cpu')
    probe = NormalizedADM1Env(E('nominal', reward_config=CMDP_REWARD,
                                obs_mode='scada', step_size=1.0))
    mc = cfg['model_cfgs']
    actor = ActorBuilder(obs_space=probe.observation_space, act_space=probe.action_space,
                         hidden_sizes=mc['actor']['hidden_sizes'],
                         activation=mc['actor']['activation'],
                         weight_initialization_mode=mc['weight_initialization_mode'],
                         ).build_actor(mc['actor_type'])
    actor.load_state_dict(params['pi'])
    actor.eval()
    norm = None
    if 'obs_normalizer' in params:
        norm = Normalizer(shape=probe.observation_space.shape, clip=5)
        norm.load_state_dict(params['obs_normalizer'])
    per = {}
    for s in scen:
        env = NormalizedADM1Env(make_env(s, reward_config=CMDP_REWARD))
        obs, _ = env.reset(seed=SEED)
        n = k = 0
        ch4 = 0.0
        for _ in range(STEPS):
            x = torch.as_tensor(obs, dtype=torch.float32)
            if norm is not None:
                # frozen: the stored training statistics, not updated here
                # (Normalizer.normalize would keep accumulating them)
                x = torch.clamp((x - norm.mean) / norm.std, -5.0, 5.0)
            with torch.no_grad():
                act = actor.predict(x, deterministic=True).numpy()
            obs, _, te, tr, _ = env.step(act)
            n += 1
            k += int(_vfa(env.env.solver.state) > VFA_SOFT)
            ch4 += env.env.solver.q_ch4
            if te or tr:
                break
        per[s['id']] = {'steps': n, 'viol': k, 'ch4_sum': ch4}
    return dict(meta, run_dir=run_dir, per_scenario=per)


# ------------------------------------------------------------- pooling
def pool(per, ids):
    n = sum(per[i]['steps'] for i in ids)
    k = sum(per[i]['viol'] for i in ids)
    c = sum(per[i]['ch4_sum'] for i in ids)
    full = STEPS * len(ids)
    return {'viol_rate': 100 * k / n, 'fa_rate': 100 * (k + full - n) / full,
            'methane': c / n, 'steps': n, 'zero_scen': sum(per[i]['viol'] == 0 and
                                                         per[i]['steps'] == STEPS for i in ids),
            'terminated_scen': sum(per[i]['steps'] < STEPS for i in ids)}


# ------------------------------------------------------------- stages
def gen(out):
    rng = random.Random(20261111)
    C = []
    for i in range(200):
        init = rng.choices(['steady_state', 'prestressed', 'elevated'], [0.5, 0.25, 0.25])[0]
        ev = None
        if rng.random() >= 0.5:
            ev = {'mag': round(rng.uniform(1.5, 3.0), 3), 'start': rng.randint(5, 45),
                  'dur': round(rng.uniform(2, 7), 2),
                  'kind': rng.choice(['lipid', 'carbohydrate', 'particulate'])}
        C.append({'id': f'cand{i:03d}', 'start_day': rng.randint(60, 220),
                  'mult': round(rng.uniform(0.6, 2.4), 3), 'init': init, 'event': ev})
    json.dump(C, open(os.path.join(out, 'candidates.json'), 'w'), indent=1)
    print(f'{len(C)} candidates')


def screen_one(spec):
    import build_tables as BT
    _, E = BT.envelope()
    duals = [n for _, _, n in E if n.startswith('dual_')]
    tried = []
    for nm in duals:
        p = {k: float(v) for k, v in re.findall(r'_(sp|Kc|ti|Km|b)([0-9.]+)', nm)}
        r = run_conventional(factory('dual', p), spec)
        tried.append(nm)
        if r['steps'] == STEPS:
            return spec['id'], True, nm
    for q in (41, 80, 120, 159):
        r = run_conventional(factory('const', {'q': q, 'mult': 1.0}), spec)
        if r['steps'] == STEPS:
            return spec['id'], True, f'const_q{q}'
    return spec['id'], False, None


def screen(out, workers):
    C = json.load(open(os.path.join(out, 'candidates.json')))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(screen_one, C))
    ok = [c for c, (_, f, _) in zip(C, res) if f]
    log = [{'id': i, 'feasible': f, 'witness': w} for i, f, w in res]
    json.dump(log, open(os.path.join(out, 'screen_log.json'), 'w'), indent=1)
    assert len(ok) >= 40, f'only {len(ok)} feasible'
    json.dump({'validation': ok[:10], 'test': ok[10:40]},
              open(os.path.join(out, 'scenarios.json'), 'w'), indent=1)
    print(f'{len(ok)} of {len(C)} feasible; validation = first 10, test = next 30')


def sweep(out, workers, split, fname):
    S = json.load(open(os.path.join(out, 'scenarios.json')))[split]
    specs = conventional_specs()
    print(f'{len(specs)} conventional configurations x {len(S)} {split} scenarios', flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(conv_job, [(s, S) for s in specs], chunksize=4), 1):
            res.append(r)
            if i % 200 == 0:
                print(f'  {i}/{len(specs)}', flush=True)
    json.dump(res, open(os.path.join(out, fname), 'w'))


def select(out):
    S = json.load(open(os.path.join(out, 'scenarios.json')))
    ids = [s['id'] for s in S['validation']]
    V = load_json(os.path.join(out, 'val_conventional.json'))
    sel = {}
    for b in BUDGETS:
        best = None
        for r in V:
            p = pool(r['per_scenario'], ids)
            if p['fa_rate'] <= b + 1e-9 and (best is None or p['methane'] > best[1]['methane']):
                best = (r, p)
        sel[str(b)] = None if best is None else {'name': best[0]['name'], 'kind': best[0]['kind'],
                                                 'validation': best[1]}
        print(f'budget {b:>2} %: ' + ('none' if best is None else
              f"{best[0]['name']}  val methane {best[1]['methane']:.0f}  fa {best[1]['fa_rate']:.2f}"))
    json.dump(sel, open(os.path.join(out, 'selected.json'), 'w'), indent=1)


def testconv(out, workers):
    S = json.load(open(os.path.join(out, 'scenarios.json')))['test']
    sel = json.load(open(os.path.join(out, 'selected.json')))
    specs = {n: (n, k, p) for n, k, p in conventional_specs()}
    jobs = sorted({v['name'] for v in sel.values() if v})
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(conv_job, [(specs[n], S) for n in jobs]))
    json.dump(res, open(os.path.join(out, 'test_conventional_selected.json'), 'w'))
    ids = [s['id'] for s in S]
    for b, v in sel.items():
        if v:
            r = next(x for x in res if x['name'] == v['name'])
            print(b, v['name'], {k: round(x, 2) for k, x in pool(r['per_scenario'], ids).items()})


def testrl(out, workers):
    S = json.load(open(os.path.join(out, 'scenarios.json')))['test']
    rows = [r for r in csv.DictReader(open(os.path.join(FD, 'rl_original_runs.csv')))
            if r['dataset'] == 'main' and r['formulation'] == 'constrained']
    jobs = {}
    for r in rows:
        g1 = float(r['vfa_violation_rate']) == 0.0
        g2 = r['cost_limit'] == '0.6'
        if g1 or g2:
            jobs[r['model_path']] = {'algorithm': r['algorithm'], 'seed': r['seed'],
                                     'cost_limit': r['cost_limit'], 'G1': int(g1), 'G2': int(g2),
                                     'reported_methane': float(r['methane_mean']),
                                     'reported_viol': float(r['vfa_violation_rate'])}
    print(f'{len(jobs)} learned runs (G1 {sum(v["G1"] for v in jobs.values())}, '
          f'G2 {sum(v["G2"] for v in jobs.values())}) x {len(S)} test scenarios', flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(rl_job, [(d, m, S) for d, m in jobs.items()]), 1):
            res.append(r)
            if i % 20 == 0:
                print(f'  {i}/{len(jobs)}', flush=True)
    json.dump(res, open(os.path.join(out, 'test_rl.json'), 'w'))


def main():
    stage, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    os.makedirs(out, exist_ok=True)
    {'gen': lambda: gen(out), 'screen': lambda: screen(out, workers),
     'val': lambda: sweep(out, workers, 'validation', 'val_conventional.json'),
     'select': lambda: select(out), 'testconv': lambda: testconv(out, workers),
     'testrl': lambda: testrl(out, workers),
     'oracle': lambda: sweep(out, workers, 'test', 'test_conventional_all.json')}[stage]()


if __name__ == '__main__':
    main()
