#!/usr/bin/env python3
"""Score the methane-only arm against the same conventional envelope.

Experiment A retrains a representative set with the feed-rate and
flow-smoothness terms of the reward zeroed and the safety channel unchanged.
This evaluates those policies under the reported protocol and puts them on
the same envelope as the reported runs, so the two arms differ only in the
part of the objective the conventional envelope does not share.

    python3 \
        scripts/score_methane_only.py <runs-dir>
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

SC = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
      'fog_surge', 'elevated_start']
VFA_SOFT = 0.320
STEPS = 60
SEED = 7


def evaluate(run_dir):
    """One policy over the scenario set, exactly as eval_cmdp.py scores it."""
    import torch
    torch.set_num_threads(1)
    from omnisafe.models.actor import ActorBuilder
    from omnisafe.common.normalizer import Normalizer
    from env.adm1_gym_env_std import ADM1Env_Std as E
    from env.normalized_wrapper import NormalizedADM1Env
    from training.omnisafe_env import CMDP_REWARD

    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    ck = sorted(glob.glob(os.path.join(run_dir, 'torch_save', 'epoch-*.pt')),
                key=lambda q: int(q.split('epoch-')[1].split('.pt')[0]))
    if not ck:
        return None
    params = torch.load(ck[-1], map_location='cpu')
    probe = NormalizedADM1Env(E('nominal', reward_config=CMDP_REWARD,
                                obs_mode='scada', step_size=1.0))
    mc = cfg['model_cfgs']
    actor = ActorBuilder(
        obs_space=probe.observation_space, act_space=probe.action_space,
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

    V, C, PS = [], [], {}
    for s in SC:
        n0 = len(V)
        env = NormalizedADM1Env(E(s, reward_config=CMDP_REWARD,
                                  obs_mode='scada', step_size=1.0))
        obs, _ = env.reset(seed=SEED)
        for _ in range(STEPS):
            x = torch.as_tensor(obs, dtype=torch.float32)
            if norm is not None:
                x = norm.normalize(x)
            with torch.no_grad():
                act = actor.predict(x, deterministic=True).numpy()
            obs, _, term, trunc, _ = env.step(act)
            stt = env.env.solver.state
            V.append(sum(stt[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac')))
            C.append(env.env.solver.q_ch4)
            if term or trunc:
                break
        PS[s] = {'steps': len(V) - n0,
                 'violating': int(sum(1 for x in V[n0:] if x > VFA_SOFT))}
    V = np.array(V)
    name = os.path.basename(os.path.dirname(run_dir.rstrip('/')))
    return {'run_dir': run_dir, 'algo': cfg.get('algo') or name.split('-')[0],
            'viol': float(100 * np.mean(V > VFA_SOFT)),
            'ch4': float(np.mean(C)), 'steps': int(len(V)), 'per_scenario': PS,
            'seed': cfg.get('seed'),
            'cost_limit': float(cfg['algo_cfgs'].get('cost_limit')
                                or cfg.get('lagrange_cfgs', {}).get('cost_limit', 0))}


def main():
    runs = sys.argv[1]
    # each job writes into its own directory, so the run dirs sit one or two
    # levels down; quarantined runs are skipped
    dirs = sorted(d for d in glob.glob(os.path.join(runs, '**', 'seed-*'),
                                       recursive=True)
                  if '_collided' not in d and '_crashed' not in d)
    # only runs that reached the final epoch; a run still training has
    # intermediate checkpoints that must not be scored as if finished
    dirs = [d for d in dirs
            if os.path.exists(os.path.join(d, 'torch_save', 'epoch-75.pt'))]
    print(f'{len(dirs)} methane-only runs to score')
    recs = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for r in ex.map(evaluate, dirs):
            if r:
                recs.append(r)
    out = os.path.join(runs, 'scored.json')
    json.dump(recs, open(out, 'w'))
    print(f'wrote {out}  ({len(recs)} scored)')


if __name__ == '__main__':
    main()
