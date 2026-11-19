#!/usr/bin/env python3
"""What the density of the conventional reference does to a learned result.

The conventional set is swept twice.  The static-multiplier families hold the
feed strength at a value chosen for the episode; the dual-loop family sets
both inputs at every control step, as a learned policy does.  Adding it takes
the conventional cloud from 330 evaluations to 2490 and the envelope from 79
points to 558, and it lifts the envelope most where the margin is expensive.

Scoring the same learned runs against the two envelopes therefore moves them
without changing anything about the runs.  Two families cross zero: PPO and
TRPO sit above the sparse envelope and below the dense one.

    python3 scripts/plot_density.py [outdir]

``outdir`` defaults to the manuscript directory.
"""
from __future__ import annotations

import collections
import glob
import importlib
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_tables as BT
from plot_figures import C_CONV, C_ENV, C_GRID, C_OFF, C_ON

PAPER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')

# The dual-loop family is what the dense sweep adds, so it carries its own
# tone; the static families are the sparse set and keep the conventional grey.
C_DUAL = '#CC79A7'          # Okabe-Ito reddish purple
C_SPARSE = '#6E6E6E'

LABEL = {'ppo': 'PPO', 'a2c': 'A2C', 'trpo': 'TRPO',
         'recurrentppo': 'RecurrentPPO', 'ars': 'ARS', 'sac': 'SAC',
         'tqc': 'TQC', 'ddpg': 'DDPG', 'td3': 'TD3', 'crossq': 'CrossQ'}
ON = ('ppo', 'a2c', 'trpo', 'recurrentppo')
OFF = ('sac', 'tqc', 'ddpg', 'td3', 'crossq')

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
import figstyle
figstyle.apply()


def envelopes():
    """The sparse and dense conventional sets, from the same loader."""
    importlib.reload(BT)
    dense_cloud, dense_env = BT.envelope()
    os.environ['ADM1_STATIC_ONLY'] = '1'
    importlib.reload(BT)
    sparse_cloud, sparse_env = BT.envelope()
    os.environ.pop('ADM1_STATIC_ONLY', None)
    importlib.reload(BT)
    return dense_cloud, dense_env, sparse_cloud, sparse_env


def learned_runs():
    """Reward-penalty runs under the filter build_tables applies."""
    R = collections.defaultdict(list)
    for f in glob.glob(f'{BT.DATA}/evres/*.json'):
        d = json.load(open(f))
        if d.get('steps') != 150000 or '_h5_' in d.get('model', ''):
            continue
        if d.get('w') not in BT.WEIGHTS:
            continue
        a = (d.get('algo') or 'sac').lower()
        want = BT.CORRECTED.get(a)
        if want and not d.get('model', '').startswith(want):
            continue
        R[a].append(d)
    for a in list(R):
        seen = {}
        for d in sorted(R[a], key=lambda x: x.get('model', '')):
            seen.setdefault((d.get('w'), d.get('seed')), d)
        R[a] = list(seen.values())
    return R


def panel_clouds(ax, dense_cloud, dense_env, sparse_env):
    """The two conventional sets and the envelope each one defines."""
    xs_d = [a for a, _, _ in dense_env]
    ys_d = [b for _, b, _ in dense_env]
    xs_s = [a for a, _, _ in sparse_env]
    ys_s = [b for _, b, _ in sparse_env]

    # The sparse set is a subset of the dense one, so the clouds are drawn
    # from the same records: the dual-loop points are what the dense sweep
    # adds and are the only ones tinted.
    stat = [(b['viol'], b['ch4']) for b in dense_cloud
            if b.get('kind') != 'dual']
    dual = [(b['viol'], b['ch4']) for b in dense_cloud
            if b.get('kind') == 'dual']
    ax.scatter([v for v, _ in dual], [c for _, c in dual], s=2.2,
               c=C_DUAL, alpha=0.30, lw=0, zorder=1,
               label=f'Dual-loop PI ({len(dual)})')
    ax.scatter([v for v, _ in stat], [c for _, c in stat], s=4.5,
               c=C_SPARSE, alpha=0.55, lw=0, zorder=2,
               label=f'Static-multiplier families ({len(stat)})')

    ax.plot(xs_s, ys_s, '-', color=C_SPARSE, lw=1.5, zorder=4,
            label=f'Sparse envelope ({len(sparse_env)} points)')
    ax.plot(xs_d, ys_d, '-', color=C_ENV, lw=1.6, zorder=5,
            label=f'Dense envelope ({len(dense_env)} points)')

    # The band between them is the reference's own uncertainty: scoring a
    # run against one envelope or the other moves it by this much.
    grid = np.linspace(0, 40, 400)
    lo = np.interp(grid, xs_s, ys_s)
    hi = np.interp(grid, xs_d, ys_d)
    ax.fill_between(grid, lo, hi, where=hi >= lo, color=C_ENV,
                    alpha=0.10, lw=0, zorder=3)

    gap = hi - lo
    i = int(np.argmax(gap))
    ax.annotate(f'up to {gap[i]:.0f} m$^3$/d apart',
                xy=(grid[i], (lo[i] + hi[i]) / 2),
                xytext=(grid[i] + 7.0, (lo[i] + hi[i]) / 2 - 330),
                fontsize=6.8, color=C_ENV, ha='left',
                arrowprops=dict(arrowstyle='-', lw=0.7, color=C_ENV))

    ax.set_xlim(-0.8, 42)
    ax.set_ylim(600, 2200)
    ax.set_xlabel('Control steps above the soft limit (%)')
    ax.set_ylabel('Mean methane flow (m$^3$/d)')
    ax.set_title('(a) the conventional set, swept twice', loc='left')
    ax.grid(color=C_GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, handletextpad=0.5,
              labelspacing=0.32, borderpad=0.2, scatterpoints=3)


def panel_shift(ax, R, dense_env, sparse_env):
    """Each family scored against both envelopes, as a paired scatter."""
    xs_d = [a for a, _, _ in dense_env]
    ys_d = [b for _, b, _ in dense_env]
    xs_s = [a for a, _, _ in sparse_env]
    ys_s = [b for _, b, _ in sparse_env]

    def med(runs, xs, ys):
        v = [r['ch4'] - float(np.interp(r['viol'], xs, ys)) for r in runs
             if xs[0] <= r['viol'] <= xs[-1]]
        return st.median(v) if v else None

    pts = []
    for a, runs in R.items():
        d = med(runs, xs_d, ys_d)
        sp = med(runs, xs_s, ys_s)
        if d is None or sp is None:
            continue
        pts.append((d, sp, a))
    pts.sort()

    ax.axvline(0, color=C_ENV, lw=1.0, zorder=1)
    ax.axhline(0, color=C_ENV, lw=1.0, zorder=1)
    # A2C sits at -450/-7 and would stretch the axis over empty space, so
    # the panel is drawn at the scale the other nine share and A2C is
    # written in at the edge it falls beyond.
    lim = (-185, 60)
    ax.plot([-185, 16], [-185, 16], ':', color='#8a8a8a', lw=1.0,
            zorder=1)
    # Name the diagonal on the line itself, in the empty lower right.
    ax.text(-52, -76, 'Unchanged by reference', color='#6a6a6a',
            fontsize=figstyle.TEXT, rotation=45, rotation_mode='anchor',
            transform_rotates_text=True, ha='center', va='bottom', zorder=2)

    # At single-column width there is no room for a label column beside
    # the cloud, so each family is named on its own marker; the offsets
    # separate the text alone and every marker stays on its two medians.
    # PPO and TRPO sit 14 units apart at the top right, too close for
    # side labels.  TRPO is the higher marker, so its label goes above it
    # and PPO's below; the other way round each label sits level with the
    # other family's marker and reads as belonging to it.
    OFFS = {'crossq': (9, 4), 'ddpg': (9, -3), 'tqc': (-9, -5),
            'sac': (9, 5), 'td3': (-10, -6), 'recurrentppo': (9, -5),
            'ars': (10, 2), 'trpo': (0, 11), 'ppo': (0, -11)}
    for d, sp, a_ in pts:
        c = C_ON if a_ in ON else (C_OFF if a_ in OFF else '#009E73')
        flip = (d < 0) != (sp < 0)
        if d < lim[0]:                       # A2C, off the left edge
            # The two medians rest on different runs: the full envelope
            # scores 20 of A2C's runs and the static-multiplier one 31.
            ax.annotate(f'{LABEL[a_]}: {d:.0f} (20 runs),\n{sp:.0f} (31 runs)'.replace('-', '\u2212'),
                        xy=(lim[0] + 1, sp), xytext=(lim[0] + 12, sp + 26),
                        fontsize=figstyle.TEXT, color=c, va='center', ha='left',
                        arrowprops=dict(arrowstyle='-|>', lw=0.7, color=c,
                                        mutation_scale=6))
            continue
        ax.scatter([d], [sp], s=36 if flip else 20, c=c, zorder=4,
                   edgecolor=C_ENV if flip else 'none',
                   lw=0.9 if flip else 0)
        dx, dy = OFFS.get(a_, (9, 0))
        ha = 'center' if dx == 0 else ('left' if dx > 0 else 'right')
        ax.annotate(LABEL[a_], (d, sp), xytext=(dx, dy),
                    textcoords='offset points', fontsize=figstyle.TEXT, color=c,
                    va='center', ha=ha)

    # The quadrant that carries the point: above the sparse envelope and
    # below the dense one.
    ax.add_patch(plt.Rectangle((lim[0], 0), -lim[0], 60, color=C_ON,
                               alpha=0.055, lw=0, zorder=0))


    ax.set_xlim(lim[0], 22)
    ax.set_ylim(-110, 60)
    ax.set_xlabel(f'Difference from full envelope ({figstyle.M3D})')
    ax.set_ylabel(f'Difference from static-multiplier\nenvelope ({figstyle.M3D})')
    ax.grid(color=C_GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    return pts


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else PAPER
    os.makedirs(out, exist_ok=True)

    dense_cloud, dense_env, _sparse_cloud, sparse_env = envelopes()
    R = learned_runs()

    # The conventional cloud and its envelope are already drawn in the
    # frontier figure, so only the paired comparison is kept here.
    fig = plt.figure(figsize=(3.4, 3.0))
    pts = panel_shift(fig.add_subplot(1, 1, 1), R, dense_env, sparse_env)

    p = os.path.join(out, 'fig_density.pdf')
    figstyle.save(fig, p, figstyle.COLUMN)
    plt.close(fig)

    flips = [a for d, s, a in pts if (d < 0) != (s < 0)]
    print(f'fig_density.pdf  {len(dense_cloud)} vs {len(_sparse_cloud)} '
          f'conventional evaluations, {len(dense_env)} vs {len(sparse_env)} '
          f'envelope points; sign flips: {", ".join(LABEL[a] for a in flips)}')


if __name__ == '__main__':
    main()
