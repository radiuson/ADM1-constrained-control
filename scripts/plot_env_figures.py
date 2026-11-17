#!/usr/bin/env python3
"""The figures for the sections that carry none: environment and results lead.

The manuscript's first forty per cent -- reactor, constraints, observations,
scenarios, the conventional sweep and the algorithm set -- is text alone, so a
reader assembles the benchmark from prose before any figure appears.  These
three fill that gap, and the first two need no evaluation records at all: they
describe the environment the records come from.

    python3 scripts/plot_env_figures.py [outdir]

fig_benchmark.pdf  the control loop, its bands and the evaluation protocol
fig_scenarios.pdf  what the seven scenarios do to the reactor
fig_headline.pdf   the zero-violation and positive-rate readings side by side

``outdir`` defaults to the manuscript's img/ directory, as plot_figures does.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The palette, the rcParams and the record loaders are plot_figures': a figure
# here and a figure there must not disagree about what a class looks like or
# which runs are in the analysis set.
from plot_figures import (C_CONV, C_ENV, C_FREE, C_OFF, C_ON, CMDP_OFF,
                          EPISODE_STEPS, PAPER, load_cmdp, load_reward)
from build_tables import envelope
import figstyle

# Written by scenario_traces.py.
TRACES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'figures', 'scenario_traces.json')

# The bands, in mg HAc-eq/L, as Section 3.2 fixes them.
VFA_SOFT_MG, VFA_HARD_MG = 300.0, 1500.0

# The scenarios in the order the text introduces them: five loading levels,
# then the two perturbation cases.
SCEN = [
    ('low_load',       'Low load',       1.5),
    ('nominal',        'Nominal',        2.4),
    ('plant_load',     'Plant load',     3.7),
    ('high_load',      'High load',      4.9),
    ('peak_load',      'Peak load',      5.9),
    ('fog_surge',      'FOG surge',      3.7),
    ('elevated_start', 'Elevated start', 3.7),
]

C_SOFT, C_HARD = '#E8A33D', '#C0392B'


# ----------------------------------------------------------------- schematic

def _box(ax, x, y, w, h, text, face, edge, fs=7.2, weight='normal'):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.018',
        linewidth=0.9, facecolor=face, edgecolor=edge, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, zorder=4, linespacing=1.45, fontweight=weight)


def _arrow(ax, p, q, text=None, rad=0.0, dy=0.022, fs=6.6, colour='#444444'):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle='-|>', mutation_scale=7.5, linewidth=0.9,
        color=colour, connectionstyle=f'arc3,rad={rad}', zorder=2,
        shrinkA=1.5, shrinkB=1.5))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text,
                ha='center', va='bottom', fontsize=fs, color='#333333',
                zorder=4, linespacing=1.4)


def fig_benchmark(out):
    """The loop on the left, the bands the loop is scored against on the right."""
    fig = plt.figure(figsize=(7.1, 2.62))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.46, 1.0], wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    for a in (ax, axb):
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.axis('off')

    # -- (a) the control loop ------------------------------------------------
    ax.set_title('(a) Control loop and evaluation protocol', loc='left',
                 fontsize=8)

    _box(ax, 0.015, 0.615, 0.305, 0.20,
         'Controller\n'
         r'conventional loop $\vert$ policy', '#FFFFFF', '#333333',
         fs=7.0)
    _box(ax, 0.645, 0.615, 0.355, 0.20,
         'ADM1 digester\n'
         r'38 states, $V_\mathrm{liq}=1836\,$m$^3$', '#F2F2F2', '#333333',
         fs=6.7)

    # Actions out above, observations back below: the arrows are kept clear of
    # each other and of the boxes so neither label lands on a box edge.
    _arrow(ax, (0.325, 0.775), (0.640, 0.775), colour='#444444')
    ax.text(0.4875, 0.815,
            r'$q_\mathrm{ad}\!\in\![41,159]$ m$^3$/d,'
            '\n'
            r'multiplier $\in\![0.7,1.3]$',
            ha='center', va='bottom', fontsize=6.5, color='#333333',
            linespacing=1.45)

    _arrow(ax, (0.640, 0.660), (0.325, 0.660), colour='#0072B2')
    ax.text(0.4875, 0.628,
            'VFA, alkalinity, pH,\n'
            r'CH$_4$ flow, $q_\mathrm{ad}$, multiplier',
            ha='center', va='top', fontsize=6.5, color='#0072B2',
            linespacing=1.45)

    # The integration one control step hides, and the protocol around it.
    _box(ax, 0.645, 0.215, 0.355, 0.175,
         'LSODA, 15-min splitting\n96 sub-steps per step',
         '#FFFFFF', '#888888', fs=6.6)
    _arrow(ax, (0.8225, 0.615), (0.8225, 0.390), colour='#888888')

    _box(ax, 0.015, 0.215, 0.305, 0.175,
         '7 scenarios $\\times$ 60 steps\n'
         f'= {7 * EPISODE_STEPS} scored control steps',
         '#FFFFFF', '#888888', fs=6.6)
    ax.text(0.1675, 0.155, '1-day control interval, 60-day episode',
            ha='center', va='top', fontsize=6.4, color='#555555')

    # -- (b) the bands -------------------------------------------------------
    axb.set_title('(b) Total VFA bands', loc='left', fontsize=8)
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 1750)
    axb.axis('on')
    axb.set_yticks([0, 300, 1500])
    axb.set_yticklabels(['0', '300', '1500'])
    axb.set_ylabel('Total VFA (mg HAc-eq/L)', fontsize=7.4)
    axb.set_xticks([])
    axb.tick_params(axis='y', labelsize=7, direction='in', length=3)
    for s in ('top', 'right', 'bottom'):
        axb.spines[s].set_visible(False)

    axb.axhspan(0, VFA_SOFT_MG, facecolor='#E9F3E9', zorder=1)
    axb.axhspan(VFA_SOFT_MG, VFA_HARD_MG, facecolor='#FDF3E3', zorder=1)
    axb.axhspan(VFA_HARD_MG, 1750, facecolor='#FBEAE7', zorder=1)
    axb.axhline(VFA_SOFT_MG, color=C_SOFT, linewidth=1.2, zorder=3)
    axb.axhline(VFA_HARD_MG, color=C_HARD, linewidth=1.2, zorder=3)

    axb.text(0.5, 150, 'within the soft band', ha='center', va='center',
             fontsize=6.7, color='#2E6B2E')
    axb.text(0.5, 830,
             'soft limit exceeded:\nthe step counts\nas a violation',
             ha='center', va='center', fontsize=6.7, color='#8A5A14',
             linespacing=1.55)
    axb.text(0.5, 1640, 'hard limit: episode ends',
             ha='center', va='center', fontsize=6.7, color='#9E2F22')

    # The two limit values sit just under their own lines, inside the band
    # below, where nothing else is drawn.
    axb.text(0.5, VFA_SOFT_MG - 40, 'soft 300', ha='center', va='top',
             fontsize=6.6, color=C_SOFT, fontweight='semibold')
    axb.text(0.5, VFA_HARD_MG - 40, 'hard 1500', ha='center', va='top',
             fontsize=6.6, color=C_HARD, fontweight='semibold')

    # The other two termination conditions, pH and free ammonia, are the
    # caption's job: nothing explanatory is drawn on the figure itself.

    p = os.path.join(out, 'fig_benchmark.pdf')
    fig.savefig(p)
    plt.close(fig)
    print(f'fig_benchmark.pdf  loop, bands and protocol; '
          f'{7 * EPISODE_STEPS} scored steps per controller')


# ----------------------------------------------------------------- scenarios

def fig_scenarios(out):
    """What each scenario does to the reactor, at the bounding action corner."""
    if not os.path.exists(TRACES):
        print(f'fig_scenarios.pdf  SKIPPED, no {TRACES}')
        return
    T = json.load(open(TRACES))

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.3))

    # -- VFA under the maximum corner ---------------------------------------
    # The text bounds each scenario by the maximum-feed corner, so that is the
    # trace that shows which scenarios can reach the limit at all.  The five
    # loading levels take one ramp of colour so the ordering by load reads off
    # the figure; the two perturbation cases are deliberately outside it.
    LOAD_COLS = ['#BBD8EE', '#7FB2D9', '#3D7FB5', '#1C5584', '#0A3355']
    cols, dash = {}, {}
    for i, (key, _, _) in enumerate(SCEN[:5]):
        cols[key] = LOAD_COLS[i]
        dash[key] = (None, None)
    cols['fog_surge'], dash['fog_surge'] = '#D55E00', (3.2, 1.3)
    cols['elevated_start'], dash['elevated_start'] = '#009E73', (1.5, 1.3)

    for key, label, olr in SCEN:
        tr = T[key]['max']
        v = tr['vfa']
        d = np.arange(1, len(v) + 1)
        ln, = ax.plot(d, v, linewidth=1.35, color=cols[key], label=label,
                      zorder=3)
        if dash[key][0]:
            ln.set_dashes(dash[key])
        # A trace that stops short ended on the hard band.  The marker alone
        # is easy to read onto a neighbouring curve, so the day is written
        # beside it.
        if len(v) < 60:
            ax.plot(d[-1], v[-1], marker='o', markersize=4.2,
                    markerfacecolor='white', markeredgewidth=1.3,
                    color=cols[key], zorder=5)
            ax.annotate(f'day {d[-1]}', xy=(d[-1], v[-1]),
                        xytext=(d[-1] + 1.5, v[-1] * 0.80),
                        fontsize=figstyle.TEXT, color=cols[key], ha='left', va='top')

    ax.axhline(VFA_SOFT_MG, color=C_SOFT, linewidth=1.1, linestyle='--',
               zorder=2)
    ax.axhline(VFA_HARD_MG, color=C_HARD, linewidth=1.1, linestyle='--',
               zorder=2)
    # Named in the right margin, where no trace runs.
    ax.text(60.8, VFA_SOFT_MG, 'Soft\nlimit', ha='left', va='center',
            fontsize=figstyle.TEXT, color=C_SOFT, linespacing=1.0)
    ax.text(60.8, VFA_HARD_MG, 'Hard\nlimit', ha='left', va='center',
            fontsize=figstyle.TEXT, color=C_HARD, linespacing=1.0)

    ax.set_yscale('log')
    ax.set_ylim(38, 2600)
    ax.set_yticks([100, 300, 1000, 1500])
    ax.set_yticklabels(['100', '300', '1000', '1500'])
    ax.set_xlim(0, 60)
    ax.set_xticks([0, 15, 30, 45, 60])
    ax.set_xlabel('Day of episode')
    ax.set_ylabel('Total VFA (mg HAc-eq/L)')
    # No title and no marker notes drawn on the figure: what the traces show,
    # and what the two markers mean, is the caption's job.
    ax.tick_params(direction='in', top=False, right=False, length=3)
    ax.legend(fontsize=figstyle.TEXT, loc='upper center', frameon=False,
              ncol=3, handlelength=1.8, columnspacing=1.0, labelspacing=0.3,
              bbox_to_anchor=(0.45, -0.24))

    p = os.path.join(out, 'fig_scenarios.pdf')
    figstyle.save(fig, p, figstyle.COLUMN)
    plt.close(fig)
    unreach = sum(1 for k, _, _ in SCEN if max(T[k]['max']['vfa']) < VFA_SOFT_MG)
    print(f'fig_scenarios.pdf  {len(SCEN)} scenarios, {unreach} cannot reach '
          f'the soft limit at the maximum corner')


# ------------------------------------------------------------------ headline

def _zero_rows(E, Rc):
    """Constrained family medians at a realised rate of zero, and the bound."""
    best = max(c for v, c, _ in E if v <= 0.0)
    rows = []
    for fam, runs in Rc.items():
        z = [r['ch4'] for r in runs if r['viol'] <= 0.0]
        if z:
            rows.append((st.median(z), fam, len(z)))
    rows.sort()
    return rows, best


def fig_headline(out):
    """The two readings the main claim is made of, in one figure.

    The frontier figure carries the whole range, where a single zero-violation
    column is one point on a long axis.  The paper's leading claim is about
    that column and about the sign reversal away from it, so those two get a
    figure of their own at the head of the results.
    """
    B, E = envelope()
    Rc = load_cmdp()
    R = load_reward()

    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(7.1, 3.35), gridspec_kw={'wspace': 0.30,
                                                'width_ratios': [1.1, 1.0]})

    # -- (a) at zero violations ---------------------------------------------
    rows, best = _zero_rows(E, Rc)
    y = np.arange(len(rows))
    cols = [C_OFF if fam in CMDP_OFF else C_ON for _, fam, _ in rows]
    # The axis starts at zero: a floor part-way up the bars would make the
    # two weakest families look far closer to the bound than they are.
    ax.barh(y, [m for m, _, _ in rows], height=0.62, color=cols,
            alpha=0.88, zorder=3)
    ax.axvline(best, color=C_ENV, linewidth=1.4, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([fam for _, fam, _ in rows], fontsize=6.8)
    ax.set_xlim(0, 2050)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xlabel('Mean methane flow (m$^3$/d)')
    ax.set_title('(a) At zero violations, RL leads', loc='left')
    ax.tick_params(direction='in', top=False, right=False, length=3)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.6, len(rows) + 0.15)

    # How many seeds reached zero at all is part of reading the median, so it
    # is written against each bar rather than left to the caption.
    for yy, (m, fam, n) in zip(y, rows):
        ax.text(m + 42, yy, f'{m:.0f}', va='center', ha='left',
                fontsize=6.2, color='#333333')
        ax.text(26, yy, f'{n} seeds', va='center', ha='left', fontsize=5.8,
                color='#FFFFFF', zorder=5)

    # The bound is annotated above the bars, where the axis is empty, so the
    # label never sits over a family's value.
    ax.annotate(f'best conventional tuning at zero violations: {best:.0f}',
                xy=(best, len(rows) - 0.42), xytext=(best - 55,
                                                     len(rows) - 0.42),
                fontsize=6.4, color=C_ENV, ha='right', va='center')

    above = sum(1 for m, _, _ in rows if m > best)
    h_on = plt.Line2D([], [], color=C_ON, linewidth=5, alpha=0.88)
    h_off = plt.Line2D([], [], color=C_OFF, linewidth=5, alpha=0.88)
    ax.legend([h_on, h_off],
              [f'On-policy — {above} of 8 above the bound',
               'Off-policy — none above'],
              fontsize=6.4, loc='lower center', frameon=False,
              handlelength=1.1, borderpad=0.3, ncol=1,
              bbox_to_anchor=(0.5, -0.30))

    # -- (b) away from zero, the sign reverses ------------------------------
    # One point per run: its distance from the envelope at its own realised
    # rate.  Zero is the envelope, so the sign is the whole reading.
    xs = [a for a, _, _ in E]
    ys = [b for _, b, _ in E]

    def delta(d):
        if not (xs[0] <= d['viol'] <= xs[-1]):
            return None
        return d['ch4'] - float(np.interp(d['viol'], xs, ys))

    pos, neg = [], []
    for fam, runs in list(Rc.items()) + list(R.items()):
        for r in runs:
            if r['viol'] <= 0.0:
                continue
            dv = delta(r)
            if dv is None:
                continue
            (pos if dv > 0 else neg).append((r['viol'], dv))

    for pts, col, lab in ((neg, '#9AA7B4', 'below the envelope'),
                          (pos, C_FREE, 'above')):
        if pts:
            axr.scatter([a for a, _ in pts], [b for _, b in pts], s=5.5,
                        facecolor=col, edgecolor='none', alpha=0.62,
                        zorder=3, label=f'{lab} ({len(pts)})')

    axr.axhline(0, color=C_ENV, linewidth=1.4, zorder=4)
    axr.set_xlim(-0.6, 20)
    axr.set_xticks([0, 5, 10, 15, 20])
    axr.set_ylim(-1250, 320)
    axr.set_xlabel('Steps above the soft limit (%)')
    axr.set_ylabel('Methane flow minus envelope (m$^3$/d)')
    axr.set_title('(b) At positive rates, the envelope leads', loc='left')
    axr.tick_params(direction='in', top=False, right=False, length=3)
    axr.annotate('conventional envelope', xy=(19.4, 60), fontsize=6.5,
                 color=C_ENV, ha='right', va='bottom')
    axr.legend(fontsize=6.4, loc='lower right', frameon=True,
               framealpha=0.95, edgecolor='#CCCCCC', handlelength=1.0,
               borderpad=0.45, scatterpoints=1)

    p = os.path.join(out, 'fig_headline.pdf')
    fig.savefig(p)
    plt.close(fig)
    print(f'fig_headline.pdf  zero column {len(rows)} families, {above} above '
          f'{best:.0f}; positive rates {len(pos)} above / {len(neg)} below')


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else PAPER
    os.makedirs(out, exist_ok=True)
    fig_benchmark(out)
    fig_scenarios(out)
    fig_headline(out)


if __name__ == '__main__':
    main()
