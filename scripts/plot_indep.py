"""Figure for the independent-disturbance evaluation.

(a) Test scenarios: every conventional configuration, the envelope built on
    the test set, the five tunings selected on the validation set, and the
    constrained runs at the tightest cost limit.
(b) Per-run difference from the envelope, original scenarios against test
    scenarios, for the on-policy runs of both pre-specified groups.

usage: python3 scripts/plot_indep.py [eval_dir] [outdir]
"""
import csv, gzip, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle
figstyle.apply()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_ON, C_OFF = '#0072B2', '#D55E00'
C_ENV, C_GRID, C_CONV = '#333333', '#BBBBBB', '#8E8E8E'

# Panel (a) runs to the largest rate of the configurations selected on the
# validation set, so their test-set rates, the panel's main finding, are all
# drawn.  A symmetric-log axis keeps the low-violation region, where the
# learned runs and the envelope's steep rise sit, readable on the same panel.
XMAX = 45.0
LINTHRESH = 1.0


def pooled(per, ids):
    steps = sum(per[i]['steps'] for i in ids)
    viol = sum(per[i]['viol'] for i in ids)
    ch4 = sum(per[i]['ch4_sum'] for i in ids)
    return 100.0 * viol / steps, ch4 / steps


def main(d, out):
    S = json.load(open(os.path.join(d, 'scenarios.json')))
    tid = [s['id'] for s in S['test']]
    A = json.load(gzip.open(os.path.join(d, 'test_conventional_all.json.gz')))
    conv = np.array([pooled(r['per_scenario'], tid) for r in A])
    env = np.array([(float(r['test_rate']), float(r['test_methane']))
                    for r in csv.DictReader(open(os.path.join(d, 'oracle_envelope.csv')))])
    sel = list(csv.DictReader(open(os.path.join(d, 'test_runs_conventional_selected.csv'))))
    R = list(csv.DictReader(open(os.path.join(d, 'test_runs_learned.csv'))))

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.9),
                               gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.28))

    # ---------------------------------------------------------------- (a)
    m = conv[:, 0] <= XMAX
    a.scatter(conv[m, 0], conv[m, 1], s=2.5, color=C_CONV, alpha=0.35, lw=0,
              zorder=1, rasterized=True)
    a.plot(env[:, 0], env[:, 1], color=C_ENV, lw=1.2, zorder=3)
    sv = np.array([float(r['test_rate_completed']) for r in sel])
    sc = np.array([float(r['test_methane_mean']) for r in sel])
    k = sv <= XMAX
    a.plot(sv[k], sc[k], 'D', ms=5, mfc='white', mec='black', mew=0.9, zorder=6)
    for r, v, c in zip(sel, sv, sc):
        if v <= XMAX:
            # The 3 % and 5 % selections sit close together, so the 3 % label
            # goes below its marker.
            below = r['budget_pct'] == '3'
            a.annotate(f"budget {r['budget_pct']} %", (v, c), xytext=(-2, -9 if below else 7),
                       textcoords='offset points', fontsize=figstyle.TEXT, ha='right',
                       bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.8),
                       zorder=7)
    for cl, col, mk in (('on-policy', C_ON, 'o'), ('off-policy', C_OFF, '^')):
        H = [r for r in R if r['class'] == cl and r['G2'] == '1']
        v = np.array([float(r['test_rate_completed']) for r in H])
        c = np.array([float(r['test_methane_mean']) for r in H])
        k = v <= XMAX
        a.scatter(v[k], c[k], s=13, marker=mk, facecolor=col, edgecolor='white',
                  lw=0.3, alpha=0.85, zorder=5)
    a.set_xscale('symlog', linthresh=LINTHRESH, linscale=1.0)
    # Mark where the axis turns from linear to logarithmic.
    a.axvline(LINTHRESH, color='#777777', lw=0.8, ls=':', zorder=0)
    a.text(LINTHRESH, 1.01, 'linear | log', transform=a.get_xaxis_transform(),
           ha='center', va='bottom', fontsize=figstyle.TEXT - 1, color='#555555',
           clip_on=False)
    a.set_xlim(0, XMAX)
    a.set_xticks([0, 0.5, 1, 2, 5, 10, 20, 40])
    a.set_xticklabels(['0', '0.5', '1', '2', '5', '10', '20', '40'])
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.set_ylim(400, 2150)
    a.set_xlabel('Steps above the soft limit, test scenarios (%)')
    a.set_ylabel(figstyle.FLOW)
    a.grid(axis='y', color=C_GRID, lw=0.4, alpha=0.5)
    a.legend(handles=[
        Line2D([], [], ls='', marker='o', ms=3, color=C_CONV, label='Conventional configurations'),
        Line2D([], [], color=C_ENV, lw=1.2, label='Envelope on test scenarios'),
        Line2D([], [], ls='', marker='D', ms=4.5, mfc='white', mec='black',
               label='Selected on validation set (label: validation budget)'),
        Line2D([], [], ls='', marker='o', ms=4, color=C_ON, label='On-policy, $d=0.6$'),
        Line2D([], [], ls='', marker='^', ms=4, color=C_OFF, label='Off-policy, $d=0.6$'),
    ], loc='lower right', frameon=False, handletextpad=0.4, borderaxespad=0.2)
    figstyle.panel(a, 'a')

    # ---------------------------------------------------------------- (b)
    on = [r for r in R if r['class'] == 'on-policy']
    zero = [r for r in on if r['G1'] == '1']
    other = [r for r in on if r['G1'] != '1']
    lim = (-420, 260)
    b.plot(lim, lim, color=C_GRID, lw=0.8, ls=(0, (2, 2)), zorder=1)
    b.axhline(0, color=C_ENV, lw=0.7, zorder=2)
    b.axvline(0, color=C_ENV, lw=0.7, zorder=2)
    for H, fill in ((other, 'white'), (zero, C_ON)):
        x = [float(r['D_reported']) for r in H]
        y = [float(r['D_test_oracle']) for r in H]
        b.scatter(x, y, s=13, facecolor=fill, edgecolor=C_ON, lw=0.6, alpha=0.9, zorder=4)
    groups = (('Zero excursions, original', zero),
              ('All runs at $d=0.6$', [r for r in on if r['G2'] == '1']))
    for (lab, H), mk in zip(groups, ('o', 's')):
        x = np.median([float(r['D_reported']) for r in H])
        y = np.median([float(r['D_test_oracle']) for r in H])
        b.plot([x], [y], mk, ms=8, mfc='none', mec='black', mew=1.3, zorder=6)
    b.set_xlim(*lim)
    b.set_ylim(-420, 120)
    b.set_xlabel(f'Difference from envelope, original ({figstyle.M3D})')
    b.set_ylabel(f'Difference from envelope, test ({figstyle.M3D})')
    b.legend(handles=[
        Line2D([], [], ls='', marker='o', ms=4, color=C_ON,
               label='Zero violations, original evaluation'),
        Line2D([], [], ls='', marker='o', ms=4, mfc='white', mec=C_ON,
               label='Other runs at $d=0.6$'),
        Line2D([], [], ls='', marker='o', ms=6, mfc='none', mec='black', mew=1.2,
               label='Median, zero-violation runs'),
        Line2D([], [], ls='', marker='s', ms=6, mfc='none', mec='black', mew=1.2,
               label='Median, all runs at $d=0.6$'),
    ], loc='upper center', bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False,
       handletextpad=0.4, columnspacing=1.0)
    figstyle.panel(b, 'b')

    figstyle.save(fig, out, figstyle.FULL)
    out_of_range = [(r['algorithm'], float(r['test_rate_completed'])) for r in R
                    if r['G2'] == '1' and float(r['test_rate_completed']) > XMAX]
    print('learned runs beyond x range:', out_of_range)
    print('selected tunings beyond x range:', [(r['budget_pct'], round(v, 2)) for r, v in zip(sel, sv) if v > XMAX])


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results_independent')
    o = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'figures')
    os.makedirs(o, exist_ok=True)
    main(d, os.path.join(o, 'fig_indep.pdf'))
