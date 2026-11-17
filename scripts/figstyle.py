"""Shared typography and sizing for the manuscript figures.

Every figure is laid out at the width it is printed at, one column (89 mm)
or the full text width (184 mm), so the point sizes set here are the sizes
on the page: 8 pt axis labels, 7 pt tick labels, legends and annotations.
Text is set in Liberation Sans, metrically identical to Arial, and embedded
as TrueType.  Units are written with Unicode super- and subscripts (m³/d,
CH₄) so that no character is shrunk below the body size.
"""
import matplotlib
import matplotlib.pyplot as plt

COLUMN = 89 / 25.4      # inches
FULL = 184 / 25.4

LABEL = 8               # axis labels, panel letters
TEXT = 7                # tick labels, legends, annotations

M3D = 'm³/d'
FLOW = f'Mean methane flow ({M3D})'


def apply():
    plt.rcParams.update({
        'font.family': ['Liberation Sans', 'Arial', 'DejaVu Sans'],
        'font.size': TEXT,
        'axes.labelsize': LABEL,
        'axes.titlesize': LABEL,
        'legend.fontsize': TEXT,
        'legend.title_fontsize': TEXT,
        'xtick.labelsize': TEXT,
        'ytick.labelsize': TEXT,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Liberation Sans',
        'mathtext.it': 'Liberation Sans:italic',
        'mathtext.bf': 'Liberation Sans:bold',
        'axes.unicode_minus': True,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


def panel(ax, letter, x=-0.02, y=1.02):
    """Panel letter at the top left of an axes, one style for every figure."""
    ax.text(x, y, f'({letter})', transform=ax.transAxes, ha='right',
            va='bottom', fontsize=LABEL, fontweight='bold')


def save(fig, path, width, tries=6):
    """Save with a tight bounding box whose width equals the printed width.

    The tight box differs from the figure size by whatever the labels add
    or leave unused, so the figure width is corrected until the saved file
    is ``width`` inches wide; the figure is then placed at 100 % scale.
    """
    for _ in range(tries):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        got = bb.width + 2 * plt.rcParams['savefig.pad_inches']
        if abs(got - width) < 0.01:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + (width - got), h)
    fig.savefig(path)
    return got
