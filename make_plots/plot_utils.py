import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

cmap = cm.get_cmap("Paired")

colors = cmap(np.linspace(0, 1, 12))

blues = ["#1B5B99", colors[1], "#789BC3"]
greens = ['darkgreen', 'mediumseagreen' ,colors[2]]
reds = ['firebrick', colors[5],colors[4]]
yellows = ['goldenrod', 'gold',colors[-2]]


settings = {
    # Use LaTeX to write all text
    "text.usetex": True,
    # "pgf.preamble": r"\usepackage{amsmath}",
    # "backend": "pgf",
    # "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.serif": "sans-serif",
    # "svg.fonttype": "none",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 11,
    "font.size": 11,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "xtick.minor.bottom": False,
    "ytick.minor.left": False,
    "ytick.minor.right": False,
    # Custom axes
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.labelpad": 8,
    # Lines and markers
    "lines.linewidth": 3,
    'lines.markersize': 6,
    # Grids
    "grid.linestyle": ":",
    # Legend
    "legend.frameon": False,
    # Save figure
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
}

plt.rcParams.update(settings)


def set_size(width, fraction=0.8, subplots=(1, 1), ratio = None, height_ratio =[1]):
    """Set figure dimensions to avoid scaling in LaTeX.
    
    For PRX document I got the following widths:
    > 510.0pt.
    l.65 \showthe\textwidth

    > 246.0pt.
    l.66 \showthe\columnwidth

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predefined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    ratio: float or None
            Height / width ratio. If set to None, the golden ratio
            (5**0.5 - 1) / 2 is used.

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "text":
        width_pt = 469.755
        settings = {
            "axes.labelsize": 12,
            "font.size": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2,
            "xtick.major.pad": 5,
            "ytick.major.pad": 5,
            "axes.labelpad": 8,
        }
        plt.rcParams.update(settings)
    elif width == "margin":
        width_pt = 144.54
        settings = {
            "axes.labelsize": 8,
            "font.size": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.5,
            "xtick.major.pad": 3.5,
            "ytick.major.pad": 3.5,
            "axes.labelpad": 6,
        }
        plt.rcParams.update(settings)
    elif width == "colums":
        width_pt = 246.0
        settings = {
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 2,
            "xtick.major.pad": 5,
            "ytick.major.pad": 5,
            "axes.labelpad": 8,
        }
        plt.rcParams.update(settings)
    elif width == "wide":
        width_pt = 510.0
        settings = {
            "axes.labelsize": 12,
            "font.size": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2,
            "xtick.major.pad": 4,
            "ytick.major.pad": 4,
            "axes.labelpad": 8,
        }
        plt.rcParams.update(settings)


    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if ratio == "equal":
        ratio = 1.0

    if ratio is None:
        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0]*height_ratio[0] / (subplots[1]*np.sum(height_ratio)))

    return (fig_width_in, fig_height_in)
