"""Shared visual style for CPI, FFR, and S&P 500 thesis figures."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# Colour-blind-safe, print-readable palette based on Okabe-Ito.
COLORS = {
    "realized": "#222222",
    "kalshi": "#0072B2",
    "kalshi_light": "#56B4E9",
    "cleveland": "#D55E00",
    "history": "#009E73",
    "tips": "#CC79A7",
    "rf": "#56B4E9",
    "gbm": "#E69F00",
    "lstm": "#6A3D9A",
    "naive": "#7A7A7A",
    "neutral": "#8C8C8C",
    "positive": "#009E73",
    "negative": "#D55E00",
    "grid": "#D9D9D9",
}


def apply_thesis_style():
    """Apply one consistent journal-style Matplotlib configuration."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.titleweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.55,
        "grid.alpha": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, figures_dir, stem, formats=("pdf", "png"), dpi=300):
    """Save a figure under a stable thesis filename in every requested format."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in formats:
        fig.savefig(figures_dir / f"{stem}.{extension}", dpi=dpi,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def panel_label(ax, label):
    ax.text(-0.10, 1.04, label, transform=ax.transAxes,
            ha="left", va="bottom", fontweight="bold")

