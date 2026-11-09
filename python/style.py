"""Shared palette and matplotlib theme for "Cosmic Singing Bowls".

Light theme only. The same tokens live in site/css/style.css, so figures and
page match. Clay red marks the effect / resonance / "true" signal; the
greenish dark blue marks text, structure and the baseline (standard) model.
"""
from pathlib import Path

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "site" / "assets"
DATA = ROOT / "data"

# --- palette tokens -------------------------------------------------------
BG = "#FAF7F2"        # warm off-white page background
PANEL = "#F2ECE2"     # slightly darker card/panel
INK = "#1F4E5A"       # greenish dark blue: text, baseline model, structure
INK_SOFT = "#4A6F78"  # lighter teal for secondary lines and small text (>=4.5:1 on BG)
CLAY = "#B5523B"      # clay red: effect, resonance, highlights
CLAY_SOFT = "#D9967F"
SAND = "#C9BBA3"      # neutral for grids / de-emphasised data
GOLD = "#C18A2C"      # reserved for "injected / true value" markers
GOLD_TEXT = "#8A5F10"  # gold for small text labels (>=4.5:1 on BG)

FONT = "DejaVu Sans"


def apply():
    """Apply the shared matplotlib theme."""
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": FONT,
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.prop_cycle": mpl.cycler(color=[INK, CLAY, INK_SOFT, CLAY_SOFT, GOLD]),
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": SAND,
        "grid.alpha": 0.5,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "lines.linewidth": 2.2,
        "svg.fonttype": "none",
    })


apply()
