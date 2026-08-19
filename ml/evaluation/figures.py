"""
One figure: predicted vs. actual salary on the held-out test rows — the simplest
possible check of whether the model tracks the target.

Self-contained: styling is declared here rather than shared with `data viz/vizstyle.py`,
so this module has no dependency on that folder.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e6e5e1"
S1 = "#2a78d6"


def _apply_style() -> None:
    mpl.rcParams.update({
        "text.parse_math": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": ["DejaVu Sans"],
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def _money(x, _pos=None) -> str:
    if x >= 1_000_000:
        return f"${x/1e6:.1f}M"
    if x >= 1_000:
        return f"${x/1e3:.0f}k"
    return f"${x:,.0f}"


def _source_note(fig, text: str) -> None:
    fig.text(0.005, -0.02, text, fontsize=7, color=INK_MUTED, ha="left", va="top")


def predicted_vs_actual_figure(y_test_log, y_pred_log, outdir: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    _apply_style()
    actual, pred = np.exp(np.asarray(y_test_log)), np.exp(np.asarray(y_pred_log))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    lo, hi = 800, 1_100_000
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=1.0, zorder=1)
    ax.scatter(actual, pred, s=7, color=S1, alpha=0.30, linewidths=0, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_formatter(FuncFormatter(_money))
    ax.yaxis.set_major_formatter(FuncFormatter(_money))
    ax.set(xlabel="actual salary", ylabel="predicted salary")
    ax.set_title("Predicted vs. actual, held-out rows")
    ax.grid(True, which="major", axis="both")
    within = np.mean(np.abs(pred / actual - 1) <= 0.25) * 100
    ax.text(0.03, 0.97, f"n={len(actual):,}\n{within:.0f}% within ±25%",
            transform=ax.transAxes, fontsize=8, color=INK_2, va="top")

    _source_note(fig, "Stack Overflow Developer Survey 2025 (ODbL). Held-out test rows. "
                      "Points on the diagonal are exact; above it, the model overpaid.")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "predicted_vs_actual.png"
    fig.savefig(path)
    plt.close(fig)
    return path
