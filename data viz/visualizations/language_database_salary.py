"""
Why LanguageHaveWorkedWith / DatabaseHaveWorkedWith aren't modelled: median salary
barely moves with how many languages or databases someone knows. Two panels, same
story twice — knowing more tools is not the same as earning more.

Run:  .venv/bin/python "data viz/visualizations/language_database_salary.py"
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "data viz")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vizstyle as vz

SALARY_FLOOR, SALARY_CEIL = 1_000, 1_000_000   # same plausibility gate the model trains under
OUT = Path("data viz/visualizations/language_database_salary.png")

raw = pd.read_csv("data/survey.csv", na_values=["NA"], keep_default_na=False)
raw["annual_salary_usd"] = pd.to_numeric(raw["annual_salary_usd"], errors="coerce")
df = raw[raw["annual_salary_usd"].between(SALARY_FLOOR, SALARY_CEIL)].copy()


def count_items(s: pd.Series) -> pd.Series:
    return s.str.split(";").map(lambda L: len(L) if isinstance(L, list) else np.nan)


def bucket_stats(n_col: pd.Series, salary: pd.Series) -> pd.DataFrame:
    bucket = n_col.clip(upper=6).map(lambda v: "6+" if v == 6 else str(int(v)) if pd.notna(v) else None)
    out = (pd.DataFrame({"bucket": bucket, "salary": salary}).dropna()
           .groupby("bucket")["salary"].agg(median="median", n="size"))
    order = [str(i) for i in range(1, 6)] + ["6+"]
    return out.reindex([b for b in order if b in out.index])


vz.apply_style()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

for ax, col, label in [(axes[0], "LanguageHaveWorkedWith", "languages known"),
                       (axes[1], "DatabaseHaveWorkedWith", "databases known")]:
    stats = bucket_stats(count_items(df[col]), df["annual_salary_usd"])
    xpos = np.arange(len(stats))
    ax.bar(xpos, stats["median"], color=vz.S1, width=0.6, zorder=2)
    ax.set_xticks(xpos, stats.index)
    ax.yaxis.set_major_formatter(vz.money)
    ax.set(xlabel=f"number of {label}", ylabel="median annual salary")
    ax.set_title(f"Salary vs. {label}")
    ax.grid(True, axis="y")
    for x, m, n in zip(xpos, stats["median"], stats["n"]):
        ax.text(x, m + stats["median"].max() * 0.02, f"n={n:,}", ha="center",
                fontsize=7.5, color=vz.INK_2)
    ax.set_ylim(0, stats["median"].max() * 1.25)

vz.source_note(fig, "Stack Overflow Developer Survey 2025 (ODbL). Salary gated to "
                    "$1,000-$1,000,000, same as the model's training data. Flat bars = "
                    "counting tools tells you almost nothing about pay.")
fig.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}")
