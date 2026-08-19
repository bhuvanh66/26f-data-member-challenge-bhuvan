"""
The raw salary data, before any cleaning. One chart: a histogram of
`annual_salary_usd` for all 5,000 rows in `data/survey.csv`, exactly as reported.

Log-scaled x-axis because the raw values span $1 to $6.9M — a linear axis would
flatten almost every respondent into the first inch and hide the shape entirely.

Run:  .venv/bin/python "data viz/visualizations/raw_salary_distribution.py"
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

TARGET = "annual_salary_usd"
OUT = Path("data viz/visualizations/raw_salary_distribution.png")

raw = pd.read_csv("data/survey.csv", na_values=["NA"], keep_default_na=False)
salary = pd.to_numeric(raw[TARGET], errors="coerce").dropna()

vz.apply_style()
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.logspace(np.log10(max(salary.min(), 1)), np.log10(salary.max()), 60)
ax.hist(salary, bins=bins, color=vz.S1)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(vz.money)
ax.set(xlabel="annual_salary_usd (raw, unfiltered)", ylabel="respondents",
      title="Raw salary data — all 5,000 rows, as reported")
ax.grid(True, axis="y")

vz.source_note(fig, f"Stack Overflow Developer Survey 2025 (ODbL). n={len(salary):,} rows "
                    f"with a non-missing salary; {raw[TARGET].isna().sum()} rows have no "
                    f"answer and are excluded from the count, not from the axis.")
fig.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}")
