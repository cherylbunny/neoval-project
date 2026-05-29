#!/usr/bin/env python3
"""
Manual check of cleaned macro data.

Runs checks: 
    1. Diagnostic summary: row counts, date ranges, NaN counts, min/max/mean
    2. Time-series plots: one chart per variable, saved as a PNG
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------- paths ----------------------

HOME       = os.path.expanduser("~")
PAPER_PATH = os.path.join(HOME, "Documents/github/neoval-project")
DATA_PATH  = os.path.join(PAPER_PATH, "data")

# ---------------------- load data ----------------------

df = pd.read_csv(os.path.join(DATA_PATH, "macro_data.csv"), parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

cols = [c for c in df.columns if c != "time"]


# ---------------------- diagnostic summary ----------------------

print("=" * 75)
print("1. DIAGNOSTIC SUMMARY")
print("=" * 75)
print(f"\nFull dataset: {len(df)} rows  |  {df['time'].min().date()} to {df['time'].max().date()}\n")

rows = []
for col in cols:
    valid = df[col].dropna()
    rows.append({
        "variable":  col,
        "start":     str(df.loc[df[col].notna(), "time"].min().date()),
        "end":       str(df.loc[df[col].notna(), "time"].max().date()),
        "n_obs":     len(valid),
        "n_nan":     int(df[col].isna().sum()),
        "min":       round(float(valid.min()), 3),
        "max":       round(float(valid.max()), 3),
        "mean":      round(float(valid.mean()), 3),
    })

summary = pd.DataFrame(rows).set_index("variable")
print(summary.to_string())


# ----------------------time-series plots ----------------------
print("\n" + "=" * 75)
print("3. TIME-SERIES PLOTS")
print("=" * 75)

plot_meta = [
    ("mortgage_rate",               "Mortgage Rate",                "% p.a."),
    ("cash_rate",                   "Cash Rate",                    "% p.a."),
    ("bond_yield_10y",              "10-Year Bond Yield",           "% p.a."),
    ("housing_lending_rate",        "Housing Lending Rate",         "% p.a."),
    ("housing_credit_growth",       "Housing Credit Growth",        "% per month"),
    ("unemployment_rate",           "Unemployment Rate",            "%"),
    ("trimmed_mean_inflation",      "Trimmed Mean Inflation",       "% year-ended"),
    ("household_disposable_income", "Household Disposable Income",  "$ millions"),
    ("net_overseas_migration",      "Net Overseas Migration",       "000 persons / qtr"),
    ("building_approvals",          "Building Approvals",           "dwellings / month"),
    ("dwelling_commencements",      "Dwelling Commencements",       "dwellings / qtr"),
    ("cpi_rents",                   "CPI Rents",                    "index (2017-18 = 100)"),
]

fig, axes = plt.subplots(4, 3, figsize=(18, 16))
fig.suptitle("Australian Housing Market Indicators", fontsize=14, fontweight="bold")

for ax, (col, title, unit) in zip(axes.flat, plot_meta):
    sub = df[["time", col]].dropna()
    ax.plot(sub["time"], sub[col], linewidth=0.9, color="#2563eb")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_ylabel(unit, fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(DATA_PATH, "macro_data_check.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n[ok] plots saved to {out_path}")
plt.show()
