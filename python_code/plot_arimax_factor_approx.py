#!/usr/bin/env python
"""
ARIMAX factor plotting script (levels, d=0)

Plots for a chosen region:
  • Actual index y_t
  • Deterministic factor components (intercept + β'X_subset), base-aligned to y
      - market only
      - market + mining
      - market + mining + lifestyle (if present)
  • ARMAX fitted values

Notes:
  - Components are *deterministic parts* of the mean (no ARMA correction). We
    align them to the actual series so they sit on the same vertical scale.
  - Alignment can be "mean" (default) or "first" observation.
  - Add simple seasonality via --sorder P,D,Q,s (e.g. 0,0,1,12).
"""

import os
import argparse
import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

HOME = os.path.expanduser("~")
PAPER_PATH = os.path.join(HOME, "Documents/github/neoval-project")
DATA_PATH  = os.path.join(PAPER_PATH, "data")

paper_path = '/Users/wsijp/Documents/github/neoval-project'

# ----------------------- file helpers -----------------------
def make_file_path(file, base_path = os.path.join(paper_path,'figures')):
    if len(file) < 1:
        raise Exception('Empty string provided for filename')
    if file[0] == '/':
        return file
    return os.path.join(base_path, file)

def insert_subname(file, subname):
    if not subname: return file
    L = file.split('.')
    if len(L) < 2: raise Exception('Provide file format via dot')
    L[-2] = f"{L[-2]}_{subname}"
    return '.'.join(L)

def show_or_save_fig(args, handle=None, subname=''):
    handle = handle or plt
    if args.file == 'show':
        handle.show()
    else:
        file = insert_subname(args.file, subname)
        file_path = make_file_path(file)
        print('Saving figure to {0}'.format(file_path))
        kwargs = {}
        if args.fine_dpi: kwargs['dpi'] = 300
        handle.savefig(file_path, bbox_inches='tight', pad_inches=0.01, **kwargs)

# ----------------------- utils ------------------------------
def parse_order3(s: str):
    try:
        p,d,q = [int(x.strip()) for x in s.split(",")]
        return (p,d,q)
    except Exception:
        raise argparse.ArgumentTypeError("Order must be like '1,0,1'")

def parse_sorder4(s: str):
    try:
        P,D,Q,S = [int(x.strip()) for x in s.split(",")]
        return (P,D,Q,S)
    except Exception:
        raise argparse.ArgumentTypeError("Seasonal order must be like '0,0,1,12'")

def ensure_monthly_freq(obj):
    obj = obj.copy()
    obj.index = pd.DatetimeIndex(obj.index)
    return obj.asfreq("MS")

def aicc(aic: float, nobs: int, kparams: int) -> float:
    return aic + (2 * kparams * (kparams + 1)) / max(nobs - kparams - 1, 1)

def fit_arimax(y: pd.Series, X: pd.DataFrame, order=(1,0,1), sorder=(0,0,0,0)):
    mod = SARIMAX(
        endog=y, exog=X,
        order=order, seasonal_order=sorder,
        trend='c',  # include intercept
        enforce_stationarity=True, enforce_invertibility=True,
        measurement_error=False
    )
    return mod.fit(disp=False)

def lb_report(resid, lags=[6,12,18,24]):
    lb = acorr_ljungbox(resid, lags=lags, return_df=True)
    return {int(L): float(p) for L, p in zip(lb.index, lb['lb_pvalue'].values)}

def align_to_reference(series: pd.Series, ref: pd.Series, how: str = "mean") -> pd.Series:
    if len(series) == 0:
        return series
    if how == "first":
        return series + (ref.iloc[0] - series.iloc[0])
    return series + (ref.mean() - series.mean())

# ----------------------- main ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot ARIMAX projections for a region with chosen factors.")
    parser.add_argument("--region", default="SYDNEY - BLACKTOWN")
    parser.add_argument("--region_level", default="sa4_name", choices=['sa4_name','major_city'])
    parser.add_argument("--order", type=parse_order3, default=(1,0,1),
                        help="Non-seasonal order 'p,0,q' (default 1,0,1).")
    parser.add_argument("--sorder", type=parse_sorder4, default=(0,0,0,0),
                        help="Seasonal order 'P,D,Q,s' (e.g. '0,0,1,12').")
    parser.add_argument("--start_year", type=int, default=1995)
    parser.add_argument("--factors", nargs="+", default=["market","mining"])
    parser.add_argument("--align", choices=["mean","first"], default="mean",
                        help="Vertical alignment of components to actual series (default: mean).")
    parser.add_argument('-f','--file', default='show')
    parser.add_argument('--fine_dpi', action='store_true', default=False)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message="No frequency information was provided")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
    warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
    warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found")
    warnings.filterwarnings("ignore", message="Non-invertible starting seasonal moving average")
    warnings.filterwarnings("ignore", message="Non-stationary starting seasonal autoregressive")

    # --- Load
    idx_path = os.path.join(DATA_PATH, "city_indexes.csv" if args.region_level=='major_city' else "sa4_indexes.csv")
    fac_path = os.path.join(DATA_PATH, "df_factor_trends.csv")
    df_idx = pd.read_csv(idx_path); df_fac = pd.read_csv(fac_path)
    df_idx['month_date'] = pd.to_datetime(df_idx['month_date'])
    df_fac['month_date'] = pd.to_datetime(df_fac['month_date'])
    df_idx = df_idx.set_index('month_date').sort_index()
    df_fac = df_fac.set_index('month_date').sort_index()

    if args.region not in df_idx.columns:
        raise ValueError(f"Region '{args.region}' not found in {idx_path} columns.")
    for c in args.factors:
        if c not in df_fac.columns:
            raise ValueError(f"Column '{c}' missing in {fac_path}.")

    # Trim & align
    df_idx = df_idx[df_idx.index >= f"{args.start_year}-01-01"]
    df_fac = df_fac[df_fac.index >= f"{args.start_year}-01-01"]
    y = ensure_monthly_freq(df_idx[args.region].astype(float).dropna())
    X = ensure_monthly_freq(df_fac[args.factors].astype(float).dropna())
    common = y.index.intersection(X.index)
    y, X = y.loc[common], X.loc[common]

    # --- Fit (non-seasonal + optional seasonal) ------------------------------
    res = fit_arimax(y, X, order=args.order, sorder=args.sorder)

    # --- Parameters & intercept ----------------------------------------------
    param_names = getattr(res, "param_names", None)
    params = (pd.Series(res.params, index=param_names)
              if param_names is not None else pd.Series(res.params))
    exog_names = list(res.model.exog_names)
    beta_exog = params.reindex(exog_names)

    int_key = next((k for k in params.index
                    if isinstance(k, str) and (("intercept" in k.lower()) or (k.lower()=="const"))), None)
    intercept = float(params[int_key]) if int_key is not None else 0.0

    # --- Deterministic components (no ARMA), then base-align to y ------------
    def det_component(cols):
        cols = [c for c in cols if c in exog_names]
        b = beta_exog.reindex(cols).fillna(0.0)
        return intercept + X[cols].mul(b, axis=1).sum(axis=1)

    comp_raw, comp_labels = [], []

    first = args.factors[0]
    s1 = det_component([first])
    comp_raw.append(s1); comp_labels.append("Market-only" if first=="market" else f"{first}-only")

    if len(args.factors) >= 2:
        first2 = args.factors[:2]
        s2 = det_component(first2)
        comp_raw.append(s2)
        comp_labels.append("Market + Mining" if first2==['market','mining'] else " + ".join(first2))

    if len(args.factors) >= 3 and args.factors[:3] == ['market','mining','lifestyle']:
        s3 = det_component(['market','mining','lifestyle'])
        comp_raw.append(s3); comp_labels.append("Market + Mining + Lifestyle")

    comp_lines = [align_to_reference(s, y, how=args.align) for s in comp_raw]

    # Full fitted values (with ARMA & seasonality)
    y_fitted = res.fittedvalues

    # --- Diagnostics ----------------------------------------------------------
    lb = lb_report(res.resid)
    kparams = res.params.size
    print(f"\n=== ARIMAX (levels, factors={args.factors}) ===")
    print(f"Order: {args.order} | Seasonal: {args.sorder} | AIC: {res.aic:.3f} | AICc: {aicc(res.aic, res.nobs, kparams):.3f}")
    print(f"Intercept: {intercept:.6f}")
    print("Betas (exogenous):")
    print(beta_exog)
    print("Ljung–Box p-values:", lb)

    # --- Plot -----------------------------------------------------------------
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(y.index, y.values, lw=2, label=f"{args.region} (actual)")
    for s, lab in zip(comp_lines, comp_labels):
        ax.plot(s.index, s.values, lw=2, label=lab + " (aligned)")
    ax.plot(y_fitted.index, y_fitted.values, lw=2, alpha=0.9,
            label=f"ARMAX fitted (order {args.order}, seasonal {args.sorder})")

    ax.set_title(f"{args.region} — ARMAX (levels) — factors={args.factors} — order={args.order} — sorder={args.sorder} — align={args.align}", fontsize=8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index level (log units)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    if args.file == 'show':
        plt.show()
    else:
        show_or_save_fig(args, handle=fig, subname='')

if __name__ == "__main__":
    main()
