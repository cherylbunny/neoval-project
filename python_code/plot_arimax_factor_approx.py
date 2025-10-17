#!/usr/bin/env python
"""
ARIMAX factor plotting script (levels, d=0)

Plots for a chosen region:
  • Actual index y_t
  • Deterministic factor components (c + β'X_subset), base-aligned to y
      - market only
      - market + mining
      - market + mining + lifestyle (if present)
  • ARMAX fitted values

Optional remainder panel (--resid_panel):
  • r_t = ŷ_t − (c_LR + β'X_t), where c_LR = c / [ (1-Σφ_i) * (1-ΣΦ_j) ].
    This removes the spurious offset due to AR (and seasonal AR) structure.
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

def region2label(region, remove_greater=True, max_components=3, max_len=50):
    region = region.replace('AUSTRALIAN CAPITAL TERRITORY', 'ACT').replace('SYDNEY - ','')
    region = region.strip()
    if remove_greater:
        region = region.replace('GREATER ', '')
    if ',' in region:
        return ",".join([
            region2label(e, remove_greater=remove_greater,
                         max_components=max_components, max_len=max_len)
            for e in region.split(',')
        ])
    def word2lower(string):
        if string in ['NSW','QLD','WA','SA','ACT','TAS','VIC','NT']:
            return string
        return string[0].upper() + string[1:].lower()
    def to_lower(string):
        comps = [word2lower(e) for e in string.split(" ")]
        return " ".join(comps)
    def abbrev(string):
        if len(string) > max_len:
            return ''.join(e[0] for e in string.split(' '))
        return string
    components = [abbrev(to_lower(e.strip())) for e in region.split('-')][:max_components]
    return " - ".join(components).strip()

def parse_order3(s: str):
    try:
        p,d,q = [int(x.strip()) for x in s.split(",")]
        return (p,d,q)
    except Exception:
        raise argparse.ArgumentTypeError("Order must be like '1,0,1'")

def parse_sorder4(s: str):
    if s.strip() == "" or s == "0,0,0,0":
        return (0,0,0,0)
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


def cov_table(df_fac: pd.DataFrame, factors=["market","mining","lifestyle"]):
    print("\n=== Factor covariances ===")

    formatters = {
        "market":  "{0:.2f}".format,
        "mining":  "{0:.2f}".format,
        "lifestyle":  "{0:.2f}".format,
    }

    factors = [f for f in factors if f in df_fac.columns]
    df_fac = df_fac[factors].astype(float).dropna()

    corr = df_fac.corr()
    print(corr)

    latex = corr.to_latex(
            index=True,
            escape=False,
            formatters=formatters,
        )
    print(f"\n{latex}\n")


    return corr


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
    parser.add_argument("--resid_panel", action="store_true",
                        help="Add bottom panel showing r_t = ŷ_t − (c_LR + β'X_t).")
    #parser.add_argument("--resid_roll", type=int, default=12,
    #                    help="Rolling mean window (months) for remainder panel (0 disables).")
    parser.add_argument("--resid_center", action="store_true",
                        help="De-mean the remainder after LR intercept adjustment.")
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
    df_idx = pd.read_csv(idx_path)
    df_fac = pd.read_csv(fac_path)
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

    cov_table(df_fac)

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

    # Intercept
    int_key = next((k for k in params.index
                    if isinstance(k, str) and (("intercept" in k.lower()) or (k.lower()=="const"))), None)
    intercept = float(params[int_key]) if int_key is not None else 0.0

    # Long-run AR multiplier for the intercept: 1 / [ (1-Σφ_i) * (1-ΣΦ_j) ]
    # non-seasonal AR terms:
    phi_ns = float(getattr(res, "arparams", np.array([])).sum()) if hasattr(res, "arparams") else 0.0
    # seasonal AR terms (names like 'ar.S.L12', 'ar.S.L24', ...)
    phi_seas = 0.0
    if param_names is not None:
        seas_ar_names = [n for n in params.index if isinstance(n, str) and n.startswith("ar.S.")]
        if len(seas_ar_names):
            phi_seas = float(params[seas_ar_names].sum())
    # build multiplier safely
    denom = (1.0 - phi_ns) * (1.0 - phi_seas)
    lr_const_mult = (1.0 / denom) if abs(denom) > 1e-8 else 1.0
    intercept_LR = intercept * lr_const_mult

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

    # --- Full fitted values (with ARMA & seasonality) ------------------------
    y_fitted = res.fittedvalues

    # --- Dynamic remainder: ŷ_t − (c_LR + β'X_t) -----------------------------
    det_full = intercept_LR + X[exog_names].mul(beta_exog, axis=1).sum(axis=1)
    idx_r = y_fitted.index.intersection(det_full.index)
    dynamic_remainder = (y_fitted.loc[idx_r] - det_full.loc[idx_r]).dropna()
    if args.resid_center:
        dynamic_remainder = dynamic_remainder - dynamic_remainder.mean()

    # --- Diagnostics ----------------------------------------------------------
    lb = lb_report(res.resid)
    kparams = res.params.size
    print(f"\n=== ARIMAX (levels, factors={args.factors}) ===")
    print(f"Order: {args.order} | Seasonal: {args.sorder} | AIC: {res.aic:.3f} | AICc: {aicc(res.aic, res.nobs, kparams):.3f}")
    print(f"Intercept: {intercept:.6f} | Σφ(ns)={phi_ns:.3f} | ΣΦ(seas)={phi_seas:.3f} | LR const mult={lr_const_mult:.3f}")
    print("Betas (exogenous):")
    print(beta_exog)
    print("Ljung–Box p-values:", lb)

    # --- Plot -----------------------------------------------------------------
    plt.close()
    if args.resid_panel:
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(10, 5.8), sharex=True,
            gridspec_kw=dict(height_ratios=[3,1], hspace=0.05)
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 4.2))

    # Top: main series + components + fitted
    ax.plot(y.index, y.values, lw=2, label=f"{region2label(args.region)} index (actual)", color='black')
    for s, lab in zip(comp_lines, comp_labels):
        ax.plot(s.index, s.values, lw=2, label=lab)
    ax.plot(y_fitted.index, y_fitted.values, lw=1.2, alpha=0.9,
            label=f"ARIMAX fitted (order {args.order}, seasonal {args.sorder})")

    ax.set_xlabel("Time")
    ax.set_ylabel("Index level (log units)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=7)

    # Bottom: remainder panel
    if args.resid_panel:
        axr.axhline(0.0, lw=1, linestyle='--', alpha=0.7)
        axr.plot(dynamic_remainder.index, dynamic_remainder.values, lw=1.3,
                 label="Remainder")

        axr.set_ylabel("Remainder")
        axr.grid(True, linestyle="--", alpha=0.4)
        axr.legend(loc="upper left", fontsize=7)

    plt.tight_layout()

    if args.file == 'show':
        plt.show()
    else:
        show_or_save_fig(args, handle=fig, subname='')

if __name__ == "__main__":
    main()
