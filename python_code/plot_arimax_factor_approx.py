#!/usr/bin/env python
"""
ARIMAX factor plotting script
- Levels model (d = 0)
- Default order = (1,0,1), override via --order "p,0,q"
- Choose factors via --factors (default: market mining)
- Plots:
    1) Actual regional index (y)
    2) Cumulative exogenous components:
       - with 2 factors: market, market+mining
       - with 3 factors incl. lifestyle: market, market+mining, market+mining+lifestyle
    3) ARIMAX fitted values (includes ARMA error adjustment)
- Optional forecast if you provide a CSV with future exogenous factors
"""

import os
import argparse
import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

# ----------------------- Config / Paths -----------------------
HOME = os.path.expanduser("~")
PAPER_PATH = os.path.join(HOME, "Documents/github/neoval-project")
DATA_PATH  = os.path.join(PAPER_PATH, "data")

paper_path = '/Users/wsijp/Documents/github/neoval-project'

def make_file_path(file, base_path = os.path.join(paper_path,'figures')):
    if len(file) < 1:
        raise Exception('Empty string provided for filename')
    elif file[0] == '/':
        return file
    return os.path.join(base_path, file)

def insert_subname(file, subname):
    if not subname:
        return file
    L = file.split('.')
    if len(L) < 2:
        raise Exception('Provide file format via dot')
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
        if args.fine_dpi:
            kwargs['dpi'] = 300
        handle.savefig(file_path, bbox_inches='tight', pad_inches=0.01, **kwargs)

def parse_order(s: str):
    """Parse --order like '1,0,1' into tuple(int,int,int)."""
    try:
        p,d,q = [int(x.strip()) for x in s.split(",")]
        return (p,d,q)
    except Exception:
        raise argparse.ArgumentTypeError("Order must be like '1,0,1'")

def ensure_monthly_freq(df_or_series):
    """Force a DateTimeIndex to MS (month start) freq for SARIMAX niceness."""
    obj = df_or_series.copy()
    obj.index = pd.DatetimeIndex(obj.index)
    return obj.asfreq("MS")

def aicc(aic: float, nobs: int, kparams: int) -> float:
    return aic + (2 * kparams * (kparams + 1)) / max(nobs - kparams - 1, 1)

def fit_arimax_levels(y: pd.Series, X: pd.DataFrame, order=(1,0,1)):
    """ARIMAX in levels (no intercept), non-seasonal."""
    mod = SARIMAX(
        endog=y, exog=X,
        order=order, seasonal_order=(0,0,0,0),
        trend='c',
        enforce_stationarity=True, enforce_invertibility=True,
        measurement_error=False
    )
    res = mod.fit(disp=False)
    return res

def lb_report(resid, lags=[6,12,18,24]):
    lb = acorr_ljungbox(resid, lags=lags, return_df=True)
    return {int(L): float(p) for L, p in zip(lb.index, lb['lb_pvalue'].values)}

def build_future_index(last_date: pd.Timestamp, months: int):
    idx = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    return idx

# ----------------------- Main ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot ARIMAX projections for a region with chosen factors.")
    parser.add_argument("--region", default="SYDNEY - BLACKTOWN", help="Region name exactly as in sa4_indexes.csv")
    parser.add_argument("--region_level", help="Region level to model.", default="sa4_name", choices=['sa4_name', 'major_city'])
    parser.add_argument("--order", type=parse_order, default=(1,0,1), help="ARIMA order as 'p,0,q' (default '1,0,1')")
    parser.add_argument("--start_year", type=int, default=1995, help="Trim sample start (inclusive, year)")
    parser.add_argument("--factors", nargs="+", default=["market","mining"],
                        help="List of factor columns to use (e.g., --factors market mining lifestyle)")
    parser.add_argument('-f', '--file', help = "Filename to save to. To show on screen 'show'.", default = 'show' )
    parser.add_argument('--fine_dpi', action='store_true', help="Finer dpi.", default = False)
    parser.add_argument("--future_exog_csv", default=None,
                        help="Optional CSV with future exogenous columns ['month_date'] + --factors to forecast.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message="No frequency information was provided")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
    warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")

    # ---------- Load data ----------
    idx_path = os.path.join(DATA_PATH, "sa4_indexes.csv")
    fac_path = os.path.join(DATA_PATH, "df_factor_trends.csv")

    if args.region_level == 'major_city':
        idx_path = os.path.join(DATA_PATH, "city_indexes.csv")
    elif args.region_level == 'sa4_name':
        idx_path = os.path.join(DATA_PATH, "sa4_indexes.csv")
    else:
        raise ValueError(f"Invalid --region_level {args.region_level}")

    df_idx = pd.read_csv(idx_path)
    df_fac = pd.read_csv(fac_path)

    df_idx['month_date'] = pd.to_datetime(df_idx['month_date'])
    df_fac['month_date'] = pd.to_datetime(df_fac['month_date'])

    df_idx = df_idx.set_index('month_date').sort_index()
    df_fac = df_fac.set_index('month_date').sort_index()

    # Sanity checks
    if args.region not in df_idx.columns:
        raise ValueError(f"Region '{args.region}' not found in {idx_path} columns.")
    for c in args.factors:
        if c not in df_fac.columns:
            raise ValueError(f"Column '{c}' missing in {fac_path}.")

    # Trim start & align
    df_idx = df_idx[df_idx.index >= f"{args.start_year}-01-01"]
    df_fac = df_fac[df_fac.index >= f"{args.start_year}-01-01"]

    y = df_idx[args.region].astype(float)
    X = df_fac[args.factors].astype(float)

    data = pd.concat([y.rename('y'), X], axis=1, join="inner").dropna()
    y_al = ensure_monthly_freq(data['y'])
    X_al = ensure_monthly_freq(data[args.factors])

    # ---------- Fit ARIMAX ----------
    order = args.order
    res = fit_arimax_levels(y_al, X_al, order=order)

    # Components from exogenous betas
    betas = pd.Series(res.params[res.model.exog_names], index=res.model.exog_names)

    # Build cumulative components for plotting:
    # - Always plot first factor alone.
    # - If 2+ factors: plot cumulative sum of first two.
    # - If 3+ factors AND specifically ['market','mining','lifestyle']: also plot cumulative of first three.
    def cum_component(cols):
        return X_al[cols].dot(betas.reindex(cols))

    comp_lines = []
    comp_labels = []

    # First factor only
    comp_lines.append(cum_component([args.factors[0]]))
    if args.factors[0] == 'market':
        comp_labels.append("Market-only component")
    else:
        comp_labels.append(f"{args.factors[0]}-only component")

    # First two
    if len(args.factors) >= 2:
        comp_lines.append(cum_component(args.factors[:2]))
        if args.factors[:2] == ['market','mining']:
            comp_labels.append("Market + Mining component")
        else:
            comp_labels.append(" + ".join(args.factors[:2]) + " component")

    # First three (only if lifestyle is the 3rd, per your request)
    if len(args.factors) >= 3 and args.factors[:3] == ['market','mining','lifestyle']:
        comp_lines.append(cum_component(args.factors[:3]))
        comp_labels.append("Market + Mining + Lifestyle component")

    # Fitted values (include ARMA error adjustment)
    y_fitted = res.fittedvalues

    # Residual diagnostics
    lb = lb_report(res.resid)
    kparams = res.params.size
    print(f"\n=== ARIMAX (levels, factors={args.factors}) ===")
    print(f"Order: {order} | AIC: {res.aic:.3f} | AICc: {aicc(res.aic, res.nobs, kparams):.3f}")
    print("Betas (exogenous):")
    print(betas)
    print("Ljung–Box p-values:", lb)

    # ---------- Optional forecast ----------
    fc_mean = None
    fc_ci   = None
    if args.future_exog_csv:
        fut = pd.read_csv(args.future_exog_csv)
        if 'month_date' not in fut.columns:
            raise ValueError("future_exog_csv must have column 'month_date'")
        for c in args.factors:
            if c not in fut.columns:
                raise ValueError(f"future_exog_csv must include columns ['month_date'] + {args.factors}")
        fut['month_date'] = pd.to_datetime(fut['month_date'])
        fut = fut.set_index('month_date').sort_index()
        # Only future months beyond last in-sample:
        fut = fut[fut.index > X_al.index.max()]
        if len(fut) > 0:
            fut = ensure_monthly_freq(fut[args.factors])
            fc = res.get_forecast(steps=len(fut), exog=fut)
            fc_mean = fc.predicted_mean
            fc_ci   = fc.conf_int()
            print(f"\nForecasting {len(fut)} months into the future...")
        else:
            print("\n[Info] Provided future_exog_csv has no rows beyond in-sample end; skipping forecast.")

    # ---------- Plot ----------
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 4.2))

    ax.plot(y_al.index, y_al.values, lw=2, label=f"{args.region} (actual)")

    # plot components
    for s, lab in zip(comp_lines, comp_labels):
        ax.plot(s.index, s.values, lw=2, label=lab)

    ax.plot(y_fitted.index, y_fitted.values, lw=2, label=f"ARMAX fitted (order {order})", alpha=0.9)

    # Forecast overlay (if any)
    if fc_mean is not None:
        ax.plot(fc_mean.index, fc_mean.values, lw=2, linestyle='--', label="Forecast (mean)")
        if fc_ci is not None:
            ax.fill_between(fc_ci.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.2, label="Forecast 95% CI")

    ax.set_title(f"{args.region} — ARMAX (levels) — factors={args.factors} — order={order}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Index level (log units)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()

    if args.file == 'show':
        plt.show()
    else:
        for h, subname in [(fig, f'')]:
            show_or_save_fig(args, handle=h, subname=subname)

if __name__ == "__main__":
    main()
