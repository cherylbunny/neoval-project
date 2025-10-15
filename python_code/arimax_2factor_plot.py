#!/usr/bin/env python
"""
ARIMAX 2-factor plotting script (Market + Mining)
- Levels model (d = 0)
- Default order = (1,0,1), override via --order "p,q,r"
- Plots:
    1) Actual regional index (y)
    2) Market-only component: beta_market * market
    3) 2-factor component: beta_market * market + beta_mining * mining
    4) ARIMAX fitted values (includes ARMA error adjustment)
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
        trend='n',
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
    parser = argparse.ArgumentParser(description="Plot ARIMAX 2-factor projections for a region.")
    parser.add_argument("--region", default="SYDNEY - BLACKTOWN", help="Region name exactly as in indexes_city_and_sa4.csv")
    parser.add_argument("--order", type=parse_order, default=(1,0,1), help="ARIMA order as 'p,0,q' (default '1,0,1')")
    parser.add_argument("--start_year", type=int, default=1995, help="Trim sample start (inclusive, year)")
    parser.add_argument('-f', '--file', help = "Filename to save to. To show on screen 'show'.", default = 'show' )
    parser.add_argument('--fine_dpi', action='store_true', help="Finer dpi.", default = False)
    parser.add_argument("--future_exog_csv", default=None,
                        help="Optional CSV with future exogenous columns ['month_date','market','mining'] to forecast.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message="No frequency information was provided")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
    warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")

    # ---------- Load data ----------
    idx_path = os.path.join(DATA_PATH, "indexes_city_and_sa4.csv")
    fac_path = os.path.join(DATA_PATH, "df_factor_trends.csv")

    df_idx = pd.read_csv(idx_path)
    df_fac = pd.read_csv(fac_path)

    df_idx['month_date'] = pd.to_datetime(df_idx['month_date'])
    df_fac['month_date'] = pd.to_datetime(df_fac['month_date'])

    df_idx = df_idx.set_index('month_date').sort_index()
    df_fac = df_fac.set_index('month_date').sort_index()

    # Sanity checks
    if args.region not in df_idx.columns:
        raise ValueError(f"Region '{args.region}' not found in {idx_path} columns.")
    for c in ['market','mining']:
        if c not in df_fac.columns:
            raise ValueError(f"Column '{c}' missing in {fac_path}.")

    # Trim start & align
    df_idx = df_idx[df_idx.index >= f"{args.start_year}-01-01"]
    df_fac = df_fac[df_fac.index >= f"{args.start_year}-01-01"]

    y = df_idx[args.region].astype(float)
    X = df_fac[['market','mining']].astype(float)

    data = pd.concat([y.rename('y'), X], axis=1, join="inner").dropna()
    y_al = ensure_monthly_freq(data['y'])
    X_al = ensure_monthly_freq(data[['market','mining']])

    # ---------- Fit ARIMAX ----------
    order = args.order
    res = fit_arimax_levels(y_al, X_al, order=order)

    # Components from exogenous betas
    betas = res.params[res.model.exog_names]   # ['market','mining']
    # market-only & 2-factor *components* (exogenous part only)
    comp_market = X_al[['market']].dot(pd.Series({'market': betas['market']}))
    comp_2fact  = X_al.dot(pd.Series({'market': betas['market'], 'mining': betas['mining']}))

    # Fitted values (include ARMA error adjustment)
    y_fitted = res.fittedvalues

    # Residual diagnostics
    lb = lb_report(res.resid)
    kparams = res.params.size
    print("\n=== ARIMAX (levels, 2-factor) ===")
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
        for c in ['market','mining']:
            if c not in fut.columns:
                raise ValueError("future_exog_csv must have columns ['month_date','market','mining']")
        fut['month_date'] = pd.to_datetime(fut['month_date'])
        fut = fut.set_index('month_date').sort_index()
        # Only future months beyond last in-sample:
        fut = fut[fut.index > X_al.index.max()]
        if len(fut) > 0:
            fut = ensure_monthly_freq(fut[['market','mining']])
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
    ax.plot(comp_market.index, comp_market.values, lw=2, label="Market-only component")
    ax.plot(comp_2fact.index,  comp_2fact.values,  lw=2, label="Market + Mining component")
    ax.plot(y_fitted.index,    y_fitted.values,    lw=2, label=f"ARIMAX fitted (order {order})", alpha=0.9)

    # Forecast overlay (if any)
    if fc_mean is not None:
        ax.plot(fc_mean.index, fc_mean.values, lw=2, linestyle='--', label="Forecast (mean)")
        if fc_ci is not None:
            ax.fill_between(fc_ci.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.2, label="Forecast 95% CI")

    ax.set_title(f"{args.region} — ARIMAX 2-factor (levels), order={order}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Index level (log units)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()

    if args.file == 'show':
        plt.show()
    else:
        for h, subname in [ (fig, f'')]:
            show_or_save_fig(args, handle=h, subname=subname)




if __name__ == "__main__":
    main()
