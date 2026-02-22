#!/usr/bin/env python
"""
ARIMAX regression of a regional housing index on three market factors
with OLS + HAC, VIF collinearity checks, ARIMAX grid search, and optional
ARDL(1)+ARIMAX robustness (lags of exogenous factors).

New:
- --fixed_order p,d,q         (e.g., 2,0,2) forces the non-seasonal ARIMA order
- --order_2022                shorthand for --fixed_order 2,0,2
- --sorder P,D,Q,s            (e.g., 0,0,1,12) forces the seasonal SARIMA order
- --seasonal12                include a small seasonal grid at s=12 in the search
- --exclude COL [COL...]      drop exogenous cols (e.g., --exclude lifestyle)
"""

import os
import argparse
import numpy as np
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

# ---- Optional: silence common, harmless warnings ---------------------------
warnings.filterwarnings("ignore", message="No frequency information was provided")
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found")
warnings.filterwarnings("ignore", message="Non-invertible starting seasonal moving average Using zeros as starting parameters.")

# ---- Paths -----------------------------------------------------------------
home_dir   = os.path.expanduser("~")
paper_path = os.path.join(home_dir, "Documents/github/neoval-project")
data_path  = os.path.join(paper_path, "data")

# ---- Helper functions -------------------------------------------------------
def aicc(aic: float, nobs: int, kparams: int) -> float:
    """Finite-sample corrected AIC."""
    return aic + (2 * kparams * (kparams + 1)) / max(nobs - kparams - 1, 1)

def parse_sorder(s: str):
    """Parse seasonal order 'P,D,Q,s'."""
    try:
        P,D,Q,S = [int(x.strip()) for x in s.split(",")]
        return (P,D,Q,S)
    except Exception:
        raise argparse.ArgumentTypeError("Seasonal order must be like '0,0,1,12'")

def fit_arimax(y: pd.Series, X: pd.DataFrame, order=(1,0,0), seasonal_order=(0,0,0,0), trend='c'):
    """Fit an ARIMAX model with optional seasonal order."""
    mod = SARIMAX(
        endog=y, exog=X,
        order=order, seasonal_order=seasonal_order,
        trend=trend,   # centered exog -> keep intercept unless you explicitly remove it elsewhere
        enforce_stationarity=True, enforce_invertibility=True,
        measurement_error=False
    )
    return mod.fit(disp=False)

def ljungbox_report(resid, lags=[6,12,18,24]):
    """Print Ljung–Box p-values for several lags to check residual whiteness."""
    lb = acorr_ljungbox(resid, lags=lags, return_df=True)
    print("\nLjung–Box p-values:")
    for L, p in zip(lags, lb['lb_pvalue'].values):
        print(f"  lag {L:>2}: p = {p:.3f}")

def search_arimax_levels(y, X, seasonal_candidates=None, seasonal12=False):
    """
    Small ARIMA grid search over (p,0,q).
    - If seasonal_candidates is given (list of seasonal orders), search only those.
    - Else if seasonal12=True, use a small seasonal set at s=12.
    - Else search non-seasonal only.
    Returns a list of dicts sorted by AICc (lowest first).
    """
    cand_orders = [
        (0,0,0),
        (1,0,0),(0,0,1),(1,0,1),
        (2,0,0),(0,0,2),
        (2,0,1),(1,0,2),(2,0,2)
    ]
    if seasonal_candidates is None:
        seasonal_candidates = [(0,0,0,0)]
        if seasonal12:
            seasonal_candidates += [
                (0,0,1,12),  # SMA(1)
                (1,0,0,12),  # SAR(1)
                (1,0,1,12),  # SARMA(1,1)
            ]

    rows = []
    for seas in seasonal_candidates:
        for od in cand_orders:
            try:
                res = fit_arimax(y, X, order=od, seasonal_order=seas, trend='c')
                k = res.params.size
                aicc_val = aicc(res.aic, nobs=res.nobs, kparams=k)
                lb_p = acorr_ljungbox(res.resid, lags=[12], return_df=True)['lb_pvalue'].iloc[0]
                rows.append({'order': od, 'seasonal_order': seas, 'aicc': aicc_val, 'lb_p12': float(lb_p), 'res': res})
            except Exception as e:
                print(f"Order {od} / seasonal {seas} failed: {e}")
    rows.sort(key=lambda r: (r['aicc'], -r['lb_p12']))
    return rows


def ols_with_hac(y, X, maxlags=12):
    """OLS fit with HAC (Newey–West) robust SEs."""
    Xc = sm.add_constant(X)
    ols = sm.OLS(y, Xc).fit()
    ols_hac = ols.get_robustcov_results(cov_type='HAC', maxlags=maxlags)
    return ols, ols_hac

def compute_vif(X):
    """Variance Inflation Factors for exogenous regressors."""
    Xc = sm.add_constant(X)
    vif = pd.DataFrame()
    vif['variable'] = Xc.columns
    vif['VIF'] = [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
    return vif

def build_ardl1(X, y):
    """
    Build ARDL(1) design: add one lag of each factor as additional regressors.
    Drops first row to align y and lagged X.
    """
    X_lag = X.shift(1).add_suffix('_lag1')
    X_dyn = pd.concat([X, X_lag], axis=1)
    data_dyn = pd.concat([y.rename('y'), X_dyn], axis=1).dropna()
    return data_dyn['y'], data_dyn[X_dyn.columns]

# ---- Main -------------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--region', help="Region to model.", default="SYDNEY - BLACKTOWN")
    parser.add_argument('--region_level', help="Region level to model.", default="sa4_name", choices=['sa4_name', 'major_city'])

    parser.add_argument('--save_diag', action='store_true',
                        help="Save residual diagnostics plot.")
    parser.add_argument('--search', action='store_true',
                        help="Run ARIMA order grid search (p,0,q) and pick best by AICc.")
    parser.add_argument('--seasonal12', action='store_true',
                        help="Also consider a small seasonal grid at s=12 during search.")
    parser.add_argument('--sorder', type=parse_sorder, default=None,
                        help="Force seasonal order as 'P,D,Q,s' (e.g., '0,0,1,12').")
    parser.add_argument('--ardl1', action='store_true',
                        help="Also run ARDL(1)+ARIMAX robustness with lagged factors.")
    parser.add_argument('--fixed_order', type=str, default=None,
                        help="Force ARIMA order as 'p,d,q' (e.g., '2,0,2'). Overrides --search/default.")
    parser.add_argument('--order_2022', action='store_true',
                        help="Shorthand for --fixed_order 2,0,2")
    parser.add_argument('--exclude', nargs='*', default=[],
                        help="Exogenous columns to exclude (e.g., --exclude lifestyle)")

    args = parser.parse_args()
    if args.order_2022:
        args.fixed_order = "2,0,2"

    region = args.region

    # --- Load data
    df = pd.read_csv(os.path.join(data_path, 'df_factor_trends.csv'))

    if args.region_level == 'major_city':
        df_indexes= pd.read_csv(os.path.join(data_path, 'city_indexes.csv'))
    else:
        df_indexes= pd.read_csv(os.path.join(data_path, 'sa4_indexes.csv'))

    # --- Indexing & alignment
    df['month_date'] = pd.to_datetime(df['month_date'])
    df_indexes['month_date'] = pd.to_datetime(df_indexes['month_date'])

    df         = df.set_index('month_date').sort_index()
    df_indexes = df_indexes.set_index('month_date').sort_index()

    if region not in df_indexes.columns:
        raise ValueError(f"Region '{region}' not found in {('city_indexes.csv' if args.region_level=='major_city' else 'sa4_indexes.csv')} columns.")

    y = df_indexes[region].astype(float)

    # Choose factors (default 3-factor; allow exclusions)
    base_cols = ['market', 'mining', 'lifestyle']
    required_cols = [c for c in base_cols if c not in set(args.exclude)]

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in df_factor_trends.csv.")

    if len(required_cols) < 1:
        raise ValueError("No exogenous columns left after exclusions.")

    X = df[required_cols].astype(float)

    data = pd.concat([y.rename('y'), X], axis=1).dropna()
    y_al = data['y']
    X_al = data[required_cols]

    # --- OLS and HAC ----------------------------------------------------------
    ols, ols_hac = ols_with_hac(y_al, X_al)
    exog_cols = list(X_al.columns)
    ols_index = ['const'] + exog_cols

    print("\n=== OLS (levels) coefficients ===")
    print(pd.Series(ols.params, index=ols_index))

    print("\nNewey–West HAC standard errors (12 lags):")
    print(pd.Series(ols_hac.bse, index=ols_index))

    # --- VIF check ------------------------------------------------------------
    print("\n=== Variance Inflation Factors ===")
    print(compute_vif(X_al))

    # --- ARIMAX model ---------------------------------------------------------
    order = None
    seas_order = (0,0,0,0)
    res = None

    if args.fixed_order:
        # parse 'p,d,q'
        try:
            order = tuple(int(s.strip()) for s in args.fixed_order.split(','))
            if len(order) != 3:
                raise ValueError
        except Exception:
            raise ValueError("--fixed_order must be like 'p,d,q', e.g., '2,0,2'")

        # seasonal (forced) or default none
        seas_order = args.sorder if args.sorder is not None else (0,0,0,0)
        print(f"\nFitting ARIMAX with fixed orders: non-seasonal {order}, seasonal {seas_order} ...\n")
        res = fit_arimax(y_al, X_al, order=order, seasonal_order=seas_order, trend='c')
        k = res.params.size
        aicc_val = aicc(res.aic, nobs=res.nobs, kparams=k)
        lb_p12 = acorr_ljungbox(res.resid, lags=[12], return_df=True)['lb_pvalue'].iloc[0]
        print("AICc:", aicc_val)
        print("Ljung–Box p(12):", float(lb_p12))
        ljungbox_report(res.resid)

    elif args.search:
        if args.sorder is not None:
            print(f"\nRunning ARIMA grid search on (p,0,q) with seasonal order fixed to {args.sorder} ...")
            cands = search_arimax_levels(y_al, X_al, seasonal_candidates=[args.sorder])
        else:
            print("\nRunning expanded ARIMA grid search on (p,0,q) ...")
            cands = search_arimax_levels(y_al, X_al, seasonal12=args.seasonal12)

        best  = cands[0]
        order = best['order']
        seas_order = best['seasonal_order']
        res   = best['res']
        print("Best orders: non-seasonal", order, "| seasonal", seas_order)
        print("AICc:", best['aicc'])
        print("Ljung–Box p(12):", best['lb_p12'])
        ljungbox_report(res.resid)

    else:
        order = (1,0,0)
        seas_order = args.sorder if args.sorder is not None else (0,0,0,0)
        res = fit_arimax(y_al, X_al, order=order, seasonal_order=seas_order, trend='c')


    print("\n=== ARIMAX intercept & exogenous coefficients (levels) ===")
    # Make a labeled Series of all parameters
    param_names = getattr(res, "param_names", None)
    if param_names is None:  # very old statsmodels fallback
        param_s = pd.Series(res.params)
    else:
        param_s = pd.Series(res.params, index=param_names)

    exog_names = list(res.model.exog_names)
    # Try to find the intercept name across statsmodels versions
    int_key = next((k for k in param_s.index
                    if "intercept" in k.lower() or k.lower() == "const"), None)

    to_show = param_s[exog_names]
    if int_key is not None and int_key in param_s.index:
        to_show = pd.concat([param_s[[int_key]], to_show])

    print(to_show)

    # --- Optional residual diagnostics plot -----------------------------------
    if args.save_diag:
        fig = res.plot_diagnostics(figsize=(10,6))
        fig.suptitle(f"Residual diagnostics — {region} — order {order} — seasonal {seas_order} — X={exog_cols}", y=1.02)
        outdir = os.path.join(paper_path, 'figures')
        os.makedirs(outdir, exist_ok=True)
        fname = f"arimax_diag_{region.replace(' ','_').replace('/','-')}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches='tight')
        print(f"Saved diagnostics plot to {fname}")

    # --- Optional ARDL(1)+ARIMAX robustness ----------------------------------
    if args.ardl1:
        print("\nRunning ARDL(1)+ARIMAX robustness check...\n")
        y_dyn, X_dyn = build_ardl1(X_al, y_al)

        if args.fixed_order:
            print(f"Fitting ARDL(1)+ARIMAX with fixed orders: non-seasonal {order}, seasonal {seas_order} ...")
            res_dyn = fit_arimax(y_dyn, X_dyn, order=order, seasonal_order=seas_order, trend='c')
            k_dyn = res_dyn.params.size
            aicc_dyn = aicc(res_dyn.aic, nobs=res_dyn.nobs, kparams=k_dyn)
            lb_dyn = acorr_ljungbox(res_dyn.resid, lags=[12], return_df=True)['lb_pvalue'].iloc[0]
            print("ARDL(1)+ARIMAX fixed orders:", order, seas_order,
                  "AICc:", aicc_dyn,
                  "LB p(12):", float(lb_dyn))
            ljungbox_report(res_dyn.resid)
        else:
            cands_dyn = search_arimax_levels(y_dyn, X_dyn, seasonal12=args.seasonal12)
            best_dyn  = cands_dyn[0]
            res_dyn   = best_dyn['res']
            print("ARDL(1)+ARIMAX best orders:", best_dyn['order'], best_dyn['seasonal_order'],
                  "AICc:", best_dyn['aicc'],
                  "LB p(12):", best_dyn['lb_p12'])
            ljungbox_report(res_dyn.resid)

        cols = list(X_al.columns)  # contemporaneous factors
        betas_dyn = pd.Series(res_dyn.params[res_dyn.model.exog_names],
                              index=res_dyn.model.exog_names)
        comp = pd.DataFrame({
            'ARIMAX_contemp': res.params[res.model.exog_names].reindex(cols),
            'ARDL1_contemp':  betas_dyn.reindex(cols),
            'ARDL1_lag1':     betas_dyn.reindex([c + '_lag1' for c in cols]).values
        }, index=cols)
        print("\n=== Contemporaneous vs lagged betas (ARDL(1)+ARIMAX) ===")
        print(comp.round(4))

    # To see if seasonal does anything.
    #if res:
    #    print(res.params)          # look for 'seasonal_ma.L12' (or similar name)
    #    print(res.pvalues)
