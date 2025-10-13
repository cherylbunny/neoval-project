#!/usr/bin/env python
"""
ARIMAX regression of a regional housing index on three market factors
with OLS + HAC, VIF collinearity checks, ARIMAX grid search, and optional
ARDL(1)+ARIMAX robustness (lags of exogenous factors).
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

# ---- Paths -----------------------------------------------------------------
home_dir   = os.path.expanduser("~")
paper_path = os.path.join(home_dir, "Documents/github/neoval-project")
data_path  = os.path.join(paper_path, "data")

# ---- Helper functions -------------------------------------------------------
def aicc(aic: float, nobs: int, kparams: int) -> float:
    """Finite-sample corrected AIC."""
    return aic + (2 * kparams * (kparams + 1)) / max(nobs - kparams - 1, 1)

def fit_arimax(y: pd.Series, X: pd.DataFrame, order=(1,0,0)):
    """Fit an ARIMAX model (non-seasonal) with exogenous regressors X."""
    mod = SARIMAX(
        endog=y, exog=X,
        order=order, seasonal_order=(0,0,0,0),
        trend='n',
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

def search_arimax_levels(y, X):
    """
    Small ARIMA grid search over (p,0,q) using an expanded set of AR and MA orders.
    Returns a list of dicts sorted by AICc (lowest first).
    """
    cand_orders = [
        (0,0,0),
        (1,0,0),(0,0,1),(1,0,1),
        (2,0,0),(0,0,2),
        (2,0,1),(1,0,2),(2,0,2)
    ]
    rows = []
    for od in cand_orders:
        try:
            res = fit_arimax(y, X, order=od)
            k = res.params.size
            aicc_val = aicc(res.aic, nobs=res.nobs, kparams=k)
            lb_p = acorr_ljungbox(res.resid, lags=[12], return_df=True)['lb_pvalue'].iloc[0]
            rows.append({'order': od, 'aicc': aicc_val, 'lb_p12': float(lb_p), 'res': res})
        except Exception as e:
            print(f"Order {od} failed: {e}")
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
    parser.add_argument('--save_diag', action='store_true',
                        help="Save residual diagnostics plot.")
    parser.add_argument('--search', action='store_true',
                        help="Run ARIMA order grid search (p,0,q) and pick best by AICc.")
    parser.add_argument('--ardl1', action='store_true',
                        help="Also run ARDL(1)+ARIMAX robustness with lagged factors.")
    args = parser.parse_args()
    region = args.region

    # --- Load data
    df_coefs  = pd.read_csv(os.path.join(data_path, 'df_reg_coefs.csv'))
    df        = pd.read_csv(os.path.join(data_path, 'df_factor_trends.csv'))
    df_indexes= pd.read_csv(os.path.join(data_path, 'indexes_city_and_sa4.csv'))

    # --- Indexing & alignment
    df_coefs = df_coefs.set_index('region')
    df['month_date'] = pd.to_datetime(df['month_date'])
    df_indexes['month_date'] = pd.to_datetime(df_indexes['month_date'])

    df         = df.set_index('month_date').sort_index()
    df_indexes = df_indexes.set_index('month_date').sort_index()

    if region not in df_indexes.columns:
        raise ValueError(f"Region '{region}' not found in indexes_city_and_sa4.csv columns.")

    y = df_indexes[region].astype(float)
    required_cols = ['market', 'mining']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in df_factor_trends.csv.")
    X = df[required_cols].astype(float)

    data = pd.concat([y.rename('y'), X], axis=1).dropna()
    y_al = data['y']
    X_al = data[required_cols]

    # --- OLS and HAC ----------------------------------------------------------
    ols, ols_hac = ols_with_hac(y_al, X_al)
    exog_cols = list(X_al.columns)                 # e.g., ['market','mining']
    ols_index = ['const'] + exog_cols

    print("\n=== OLS (levels) coefficients ===")
    print(pd.Series(ols.params, index=ols_index))

    print("\nNewey–West HAC standard errors (12 lags):")
    print(pd.Series(ols_hac.bse, index=ols_index))


    # --- VIF check ------------------------------------------------------------
    print("\n=== Variance Inflation Factors ===")
    print(compute_vif(X_al))

    # --- ARIMAX model ---------------------------------------------------------
    if args.search:
        print("\nRunning expanded ARIMA grid search on (p,0,q)...\n")
        cands = search_arimax_levels(y_al, X_al)
        best  = cands[0]
        order = best['order']
        res   = best['res']
        print("Best order:", order)
        print("AICc:", best['aicc'])
        print("Ljung–Box p(12):", best['lb_p12'])
        ljungbox_report(res.resid)
    else:
        order = (1,0,0)  # default AR(1) if no search requested
        res   = fit_arimax(y_al, X_al, order=order)

    print("\n=== ARIMAX exogenous coefficients (levels) ===")
    print(res.params[res.model.exog_names])

    # --- Optional residual diagnostics plot -----------------------------------
    if args.save_diag:
        fig = res.plot_diagnostics(figsize=(10,6))
        fig.suptitle(f"Residual diagnostics — {region} — order {order}", y=1.02)
        outdir = os.path.join(paper_path, 'figures')
        os.makedirs(outdir, exist_ok=True)
        fname = f"arimax_diag_{region.replace(' ','_').replace('/','-')}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches='tight')
        print(f"Saved diagnostics plot to {fname}")

    # --- Optional ARDL(1)+ARIMAX robustness ----------------------------------
    if args.ardl1:
        print("\nRunning ARDL(1)+ARIMAX robustness check...\n")
        y_dyn, X_dyn = build_ardl1(X_al, y_al)
        cands_dyn = search_arimax_levels(y_dyn, X_dyn)
        best_dyn  = cands_dyn[0]
        res_dyn   = best_dyn['res']
        print("ARDL(1)+ARIMAX best order:", best_dyn['order'],
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
