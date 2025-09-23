#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

home_dir = os.path.expanduser("~")
paper_path = os.path.join(home_dir, 'Documents/github/neoval-project')
data_path = os.path.join(paper_path, 'data')

def aicc(aic: float, nobs: int, kparams: int) -> float:
    # finite-sample corrected AIC
    return aic + (2 * kparams * (kparams + 1)) / max(nobs - kparams - 1, 1)

def fit_arimax(y: pd.Series, X: pd.DataFrame, order=(1,1,1)):
    mod = SARIMAX(
        endog=y, exog=X,
        order=order, seasonal_order=(0,0,0,0),
        trend='n', enforce_stationarity=True, enforce_invertibility=True,
        measurement_error=True
    )

    res = mod.fit(disp=False)
    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--region', help="Region to model.", default="SYDNEY - BLACKTOWN", required=False)
    parser.add_argument('--save_diag', action='store_true', help="Save residual diagnostics plot.")
    args = parser.parse_args()
    region = args.region

    # --- Load data
    df_coefs = pd.read_csv(os.path.join(data_path, 'df_reg_coefs.csv'))
    df = pd.read_csv(os.path.join(data_path, 'df_factor_trends.csv'))     # has columns: month_date, market, mining, lifestyle
    df_indexes = pd.read_csv(os.path.join(data_path, 'indexes_city_and_sa4.csv'))

    # --- Indexing & alignment
    df_coefs = df_coefs.set_index('region')
    df['month_date'] = pd.to_datetime(df['month_date'])
    df_indexes['month_date'] = pd.to_datetime(df_indexes['month_date'])

    df = df.set_index('month_date').sort_index()
    df_indexes = df_indexes.set_index('month_date').sort_index()

    # Check region exists
    if region not in df_indexes.columns:
        raise ValueError(f"Region '{region}' not found in indexes_city_and_sa4.csv columns.")

    # y: regional level series μ_r
    y = df_indexes[region].astype(float)

    # X: exogenous factors in levels (U, δ_BS, δ_L)
    required_cols = ['market', 'mining', 'lifestyle']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in df_factor_trends.csv.")
    X = df[required_cols].astype(float)

    # Align on common monthly index and drop missing
    data = pd.concat([y.rename('y'), X], axis=1).dropna()
    y_al = data['y']
    X_al = data[required_cols]


    od = (0,1,0)
    res = fit_arimax(y, X, order=od)

    print(res.model.exog_names)  # ['const','market','mining','lifestyle']
    print(res.params[res.model.exog_names])

    resid = res.resid


    # Questions now: how to get y_hat and estimates of disturbance growth?


    # Residual diagnostics plot (optional)
    if args.save_diag:
        fig = res.plot_diagnostics(figsize=(10,6))
        fig.suptitle(f"Residual diagnostics — {region} — order {od}", y=1.02)
        outdir = os.path.join(paper_path, 'figures')
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"arimax_diag_{region.replace(' ','_').replace('/','-')}.png"), dpi=150, bbox_inches='tight')

    # --- (Optional) Forecast example (requires future paths for exog)
    # Build X_future with same columns and a monthly DateTimeIndex
    # X_future = pd.DataFrame({'market': ..., 'mining': ..., 'lifestyle': ...}, index=future_months_index)
    # fc = best_res.get_forecast(steps=len(X_future), exog=X_future)
    # fc_mean = fc.predicted_mean
    # fc_ci = fc.conf_int()
    # print(fc_mean.tail()); print(fc_ci.tail())
