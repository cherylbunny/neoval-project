#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rolling backtest for national ARIMAX model with interest rate scenarios.

Tests out-of-sample forecast accuracy at multiple horizons across rolling windows for three model configurations.

Model configurations tested:
    1. current    : rate_level + lags + changes, no seasonal, (1,0,0)
    2. simplified : rate_level only, no seasonal, (1,0,0)
    3. seasonal   : rate_level only, seasonal MA(1) at lag 12, (1,0,0)

Target variable:
    Cumulated log index levels over each forecast horizon, derived from monthly log returns.

Usage
-----
python rolling_backtest.py \
    --factor_csv  ../data/df_factor_trends.csv \
    --rate_csv    ../data/interest_rates.csv \
    --outdir      ../data_out \
    --start       2014-01 \
    --horizons    1 3 6 12 24

Outputs
-------
    backtest_errors.csv : raw forecast errors for every window, horizon and model
    backtest_metrics.csv : MAE, RMSE and MAPE summary statistics by model and horizon
    backtest_metrics_pivot.csv : same as backtest_metrics.csv but pivoted for easy comparison

"""

# ---------------------- imports ----------------------

import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------- imported utilities ----------------------

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_national_arimax_funnel import (
    to_month_start,
    safe_numeric,
    build_rate_features,
    load_rate_series,
)

# ---------------------- default paths ----------------------

HOME      = os.path.expanduser("~")
PAPER_PATH = os.path.join(HOME, "Documents/github/neoval-project")
DATA_PATH  = os.path.join(PAPER_PATH, "data")
DATA_OUT   = os.path.join(PAPER_PATH, "data_out")

# ---------------------- model configurations ----------------------

MODEL_CONFIGS = {
    "current": {
        "features":        ["level", "d1"],
        "rate_lags":       [0, 3, 6],
        "order":           (1, 0, 0),
        "seasonal_order":  (0, 0, 0, 0),
    },
    "simplified": {
        "features":        ["level"],
        "rate_lags":       [0],
        "order":           (1, 0, 0),
        "seasonal_order":  (0, 0, 0, 0),
    },
    "seasonal": {
        "features":        ["level"],
        "rate_lags":       [0],
        "order":           (1, 0, 0),
        "seasonal_order":  (0, 0, 1, 12),
    },
}

# ---------------------- utilities ----------------------

def load_market_series(factor_csv, market_col="market"):
    df = pd.read_csv(factor_csv)
    df["month_date"] = to_month_start(df["month_date"])
    df = df.set_index("month_date").sort_index().asfreq("MS")
    y = safe_numeric(df[market_col]).rename("market_log_index")
    return y


def fit_model(y, X, order, seasonal_order):
    mod = SARIMAX(
        endog=y,
        exog=X,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    return mod.fit(disp=False, maxiter=750)


def cumulate_log_returns(base_log, returns):
    """converts forecast returns to log index"""
    return base_log + np.cumsum(returns)


# ---------------------- backtest ----------------------

def run_backtest(y_log, rate, config, horizons, start_date):
    """
    Fits the model on expending windows and records forecast errors 
    at each horizon. 

    Returns DataFrame of errors with columns:
        window_end, horizon, actual_log, forecast_log, error
    """
    features       = config["features"]
    lags           = config["rate_lags"]
    order          = config["order"]
    seasonal_order = config["seasonal_order"]
    y_returns = y_log.diff().dropna().asfreq("MS")


    # build full interest rate feature matrix 
    X_full = build_rate_features(rate, features=features, lags=lags)

    # join returns and rate features 
    data = pd.concat([y_returns.rename("y"), X_full], axis=1).dropna()

    # set up rolling window end points 
    max_horizon    = max(horizons)
    all_dates      = data.index
    start_ts       = pd.Timestamp(start_date)
    end_ts         = all_dates[-1] - pd.offsets.MonthBegin(max_horizon)
    window_ends    = all_dates[(all_dates >= start_ts) & (all_dates <= end_ts)]
    
    records = []

    for t in window_ends:
        # training data 
        train       = data.loc[data.index <= t]
        y_train     = train["y"].asfreq("MS")
        X_train_raw = train.drop(columns=["y"]).asfreq("MS")
        mu          = X_train_raw.mean()
        X_train     = X_train_raw - mu

        # fit model
        try:
            res = fit_model(y_train, X_train, order, seasonal_order)
        except Exception as e:
            print(f"[warn] fit failed at {t.date()}: {e}")
            continue
        base_log = float(y_log.loc[t])

        # forecast each horizon
        for h in horizons:
            # future dates
            future_idx = pd.date_range(
                start=t + pd.offsets.MonthBegin(1),
                periods=h,
                freq="MS"
            )

            # mean centers actual interest rate 
            X_future_raw = X_full.loc[future_idx].copy()
            if X_future_raw.isna().any().any():
                continue
            X_future = X_future_raw - mu

            # get forecast
            try:
                pred      = res.get_forecast(steps=h, exog=X_future)
                ret_mean  = pred.predicted_mean.to_numpy()
            except Exception as e:
                print(f"[warn] forecast failed at {t.date()} h={h}: {e}")
                continue

            # cumulate returns to log index
            forecast_log = cumulate_log_returns(base_log, ret_mean)[-1]

            # actual log index at t+h
            actual_date = future_idx[-1]
            if actual_date not in y_log.index:
                continue
            actual_log = float(y_log.loc[actual_date])

            records.append({
                "window_end":   t,
                "horizon":      h,
                "actual_log":   actual_log,
                "forecast_log": forecast_log,
                "error":        forecast_log - actual_log,
            })

    return pd.DataFrame(records)


def compute_metrics(errors_df):
    """
    Compute MAE, RMSE, MAPE by horizon.
    """
    rows = []
    for h, grp in errors_df.groupby("horizon"):
        err        = grp["error"].to_numpy()
        actual_log = grp["actual_log"].to_numpy()
        mae        = float(np.mean(np.abs(err)))
        rmse       = float(np.sqrt(np.mean(err ** 2)))
        mape       = float(np.mean(np.abs(err) / np.abs(actual_log)) * 100)
        rows.append({
            "horizon":        h,
            "n_windows":      len(grp),
            "mae":            round(mae,  6),
            "rmse":           round(rmse, 6),
            "mape_pct":       round(mape, 3),
        })
    return pd.DataFrame(rows).sort_values("horizon")

# ---------------------- main ----------------------

def main():
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser()
    ap.add_argument("--factor_csv", default=os.path.join(DATA_PATH, "df_factor_trends.csv"))
    ap.add_argument("--rate_csv",   default=os.path.join(DATA_PATH, "interest_rates.csv"))
    ap.add_argument("--outdir",     default=DATA_OUT)
    ap.add_argument("--start",      default="2014-01",
                    help="Start date for rolling windows. Default: 2014-01.")
    ap.add_argument("--horizons",   nargs="+", type=int, default=[1, 3, 6, 12, 24],
                    help="Forecast horizons in months. Default: 1 3 6 12 24.")
    ap.add_argument("--models",     nargs="+",
                    choices=list(MODEL_CONFIGS.keys()),
                    default=list(MODEL_CONFIGS.keys()),
                    help="Which models to backtest. Default: all three.")
    args = ap.parse_args()
    
    # output folder 
    os.makedirs(args.outdir, exist_ok=True)

    print("[loading data]")
    y_log = load_market_series(args.factor_csv)
    rate, _ = load_rate_series(args.rate_csv, date_col="time", rate_col="")

    all_metrics = []
    all_errors  = []

    for model_name in args.models:
        config = MODEL_CONFIGS[model_name]
        print(f"\n[backtesting] {model_name}")
        print(f"  features:       {config['features']}")
        print(f"  rate_lags:      {config['rate_lags']}")
        print(f"  order:          {config['order']}")
        print(f"  seasonal_order: {config['seasonal_order']}")

        errors_df = run_backtest(
            y_log      = y_log,
            rate       = rate,
            config     = config,
            horizons   = args.horizons,
            start_date = args.start,
        )

        if errors_df.empty:
            print(f"  [warn] no results for {model_name}")
            continue

        errors_df["model"] = model_name
        all_errors.append(errors_df)

        metrics_df = compute_metrics(errors_df)
        metrics_df["model"] = model_name
        all_metrics.append(metrics_df)

        print(metrics_df[["horizon", "n_windows", "mae", "rmse", "mape_pct"]].to_string(index=False))

    if all_errors and all_metrics:
        errors_out  = pd.concat(all_errors,  ignore_index=True)
        metrics_out = pd.concat(all_metrics, ignore_index=True)

        errors_path  = os.path.join(args.outdir, "backtest_errors.csv")
        metrics_path = os.path.join(args.outdir, "backtest_metrics.csv")
        pivot_path   = os.path.join(args.outdir, "backtest_metrics_pivot.csv")

        errors_out.to_csv(errors_path,   index=False, float_format="%.8f")
        metrics_out.to_csv(metrics_path, index=False, float_format="%.6f")

        pivot = metrics_out.pivot_table(
            index="horizon",
            columns="model",
            values=["mae", "rmse", "mape_pct"]
        ).round(6)
        pivot.to_csv(pivot_path)

        print(f"\n[ok] wrote errors:      {errors_path}")
        print(f"[ok] wrote metrics:     {metrics_path}")
        print(f"[ok] wrote pivot table: {pivot_path}")

    print("\n[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())