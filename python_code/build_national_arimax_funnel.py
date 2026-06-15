#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
National housing-market ARIMAX forecast with interest-rate scenarios.

Purpose
-------
Uses the factor-model 'market' series in df_factor_trends.csv as a provisional
Australian national log house price index. Fits a SARIMAX/ARIMAX model to the
monthly log level or monthly log returns, with mortgage/interest-rate features
as exogenous regressors, and produces conditional forecast funnels under
alternative interest-rate paths.

This is intentionally scenario-based: it does not try to forecast interest rates
with a VAR. Instead, it asks what the national index forecast looks like given
specified interest-rate paths.

Typical usage
-------------
  ./build_national_arimax_funnel.py --h 60 --scenarios flat down100 up100

  # Log-level model, with the ARIMA differencing handled inside SARIMAX
  ./build_national_arimax_funnel.py \
      --h 84 \
      --features level d1 \
      --rate_lags 0 3 6 \
      --fixed_order 1,1,1

  # Monthly-return model; forecasts are cumulated back to log levels for plots
  ./build_national_arimax_funnel.py \
      --target monthly_return \
      --h 84 \
      --features level d1 d3 d6 d12 \
      --rate_lags 0 3 6 12

Outputs
-------
CSV forecasts, model summary CSV, and PNG/PDF plots are written to data_out by
default.
"""

import os
import argparse
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

# ---------------------- default paths ----------------------
HOME = os.path.expanduser("~")
PAPER_PATH = os.path.join(HOME, "Documents/github/neoval-project")
DATA_PATH  = os.path.join(PAPER_PATH, "data")
DATA_OUT   = os.path.join(PAPER_PATH, "data_out")

# ---------------------- small utilities ----------------------

def expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def aicc_from_res(res) -> float:
    """Finite-sample corrected AIC from a fitted statsmodels result."""
    k = int(res.params.shape[0])
    n = int(res.nobs)
    if n - k - 1 <= 0:
        return float(res.aic)
    return float(res.aic + (2.0 * k * (k + 1)) / (n - k - 1))


def parse_order(s: str) -> Tuple[int, int, int]:
    try:
        out = tuple(int(x.strip()) for x in s.split(","))
        if len(out) != 3:
            raise ValueError
        return out  # type: ignore[return-value]
    except Exception as exc:
        raise argparse.ArgumentTypeError("Order must be like '1,1,1'.") from exc


def to_month_start(x: pd.Series) -> pd.DatetimeIndex:
    """Parse dates and coerce them to month start."""
    # pd.DatetimeIndex handles strings like 'Jan-1959' in the user's file.
    dt = pd.DatetimeIndex(x)
    return dt.to_period("M").to_timestamp(how="start")


def ensure_monthly_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = to_month_start(df[date_col])
    df = df.set_index(date_col).sort_index()
    # Multiple entries in same month are averaged; normally this should not happen.
    df = df.groupby(df.index).mean(numeric_only=True)
    return df.asfreq("MS")


def lb_pvals(resid: Sequence[float], lags: Sequence[int] = (12, 24)) -> Dict[int, float]:
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    out = {int(L): np.nan for L in lags}
    if r.size < max(lags) + 5:
        return out
    tab = acorr_ljungbox(r, lags=list(lags), return_df=True)
    for L in lags:
        if L in tab.index:
            out[int(L)] = float(tab.loc[L, "lb_pvalue"])
    return out


def safe_numeric(s: pd.Series) -> pd.Series:
    """Convert RBA-style strings to numeric values."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("%", "", regex=False)
         .str.replace("–", "-", regex=False)
         .replace({"..": np.nan, "nan": np.nan, "None": np.nan, "": np.nan}),
        errors="coerce"
    )

# ---------------------- data loading ----------------------

def load_market_series(factor_csv: str, market_col: str) -> pd.Series:
    df = pd.read_csv(expand(factor_csv))
    if "month_date" not in df.columns:
        raise ValueError(f"Expected a 'month_date' column in {factor_csv}.")
    df["month_date"] = to_month_start(df["month_date"])
    df = df.set_index("month_date").sort_index().asfreq("MS")
    if market_col not in df.columns:
        raise ValueError(
            f"Column '{market_col}' not found in {factor_csv}. "
            f"Available columns include: {list(df.columns)[:12]}"
        )
    y = safe_numeric(df[market_col]).rename("market_log_index").dropna()
    if y.empty:
        raise ValueError(f"Column '{market_col}' has no usable numeric values.")
    return y


def make_model_target(y_log: pd.Series, target: str) -> pd.Series:
    """Return the series actually modelled by SARIMAX."""
    target = target.lower()
    if target == "log_level":
        return y_log.rename("market_log_index")
    if target == "monthly_return":
        # Monthly log return. Forecasts from this target are cumulated back to
        # log levels for the plots and output columns.
        return y_log.diff().rename("market_log_return").dropna()
    raise ValueError("target must be either 'log_level' or 'monthly_return'.")


def choose_rate_column(df: pd.DataFrame, date_col: str, requested: str = "") -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested rate_col '{requested}' not found. Columns: {list(df.columns)}")
        return requested

    # Prefer common names / RBA IDs if present. The first two are useful if the
    # user has kept the RBA F5 series IDs after processing.
    priority_exact = [
        "FILRHLBVD",      # discounted variable housing loan rate, owner-occupier, banks
        "FILRHLBVS",      # standard variable housing loan rate, owner-occupier, banks
        "mortgage_rate",
        "housing_rate",
        "variable_mortgage_rate",
        "rate",
        "cash_rate",
    ]
    for col in priority_exact:
        if col in df.columns:
            return col

    candidates = []
    for col in df.columns:
        if col == date_col:
            continue
        vals = safe_numeric(df[col])
        n = int(vals.notna().sum())
        if n == 0:
            continue
        name = col.lower()
        score = n
        if any(tok in name for tok in ["mortgage", "housing", "lending", "loan", "rate", "filr"]):
            score += 10_000
        candidates.append((score, n, col))

    if not candidates:
        raise ValueError("Could not find a numeric interest-rate column in the interest-rate CSV.")

    candidates.sort(reverse=True)
    return candidates[0][2]


def load_rate_series(rate_csv: str, date_col: str, rate_col: str) -> Tuple[pd.Series, str]:
    df = pd.read_csv(expand(rate_csv))

    if date_col not in df.columns:
        # Fall back to common alternatives.
        alternatives = ["time", "month_date", "date", "Date", "month", "Month"]
        found = next((c for c in alternatives if c in df.columns), None)
        if found is None:
            raise ValueError(
                f"Date column '{date_col}' not found in {rate_csv}. "
                f"Available columns: {list(df.columns)}"
            )
        date_col = found

    chosen = choose_rate_column(df, date_col, requested=rate_col)
    df[date_col] = to_month_start(df[date_col])
    s = safe_numeric(df[chosen])
    out = pd.Series(s.values, index=df[date_col], name="rate").sort_index()
    out = out.groupby(out.index).mean().asfreq("MS").dropna()
    if out.empty:
        raise ValueError(f"Rate column '{chosen}' has no usable numeric values.")
    return out, chosen

# ---------------------- exogenous feature construction ----------------------

def build_rate_features(rate: pd.Series,
                        features: Sequence[str],
                        lags: Sequence[int]) -> pd.DataFrame:
    """
    Construct interest-rate features.

    Supported base features:
      level : rate in percentage points, e.g. 6.25
      d1    : one-month change in percentage points
      d3    : three-month change in percentage points
      d6    : six-month change in percentage points
      d12   : twelve-month change in percentage points
    """
    rate = rate.astype(float).sort_index().asfreq("MS")
    base = {}
    for f in features:
        f = f.lower()
        if f == "level":
            base["rate_level"] = rate
        elif f == "d1":
            base["d_rate_1m"] = rate.diff(1)
        elif f == "d3":
            base["d_rate_3m"] = rate.diff(3)
        elif f == "d6":
            base["d_rate_6m"] = rate.diff(6)
        elif f == "d12":
            base["d_rate_12m"] = rate.diff(12)
        else:
            raise ValueError(f"Unsupported feature '{f}'. Use one of: level, d1, d3, d6, d12.")

    X = pd.DataFrame(index=rate.index)
    for name, ser in base.items():
        for lag in lags:
            lag = int(lag)
            if lag < 0:
                raise ValueError("rate_lags must be non-negative.")
            col = name if lag == 0 else f"{name}_lag{lag}"
            X[col] = ser.shift(lag)
    return X


# ---------------------- macro (multi-series) exogenous features ----------------------

# Default baseline regressor set agreed for the national funnel: full-history series
# with defensible signal on monthly returns. Each spec is "series:transform[:lag]".
#   transform in {level, d1, d3, d6, d12, yoy}
#     level : the series as-is (percentage points, index level, etc.)
#     d1/d3/d6/d12 : k-month difference of the level
#     yoy   : 12-month log growth, i.e. log(s) - log(s).shift(12); for strictly
#             positive flow/stock series (approvals, commencements, income, NOM, CPI)
# Lags are applied after the transform, in months (default 0).
DEFAULT_MACRO_FEATURES = [
    "mortgage_rate:level",
    "mortgage_rate:d3",
    "building_approvals:yoy",
    "housing_credit_growth:level",
    "unemployment_rate:level",
]


def parse_macro_spec(spec: str) -> Tuple[str, str, int]:
    """Parse 'series:transform[:lag]' into (series, transform, lag)."""
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) == 2:
        series, transform = parts
        lag = 0
    elif len(parts) == 3:
        series, transform, lag_s = parts
        lag = int(lag_s)
    else:
        raise argparse.ArgumentTypeError(
            f"Macro spec '{spec}' must be 'series:transform' or 'series:transform:lag'."
        )
    transform = transform.lower()
    if transform not in {"level", "d1", "d3", "d6", "d12", "yoy"}:
        raise argparse.ArgumentTypeError(
            f"Unsupported transform '{transform}' in '{spec}'. "
            f"Use level, d1, d3, d6, d12, or yoy."
        )
    if lag < 0:
        raise argparse.ArgumentTypeError("Macro feature lag must be non-negative.")
    return series, transform, lag


def transform_macro_series(s: pd.Series, transform: str) -> pd.Series:
    """Apply a single transform to one macro series already on a monthly index."""
    s = s.astype(float).sort_index().asfreq("MS")
    if transform == "level":
        out = s
    elif transform == "yoy":
        # 12-month log growth; guard against non-positive values (e.g. net
        # overseas migration went negative during the 2020-21 border closure).
        out = np.log(s.where(s > 0)).diff(12)
    else:
        k = {"d1": 1, "d3": 3, "d6": 6, "d12": 12}[transform]
        out = s.diff(k)
    # Never let +/-inf leak into the design matrix; treat as missing so the
    # downstream dropna / missing-value checks handle it explicitly.
    return out.replace([np.inf, -np.inf], np.nan)


def load_macro_frame(macro_csv: str, date_col: str) -> pd.DataFrame:
    """Load the wide macro CSV onto a clean monthly index, numeric columns only."""
    df = pd.read_csv(expand(macro_csv))
    if date_col not in df.columns:
        alternatives = ["time", "month_date", "date", "Date", "month", "Month"]
        found = next((c for c in alternatives if c in df.columns), None)
        if found is None:
            raise ValueError(
                f"Date column '{date_col}' not found in {macro_csv}. "
                f"Available columns: {list(df.columns)}"
            )
        date_col = found
    df[date_col] = to_month_start(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df.apply(safe_numeric)
    df = df.groupby(df.index).mean(numeric_only=True).asfreq("MS")
    return df


def build_macro_features(macro: pd.DataFrame,
                         specs: Sequence[Tuple[str, str, int]]) -> pd.DataFrame:
    """
    Build the macro exogenous matrix from parsed specs.

    Column naming mirrors the rate features: '<series>_<transform>' with an
    optional '_lag<k>' suffix, so the training and forecast matrices align.
    """
    X = pd.DataFrame(index=macro.index)
    for series, transform, lag in specs:
        if series not in macro.columns:
            raise ValueError(
                f"Macro series '{series}' not found. Available: {list(macro.columns)}"
            )
        base = transform_macro_series(macro[series], transform)
        name = f"{series}_{transform}"
        if lag:
            name = f"{name}_lag{lag}"
        X[name] = base.shift(lag)
    return X


def extend_macro_future(macro: pd.DataFrame,
                        specs: Sequence[Tuple[str, str, int]],
                        cutoff: pd.Timestamp,
                        future_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build the macro feature matrix over the forecast horizon.

    Future regressor levels are held flat at each series' last observed value as
    of the forecast origin. Under that assumption, difference transforms taper to
    zero and yoy growth decays as the flat level laps the final 12 observed months
    -- the honest 'no further news' baseline for a conditional rate-scenario
    forecast. Long-horizon non-rate regressors therefore contribute their last
    known momentum and then fade, rather than being silently extrapolated.
    """
    needed = max(48, len(future_index) + 13)  # enough history for d12 / yoy laps
    hist = macro.loc[macro.index <= cutoff].tail(needed).copy()
    last_vals = hist.ffill().iloc[-1]
    future_levels = pd.DataFrame(
        np.repeat(last_vals.values[None, :], len(future_index), axis=0),
        index=future_index, columns=hist.columns,
    )
    ext = pd.concat([hist, future_levels]).sort_index()
    ext = ext[~ext.index.duplicated(keep="last")].asfreq("MS")
    Xext = build_macro_features(ext, specs)
    return Xext.loc[future_index].copy()


def center_or_standardize(X_train: pd.DataFrame,
                          X_future: pd.DataFrame,
                          standardize: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mu = X_train.mean(axis=0)
    if standardize:
        sd = X_train.std(axis=0).replace(0.0, 1.0)
    else:
        sd = pd.Series(1.0, index=X_train.columns)
    return (X_train - mu) / sd, (X_future - mu) / sd, mu, sd

# ---------------------- model fitting ----------------------

def fit_one_arimax(y: pd.Series,
                   X: pd.DataFrame,
                   order: Tuple[int, int, int],
                   enforce_stationarity: bool = True):
    mod = SARIMAX(
        endog=y,
        exog=X,
        order=order,
        seasonal_order=(0, 0, 0, 0),
        trend="c",
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=True,
        measurement_error=False,
    )
    return mod.fit(disp=False, maxiter=750)


def fit_arimax_grid(y: pd.Series,
                    X: pd.DataFrame,
                    pmax: int,
                    qmax: int,
                    d_values: Sequence[int],
                    parsimony_delta: float,
                    enforce_stationarity: bool = True):
    rows = []
    for d in d_values:
        for p in range(pmax + 1):
            for q in range(qmax + 1):
                order = (p, int(d), q)
                try:
                    res = fit_one_arimax(y, X, order, enforce_stationarity=enforce_stationarity)
                    rows.append({
                        "order": order,
                        "aic": float(res.aic),
                        "aicc": aicc_from_res(res),
                        "bic": float(res.bic),
                        "res": res,
                    })
                except Exception as exc:
                    print(f"[warn] order={order} failed: {exc}")
    if not rows:
        raise RuntimeError("No ARIMAX candidate converged.")

    rows.sort(key=lambda r: r["aicc"])
    best = rows[0]
    shortlist = [r for r in rows if r["aicc"] <= best["aicc"] + parsimony_delta]
    shortlist.sort(key=lambda r: (r["order"][0] + r["order"][2], r["order"][1], r["order"][2], r["order"][0], r["aicc"]))
    chosen = shortlist[0]
    return chosen["order"], chosen["res"], pd.DataFrame([{k: v for k, v in r.items() if k != "res"} for r in rows])

# ---------------------- scenarios and forecasts ----------------------

def make_future_rate_path(last_rate: float,
                          start_date: pd.Timestamp,
                          h: int,
                          scenario: str,
                          shock_bp: float,
                          transition_months: int) -> pd.Series:
    """Create future rate path in percentage points."""
    idx = pd.date_range(start=start_date, periods=h, freq="MS")
    scenario = scenario.lower()
    shock_pp = shock_bp / 100.0

    if scenario == "flat":
        vals = np.repeat(last_rate, h)
    elif scenario.startswith("down"):
        # down100 means -100bp; down50 means -50bp if supplied.
        digits = "".join(ch for ch in scenario if ch.isdigit())
        pp = (float(digits) / 100.0) if digits else shock_pp
        target = last_rate - pp
        n = max(1, min(int(transition_months), h))
        vals = np.r_[np.linspace(last_rate, target, n), np.repeat(target, h - n)]
    elif scenario.startswith("up"):
        digits = "".join(ch for ch in scenario if ch.isdigit())
        pp = (float(digits) / 100.0) if digits else shock_pp
        target = last_rate + pp
        n = max(1, min(int(transition_months), h))
        vals = np.r_[np.linspace(last_rate, target, n), np.repeat(target, h - n)]
    else:
        raise ValueError(f"Unknown built-in scenario '{scenario}'. Use flat, down100, up100, etc.")
    return pd.Series(vals, index=idx, name="rate")


def load_custom_future_rate_path(path: str, date_col: str, rate_col: str, h: int, start_date: pd.Timestamp) -> pd.Series:
    s, chosen = load_rate_series(path, date_col=date_col, rate_col=rate_col)
    s = s.loc[s.index >= start_date]
    if s.shape[0] < h:
        raise ValueError(
            f"Custom rate path '{path}' has only {s.shape[0]} months from {start_date.date()}, "
            f"but h={h}. Rate column used: {chosen}."
        )
    return s.iloc[:h].rename("rate")


def forecast_scenario(res,
                      y_train: pd.Series,
                      y_log_train: pd.Series,
                      target: str,
                      rate_hist: pd.Series,
                      future_rate: pd.Series,
                      features: Sequence[str],
                      lags: Sequence[int],
                      train_mu: pd.Series,
                      train_sd: pd.Series,
                      macro: Optional[pd.DataFrame] = None,
                      macro_specs: Optional[Sequence[Tuple[str, str, int]]] = None,
                      exog_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    # Include enough history to build lagged exogenous features for the forecast months.
    cutoff = y_train.index[-1]
    rate_upto_cutoff = rate_hist.loc[rate_hist.index <= cutoff]
    rate_ext = pd.concat([rate_upto_cutoff, future_rate])
    rate_ext = rate_ext[~rate_ext.index.duplicated(keep="last")].sort_index().asfreq("MS")

    X_ext = build_rate_features(rate_ext, features=features, lags=lags)
    X_future = X_ext.loc[future_rate.index].copy()

    # Append macro regressors over the same horizon (levels held flat at the
    # forecast origin; see extend_macro_future for the rationale).
    if macro is not None and macro_specs:
        X_macro_future = extend_macro_future(
            macro=macro, specs=macro_specs, cutoff=cutoff,
            future_index=future_rate.index,
        )
        X_future = pd.concat([X_future, X_macro_future], axis=1)

    # Reorder to exactly match the training design matrix.
    if exog_columns is not None:
        X_future = X_future.reindex(columns=list(exog_columns))

    if X_future.isna().any().any():
        bad = X_future.columns[X_future.isna().any()].tolist()
        raise ValueError(f"Future exogenous matrix has missing values in columns: {bad}")

    X_future = (X_future - train_mu) / train_sd
    pred = res.get_forecast(steps=future_rate.shape[0], exog=X_future)
    mean = pred.predicted_mean
    ci95 = pred.conf_int(alpha=0.05)
    ci80 = pred.conf_int(alpha=0.20)
    ci50 = pred.conf_int(alpha=0.50)

    last_log = float(y_log_train.iloc[-1])
    target = target.lower()

    if target == "monthly_return":
        # SARIMAX forecasts monthly log returns. Convert the mean path to a log
        # index by cumulating from the final observed log index. For intervals,
        # use a transparent approximation: cumulative forecast variance is the
        # cumulative sum of h-step return forecast variances. This ignores
        # cross-step forecast-error covariance, so treat the bands as a practical
        # first approximation rather than a full simulation/parameter-uncertainty band.
        ret_mean = pd.Series(mean.to_numpy(), index=mean.index, name="pred_mean_return")
        try:
            se_ret = np.asarray(pred.se_mean, dtype=float)
        except Exception:
            se_ret = (ci95.iloc[:, 1].to_numpy() - ci95.iloc[:, 0].to_numpy()) / (2.0 * 1.96)

        cum_mean_log = last_log + np.cumsum(ret_mean.to_numpy())
        cum_se = np.sqrt(np.cumsum(se_ret ** 2))

        pred_mean_log = cum_mean_log
        lower_95_log = cum_mean_log - 1.959963984540054 * cum_se
        upper_95_log = cum_mean_log + 1.959963984540054 * cum_se
        lower_80_log = cum_mean_log - 1.2815515655446004 * cum_se
        upper_80_log = cum_mean_log + 1.2815515655446004 * cum_se
        lower_50_log = cum_mean_log - 0.6744897501960817 * cum_se
        upper_50_log = cum_mean_log + 0.6744897501960817 * cum_se

        model_mean = ret_mean.to_numpy()
        model_lower_95 = ci95.iloc[:, 0].to_numpy()
        model_upper_95 = ci95.iloc[:, 1].to_numpy()

    elif target == "log_level":
        pred_mean_log = mean.to_numpy()
        lower_95_log = ci95.iloc[:, 0].to_numpy()
        upper_95_log = ci95.iloc[:, 1].to_numpy()
        lower_80_log = ci80.iloc[:, 0].to_numpy()
        upper_80_log = ci80.iloc[:, 1].to_numpy()
        lower_50_log = ci50.iloc[:, 0].to_numpy()
        upper_50_log = ci50.iloc[:, 1].to_numpy()

        model_mean = mean.to_numpy()
        model_lower_95 = lower_95_log
        model_upper_95 = upper_95_log

    else:
        raise ValueError("target must be either 'log_level' or 'monthly_return'.")

    out = pd.DataFrame({
        "date": mean.index,
        "rate": future_rate.reindex(mean.index).to_numpy(),
        "pred_mean_log": pred_mean_log,
        "lower_95_log": lower_95_log,
        "upper_95_log": upper_95_log,
        "lower_80_log": lower_80_log,
        "upper_80_log": upper_80_log,
        "lower_50_log": lower_50_log,
        "upper_50_log": upper_50_log,
        "model_target_mean": model_mean,
        "model_target_lower_95": model_lower_95,
        "model_target_upper_95": model_upper_95,
    })
    for col in ["pred_mean", "lower_95", "upper_95", "lower_80", "upper_80", "lower_50", "upper_50"]:
        log_col = col + "_log" if col != "pred_mean" else "pred_mean_log"
        out[col + "_last100"] = 100.0 * np.exp(out[log_col] - last_log)
    out["step"] = np.arange(1, len(out) + 1, dtype=int)
    return out

# ---------------------- plotting ----------------------

def plot_one_funnel(y: pd.Series,
                    fc: pd.DataFrame,
                    scenario: str,
                    outbase: str,
                    plot_start: str = "2000-01") -> None:
    y_plot = y.copy()
    if plot_start:
        y_plot = y_plot.loc[y_plot.index >= pd.to_datetime(plot_start)]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.plot(y_plot.index, y_plot.values, lw=1.5, label="Observed")
    ax.plot(fc["date"], fc["pred_mean_log"], lw=1.8, label=f"Forecast: {scenario}")
    ax.fill_between(fc["date"], fc["lower_95_log"], fc["upper_95_log"], alpha=0.16, label="95% interval")
    ax.fill_between(fc["date"], fc["lower_80_log"], fc["upper_80_log"], alpha=0.22, label="80% interval")
    ax.fill_between(fc["date"], fc["lower_50_log"], fc["upper_50_log"], alpha=0.28, label="50% interval")
    ax.set_title(f"National market factor forecast — {scenario}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Log index")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{outbase}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_one_rebased_funnel(y: pd.Series,
                            fc: pd.DataFrame,
                            scenario: str,
                            outbase: str,
                            plot_start: str = "2010-01") -> None:
    last_y = float(y.iloc[-1])
    y_rebased = 100.0 * np.exp(y - last_y)
    if plot_start:
        y_rebased = y_rebased.loc[y_rebased.index >= pd.to_datetime(plot_start)]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.plot(y_rebased.index, y_rebased.values, lw=1.5, label="Observed, last month = 100")
    ax.plot(fc["date"], fc["pred_mean_last100"], lw=1.8, label=f"Forecast: {scenario}")
    ax.fill_between(fc["date"], fc["lower_95_last100"], fc["upper_95_last100"], alpha=0.16, label="95% interval")
    ax.fill_between(fc["date"], fc["lower_80_last100"], fc["upper_80_last100"], alpha=0.22, label="80% interval")
    ax.fill_between(fc["date"], fc["lower_50_last100"], fc["upper_50_last100"], alpha=0.28, label="50% interval")
    ax.axhline(100.0, lw=0.8, ls="--", alpha=0.7)
    ax.set_title(f"National market factor forecast, rebased — {scenario}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Index, forecast origin = 100")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{outbase}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_combined_scenarios(y: pd.Series,
                            forecasts: Dict[str, pd.DataFrame],
                            outbase: str,
                            plot_start: str = "2010-01") -> None:
    last_y = float(y.iloc[-1])
    y_rebased = 100.0 * np.exp(y - last_y)
    if plot_start:
        y_rebased = y_rebased.loc[y_rebased.index >= pd.to_datetime(plot_start)]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.plot(y_rebased.index, y_rebased.values, lw=1.5, label="Observed, last month = 100")
    for name, fc in forecasts.items():
        ax.plot(fc["date"], fc["pred_mean_last100"], lw=1.8, label=name)
    ax.axhline(100.0, lw=0.8, ls="--", alpha=0.7)
    ax.set_title("National market factor forecast: scenario comparison")
    ax.set_xlabel("Month")
    ax.set_ylabel("Index, forecast origin = 100")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{outbase}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

# ---------------------- main ----------------------

def main() -> int:
    warnings.filterwarnings("ignore", message="No frequency information was provided")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
    warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
    warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found")

    ap = argparse.ArgumentParser()
    ap.add_argument("--factor_csv", default=os.path.join(DATA_PATH, "df_factor_trends.csv"),
                    help="CSV containing month_date and the market factor.")
    ap.add_argument("--market_col", default="market",
                    help="Column in factor_csv to use as the national log index; default: market.")
    ap.add_argument("--target", choices=["log_level", "monthly_return"], default="log_level",
                    help="Series to model. 'log_level' models the index directly; "
                         "'monthly_return' models first differences and cumulates forecasts back to levels. "
                         "Default: log_level.")
    ap.add_argument("--rate_csv", default=os.path.join(DATA_PATH, "interest_rates.csv"),
                    help="CSV containing monthly interest rates; default: data/interest_rates.csv.")
    ap.add_argument("--rate_date_col", default="time",
                    help="Date column in the interest-rate CSV; default: time.")
    ap.add_argument("--rate_col", default="",
                    help="Interest-rate column to use. If omitted, the script tries to infer it.")

    # ---- macro (multi-series) exogenous regressors ----
    ap.add_argument("--macro_csv", default=os.path.join(DATA_PATH, "macro_data.csv"),
                    help="Wide CSV of macro/housing series; default: data/macro_data.csv.")
    ap.add_argument("--macro_date_col", default="time",
                    help="Date column in the macro CSV; default: time.")
    ap.add_argument("--macro_features", nargs="*", default=None,
                    help="Macro regressor specs 'series:transform[:lag]'. "
                         "Transforms: level, d1, d3, d6, d12, yoy. "
                         "If omitted, the agreed baseline set is used. "
                         "Pass --no_macro to disable macro regressors entirely.")
    ap.add_argument("--no_macro", action="store_true",
                    help="Disable macro regressors; reproduces the rate-only model.")
    ap.add_argument("--outdir", default=DATA_OUT,
                    help="Output directory; default: ~/Documents/github/neoval-project/data_out.")

    ap.add_argument("--h", type=int, default=60, help="Forecast horizon in months; default: 60.")
    ap.add_argument("--features", nargs="+", default=["level", "d1"],
                    choices=["level", "d1", "d3", "d6", "d12"],
                    help="Interest-rate base features. Default: level d1.")
    ap.add_argument("--rate_lags", nargs="+", type=int, default=[0, 3, 6],
                    help="Lags, in months, applied to each rate feature. Default: 0 3 6.")
    ap.add_argument("--standardize_exog", action="store_true",
                    help="Z-score exogenous variables; otherwise only mean-center them.")

    ap.add_argument("--fixed_order", type=parse_order, default=None,
                    help="Force ARIMA order p,d,q, e.g. 1,1,1. If omitted, a grid search is run.")
    ap.add_argument("--pmax", type=int, default=3, help="Max AR order for grid search. Default: 3.")
    ap.add_argument("--qmax", type=int, default=2, help="Max MA order for grid search. Default: 2.")
    ap.add_argument("--d_values", nargs="+", type=int, default=None,
                    help="d values to search over. Default: 1 for log_level, 0 for monthly_return.")
    ap.add_argument("--parsimony_delta", type=float, default=2.0,
                    help="Pick simplest model within this AICc gap. Default: 2.0.")
    ap.add_argument("--no_enforce_stationarity", action="store_true",
                    help="Pass enforce_stationarity=False to SARIMAX.")

    ap.add_argument("--scenarios", nargs="+", default=["flat", "down100", "up100"],
                    help="Built-in rate scenarios: flat, down100, up100, down200, up200, etc.")
    ap.add_argument("--shock_bp", type=float, default=100.0,
                    help="Fallback shock size in bp for 'up'/'down' without digits. Default: 100.")
    ap.add_argument("--transition_months", type=int, default=24,
                    help="Months over which up/down scenarios move to their target. Default: 24.")
    ap.add_argument("--custom_rate_csv", default="",
                    help="Optional CSV containing a custom future rate path. If supplied, adds scenario 'custom'.")
    ap.add_argument("--custom_rate_col", default="",
                    help="Rate column in custom_rate_csv. If omitted, inferred as for the historical rate file.")

    ap.add_argument("--start", default="",
                    help="Optional estimation start month, e.g. 1988-01.")
    ap.add_argument("--end", default="",
                    help="Optional estimation end month, e.g. 2025-12.")
    ap.add_argument("--plot_start", default="2000-01",
                    help="Start month for log-level plots. Default: 2000-01.")
    ap.add_argument("--plot_start_rebased", default="2010-01",
                    help="Start month for rebased plots. Default: 2010-01.")

    args = ap.parse_args()
    if args.d_values is None:
        args.d_values = [1] if args.target == "log_level" else [0]
    outdir = expand(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    y_log = load_market_series(args.factor_csv, args.market_col)
    y_model = make_model_target(y_log, args.target)
    rate, chosen_rate_col = load_rate_series(args.rate_csv, args.rate_date_col, args.rate_col)

    X_raw = build_rate_features(rate, features=args.features, lags=args.rate_lags)

    # Resolve and build macro regressors (default baseline unless disabled).
    macro_specs: List[Tuple[str, str, int]] = []
    macro_frame: Optional[pd.DataFrame] = None
    if not args.no_macro:
        spec_strings = args.macro_features if args.macro_features is not None else DEFAULT_MACRO_FEATURES
        macro_specs = [parse_macro_spec(s) for s in spec_strings]
        if macro_specs:
            macro_frame = load_macro_frame(args.macro_csv, args.macro_date_col)
            X_macro = build_macro_features(macro_frame, macro_specs)
            X_raw = pd.concat([X_raw, X_macro], axis=1)

    data = pd.concat([y_model.rename("y"), X_raw], axis=1).dropna()
    if args.start:
        data = data.loc[data.index >= pd.to_datetime(args.start)]
    if args.end:
        data = data.loc[data.index <= pd.to_datetime(args.end)]

    if data.shape[0] < 80:
        raise ValueError(f"Only {data.shape[0]} usable monthly observations after alignment; check dates and columns.")

    y_train = data["y"].asfreq("MS")
    y_log_train = y_log.loc[y_log.index <= y_train.index[-1]].asfreq("MS").dropna()
    X_train_raw = data.drop(columns=["y"]).asfreq("MS")
    exog_columns = list(X_train_raw.columns)  # canonical order for the future matrix

    # SARIMAX requires a contiguous monthly index, so the .asfreq above can
    # reintroduce rows that dropna() had removed -- this happens when a chosen
    # regressor has *interior* missing months over the estimation window (e.g.
    # net_overseas_migration went negative during the 2020-21 border closure, so
    # its yoy log-growth is undefined there). Such a regressor cannot enter the
    # model as-is; fail loudly and name the culprit rather than passing NaNs to
    # the fit. The padded rows make every column look NaN, so identify the real
    # offender(s) from the pre-pad (dropna'd) rows that are now missing.
    if X_train_raw.isna().any().any():
        padded = X_train_raw.index.difference(data.index)  # rows reinserted by asfreq
        # Find which regressor is actually undefined on the padded months by
        # rebuilding each feature on the full grid and testing it there.
        full_feats = X_raw  # built above on the contiguous monthly grid, pre-dropna
        culprits = [c for c in exog_columns
                    if c in full_feats.columns and full_feats.loc[padded, c].isna().any()]
        n_bad = len(padded)
        raise ValueError(
            "Regressor(s) have missing values inside the estimation window and "
            f"cannot be used as specified: {culprits or exog_columns} "
            f"({n_bad} interior month(s) affected -- e.g. negative net overseas "
            "migration breaks a yoy log transform). Drop the regressor, choose a "
            "different transform (e.g. d12 instead of yoy), or restrict "
            "--start/--end to a gap-free span."
        )

    # Mean-center or z-score exog using only the estimation sample.
    # The future exog will be transformed with the same mean/std.
    X_train, _, train_mu, train_sd = center_or_standardize(
        X_train_raw, X_train_raw.copy(), standardize=args.standardize_exog
    )

    if args.fixed_order is None:
        order, res, grid = fit_arimax_grid(
            y_train, X_train,
            pmax=args.pmax,
            qmax=args.qmax,
            d_values=args.d_values,
            parsimony_delta=args.parsimony_delta,
            enforce_stationarity=not args.no_enforce_stationarity,
        )
        grid_path = os.path.join(outdir, "national_arimax_grid.csv")
        grid.to_csv(grid_path, index=False, float_format="%.6f")
        print(f"[ok] wrote grid search table: {grid_path}")
    else:
        order = args.fixed_order
        res = fit_one_arimax(
            y_train, X_train, order,
            enforce_stationarity=not args.no_enforce_stationarity,
        )

    lbp = lb_pvals(res.resid, lags=(12, 24))
    print("\n=== National ARIMAX fit ===")
    print(f"market column: {args.market_col}")
    print(f"target:        {args.target}")
    print(f"rate column:   {chosen_rate_col}")
    if macro_specs:
        print(f"macro regs:    {[f'{s}:{t}' + (f':{l}' if l else '') for s, t, l in macro_specs]}")
    else:
        print("macro regs:    (none)")
    print(f"sample:        {y_train.index[0].date()} to {y_train.index[-1].date()}  (n={len(y_train)})")
    print(f"features:      {list(X_train.columns)}")
    print(f"order:         {order}")
    print(f"AIC:           {res.aic:.2f}")
    print(f"AICc:          {aicc_from_res(res):.2f}")
    print(f"BIC:           {res.bic:.2f}")
    print(f"LB p(12):      {lbp[12]:.3f}")
    print(f"LB p(24):      {lbp[24]:.3f}")

    params = pd.DataFrame({
        "param": res.params.index,
        "coef": res.params.values,
        "se": res.bse.values,
        "z": res.params.values / res.bse.values,
    })
    params_path = os.path.join(outdir, "national_arimax_params.csv")
    params.to_csv(params_path, index=False, float_format="%.8f")
    print(f"[ok] wrote parameter table: {params_path}")

    summary = pd.DataFrame([{
        "market_col": args.market_col,
        "rate_col": chosen_rate_col,
        "target": args.target,
        "sample_start": y_train.index[0].strftime("%Y-%m-%d"),
        "sample_end": y_train.index[-1].strftime("%Y-%m-%d"),
        "nobs": int(len(y_train)),
        "order": str(order),
        "aic": float(res.aic),
        "aicc": float(aicc_from_res(res)),
        "bic": float(res.bic),
        "lb_p12": float(lbp[12]),
        "lb_p24": float(lbp[24]),
        "features": ",".join(X_train.columns),
        "standardize_exog": bool(args.standardize_exog),
    }])
    summary_path = os.path.join(outdir, "national_arimax_summary.csv")
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"[ok] wrote model summary: {summary_path}")

    forecast_start = y_train.index[-1] + pd.offsets.MonthBegin(1)
    # Use the rate at the model's forecast origin. If rate extends beyond y_train,
    # ignore that by default: scenarios begin at the last month used in the fit.
    rate_origin = rate.loc[rate.index <= y_train.index[-1]].iloc[-1]

    scenario_names = list(dict.fromkeys([s.lower() for s in args.scenarios]))
    future_rate_paths: Dict[str, pd.Series] = {}
    for sc in scenario_names:
        future_rate_paths[sc] = make_future_rate_path(
            last_rate=float(rate_origin),
            start_date=forecast_start,
            h=args.h,
            scenario=sc,
            shock_bp=args.shock_bp,
            transition_months=args.transition_months,
        )

    if args.custom_rate_csv:
        future_rate_paths["custom"] = load_custom_future_rate_path(
            args.custom_rate_csv,
            date_col=args.rate_date_col,
            rate_col=args.custom_rate_col,
            h=args.h,
            start_date=forecast_start,
        )

    forecast_dfs: Dict[str, pd.DataFrame] = {}
    for sc, f_rate in future_rate_paths.items():
        fc = forecast_scenario(
            res=res,
            y_train=y_train,
            y_log_train=y_log_train,
            target=args.target,
            rate_hist=rate,
            future_rate=f_rate,
            features=args.features,
            lags=args.rate_lags,
            train_mu=train_mu,
            train_sd=train_sd,
            macro=macro_frame,
            macro_specs=macro_specs,
            exog_columns=exog_columns,
        )
        fc.insert(0, "scenario", sc)
        forecast_dfs[sc] = fc

        fc_path = os.path.join(outdir, f"national_arimax_forecast_{sc}_h{args.h}.csv")
        fc.to_csv(fc_path, index=False, float_format="%.8f")
        print(f"[ok] wrote forecast CSV: {fc_path}")

        plot_one_funnel(
            y_log_train, fc, sc,
            outbase=os.path.join(outdir, f"national_arimax_funnel_log_{sc}"),
            plot_start=args.plot_start,
        )
        plot_one_rebased_funnel(
            y_log_train, fc, sc,
            outbase=os.path.join(outdir, f"national_arimax_funnel_rebased_{sc}"),
            plot_start=args.plot_start_rebased,
        )

    all_fc = pd.concat(forecast_dfs.values(), ignore_index=True)
    all_path = os.path.join(outdir, f"national_arimax_forecasts_all_h{args.h}.csv")
    all_fc.to_csv(all_path, index=False, float_format="%.8f")
    print(f"[ok] wrote combined forecast CSV: {all_path}")

    if len(forecast_dfs) > 1:
        plot_combined_scenarios(
            y_log_train,
            forecast_dfs,
            outbase=os.path.join(outdir, "national_arimax_scenario_comparison"),
            plot_start=args.plot_start_rebased,
        )
        print(f"[ok] wrote scenario comparison plot to {outdir}")

    print("\n[done]")
    print("Note: intervals are conditional on the specified future rate path and on the fitted model;")
    print("      they do not include uncertainty about the interest-rate scenario itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
