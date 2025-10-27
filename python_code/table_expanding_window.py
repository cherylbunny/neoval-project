#!/usr/bin/env python
"""
Build a table of ARIMAX factor loadings ($\\beta_r, \\lambda_r, \\gamma_r$) for major cities
by averaging (mean/median) their expanding-window coefficients over a chosen aggregation window,
and export per-region idiosyncratic forecast variance sigma2_epsilon at horizon h.

NEW (default behavior):
- If data_out/arimax_orders_by_city.csv exists, per-city (p,0,q) and seasonal (P,0,Q)[s] orders
  are loaded and applied automatically. If a city is missing in that file, we fall back to
  --order/--sorder for that city only. If the file is absent, we use --order/--sorder for all.

Key point: sigma2_epsilon(h) here is the **multi-step forecast variance of the idiosyncratic ARMA remainder**
conditional on the resolved factors (exog) — not just the one-step white-noise innovation variance.

Outputs:
  - data_out/expanding_loadings_{agg}_{agg_start}_{agg_end}_step{step}.csv
  - data_out/sigma2_epsilon_h{h}.csv   (Region, sigma2_epsilon, h_months)
  - optional LaTeX printout to stdout or file
"""

import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----------------------- paths -----------------------
HOME = os.path.expanduser("~")
PAPER_PATH = os.path.join(HOME, "Documents/github/neoval-project")
DATA_PATH  = os.path.join(PAPER_PATH, "data")
DATA_OUT   = os.path.join(PAPER_PATH, "data_out")

ORDERS_CSV_DEFAULT = os.path.join(DATA_OUT, "arimax_orders_by_city.csv")

# ----------------------- city list -------------------
CITY_LIST_DEFAULT = [
    "GREATER SYDNEY",
    "GREATER MELBOURNE",
    "GREATER BRISBANE",
    "GREATER ADELAIDE",
    "GREATER PERTH",
    "GREATER HOBART",
    "GREATER DARWIN",
    "AUSTRALIAN CAPITAL TERRITORY",
    "REST OF VIC.",
    "REST OF NSW",
    "REST OF QLD",
    "REST OF SA",
    "REST OF WA",
    "REST OF TAS.",
]

# ----------------------- small utils -----------------
def region2label(region, remove_greater=True, max_components=2, max_len=30):
    region = region.replace('AUSTRALIAN CAPITAL TERRITORY', 'ACT').replace('SYDNEY - ','')
    region = region.strip()
    if remove_greater:
        region = region.replace('GREATER ', '')
    if ',' in region:
        return ",".join([region2label(e, remove_greater=remove_greater,
                                      max_components=max_components, max_len=max_len)
                         for e in region.split(',')])
    def word2lower(s):
        if s in ['NSW','QLD','WA','SA','ACT','TAS','VIC','NT']:
            return s
        return s[0].upper() + s[1:].lower()
    def to_lower(s):
        return " ".join([word2lower(e) for e in s.split(" ")])
    def abbrev(s):
        return ''.join(e[0] for e in s.split(' ')) if len(s) > max_len else s
    components = [abbrev(to_lower(e.strip())) for e in region.split('-')][:max_components]
    return " - ".join(components).strip()

def keyize(name: str) -> str:
    """Normalize region strings so we can match raw CSV names to pretty labels robustly."""
    if name is None:
        return ""
    pretty = region2label(str(name), remove_greater=True, max_components=2, max_len=30)
    # Lowercase and strip non-alphanumeric to avoid '.' or spaces mismatches
    return re.sub(r'[^a-z0-9]', '', pretty.lower())

def ensure_ms(obj):
    obj = obj.copy()
    obj.index = pd.DatetimeIndex(obj.index)
    return obj.asfreq("MS")

def parse_order3(s: str):
    try:
        p, d, q = [int(x.strip()) for x in s.split(",")]
        if d > 1:
            raise argparse.ArgumentTypeError("This script assumes levels (d=0).")
        return (p, d, q)
    except Exception:
        raise argparse.ArgumentTypeError("Non-seasonal order must be like '2,0,1'")

def parse_sorder4(s: str):
    if isinstance(s, tuple) and len(s) == 4:
        return s
    if isinstance(s, str) and (s.strip() == "" or s.strip() == "0,0,0,0"):
        return (0,0,0,0)
    try:
        P, D, Q, S = [int(x.strip()) for x in str(s).split(",")]
        return (P, D, Q, S)
    except Exception:
        raise argparse.ArgumentTypeError("Seasonal order must be like '0,0,1,12'")

def fit_arimax(y: pd.Series, X: pd.DataFrame, order=(2,0,1), sorder=(0,0,1,12)):
    mod = SARIMAX(
        endog=y, exog=X,
        order=order, seasonal_order=sorder,
        trend='c',
        enforce_stationarity=True, enforce_invertibility=True,
        measurement_error=False
    )
    return mod.fit(method="lbfgs", disp=False, maxiter=300)

def epsilon_variance_h(y_full: pd.Series,
                       X_full: pd.DataFrame,
                       order, sorder, h: int,
                       verbose: bool = False) -> float:
    """
    Return sigma2_epsilon(h): the h-step-ahead **forecast variance** of the idiosyncratic remainder
    (the ARMA disturbance in y after conditioning on exogenous factors).
    Implementation: fit ARIMAX on the aligned sample, then call get_forecast(h, exog=0)
    and read the variance from se_mean (or the 95% CI fallback). This variance includes the
    AR/MA propagation over h steps; it is not just the one-step innovation variance.
    """
    # Align indices & drop NAs
    common = y_full.index.intersection(X_full.index)
    y = y_full.loc[common].astype(float)
    X = X_full.loc[common].astype(float)
    y = y.dropna()
    X = X.dropna()
    common2 = y.index.intersection(X.index)
    y, X = y.loc[common2], X.loc[common2]

    if len(y) < 24 or X.shape[0] < 24:
        if verbose:
            print("[warn] too few aligned obs for epsilon variance; returning NaN")
        return float('nan')

    try:
        res = fit_arimax(y, X, order=order, sorder=sorder)
    except Exception as e:
        if verbose:
            print(f"[warn] fit_arimax failed: {e}")
        return float('nan')

    # Build a zero exog path (deterministic) for h months; variance does not depend on exog values
    future_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")
    exog_future = pd.DataFrame(0.0, index=future_idx, columns=list(X.columns), dtype=float)

    try:
        pred = res.get_forecast(steps=h, exog=exog_future)
        se = getattr(pred, "se_mean", None)
        if se is None:
            ci = pred.conf_int(alpha=0.05)
            lower = ci.iloc[:, 0]
            upper = ci.iloc[:, -1]
            se = (upper - lower) / (2.0 * 1.96)
        var_last = float(np.asarray(se)[-1] ** 2)
        if verbose:
            # Innovation variance vs. h-step variance (for reassurance)
            try:
                innov_var = float(res.filter_results.obs_cov[0, 0])
            except Exception:
                innov_var = float(getattr(res, "sigma2", np.nan))
            ratio = var_last / innov_var if np.isfinite(innov_var) and innov_var > 0 else np.nan
            print(f"    [diag] innov_var={innov_var:.6f}, sigma2_epsilon(h={h})={var_last:.6f}, ratio={ratio:.2f}")
        return var_last
    except Exception as e:
        if verbose:
            print(f"[warn] get_forecast failed: {e}")
        return float('nan')

# ----------------------- orders I/O -------------------
def _parse_order_tuple_from_str(s: str):
    """Parse '(p,0,q)' -> (p,0,q) with ints."""
    m = re.match(r'\s*\((\-?\d+)\s*,\s*(\-?\d+)\s*,\s*(\-?\d+)\)\s*$', str(s))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

def _parse_seasonal_from_str(s: str):
    """Parse '(P,D,Q)[s]' -> (P,D,Q,s) with ints."""
    m = re.match(r'\s*\((\-?\d+)\s*,\s*(\-?\d+)\s*,\s*(\-?\d+)\)\s*\[\s*(\-?\d+)\s*\]\s*$', str(s))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))

def load_orders_map(orders_csv_path: str,
                    fallback_order=(2,0,1),
                    fallback_sorder=(0,0,1,12)):
    """
    Return dict: normalized region key -> (order, sorder).
    Accepts CSVs with either numeric columns (p,d,q,P,D,Q,S) or older names,
    or string columns ('ARIMA(p,d,q)', 'Seasonal (P,D,Q)[s]', etc.).
    """
    if not os.path.exists(orders_csv_path):
        return {}

    df = pd.read_csv(orders_csv_path)

    # Find a name column
    name_col = None
    for cand in ["City", "Region", "Name", "city", "region"]:
        if cand in df.columns:
            name_col = cand
            break
    if name_col is None:
        return {}

    def extract_order(row):
        # New numeric style
        if all(c in row.index for c in ["p","d","q"]) and pd.notna(row["p"]) and pd.notna(row["d"]) and pd.notna(row["q"]):
            return (int(row["p"]), int(row["d"]), int(row["q"]))
        # Older numeric style
        if all(c in row.index for c in ["order_p","order_d","order_q"]):
            return (int(row["order_p"]), int(row["order_d"]), int(row["order_q"]))
        # String styles
        for c in ["order_str", "ARIMA(p,d,q)", "ARIMA$(p,d,q)$", "ARIMA$(p,0,q)$"]:
            if c in row.index and pd.notna(row[c]):
                tup = _parse_order_tuple_from_str(row[c])
                if tup:
                    return tup
        return fallback_order

    def extract_sorder(row):
        # New numeric style
        if all(c in row.index for c in ["P","D","Q","S"]) and \
           pd.notna(row["P"]) and pd.notna(row["D"]) and pd.notna(row["Q"]) and pd.notna(row["S"]):
            return (int(row["P"]), int(row["D"]), int(row["Q"]), int(row["S"]))
        # Older numeric style
        if all(c in row.index for c in ["seas_P","seas_D","seas_Q","seas_s"]):
            return (int(row["seas_P"]), int(row["seas_D"]), int(row["seas_Q"]), int(row["seas_s"]))
        # String styles
        for c in ["seasonal_str", "Seasonal (P,D,Q)[s]", "Seasonal $(P,D,Q)[s]$"]:
            if c in row.index and pd.notna(row[c]):
                tup = _parse_seasonal_from_str(row[c])
                if tup:
                    return tup
        return fallback_sorder

    mapping = {}
    for _, row in df.iterrows():
        raw_name = str(row[name_col])

        od  = extract_order(row)
        sod = extract_sorder(row)

        # Enforce I(0) by construction
        od  = (int(od[0]), 0, int(od[2]))
        sod = (int(sod[0]), 0, int(sod[2]), int(sod[3]))

        # Make robust keys using your pretty-normalization (removes 'GREATER ', punctuation, case)
        k = keyize(raw_name)                                # normalized pretty key
        k_alt = keyize(region2label(raw_name))              # same in practice, but keep both for safety
        for kk in {k, k_alt}:
            if kk:
                mapping[kk] = (od, sod)

    return mapping


# ----------------------- core ------------------------
def coef_paths_for_city(y_full, X_full, order, sorder, min_years, step_months):
    """Expanding-window exog coefficients for a single city; returns DataFrame indexed by t_end."""
    common = y_full.index.intersection(X_full.index)
    y_full, X_full = y_full.loc[common], X_full.loc[common]

    if len(y_full) < min_years * 12 + 2:
        return pd.DataFrame(columns=["nobs","aic","aicc"] + list(X_full.columns))

    min_obs = int(min_years * 12)
    ends = y_full.index[min_obs-1:]
    ends = ends[::max(1, step_months)]

    rows = []
    for t_end in ends:
        y = y_full.loc[:t_end]
        X = X_full.loc[:t_end]
        try:
            res = fit_arimax(y, X, order=order, sorder=sorder)
            names = getattr(res, "param_names", None)
            params = (pd.Series(res.params, index=names)
                      if names is not None else pd.Series(res.params))
            exog_names = list(res.model.exog_names)
            beta_exog = params.reindex(exog_names)
            row = {"t_end": t_end, "nobs": int(res.nobs), "aic": float(res.aic), "aicc": float(res.aic)}
            for col in X.columns:
                row[col] = float(beta_exog.get(col, np.nan))
        except Exception:
            row = {"t_end": t_end, "nobs": len(y), "aic": np.nan, "aicc": np.nan}
            for col in X.columns:
                row[col] = np.nan
        rows.append(row)

    return pd.DataFrame.from_records(rows).set_index("t_end").sort_index()


def aggregate_city(coef_df, agg, agg_start, agg_end, factors):
    """Aggregate (mean/median) per factor over t_end in [agg_start, agg_end].
       If the mask is empty and agg_start==agg_end (a single month), fall back to
       the nearest available window <= agg_end (or >= agg_start if none ≤)."""
    if coef_df.empty:
        d = {f: np.nan for f in factors}
        d["N"] = 0
        return d

    mask = (coef_df.index >= agg_start) & (coef_df.index <= agg_end)
    S = coef_df.loc[mask, factors]

    # Fallback for single-month requests that don't land on a window end
    if S.empty and agg_start == agg_end:
        idx_le = coef_df.index[coef_df.index <= agg_end]
        if len(idx_le) > 0:
            last = idx_le.max()
            S = coef_df.loc[[last], factors]
        else:
            idx_ge = coef_df.index[coef_df.index >= agg_start]
            if len(idx_ge) > 0:
                first = idx_ge.min()
                S = coef_df.loc[[first], factors]

    if S.empty:
        d = {f: np.nan for f in factors}
        d["N"] = 0
        return d

    if agg == "median":
        vals = S.median(skipna=True)
    else:
        vals = S.mean(skipna=True)

    out = {f: float(vals.get(f, np.nan)) for f in factors}
    out["N"] = int(S.shape[0])  # number of matched windows
    return out




# ----------------------- main ------------------------
def main():
    parser = argparse.ArgumentParser(description="Expanding-window ARIMAX factor loadings and epsilon funnel export.")
    parser.add_argument("--cities", nargs="*", default=CITY_LIST_DEFAULT,
                        help="Subset of major cities to include.")
    parser.add_argument("--order", type=parse_order3, default=(2,0,1),
                        help="Fallback non-seasonal order 'p,0,q' (default 2,0,1).")
    parser.add_argument("--sorder", type=parse_sorder4, default=(0,0,1,12),
                        help="Fallback seasonal order 'P,D,Q,s' (default 0,0,1,12).")
    parser.add_argument("--start_year", type=int, default=1995,
                        help="Trim sample start year (inclusive).")
    parser.add_argument("--min_years", type=int, default=14,
                        help="Minimum expanding-window length in years (default 14).")
    parser.add_argument("--step_months", type=int, default=3,
                        help="Advance window end by this many months (default 3 = quarterly).")
    parser.add_argument("--agg", choices=["median","mean"], default="median",
                        help="Aggregate across windows (median or mean).")
    parser.add_argument("--agg_start", default="2014-01", help="Aggregation start YYYY-MM (on window end dates).")
    parser.add_argument("--agg_end",   default="2024-12", help="Aggregation end   YYYY-MM (on window end dates).")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--tex_out", default="", help="Optional path to save LaTeX table ('' = print only).")
    parser.add_argument("--csv_out", default="", help="Optional path to save the aggregated CSV ('' = data_out default).")
    parser.add_argument("--h_months", type=int, default=120,
                        help="Forecast horizon for epsilon variance export (default 120).")
    parser.add_argument("--eps_out", default="", help="Override path for epsilon CSV ('' = default in data_out).")
    args = parser.parse_args()

    # Warnings hygiene
    warnings.filterwarnings("ignore", message="No frequency information was provided")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
    warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
    warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found")
    warnings.filterwarnings("ignore", message="Non-invertible starting seasonal moving average")
    warnings.filterwarnings("ignore", message="Non-stationary starting seasonal autoregressive")

    # --- Load per-city orders (preferred), else fall back to CLI defaults
    orders_map = load_orders_map(ORDERS_CSV_DEFAULT,
                                 fallback_order=args.order,
                                 fallback_sorder=args.sorder)
    if orders_map:
        print(f"[info] Loaded per-city ARIMAX orders from {ORDERS_CSV_DEFAULT} "
              f"(matches will override --order/--sorder).")
    else:
        print(f"[warn] No usable orders file at {ORDERS_CSV_DEFAULT}; "
              f"using fallback --order={args.order}, --sorder={args.sorder} for all cities.")

    # --- Load data
    idx_path = os.path.join(DATA_PATH, "city_indexes.csv")
    fac_path = os.path.join(DATA_PATH, "df_factor_trends.csv")
    df_idx = pd.read_csv(idx_path)
    df_fac = pd.read_csv(fac_path)

    df_idx['month_date'] = pd.to_datetime(df_idx['month_date'])
    df_fac['month_date'] = pd.to_datetime(df_fac['month_date'])
    df_idx = df_idx.set_index('month_date').sort_index()
    df_fac = df_fac.set_index('month_date').sort_index()

    # Trim start
    df_idx = df_idx[df_idx.index >= f"{args.start_year}-01-01"]
    df_fac = df_fac[df_fac.index >= f"{args.start_year}-01-01"]

    # Factors
    factors = ["market","mining","lifestyle"]
    for c in factors:
        if c not in df_fac.columns:
            raise ValueError(f"Factor column '{c}' missing in {fac_path}.")

    agg_start = pd.to_datetime(args.agg_start + "-01")
    agg_end   = pd.to_datetime(args.agg_end   + "-01")

    # Prepare outputs
    rows = []
    eps_rows = []

    # Do each city
    for city in args.cities:
        print("[info] Processing city:", city)
        if city not in df_idx.columns:
            print(f"[warn] City '{city}' not found in {idx_path}; skipping.")
            continue

        y_full = ensure_ms(df_idx[city].astype(float)).dropna()
        X_full = ensure_ms(df_fac[factors].astype(float)).dropna()

        # Pick orders: prefer per-city from CSV, else fall back
        k_raw   = keyize(city)
        k_pretty = keyize(region2label(city, remove_greater=True, max_components=2, max_len=30))
        if k_raw in orders_map:
            order_city, sorder_city = orders_map[k_raw]
        elif k_pretty in orders_map:
            order_city, sorder_city = orders_map[k_pretty]
        else:
            order_city, sorder_city = args.order, args.sorder

        print(f"       Using orders: ARIMA{order_city} + seasonal{(sorder_city[0], sorder_city[1], sorder_city[2])}[{sorder_city[3]}]")

        # Coefficient paths -> aggregated loadings
        coef_df = coef_paths_for_city(y_full, X_full, order_city, sorder_city,
                                      args.min_years, args.step_months)
        agg_vals = aggregate_city(coef_df, args.agg, agg_start, agg_end, factors)
        row = {
            "City": city,
            r"$\beta_r$": agg_vals["market"],
            r"$\lambda_r$": agg_vals["mining"],
            r"$\gamma_r$": agg_vals["lifestyle"],
            "N": agg_vals["N"],
        }
        rows.append(row)

        # Epsilon variance at horizon h (fit on full sample, exog future zeros)
        s2_eps = epsilon_variance_h(y_full, X_full, order_city, sorder_city, args.h_months, verbose=args.verbose)
        eps_rows.append({
            "Region": region2label(city, remove_greater=True, max_components=2, max_len=30),
            "sigma2_epsilon": float(s2_eps),
            "h_months": int(args.h_months),
        })

    if not rows:
        raise RuntimeError("No cities processed; nothing to report.")

    # DataFrame for loadings table
    df = pd.DataFrame(rows)
    df["City"] = df["City"].map(lambda x: region2label(x, remove_greater=True, max_components=2, max_len=30))
    df = df.sort_values(r"$\beta_r$", ascending=False)

    # Rounding
    for c in [r"$\beta_r$", r"$\lambda_r$", r"$\gamma_r$"]:
        df[c] = df[c].astype(float).round(2)

    # Save loadings CSV
    os.makedirs(DATA_OUT, exist_ok=True)
    csv_default = os.path.join(
        DATA_OUT,
        f"expanding_loadings_{args.agg}_{args.agg_start}_{args.agg_end}_step{args.step_months}.csv"
    )
    csv_path = args.csv_out if args.csv_out else csv_default
    df.to_csv(csv_path, index=False)
    print(f"[ok] Wrote aggregated table to {csv_path}")

    # LaTeX table print
    formatters = {r"$\beta_r$": "{:.2f}".format,
                  r"$\lambda_r$": "{:.2f}".format,
                  r"$\gamma_r$": "{:.2f}".format}
    try:
        latex = df[["City", r"$\beta_r$", r"$\lambda_r$", r"$\gamma_r$", "N"]].to_latex(
            index=False, escape=False, formatters=formatters,
            column_format="lrrrr", hrules=False
        )
    except TypeError:
        latex = df[["City", r"$\beta_r$", r"$\lambda_r$", r"$\gamma_r$", "N"]].to_latex(
            index=False, escape=False, formatters=formatters, column_format="lrrrr"
        )
        latex = (latex.replace("\\toprule\n","").replace("\\midrule\n","").replace("\\bottomrule\n",""))
    print("\n" + latex + "\n")
    if args.tex_out:
        with open(args.tex_out, "w") as f:
            f.write(latex)
        print(f"[ok] Wrote LaTeX to {args.tex_out}")

    # Save epsilon variances CSV (Region labels to match growth-mapping table)
    if eps_rows:
        eps_df = pd.DataFrame(eps_rows)
        eps_default = os.path.join(DATA_OUT, f"sigma2_epsilon_h{args.h_months}.csv")
        eps_path = args.eps_out if args.eps_out else eps_default
        eps_df.to_csv(eps_path, index=False, float_format="%.6f")
        print(f"[ok] Wrote epsilon funnel variances to {eps_path}")

if __name__ == "__main__":
    main()
