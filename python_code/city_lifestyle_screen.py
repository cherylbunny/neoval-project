#!/usr/bin/env python
"""
Compare ARIMAX with/without Lifestyle across major cities and output a plain LaTeX tabular.

Kept options (everything else removed):
- --cities: which major cities to run
- --order p,0,q: fixed non-seasonal order for BOTH specs (default: auto grid if omitted)
- --sorder P,D,Q,s: fixed seasonal order for BOTH specs (default: 0,0,0,0 if omitted)
- --rule_aicc: include Lifestyle if ΔAICc (3f-2f) <= rule_aicc (default -2.0)
- --rule_lbmin: require LB p(12) for 3f >= rule_lbmin (default 0.05)
- --start_year: sample start
- --out: .tex output path

Notes:
- Factors are used as-is from df_factor_trends.csv (assumed centered).
- Intercept is included (trend='c').
- If neither --order nor --sorder is provided, each spec picks its own non-seasonal order from a small grid via AICc.
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

# ----- Paths -----
HOME = os.path.expanduser("~")
PAPER_PATH = os.path.join(HOME, "Documents/github/neoval-project")
DATA_PATH  = os.path.join(PAPER_PATH, "data")

CITY_LIST_DEFAULT = [
    "GREATER SYDNEY",
    "GREATER MELBOURNE",
    "GREATER BRISBANE",
    "GREATER ADELAIDE",
    "GREATER PERTH",
    "GREATER HOBART",
    "GREATER DARWIN",
    "AUSTRALIAN CAPITAL TERRITORY",
]

# ----- Helpers -----
def ensure_ms(obj):
    obj = obj.copy()
    obj.index = pd.DatetimeIndex(obj.index)
    return obj.asfreq("MS")

def aicc(aic: float, nobs: int, kparams: int) -> float:
    return aic + (2 * kparams * (kparams + 1)) / max(nobs - kparams - 1, 1)

def parse_order3(s: str):
    try:
        p,d,q = [int(x.strip()) for x in s.split(",")]
        return (p,d,q)
    except Exception:
        raise argparse.ArgumentTypeError("Non-seasonal order must be like '2,0,1'")

def parse_sorder4(s: str):
    try:
        P,D,Q,S = [int(x.strip()) for x in s.split(",")]
        return (P,D,Q,S)
    except Exception:
        raise argparse.ArgumentTypeError("Seasonal order must be like '0,0,1,12'")

# Small non-seasonal grid for levels
CAND_ORDERS_LEVELS = [
    (0,0,0),
    (1,0,0), (0,0,1), (1,0,1),
    (2,0,0), (0,0,2),
    (2,0,1), (1,0,2), (2,0,2),
]

def fit_one(y: pd.Series, X: pd.DataFrame,
            order=(2,0,1), seasonal_order=(0,0,0,0),
            trend='c', maxiter=400):
    """Fit ARIMAX; return (res, aicc, lb12)."""
    mod = SARIMAX(
        endog=y, exog=X,
        order=order, seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=True, enforce_invertibility=True,
        measurement_error=False,
    )
    res = mod.fit(method="lbfgs", disp=False, maxiter=maxiter)
    k = res.params.size
    aicc_val = aicc(res.aic, nobs=res.nobs, kparams=k)
    lb12 = float(acorr_ljungbox(res.resid, lags=[12], return_df=True)['lb_pvalue'].iloc[0])
    return res, aicc_val, lb12

def fit_best_nonseasonal(y: pd.Series, X: pd.DataFrame, trend='c'):
    """Grid-search non-seasonal orders only; pick best by AICc."""
    best = None
    for od in CAND_ORDERS_LEVELS:
        try:
            res, aiccv, lb12 = fit_one(y, X, order=od, seasonal_order=(0,0,0,0), trend=trend)
            if (best is None) or (aiccv < best['aicc']):
                best = {'order': od, 'seasonal_order': (0,0,0,0), 'res': res, 'aicc': aiccv, 'lb12': lb12}
        except Exception:
            continue
    if best is None:
        raise RuntimeError("No ARIMAX order converged.")
    return best

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

def _fmt_city_list(cities):
    # brief, readable
    return ", ".join([c.replace("AUSTRALIAN CAPITAL TERRITORY","ACT").replace("GREATER ","") for c in cities])

def print_run_summary(args, df_idx):
    # sample window after trimming
    if len(df_idx.index):
        start = df_idx.index.min().strftime("%Y-%m")
        end   = df_idx.index.max().strftime("%Y-%m")
        sample = f"{start} to {end} (monthly)"
    else:
        sample = "(empty after trim)"
    nonseas = f"{args.order}" if args.order else "auto grid (per spec)"
    seas    = f"{args.sorder}" if args.sorder else "(0,0,0,0)"
    print("=== Screening setup ===")
    print(f"Cities: {_fmt_city_list(args.cities)}")
    print(f"Sample: {sample}")
    print(f"Orders: non-seasonal={nonseas}, seasonal={seas}")
    print(f"Decision rule: include Lifestyle if ΔAICc ≤ {args.rule_aicc:.2f} and LB p(12) ≥ {args.rule_lbmin:.2f}.")
    print("Notes: AICc is AIC with a small-sample penalty; smaller is better.")
    print("       ΔAICc = AICc(3f) − AICc(2f); negative values favor the 3-factor model.")
    print("       LB p(12) is the Ljung–Box p-value at lag 12 on residuals; higher p ⇒ whiter residuals.")
    print("       Each model includes an intercept (trend='c'). Exogenous sets: 2f=[market,mining], 3f=[market,mining,lifestyle].\n")

# ----- Main -----
def main():
    parser = argparse.ArgumentParser(description="Compare ARIMAX with/without Lifestyle across major cities; output LaTeX table.")
    parser.add_argument("--cities", nargs="*", default=CITY_LIST_DEFAULT,
                        help="City names as in city_indexes.csv.")
    parser.add_argument("--order", type=parse_order3, default=None,
                        help="Fixed non-seasonal order 'p,0,q' for BOTH specs (e.g., '2,0,1').")
    parser.add_argument("--sorder", type=parse_sorder4, default=None,
                        help="Fixed seasonal order 'P,D,Q,s' for BOTH specs (e.g., '0,0,1,12').")
    parser.add_argument("--start_year", type=int, default=1995,
                        help="Trim sample start (inclusive, year).")
    parser.add_argument("--rule_aicc", type=float, default=-2.0,
                        help="Include Lifestyle if ΔAICc (3f-2f) <= this threshold (default -2).")
    parser.add_argument("--rule_lbmin", type=float, default=0.05,
                        help="Include Lifestyle if LB p(12) for 3f >= this (default 0.05).")
    parser.add_argument("--out", default=os.path.join(PAPER_PATH, "tables", "city_lifestyle_comparison.tex"),
                        help="Output .tex path (simple tabular).")
    args = parser.parse_args()

    # Quiet common warnings
    warnings.filterwarnings("ignore", message="No frequency information was provided")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
    warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
    warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found")
    warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")
    warnings.filterwarnings("ignore", message="Non-invertible starting seasonal moving average Using zeros as starting parameters.")

    # --- Load data ---
    idx_path = os.path.join(DATA_PATH, "city_indexes.csv")
    fac_path = os.path.join(DATA_PATH, "df_factor_trends.csv")

    df_idx = pd.read_csv(idx_path)
    df_fac = pd.read_csv(fac_path)

    df_idx['month_date'] = pd.to_datetime(df_idx['month_date'])
    df_fac['month_date'] = pd.to_datetime(df_fac['month_date'])

    df_idx = df_idx.set_index('month_date').sort_index()
    df_fac = df_fac.set_index('month_date').sort_index()

    # Trim start and announce setup
    df_idx = df_idx[df_idx.index >= f"{args.start_year}-01-01"]
    df_fac = df_fac[df_fac.index >= f"{args.start_year}-01-01"]
    print_run_summary(args, df_idx)

    cols_2f = ['market', 'mining']
    cols_3f = ['market', 'mining', 'lifestyle']
    for c in cols_3f:
        if c not in df_fac.columns:
            raise ValueError(f"Column '{c}' missing in {fac_path}.")

    rows = []
    for city in args.cities:
        if city not in df_idx.columns:
            print(f"[warn] City '{city}' not found in {idx_path}; skipping.")
            continue

        # Levels data
        y = ensure_ms(df_idx[city].astype(float)).dropna()
        X2 = ensure_ms(df_fac[cols_2f].astype(float)).dropna()
        X3 = ensure_ms(df_fac[cols_3f].astype(float)).dropna()

        # Common inner join
        idx_common = y.index.intersection(X2.index).intersection(X3.index)
        y, X2, X3 = y.loc[idx_common], X2.loc[idx_common], X3.loc[idx_common]

        # Decide orders
        trend = 'c'
        try:
            if (args.order is not None) or (args.sorder is not None):
                nonseas = args.order if args.order is not None else (2,0,1)
                seas    = args.sorder if args.sorder is not None else (0,0,0,0)
                res2, aicc2, lb2 = fit_one(y, X2, order=nonseas, seasonal_order=seas, trend=trend)
                res3, aicc3, lb3 = fit_one(y, X3, order=nonseas, seasonal_order=seas, trend=trend)
                order2 = order3 = nonseas
                seas2 = seas3 = seas
            else:
                best2 = fit_best_nonseasonal(y, X2, trend=trend)
                order2, seas2 = best2['order'], best2['seasonal_order']
                res2, aicc2, lb2 = best2['res'], best2['aicc'], best2['lb12']

                best3 = fit_best_nonseasonal(y, X3, trend=trend)
                order3, seas3 = best3['order'], best3['seasonal_order']
                res3, aicc3, lb3 = best3['res'], best3['aicc'], best3['lb12']

        except Exception as e:
            print(f"[warn] Fitting failed for {city}: {e}")
            continue

        # Debug print (3f spec): intercept & implied mean
        p3 = pd.Series(res3.params, index=getattr(res3, 'param_names', None)) \
             if getattr(res3, 'param_names', None) is not None else pd.Series(res3.params)
        int_key = next((k for k in p3.index if k and ('intercept' in k.lower() or k.lower()=='const')), None)
        phi_sum = float(getattr(res3, 'arparams', np.array([])).sum()) if hasattr(res3,'arparams') else 0.0
        if int_key:
            mu = float(p3[int_key]) / (1.0 - phi_sum) if abs(1.0 - phi_sum) > 1e-8 else np.nan
            print(f"{region2label(city, True, 2, 30)} | 2f {order2},{seas2} | 3f {order3},{seas3} | "
                  f"intercept={p3[int_key]:.6f}, sum(phi)={phi_sum:.3f}, implied mean={mu:.3f}")

        # Decision
        dAICc  = aicc3 - aicc2   # negative favors Lifestyle
        beta_L = float(res3.params.get('lifestyle', np.nan))
        use_L  = "Yes" if (dAICc <= args.rule_aicc and lb3 >= args.rule_lbmin) else "No"

        rows.append({
            "City": city,
            "AICc 2f": aicc2,
            "AICc 3f": aicc3,
            "dAICc": dAICc,
            "LB12 2f": lb2,
            "LB12 3f": lb3,
            "beta L": beta_L,
            "Use L": use_L,
        })

    if not rows:
        raise RuntimeError("No cities processed successfully; nothing to report.")

    df = pd.DataFrame(rows).sort_values("dAICc")

    # --- Formatting (no Styler/Jinja2 dependency beyond to_latex) ---
    df["AICc 2f"] = df["AICc 2f"].round(2)
    df["AICc 3f"] = df["AICc 3f"].round(2)
    df["dAICc"]   = df["dAICc"].round(2)
    df["LB12 2f"] = df["LB12 2f"].round(3)
    df["LB12 3f"] = df["LB12 3f"].round(3)
    df["beta L"]  = df["beta L"].round(3)

    df = df[["City", "AICc 2f", "AICc 3f", "dAICc", "LB12 2f", "LB12 3f", "beta L", "Use L"]]
    df["City"] = df["City"].map(lambda x: region2label(x, remove_greater=True, max_components=2, max_len=30))

    fmt2 = lambda x: "" if pd.isna(x) else f"{x:.2f}"
    fmt3 = lambda x: "" if pd.isna(x) else f"{x:.3f}"
    formatters = {
        "AICc 2f": fmt2,
        "AICc 3f": fmt2,
        "dAICc":   fmt2,
        "LB12 2f": fmt3,
        "LB12 3f": fmt3,
        "beta L":  fmt3,
    }

    try:
        latex = df.to_latex(
            index=False,
            escape=False,
            formatters=formatters,
            column_format="lrrrrrrl",
            hrules=False,
        )
    except TypeError:
        latex = df.to_latex(
            index=False,
            escape=False,
            formatters=formatters,
            column_format="lrrrrrrl",
        )
        latex = (latex
                 .replace("\\toprule\n", "")
                 .replace("\\midrule\n", "")
                 .replace("\\bottomrule\n", ""))

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(latex)

    print(latex)
    print(f"[ok] LaTeX table written to: {out_path}")

if __name__ == "__main__":
    main()
