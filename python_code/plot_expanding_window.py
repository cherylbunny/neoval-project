#!/usr/bin/env python
"""
Plot ARIMAX factor loadings (β_r, λ_r, γ_r) over expanding or rolling windows.

- Loads monthly indexes and factor trends.
- For a chosen region, fits ARIMAX repeatedly on:
    * expanding windows: [start .. t_end]
    * rolling windows   : [t_end - (W-1) .. t_end], W = rolling_years*12
- Extracts exogenous coefficients for the requested factors.
- Plots coefficient paths versus window end date.
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----------------------- paths -----------------------
HOME = os.path.expanduser("~")
PAPER_PATH = os.path.join(HOME, "Documents/github/neoval-project")
DATA_PATH  = os.path.join(PAPER_PATH, "data")

# ----------------------- file helpers -----------------------
def make_file_path(file, base_path=os.path.join(PAPER_PATH, 'figures')):
    if len(file) < 1:
        raise Exception('Empty string provided for filename')
    if file[0] == '/':
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
        print(f"Saving figure to {file_path}")
        kwargs = {}
        if args.fine_dpi:
            kwargs['dpi'] = 300
        handle.savefig(file_path, bbox_inches='tight', pad_inches=0.01, **kwargs)

def slugify(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)

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
        p, d, q = [int(x.strip()) for x in s.split(",")]
        if d > 1:
            raise argparse.ArgumentTypeError("This script assumes levels (d=0).")
        return (p, d, q)
    except Exception:
        raise argparse.ArgumentTypeError("Non-seasonal order must be like '2,0,1'")

def parse_sorder4(s: str):
    if s.strip() == "" or s.strip() == "0,0,0,0":
        return (0,0,0,0)
    try:
        P, D, Q, S = [int(x.strip()) for x in s.split(",")]
        return (P, D, Q, S)
    except Exception:
        raise argparse.ArgumentTypeError("Seasonal order must be like '0,0,1,12'")

def ensure_monthly_freq(obj):
    obj = obj.copy()
    obj.index = pd.DatetimeIndex(obj.index)
    return obj.asfreq("MS")

def aicc(aic: float, nobs: int, kparams: int) -> float:
    return aic + (2 * kparams * (kparams + 1)) / max(nobs - kparams - 1, 1)

def fit_arimax(y: pd.Series, X: pd.DataFrame, order=(2,0,1), sorder=(0,0,0,0)):
    mod = SARIMAX(
        endog=y, exog=X,
        order=order, seasonal_order=sorder,
        trend='c',  # include intercept
        enforce_stationarity=True, enforce_invertibility=True,
        measurement_error=False
    )
    return mod.fit(method="lbfgs", disp=False, maxiter=400)

# ----------------------- main ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot ARIMAX factor loadings over expanding or rolling windows.")
    parser.add_argument("--region", default="GREATER SYDNEY")
    parser.add_argument("--region_level", default="major_city", choices=['sa4_name','major_city'])
    parser.add_argument("--factors", nargs="+", default=["market","mining","lifestyle"],
                        help="Subset of factors to include, e.g., market mining [lifestyle].")
    parser.add_argument("--order", type=parse_order3, default=(2,0,1),
                        help="Non-seasonal order 'p,0,q' (default 2,0,1).")
    parser.add_argument("--sorder", type=parse_sorder4, default=(0,0,1,12),
                        help="Seasonal order 'P,D,Q,s' (e.g. '0,0,1,12').")
    parser.add_argument("--start_year", type=int, default=1995,
                        help="Trim sample start (inclusive).")
    parser.add_argument("--min_years", type=int, default=14,
                        help="(Expanding) Minimum window length in years for first fit.")
    parser.add_argument("--step_months", type=int, default=3,
                        help="Advance the window end by this many months each step.")
    parser.add_argument("--csv_out", default="",
                        help="Optional CSV path to save coefficient paths ('' disables).")

    # Window mode and size for rolling
    parser.add_argument("--window_mode", choices=["expanding","rolling"], default="expanding",
                        help="Use 'rolling' for fixed-length moving windows; default expanding.")
    parser.add_argument("--rolling_years", type=int, default=10,
                        help="Rolling window length in years (used if --window_mode rolling).")

    # Aggregation line
    parser.add_argument("--agg", choices=["none","median","mean","trimmed"], default="none",
                        help="Aggregate across windows and draw a horizontal reference line.")
    parser.add_argument("--agg_trim", type=float, default=0.10,
                        help="Trim fraction per tail for --agg trimmed (e.g., 0.10 = 10%% each tail).")
    parser.add_argument("--agg_start", default="", help="Optional YYYY-MM start for aggregation window.")
    parser.add_argument("--agg_end",   default="", help="Optional YYYY-MM end for aggregation window.")

    # Optional: drop highly collinear factors with Market per-window
    parser.add_argument("--collinear_guard", type=float, default=0.0,
                        help="If >0, drop non-market factors when |corr(factor, market)| ≥ threshold (e.g., 0.95).")

    # I(1) option — first-difference y and all factors before fitting
    parser.add_argument("--i1", action="store_true",
                        help="First-difference y and all selected factors (estimate on Δ series).")

    # Display-only tail exclusion
    parser.add_argument("--tail_exclude", type=int, default=0,
                        help="Drop the last N window-ends from the figure (display only).")

    parser.add_argument('-f','--file', default='show',
                        help="'show' or a filename like 'loading_paths_region.pdf'.")
    parser.add_argument('--fine_dpi', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    # Warnings hygiene
    warnings.filterwarnings("ignore", message="No frequency information was provided")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
    warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
    warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found")
    warnings.filterwarnings("ignore", message="Non-invertible starting seasonal moving average")
    warnings.filterwarnings("ignore", message="Non-stationary starting seasonal autoregressive")

    # --- Load data
    idx_path = os.path.join(DATA_PATH, "city_indexes.csv" if args.region_level=='major_city' else "sa4_indexes.csv")
    fac_path = os.path.join(DATA_PATH, "df_factor_trends.csv")
    df_idx = pd.read_csv(idx_path)
    df_fac = pd.read_csv(fac_path)

    # Parse dates and index
    df_idx['month_date'] = pd.to_datetime(df_idx['month_date'])
    df_fac['month_date'] = pd.to_datetime(df_fac['month_date'])
    df_idx = df_idx.set_index('month_date').sort_index()
    df_fac = df_fac.set_index('month_date').sort_index()

    # Region column exists?
    if args.region not in df_idx.columns:
        raise ValueError(f"Region '{args.region}' not found in {idx_path}.")

    # Factor columns exist?
    for c in args.factors:
        if c not in df_fac.columns:
            raise ValueError(f"Factor column '{c}' missing in {fac_path}.")

    # Trim start
    df_idx = df_idx[df_idx.index >= f"{args.start_year}-01-01"]
    df_fac = df_fac[df_fac.index >= f"{args.start_year}-01-01"]

    # Series
    y_full = ensure_monthly_freq(df_idx[args.region].astype(float).dropna())
    X_full = ensure_monthly_freq(df_fac[args.factors].astype(float).dropna())

    # Common index
    common = y_full.index.intersection(X_full.index)
    y_full, X_full = y_full.loc[common], X_full.loc[common]

    # ---- I(1) transform (manual Δ on both y and X; keep order d=0) ----
    if args.i1:
        y_full = y_full.diff().dropna()
        X_full = X_full.diff().dropna()
        common = y_full.index.intersection(X_full.index)
        y_full, X_full = y_full.loc[common], X_full.loc[common]
        if args.verbose:
            print(f"[i1] Using first differences: sample now {y_full.index.min():%Y-%m} to {y_full.index.max():%Y-%m}")

    # ---------- Build window ends ----------
    if args.window_mode == "expanding":
        if len(y_full) < args.min_years * 12 + 2:
            raise RuntimeError("Not enough observations after trimming to build an expanding window.")
        min_obs = int(args.min_years * 12)
        ends = y_full.index[min_obs-1:]  # inclusive end
    else:  # rolling
        roll_obs = int(args.rolling_years * 12)
        if len(y_full) < roll_obs:
            raise RuntimeError("Not enough observations after trimming to build a rolling window.")
        ends = y_full.index[roll_obs-1:]  # first window end with full length

    ends = ends[::max(1, args.step_months)]

    if args.verbose:
        print("=== Loading-path setup ===")
        print(f"Region: {region2label(args.region)} | Level: {args.region_level}")
        print(f"Sample: {y_full.index.min():%Y-%m} to {y_full.index.max():%Y-%m} (monthly)")
        print(f"Order: {args.order} | Seasonal: {args.sorder} | Factors: {args.factors}")
        print(f"I(1): {'yes (Δy, ΔX)' if args.i1 else 'no (levels)'}")
        if args.window_mode == "expanding":
            print(f"Mode: expanding | First window length: {min_obs} months | Steps: {len(ends)}")
        else:
            print(f"Mode: rolling({args.rolling_years}y) | Window length: {roll_obs} months | Steps: {len(ends)}")

    # ---------- Collect coefficients across windows ----------
    records = []
    for t_end in ends:
        if args.window_mode == "expanding":
            y = y_full.loc[:t_end]
            X = X_full.loc[:t_end].copy()
        else:
            roll_obs = int(args.rolling_years * 12)
            start = t_end - pd.DateOffset(months=roll_obs-1)
            y = y_full.loc[start:t_end]
            X = X_full.loc[start:t_end].copy()

        # Optional per-window collinearity guard
        if args.collinear_guard and "market" in X.columns and len(X.columns) > 1:
            corr = X.corr()
            to_drop = []
            for col in X.columns:
                if col == "market":
                    continue
                val = corr.loc["market", col] if ("market" in corr.index and col in corr.columns) else np.nan
                if np.isfinite(val) and abs(val) >= args.collinear_guard:
                    to_drop.append(col)
            if to_drop:
                if args.verbose:
                    print(f"[guard] {t_end.date()}: dropping {to_drop} (|corr|≥{args.collinear_guard})")
                X = X.drop(columns=to_drop, errors="ignore")

        try:
            res = fit_arimax(y, X, order=args.order, sorder=args.sorder)
            param_names = getattr(res, "param_names", None)
            params = (pd.Series(res.params, index=param_names)
                      if param_names is not None else pd.Series(res.params))
            exog_names = list(res.model.exog_names)
            beta_exog = params.reindex(exog_names)

            row = {"t_end": t_end, "nobs": int(res.nobs), "aic": float(res.aic),
                   "aicc": float(aicc(res.aic, res.nobs, params.size))}
            for col in X.columns:
                row[col] = float(beta_exog.get(col, np.nan))
            for col in set(args.factors) - set(X.columns):
                row[col] = np.nan
        except Exception as e:
            if args.verbose:
                print(f"[warn] window ending {t_end.date()}: fit failed ({e}); recording NaNs")
            row = {"t_end": t_end, "nobs": len(y), "aic": np.nan, "aicc": np.nan}
            for col in args.factors:
                row[col] = np.nan

        records.append(row)

    df_coef = pd.DataFrame.from_records(records).set_index("t_end").sort_index()

    # --------- ALWAYS write CSV to PAPER_PATH/data_out ----------
    data_out_dir = os.path.join(PAPER_PATH, "data_out")
    os.makedirs(data_out_dir, exist_ok=True)
    reg_slug  = slugify(region2label(args.region))
    mode_slug = f"roll{args.rolling_years}y" if args.window_mode=="rolling" else "expand"
    i_slug    = "I1" if args.i1 else "I0"
    csv_name  = f"coef_paths_{reg_slug}_{mode_slug}_{i_slug}.csv"
    out_full  = os.path.join(data_out_dir, csv_name)
    df_coef.to_csv(out_full, index_label="t_end")
    print(f"[ok] Saved coefficient paths to {out_full}")

    # -------- Optional CSV export (legacy flag) ----------
    if args.csv_out:
        out_path = args.csv_out
        if out_path.lower() != "show":
            if not out_path.lower().endswith(".csv"):
                out_path += ".csv"
            full = make_file_path(out_path, base_path=os.path.join(PAPER_PATH, "tables"))
            os.makedirs(os.path.dirname(full), exist_ok=True)
            df_coef.to_csv(full)
            if args.verbose:
                print(f"[ok] Saved coefficient paths to {full}")

    # -------- DISPLAY-ONLY tail exclusion for plotting/aggregation --------
    df_plot = df_coef.copy()
    if args.tail_exclude and args.tail_exclude > 0:
        n = int(args.tail_exclude)
        if n >= len(df_plot):
            raise RuntimeError(f"--tail_exclude={n} removes all points; decrease N.")
        df_plot = df_plot.iloc[:-n]

    # ----------------------- Plot -----------------------
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 4.2))

    label_map = {
        "market": r"$\beta_r$ (Market)" if not args.i1 else r"$\Delta\beta_r$ (Market)",
        "mining": r"$\lambda_r$ (Mining)" if not args.i1 else r"$\Delta\lambda_r$ (Mining)",
        "lifestyle": r"$\gamma_r$ (Lifestyle)" if not args.i1 else r"$\Delta\gamma_r$ (Lifestyle)",
    }
    short_map = {
        "market": r"$\beta_r$",
        "mining": r"$\lambda_r$",
        "lifestyle": r"$\gamma_r$",
    }

    plotted_any = False
    for col in args.factors:
        if col in df_plot.columns:
            ax.plot(df_plot.index, df_plot[col], lw=1.6, label=label_map.get(col, col))
            plotted_any = True
    if not plotted_any:
        raise RuntimeError("No factor columns available to plot (after tail exclusion).")

    # ----- Aggregation lines (on plotted subset) -----
    if args.agg != "none":
        mask = pd.Series(True, index=df_plot.index)
        if args.agg_start:
            mask &= (df_plot.index >= pd.to_datetime(args.agg_start + "-01"))
        if args.agg_end:
            mask &= (df_plot.index <= pd.to_datetime(args.agg_end + "-01"))
        xmin, xmax = df_plot.index.min(), df_plot.index.max()

        for col in args.factors:
            if col not in df_plot.columns:
                continue
            s = df_plot.loc[mask, col].dropna()
            if not len(s):
                continue

            # compute aggregate
            if args.agg == "median":
                agg_val = float(s.median())
                # draw line WITHOUT legend entry
                ax.hlines(agg_val, xmin, xmax, linestyles="--", linewidth=1.0, alpha=0.6)
                # print the value just to the right of the line
                ax.annotate(f"{agg_val:.2f}",
                            xy=(xmax, agg_val), xytext=(5, 0),
                            textcoords="offset points", va="center", ha="left",
                            fontsize=10, annotation_clip=False)
                if args.verbose:
                    print(f"[agg] {col}: median={agg_val:.3f} (printed at right edge)")
            elif args.agg == "mean":
                agg_val = float(s.mean())
                if np.isfinite(agg_val):
                    ax.hlines(agg_val, xmin, xmax, linestyles="--", linewidth=1.0, alpha=0.6,
                              label=f"{label_map.get(col,col)} [mean={agg_val:.3f}]")
                    if args.verbose:
                        print(f"[agg] {col}: mean={agg_val:.3f}")
            else:  # trimmed
                lo = s.quantile(args.agg_trim)
                hi = s.quantile(1.0 - args.agg_trim)
                trimmed = s[(s >= lo) & (s <= hi)]
                agg_val = float(trimmed.mean()) if len(trimmed) else float('nan')
                if np.isfinite(agg_val):
                    ax.hlines(agg_val, xmin, xmax, linestyles="--", linewidth=1.0, alpha=0.6,
                              label=f"{label_map.get(col,col)} [trimmed({int(100*args.agg_trim)}%)={agg_val:.3f}]")
                    if args.verbose:
                        print(f"[agg] {col}: trimmed({int(100*args.agg_trim)}%)={agg_val:.3f}")

    #ax.set_xlabel("Window end date")
    ax.set_ylabel("Loading (coefficient)" if not args.i1 else "Short-run loading (Δ coefficient)")
    title_region = region2label(args.region)
    mode_str = f"rolling {args.rolling_years}y" if args.window_mode=="rolling" else "expanding"
    prefix = "Δ " if args.i1 else ""
    #ax.set_title(f"{title_region}: {prefix}ARIMAX factor loadings ({mode_str} windows)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Legend: only the series (no median entries when --agg median)
    ax.legend(loc="lower left", fontsize=12)
    plt.tight_layout()

    if args.file == 'show':
        plt.show()
    else:
        show_or_save_fig(args, handle=fig, subname="")

if __name__ == "__main__":
    main()
