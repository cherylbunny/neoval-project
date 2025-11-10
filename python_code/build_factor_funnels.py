#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor ARIMAX modelling for Mining (PS) and Lifestyle, with forecast funnels.

Key features
- Optional boom/decline dummies for Mining (off by default).
- ARIMA(p,d,q) grid search with AICc; parsimony rule picks the simplest model
  among those within ΔAICc <= parsimony_delta (default 2.0).
- h-step forecasts with state-space funnels; outputs CSV + paper-ready plots.
- Summary CSV (and optional LaTeX) with orders, information criteria, and LB tests.
- Optional AR/MA coefficient table printed to stdout unless --no_table is set.

Usage (defaults assume your repo layout):
  ./build_factor_funnels.py \
      --h 120 --factors mining lifestyle \
      --tex_out ~/Documents/github/neoval-project/data_out/factor_arimax_summary.tex
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Optional, List

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# ---------------------- default paths ----------------------
HOME = os.path.expanduser("~")
PAPER_PATH = os.path.join(HOME, "Documents/github/neoval-project")
DATA_PATH  = os.path.join(PAPER_PATH, "data")
DATA_OUT   = os.path.join(PAPER_PATH, "data_out")

# ---------------------- utils ----------------------

def aicc(llf: float, k: int, n: int) -> float:
    aic = 2.0 * k - 2.0 * llf
    if n - k - 1 <= 0:
        return aic
    return aic + (2.0 * k * (k + 1)) / (n - k - 1)

def ensure_ms(df: pd.DataFrame, date_col="month_date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df.asfreq("MS")

def lb_pvals(resid: np.ndarray, lags: List[int]) -> Dict[int, float]:
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    out = {L: np.nan for L in lags}
    if resid.size < max(lags) + 5:
        return out
    df = acorr_ljungbox(resid, lags=max(lags), return_df=True)
    for L in lags:
        if L in df.index:
            out[L] = float(df.loc[L, 'lb_pvalue'])
    return out

def try_import_ruptures():
    try:
        import ruptures as rpt  # noqa
        return True
    except Exception:
        return False

# ---------------------- modelling helpers ----------------------

def make_dummies(idx: pd.DatetimeIndex,
                 boom_start: Optional[str],
                 boom_end: Optional[str],
                 decline_end: Optional[str]) -> pd.DataFrame:
    """Build boom/decline dummies over a monthly index."""
    boom = pd.Series(0, index=idx, dtype=int)
    decline = pd.Series(0, index=idx, dtype=int)
    if boom_start and boom_end:
        bs = pd.to_datetime(boom_start)
        be = pd.to_datetime(boom_end)
        boom.loc[(idx >= bs) & (idx <= be)] = 1
        if decline_end:
            de = pd.to_datetime(decline_end)
            if de > be:
                decline.loc[(idx > be) & (idx <= de)] = 1
    return pd.DataFrame({"boom": boom, "decline": decline})

def autodetect_boom_decline(y: pd.Series,
                            pen: float = 10.0) -> Tuple[Optional[pd.Timestamp],
                                                        Optional[pd.Timestamp],
                                                        Optional[pd.Timestamp]]:
    """
    Auto-detects a 'boom' as the segment containing the global max (Pelt, rbf),
    and 'decline' as the following segment if present.
    """
    if not try_import_ruptures():
        return None, None, None
    import ruptures as rpt  # type: ignore

    yv = y.values.astype(float)
    algo = rpt.Pelt(model="rbf").fit(yv)
    bkps = algo.predict(pen=pen)
    ends = [min(b, len(yv)) - 1 for b in bkps]
    starts = [0] + [e+1 for e in ends[:-1]]
    segs = list(zip(starts, ends))
    if not segs:
        return None, None, None

    imax = int(np.nanargmax(yv))
    seg_idx = next((i for i, (s, e) in enumerate(segs) if s <= imax <= e), None)
    if seg_idx is None:
        return None, None, None

    s_boom, e_boom = segs[seg_idx]
    boom_start = y.index[s_boom]
    boom_end   = y.index[e_boom]

    if seg_idx + 1 < len(segs):
        s_dec, e_dec = segs[seg_idx + 1]
        decline_end = y.index[e_dec]
    else:
        decline_end = None

    return boom_start, boom_end, decline_end


def fit_arimax_grid_parsimony(y: pd.Series,
                              exog: Optional[pd.DataFrame],
                              p_max: int = 3,
                              q_max: int = 2,
                              parsimony_delta: float = 2.0):
    """
    Fit ARIMA(p,d,q) with d in {0,1} over p=0..p_max, q=0..q_max, trend='c', optional exog.
    Returns exactly (chosen_order, chosen_result, metrics_dict).

    Selection:
      1) Fit full grid for d=0 and d=1.
      2) Prefer d=0 unless the best d=1 beats the best d=0 by >= AICC_ADV_D1 (AICc units).
      3) Within the chosen d, among models with AICc <= best_aicc_d + parsimony_delta,
         pick the simplest (min p+q, then min q, then min p), then lowest AICc.
    """
    AICC_ADV_D1 = 10.0  # d=1 must win by at least this much

    y_fit = y.dropna()
    n = int(y_fit.shape[0])

    candidates = {0: [], 1: []}  # per-d

    for d in (0, 1):
        for p in range(p_max + 1):
            for q in range(q_max + 1):
                try:
                    mod = SARIMAX(endog=y, exog=exog, order=(p, d, q),
                                  seasonal_order=(0, 0, 0, 0), trend='c',
                                  enforce_stationarity=True, enforce_invertibility=True)
                    res = mod.fit(disp=False, maxiter=500)
                    k = int(res.params.shape[0])
                    llf = float(res.llf)
                    aicc_val = aicc(llf, k, n)
                    candidates[d].append({
                        "order": (p, d, q),
                        "aic": float(res.aic),
                        "aicc": float(aicc_val),
                        "bic": float(res.bic),
                        "k": k,
                        "res": res
                    })
                except Exception:
                    continue

    if not candidates[0] and not candidates[1]:
        raise RuntimeError("No ARIMAX candidate converged. Consider relaxing the grid.")

    best_d0 = min(candidates[0], key=lambda c: c["aicc"]) if candidates[0] else None
    best_d1 = min(candidates[1], key=lambda c: c["aicc"]) if candidates[1] else None

    if best_d0 is None and best_d1 is None:
        raise RuntimeError("No ARIMAX candidate converged for d=0 or d=1.")
    elif best_d0 is None:
        chosen_d = 1; ref_best = best_d1
    elif best_d1 is None:
        chosen_d = 0; ref_best = best_d0
    else:
        chosen_d = 1 if (best_d1["aicc"] <= best_d0["aicc"] - AICC_ADV_D1) else 0
        ref_best = best_d1 if chosen_d == 1 else best_d0

    shortlist = [c for c in candidates[chosen_d] if c["aicc"] <= ref_best["aicc"] + parsimony_delta]
    if not shortlist:
        shortlist = [ref_best]

    shortlist.sort(key=lambda c: ((c["order"][0] + c["order"][2]),  # p+q
                                  c["order"][2],                    # q
                                  c["order"][0],                    # p
                                  c["aicc"]))                       # then AICc

    chosen = shortlist[0]
    res = chosen["res"]

    resid = res.resid
    resid = resid[np.isfinite(resid)]
    L_p = lb_pvals(resid, lags=[12, 24])

    dummy_stats = {}
    if exog is not None:
        for col in exog.columns:
            if col.lower() in ("boom", "decline"):
                if col in res.params.index:
                    val = float(res.params[col]); se = float(res.bse[col])
                    t = val / se if se > 0 else np.nan
                    p = 2.0 * (1 - stats.t.cdf(abs(t), df=max(1, n - chosen["k"])))
                    dummy_stats[col] = {"coef": val, "se": se, "t": t, "p": p}
                else:
                    dummy_stats[col] = {"coef": np.nan, "se": np.nan, "t": np.nan, "p": np.nan}

    all_cands = (candidates[0] if candidates[0] else []) + (candidates[1] if candidates[1] else [])
    best_overall = min(all_cands, key=lambda c: c["aicc"])
    best_order_overall = best_overall["order"]; best_aicc_overall = best_overall["aicc"]

    metrics = {
        "order": chosen["order"],
        "aic": chosen["aic"], "aicc": chosen["aicc"], "bic": chosen["bic"],
        "lb_p12": L_p.get(12, np.nan), "lb_p24": L_p.get(24, np.nan),
        "dummy_stats": dummy_stats,
        "selected_by_parsimony": (chosen["order"] != ref_best["order"]),
        "chosen_d": chosen_d,
        "best_order_within_d": ref_best["order"],
        "best_aicc_within_d": ref_best["aicc"],
        "best_order_overall": best_order_overall,
        "best_aicc_overall": best_aicc_overall,
        "best_aicc_d0": (best_d0["aicc"] if best_d0 else np.nan),
        "best_aicc_d1": (best_d1["aicc"] if best_d1 else np.nan)
    }
    return chosen["order"], res, metrics



def forecast_with_funnel(res, h: int, exog_future: Optional[pd.DataFrame]) -> pd.DataFrame:
    """h-step forecast producing: date, step, pred_mean, lower_95, upper_95, se_mean, var"""
    pred = res.get_forecast(steps=h, exog=exog_future)
    mean = pred.predicted_mean
    ci   = pred.conf_int(alpha=0.05)
    lower = ci.iloc[:, 0]
    upper = ci.iloc[:, 1]
    try:
        se = pred.se_mean
    except Exception:
        se = (upper - lower) / (2 * 1.96)
    var = se**2
    return pd.DataFrame({
        "date": mean.index,
        "step": np.arange(1, h+1, dtype=int),
        "pred_mean": mean.to_numpy(),
        "lower_95": lower.to_numpy(),
        "upper_95": upper.to_numpy(),
        "se_mean":  se.to_numpy(),
        "var":      var.to_numpy()
    })

def make_forecast_plot(y: pd.Series, fc_df: pd.DataFrame, title: str, outpath_base: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(y.index, y.values, lw=1.75, label="Observed")
    ax.plot(fc_df["date"], fc_df["pred_mean"], lw=1.75, label="Forecast")
    ax.fill_between(fc_df["date"], fc_df["lower_95"], fc_df["upper_95"], alpha=0.20, label="95% interval")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Level (log)")
    ax.legend(loc="upper left", frameon=False)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(ymin, 0.0), max(ymax, 0.0))
    ax.axhline(0.0, lw=0.8, ls="--", alpha=0.5)

    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(f"{outpath_base}.{ext}", dpi=300 if ext=="png" else None)
    plt.close(fig)

def make_combined_funnel_plot(y1: pd.Series, fc1: pd.DataFrame, title1: str,
                              y2: pd.Series, fc2: pd.DataFrame, title2: str,
                              outpath_base: str,
                              panel_labels=("a)", "b)")):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    ax = axes[0]
    ax.plot(y1.index, y1.values, lw=1.75, label="Observed")
    ax.plot(fc1["date"], fc1["pred_mean"], lw=1.75, label="Forecast")
    ax.fill_between(fc1["date"], fc1["lower_95"], fc1["upper_95"], alpha=0.20, label="95% interval")
    ax.set_title(f"{panel_labels[0]} {title1}" if panel_labels else title1)
    ax.set_xlabel("Month")
    ax.set_ylabel("Level (log)")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    ax.plot(y2.index, y2.values, lw=1.75, label="Observed")
    ax.plot(fc2["date"], fc2["pred_mean"], lw=1.75, label="Forecast")
    ax.fill_between(fc2["date"], fc2["lower_95"], fc2["upper_95"], alpha=0.20, label="95% interval")
    ax.set_title(f"{panel_labels[1]} {title2}" if panel_labels else title2)
    ax.set_xlabel("Month")
    ax.legend(loc="upper left", frameon=False)

    min1 = min(np.nanmin(y1.values), np.nanmin(fc1["lower_95"].values))
    max1 = max(np.nanmax(y1.values), np.nanmax(fc1["upper_95"].values))
    min2 = min(np.nanmin(y2.values), np.nanmin(fc2["lower_95"].values))
    max2 = max(np.nanmax(y2.values), np.nanmax(fc2["upper_95"].values))
    y_min = min(min1, min2, 0.0)
    y_max = max(max1, max2, 0.0)
    for ax in axes:
        ax.set_ylim(y_min, y_max)
        ax.axhline(0.0, lw=0.8, ls="--", alpha=0.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fname = f"{outpath_base}.{ext}"
        fig.savefig(fname, dpi=300 if ext == "png" else None)
        print(f"Saved combined funnel plot to {fname}")
    plt.close(fig)



# ---------------------- main ----------------------

def main():
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser()
    ap.add_argument("--factor_csv", default=os.path.join(DATA_PATH, "df_factor_trends.csv"),
                    help="Path to df_factor_trends.csv (default: ~/Documents/github/neoval-project/data/df_factor_trends.csv)")
    ap.add_argument("--outdir", default=DATA_OUT,
                    help="Output directory (default: ~/Documents/github/neoval-project/data_out)")
    ap.add_argument("--factors", nargs="+", default=["mining","lifestyle"],
                    help="Subset of factors to model: mining, lifestyle")
    ap.add_argument("--h", type=int, default=120, help="Forecast horizon in months (default 120=10y)")
    ap.add_argument("--pmax", type=int, default=3, help="Max AR order p (default 3)")
    ap.add_argument("--qmax", type=int, default=2, help="Max MA order q (default 2)")
    ap.add_argument("--parsimony_delta", type=float, default=2.0,
                    help="Choose simplest model among those with AICc within this gap (default 2.0)")

    # Mining boom/decline (optional)
    ap.add_argument("--mining_dummies", choices=["none","auto","manual"], default="none",
                    help="Dummy strategy for Mining: 'none' (default), 'auto' (ruptures or provided dates), 'manual' (requires dates).")
    ap.add_argument("--boom_start", default="", help="YYYY-MM for boom start (manual/auto override)")
    ap.add_argument("--boom_end", default="", help="YYYY-MM for boom end (manual/auto override)")
    ap.add_argument("--decline_end", default="", help="YYYY-MM for decline end (optional)")

    ap.add_argument("--tex_out", default="", help="Optional LaTeX table output")

    ap.add_argument("--print-fit-summary", action="store_true",
                    help="Print a compact factor-level fit + 10y uncertainty table to stdout")
    ap.add_argument("--tex-fit-summary", default="",
                    help="Optional LaTeX output for the same table")

    ap.add_argument("--no_table", action="store_true",
        help="Turn off ARIMA parameter table output (printed to stdout)")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load factors
    df = pd.read_csv(args.factor_csv)
    df = ensure_ms(df, "month_date")
    for col in ["mining","lifestyle"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {args.factor_csv}.")

    summary_rows = []
    horizon_vars = {}
    detected_dates = {}
    fit_results = {}  # factor -> dict(order=(p,d,q), res=results)

    y_mining = None
    fc_mining = None
    y_lifestyle = None
    fc_lifestyle = None

    # ---------------- Mining ----------------
    if "mining" in [x.lower() for x in args.factors]:
        y = df["mining"].astype(float).dropna()

        mode = args.mining_dummies.lower()
        boom_start = boom_end = decline_end = ""
        X = None

        if mode == "none":
            detected_dates["mining_boom"] = (None, None, None)
            print("[MINING] using NO dummies (mining_dummies=none).")
        elif mode == "manual":
            if not (args.boom_start and args.boom_end):
                raise ValueError("mining_dummies=manual but --boom_start/--boom_end not provided.")
            boom_start = args.boom_start
            boom_end   = args.boom_end
            decline_end = args.decline_end if args.decline_end else ""
            X = make_dummies(y.index, boom_start, boom_end, decline_end)
            detected_dates["mining_boom"] = (boom_start, boom_end, decline_end or None)
            print(f"[MINING] using MANUAL dummies: start={boom_start}, end={boom_end}, decline_end={decline_end or None}")
        else:  # auto
            if args.boom_start and args.boom_end:
                boom_start = args.boom_start
                boom_end   = args.boom_end
                decline_end = args.decline_end if args.decline_end else ""
                X = make_dummies(y.index, boom_start, boom_end, decline_end)
                detected_dates["mining_boom"] = (boom_start, boom_end, decline_end or None)
                print(f"[MINING] using PROVIDED dummies (auto mode): start={boom_start}, end={boom_end}, decline_end={decline_end or None}")
            else:
                bs, be, de = autodetect_boom_decline(y)
                if bs is not None and be is not None:
                    boom_start = bs.strftime("%Y-%m")
                    boom_end   = be.strftime("%Y-%m")
                    decline_end = de.strftime("%Y-%m") if de is not None else ""
                    X = make_dummies(y.index, boom_start, boom_end, decline_end)
                    detected_dates["mining_boom"] = (boom_start, boom_end, decline_end or None)
                    print(f"[MINING] using AUTO-detected dummies: start={boom_start}, end={boom_end}, decline_end={decline_end or None}")
                else:
                    detected_dates["mining_boom"] = (None, None, None)
                    print("[MINING] auto-detect found no clear boom; using NO dummies.")

        order, res, m = fit_arimax_grid_parsimony(y, X, p_max=args.pmax, q_max=args.qmax,
                                                  parsimony_delta=args.parsimony_delta)
        fit_results["Mining"] = {"order": order, "res": res}

        # Forecast (zeros for dummies OOS; else no exog)
        if X is None:
            exog_future = None
        else:
            exog_future = pd.DataFrame(
                {"boom": np.zeros(args.h, dtype=int),
                 "decline": np.zeros(args.h, dtype=int)},
                index=pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=args.h, freq="MS")
            )
        fc = forecast_with_funnel(res, args.h, exog_future)
        fc_path = os.path.join(args.outdir, f"factor_forecast_mining_h{args.h}.csv")
        fc.to_csv(fc_path, index=False)

        y_mining = y
        fc_mining = fc

        plot_base = os.path.join(args.outdir, f"factor_funnel_mining")
        make_forecast_plot(y, fc, "Mining factor: forecast and 95% interval", plot_base)

        dboom = m["dummy_stats"].get("boom", {})
        ddecl = m["dummy_stats"].get("decline", {})
        parsimony_note = (" (picked by parsimony; best AICc at "
                          f"{m['best_order']} = {m['best_aicc']:.2f})") if m["selected_by_parsimony"] else ""
        print(f"[MINING] order={order}, AIC={m['aic']:.2f}, AICc={m['aicc']:.2f}, BIC={m['bic']:.2f}{parsimony_note}")
        if X is not None:
            print(f"         boom={dboom.get('coef',np.nan):.4f} (se {dboom.get('se',np.nan):.4f}, p={dboom.get('p',np.nan):.3f}), "
                  f"decline={ddecl.get('coef',np.nan):.4f} (se {ddecl.get('se',np.nan):.4f}, p={ddecl.get('p',np.nan):.3f})")
        else:
            print("         (no dummies in model)")
        print(f"         Ljung–Box p(12)={m['lb_p12']:.3f}, p(24)={m['lb_p24']:.3f}")
        ds = detected_dates.get("mining_boom", (None, None, None))
        print(f"         boom/decline dates used: start={ds[0]}, end={ds[1]}, decline_end={ds[2]}")

        summary_rows.append({
            "Factor": "Mining",
            "Order": f"({order[0]},0,{order[2]})",
            "AIC": m["aic"], "AICc": m["aicc"], "BIC": m["bic"],
            "boom": dboom.get("coef", np.nan) if X is not None else np.nan,
            "boom_se": dboom.get("se", np.nan) if X is not None else np.nan,
            "boom_p": dboom.get("p", np.nan) if X is not None else np.nan,
            "decline": ddecl.get("coef", np.nan) if X is not None else np.nan,
            "decline_se": ddecl.get("se", np.nan) if X is not None else np.nan,
            "decline_p": ddecl.get('p', np.nan) if X is not None else np.nan,
            "LB_p12": m["lb_p12"], "LB_p24": m["lb_p24"],
            "boom_start": ds[0], "boom_end": ds[1], "decline_end": ds[2]
        })
        horizon_vars["sigma2_PS"] = float(fc["var"].iloc[-1])

    # ---------------- Lifestyle ----------------
    if "lifestyle" in [x.lower() for x in args.factors]:
        y = df["lifestyle"].astype(float).dropna()
        X = None
        order, res, m = fit_arimax_grid_parsimony(y, X, p_max=args.pmax, q_max=args.qmax,
                                                  parsimony_delta=args.parsimony_delta)
        fit_results["Lifestyle"] = {"order": order, "res": res}

        fc = forecast_with_funnel(res, args.h, exog_future=None)
        fc_path = os.path.join(args.outdir, f"factor_forecast_lifestyle_h{args.h}.csv")
        fc.to_csv(fc_path, index=False)

        y_lifestyle = y
        fc_lifestyle = fc

        plot_base = os.path.join(args.outdir, f"factor_funnel_lifestyle")
        make_forecast_plot(y, fc, "Lifestyle factor: forecast and 95% interval", plot_base)

        parsimony_note = (" (picked by parsimony; best AICc at "
                          f"{m['best_order']} = {m['best_aicc']:.2f})") if m["selected_by_parsimony"] else ""
        print(f"[LIFESTYLE] order={order}, AIC={m['aic']:.2f}, AICc={m['aicc']:.2f}, BIC={m['bic']:.2f}{parsimony_note}")
        print(f"           Ljung–Box p(12)={m['lb_p12']:.3f}, p(24)={m['lb_p24']:.3f}")

        summary_rows.append({
            "Factor": "Lifestyle",
            "Order": f"({order[0]},0,{order[2]})",
            "AIC": m["aic"], "AICc": m["aicc"], "BIC": m["bic"],
            "boom": np.nan, "boom_se": np.nan, "boom_p": np.nan,
            "decline": np.nan, "decline_se": np.nan, "decline_p": np.nan,
            "LB_p12": m["lb_p12"], "LB_p24": m["lb_p24"],
            "boom_start": None, "boom_end": None, "decline_end": None
        })
        horizon_vars["sigma2_L"] = float(fc["var"].iloc[-1])

    # ---------------- combined figure (if both present) ----------------
    if (y_mining is not None) and (y_lifestyle is not None) and \
       (fc_mining is not None) and (fc_lifestyle is not None):
        combined_base = os.path.join(args.outdir, "factor_funnels_combined")
        make_combined_funnel_plot(y_mining, fc_mining, "Mining",
                                  y_lifestyle, fc_lifestyle, "Lifestyle",
                                  combined_base)

    # ---------------- summary outputs ----------------
    if summary_rows:
        summ = pd.DataFrame(summary_rows)
        summ_path = os.path.join(args.outdir, "factor_arimax_summary.csv")
        summ.to_csv(summ_path, index=False, float_format="%.4f")
        print(f"[ok] Wrote model summary CSV: {summ_path}")

        if args.tex_out:
            tex_df = summ[["Factor","Order","AICc","LB_p12","LB_p24","boom","decline"]].copy()
            for c in ["AICc","LB_p12","LB_p24","boom","decline"]:
                tex_df[c] = tex_df[c].astype(float).round(3)
            tex_df.columns = ["Factor","ARIMA$(p,0,q)$","AICc","LB12","LB24","Boom coef","Decline coef"]
            tex = tex_df.to_latex(index=False, escape=False, column_format="lrrrrrr")
            with open(args.tex_out, "w") as f:
                f.write(tex)
            print(f"[ok] Wrote LaTeX summary: {args.tex_out}")

    if horizon_vars:
        hsum = pd.DataFrame([{"h_months": args.h,
                              "sigma2_PS": horizon_vars.get("sigma2_PS", np.nan),
                              "sigma2_L":  horizon_vars.get("sigma2_L",  np.nan)}])
        hsum_path = os.path.join(args.outdir, f"factor_funnels_h{args.h}.csv")
        hsum.to_csv(hsum_path, index=False, float_format="%.6f")
        print(f"[ok] Wrote horizon-variance summary: {hsum_path}")

    # ---- Optional: print/write a factor-level fit + 10y uncertainty table ----
    if summary_rows and horizon_vars and args.print_fit_summary:
        summ = pd.DataFrame(summary_rows)
        var_map = {
            "Mining": horizon_vars.get("sigma2_PS", np.nan),
            "Lifestyle": horizon_vars.get("sigma2_L", np.nan),
        }
        rows = []
        for _, r in summ.iterrows():
            fac = str(r["Factor"])
            s2 = float(var_map.get(fac, np.nan))
            sd = float(np.sqrt(s2)) if np.isfinite(s2) else np.nan
            x95 = float(np.exp(1.96 * sd)) if np.isfinite(sd) else np.nan
            rows.append({
                "Factor": fac,
                "ARIMA(p,0,q)": r["Order"],
                "AICc": float(r["AICc"]),
                "LB12": float(r["LB_p12"]),
                "LB24": float(r["LB_p24"]),
                "h_months": int(args.h),
                "sigma2_10y": s2,
                "sigma_10y": sd,
                "x95_factor": x95,
            })

        fit_tab = pd.DataFrame(rows)
        view = fit_tab.copy()
        view["AICc"] = view["AICc"].round(2)
        view["LB12"] = view["LB12"].round(3)
        view["LB24"] = view["LB24"].round(3)
        view["sigma2_10y"] = view["sigma2_10y"].round(6)
        view["sigma_10y"] = view["sigma_10y"].round(4)
        view["x95_factor"] = view["x95_factor"].round(3)
        show_cols = ["Factor","ARIMA(p,0,q)","AICc","LB12","LB24","sigma_10y","x95_factor"]

        print("\n[Factor-level fit and 10y uncertainty (NOT region-scaled)]")
        print(view[show_cols].to_string(index=False))
        print("Note: region bands use exp(1.96*sqrt(lambda_r^2*sigma2_PS + gamma_r^2*sigma2_L + sigma2_epsilon)).\n")

    # ---------------- AR/MA coefficient LaTeX (stdout) ----------------
    if (not args.no_table) and fit_results:
        # First pass: collect rows and find max AR/MA length
        rows = []
        max_p = 0
        max_q = 0
        for fac, fx in fit_results.items():
            p, d, q = fx["order"]
            res = fx["res"]

            # AR/MA params
            ar = np.asarray(getattr(res, "arparams", np.array([])), dtype=float)
            ma = np.asarray(getattr(res, "maparams", np.array([])), dtype=float)
            max_p = max(max_p, len(ar))
            max_q = max(max_q, len(ma))

            # intercept
            intercept = np.nan
            if hasattr(res, "params"):
                try:
                    # prefer 'const', fall back to 'intercept'
                    if "const" in res.params.index:
                        intercept = float(res.params["const"])
                    elif "intercept" in res.params.index:
                        intercept = float(res.params["intercept"])
                except Exception:
                    intercept = np.nan

            # sigma^2: prefer explicit param; else scale (if informative); else residual variance
            sigma2_candidates = []
            try:
                if hasattr(res, "params") and "sigma2" in res.params.index:
                    sigma2_candidates.append(float(res.params["sigma2"]))
            except Exception:
                pass
            try:
                if hasattr(res, "scale") and np.isfinite(res.scale):
                    sigma2_candidates.append(float(res.scale))
            except Exception:
                pass
            try:
                resid = res.resid
                resid = resid[np.isfinite(resid)]
                if resid.size > 5:
                    sigma2_candidates.append(float(np.var(resid, ddof=1)))
            except Exception:
                pass

            sigma2 = np.nan
            for s2 in sigma2_candidates:
                if np.isfinite(s2) and s2 > 0:
                    sigma2 = s2
                    break

            sigma = float(np.sqrt(sigma2)) if np.isfinite(sigma2) else np.nan

            rows.append({
                "Factor": fac,
                "Order": f"({p},{d},{q})",
                "Intercept": intercept,
                "sigma2": sigma2,
                "sigma": sigma,
                "ar_list": ar,
                "ma_list": ma,
            })

        # Build table rows with phi/theta and sigma columns
        out_rows = []
        for r in rows:
            row = {
                "Factor": r["Factor"],
                "ARIMA$(p,d,q)$": r["Order"],
                "Intercept": r["Intercept"],
            }
            for i in range(max_p):
                row[f"$\\phi_{i+1}$"] = float(r["ar_list"][i]) if i < len(r["ar_list"]) else np.nan
            for j in range(max_q):
                row[f"$\\theta_{j+1}$"] = float(r["ma_list"][j]) if j < len(r["ma_list"]) else np.nan
            row["$\\sigma$ (innov)"] = r["sigma"]
            row["$\\sigma^2$"] = r["sigma2"]
            out_rows.append(row)

        params_df = pd.DataFrame(out_rows)

        # Round general numerics to 3 decimals
        phi_cols = [c for c in params_df.columns if c.startswith("$\\phi_")]
        theta_cols = [c for c in params_df.columns if c.startswith("$\\theta_")]
        round_cols = ["Intercept", "$\\sigma$ (innov)"] + phi_cols + theta_cols
        for c in round_cols:
            if c in params_df.columns:
                params_df[c] = params_df[c].astype(float).round(3)

        # sigma^2 in scientific notation
        if "$\\sigma^2$" in params_df.columns:
            params_df["$\\sigma^2$"] = params_df["$\\sigma^2$"].apply(
                lambda x: "" if (not np.isfinite(x)) else f"{x:.3e}"
            )

        print(params_df)

        tex = params_df.to_latex(index=False, escape=False,
                                 column_format="l" + "r"*(len(params_df.columns)-1), float_format="%.4f")

        print(f"\n[ARIMA coefficient table]\n{tex}\n")

    print("[done]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
