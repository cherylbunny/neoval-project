#!/usr/bin/env python3
"""
Build a "market doubling" table from precomputed expanding-window loadings,
optionally add a slope-based sanity check for beta, and (optionally) append
10-year uncertainty columns relative to a market-only proxy.

Outputs (ASCII column names):
  - data_out/growth_mapping_fM{xx.xx}.csv / .tex
    Columns: Region, beta_r, lambda_r, gamma_r, fr_if_fM_{...}, Doubling time,
             (optional) x95, x95_total, [optional components if requested]
  - data_out/beta_sanity_check.csv   (optional)

Defaults assume your repo layout.
"""

import os, re, glob, argparse
import numpy as np
import pandas as pd

HOME = os.path.expanduser("~")
ROOT = os.path.join(HOME, "Documents/github/neoval-project")
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "data_out")

# ---------------- utils ----------------
def find_latest_agg_csv():
    cand = glob.glob(os.path.join(OUT, "expanding_loadings_*.csv"))
    return max(cand, key=os.path.getmtime) if cand else ""

def pick_col(df, candidates):
    cols = list(df.columns)
    for pat in candidates:
        for c in cols:
            if (isinstance(pat, str) and pat.lower() in c.lower()) or \
               (not isinstance(pat, str) and re.search(pat, c)):
                return c
    raise KeyError(f"No column matched any of {candidates} in {cols}")

def region2label(region, remove_greater=True, max_components=2, max_len=30):
    s = str(region)
    s = s.replace('AUSTRALIAN CAPITAL TERRITORY', 'ACT').replace('SYDNEY - ','').strip()
    if remove_greater:
        s = s.replace('GREATER ', '')
    def word2lower(w):
        return w if w in ['NSW','QLD','WA','SA','ACT','TAS','VIC','NT'] else w[0].upper()+w[1:].lower()
    def to_lower(ss): return " ".join([word2lower(e) for e in ss.split(" ")])
    def abbrev(ss):   return ''.join(e[0] for e in ss.split(' ')) if len(ss) > max_len else ss
    components = [abbrev(to_lower(e.strip())) for e in s.split('-')][:max_components]
    return " - ".join(components).strip()

def ols_slope_per_month(dates, y):
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    base = idx[0].year * 12 + idx[0].month
    months = (idx.year * 12 + idx.month) - base
    t = months.to_numpy(dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(y)
    t, y = t[mask], y[mask]
    if y.size < 2:
        return np.nan
    t_c = t - t.mean()
    y_c = y - y.mean()
    denom = float(np.dot(t_c, t_c))
    if denom <= 0:
        return np.nan
    return float(np.dot(t_c, y_c) / denom)

def endpoint_slope_per_month(dates, y):
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    i0 = np.argmax(mask)
    iT = len(mask) - 1 - np.argmax(mask[::-1])
    mspan = (idx[iT].year - idx[i0].year) * 12 + (idx[iT].month - idx[i0].month)
    if mspan <= 0:
        return np.nan
    return float((y[iT] - y[i0]) / mspan)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg_csv", default="", help="Aggregated loadings CSV (default: latest in data_out)")
    ap.add_argument("--f_m", type=float, nargs="*", default=[2.0], help="Market growth factors to map (default 2.0)")
    ap.add_argument("--round_beta", type=int, default=2)
    ap.add_argument("--round_growth", type=int, default=2)
    ap.add_argument("--round_time", type=int, default=2)
    ap.add_argument("--tex_out", default="", help="Override LaTeX out path")
    ap.add_argument("--csv_out", default="", help="Override CSV out path")

    # Market doubling time estimation
    ap.add_argument("--tm_years", type=float, default=None,
                    help="If set, use this market doubling time T_M (years). Overrides tm_method.")
    ap.add_argument("--tm_method", choices=["ols","endpoints"], default="ols",
                    help="How to estimate T_M from factor file if tm_years not set.")
    ap.add_argument("--tm_start", default="", help="YYYY-MM subset for T_M start (optional)")
    ap.add_argument("--tm_end",   default="", help="YYYY-MM subset for T_M end (optional)")
    ap.add_argument("--factor_csv", default=os.path.join(DATA, "df_factor_trends.csv"),
                    help="Factor file with 'month_date' and 'market' (for T_M estimation)")

    # Uncertainty (10y) relative to market-only proxy
    ap.add_argument("--include_uncertainty", action="store_true",
                    help="Append 10y uncertainty columns relative to market-only proxy.")
    ap.add_argument("--include_components", action="store_true",
                    help="If set with --include_uncertainty, also add lam_sigma_PS, gam_sigma_L, sigma_epsilon columns.")
    ap.add_argument("--h_months", type=int, default=120,
                    help="Horizon in months to match funnels (default 120).")
    ap.add_argument("--funnels_csv", default="",
                    help="Path to factor funnels CSV with sigma2_PS, sigma2_L (default: data_out/factor_funnels_h{h}.csv).")
    ap.add_argument("--round_sigma", type=int, default=3,
                    help="Rounding for sigma-derived component columns (default 3).")
    ap.add_argument("--round_band", type=int, default=2,
                    help="Rounding for x95 columns (default 2).")

    # Optional sanity check
    ap.add_argument("--do_sanity_check", action="store_true")
    ap.add_argument("--city_idx_csv", default=os.path.join(DATA, "city_indexes.csv"),
                    help="City index file with 'month_date' and region columns")

    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ---- load aggregated loadings (beta, lambda, gamma) ----
    agg_csv = args.agg_csv or find_latest_agg_csv()
    if not agg_csv or not os.path.exists(agg_csv):
        raise FileNotFoundError("No aggregated loadings CSV found in data_out/. Run table_expanding_window.py first.")
    agg = pd.read_csv(agg_csv)

    city_col = pick_col(agg, ["City","Region","city"])
    beta_col = pick_col(agg, [r"\$\\beta_r\$", "beta_r", "beta", "br", "b_r"])
    lam_col  = pick_col(agg, [r"\$\\lambda_r\$", "lambda_r", "lambda", "lr", "l_r"])
    gam_col  = pick_col(agg, [r"\$\\gamma_r\$",  "gamma_r",  "gamma",  "gr", "g_r"])

    out = agg[[city_col, beta_col]].copy()
    out.rename(columns={city_col:"Region", beta_col:"beta_r"}, inplace=True)
    out["beta_r"] = out["beta_r"].astype(float)

    # Loadings (to include by default)
    loadings = agg[[city_col, lam_col, gam_col]].copy()
    loadings.rename(columns={city_col:"Region", lam_col:"lambda_r", gam_col:"gamma_r"}, inplace=True)

    # Always include loadings in the working table (keep unrounded for now)
    out = out.merge(loadings, on="Region", how="left")

    # ---- market doubling time T_M ----
    if args.tm_years is not None:
        T_M_years = float(args.tm_years)
    else:
        fac = pd.read_csv(args.factor_csv)
        if "month_date" not in fac or "market" not in fac:
            raise ValueError(f"{args.factor_csv} must have 'month_date' and 'market'.")
        fac["month_date"] = pd.to_datetime(fac["month_date"])
        fac = fac.sort_values("month_date")
        if args.tm_start:
            fac = fac[fac["month_date"] >= pd.to_datetime(args.tm_start + "-01")]
        if args.tm_end:
            fac = fac[fac["month_date"] <= pd.to_datetime(args.tm_end + "-01")]
        fac = fac.dropna(subset=["market"])
        if len(fac) < 2:
            raise ValueError("Not enough observations to compute T_M.")
        slope = ols_slope_per_month(fac["month_date"], fac["market"]) if args.tm_method=="ols" \
                else endpoint_slope_per_month(fac["month_date"], fac["market"])
        T_months = np.log(2.0) / slope
        T_M_years = float(T_months / 12.0)

    # ---- growth mapping columns (use unrounded beta_r) ----
    for f in args.f_m:
        out[f"fr_if_fM_{f:g}"] = np.power(f, out["beta_r"].values)
    out["Doubling time"] = (T_M_years / out["beta_r"]).astype(float)

    # ---- optional uncertainty (10y) ----
    if args.include_uncertainty:
        funnels_csv = args.funnels_csv or os.path.join(OUT, f"factor_funnels_h{args.h_months}.csv")
        if not os.path.exists(funnels_csv):
            raise FileNotFoundError(f"Funnels file not found: {funnels_csv}. Run build_factor_funnels.py first.")
        fv = pd.read_csv(funnels_csv)
        if not set(["sigma2_PS","sigma2_L"]).issubset(fv.columns):
            raise ValueError(f"{funnels_csv} must contain sigma2_PS and sigma2_L.")
        sigma2_PS = float(fv["sigma2_PS"].iloc[0])
        sigma2_L  = float(fv["sigma2_L"].iloc[0])

        # Auto-load epsilon variances (per-region)
        eps_path = os.path.join(OUT, f"sigma2_epsilon_h{args.h_months}.csv")
        sigma2_eps_map = {}
        if os.path.exists(eps_path):
            eps_df = pd.read_csv(eps_path)
            reg_col = pick_col(eps_df, ["Region","City","region","city"])
            var_col = pick_col(eps_df, ["sigma2_epsilon","sigma2_eps","sigma2_e"])
            eps_df = eps_df[[reg_col, var_col]].copy()
            eps_df.columns = ["Region_raw", "sigma2_epsilon"]
            eps_df["Region"] = eps_df["Region_raw"].map(lambda x: region2label(x, remove_greater=True, max_components=2, max_len=30))
            sigma2_eps_map = {str(r["Region"]): float(r["sigma2_epsilon"]) for _, r in eps_df.iterrows()}
            matched = sum([1 for r in out["Region"] if r in sigma2_eps_map])
            print(f"[info] Loaded epsilon variances from {eps_path}; matched {matched}/{len(out)} regions.")
        else:
            print(f"[warn] No epsilon file found at {eps_path}; proceeding without epsilon contribution.")

        # total log-SDs (compute with unrounded loadings)
        lam = out["lambda_r"].astype(float)
        gam = out["gamma_r"].astype(float)
        s_eps2 = np.array([sigma2_eps_map.get(str(reg), 0.0) for reg in out["Region"]], dtype=float)

        sigma_rel_10y        = np.sqrt(lam**2 * sigma2_PS + gam**2 * sigma2_L)
        sigma_rel_10y_total  = np.sqrt(lam**2 * sigma2_PS + gam**2 * sigma2_L + s_eps2)

        out["x95"]        = np.exp(1.96 * sigma_rel_10y)
        out["x95_total"]  = np.exp(1.96 * sigma_rel_10y_total)

        # optional component columns (magnitudes, log units)
        if args.include_components:
            out["lam_sigma_PS"] = np.abs(lam) * np.sqrt(sigma2_PS)
            out["gam_sigma_L"]  = np.abs(gam) * np.sqrt(sigma2_L)
            out["sigma_epsilon"] = np.sqrt([sigma2_eps_map.get(str(reg), np.nan) for reg in out["Region"]])
            for c in ["lam_sigma_PS","gam_sigma_L","sigma_epsilon"]:
                out[c] = out[c].astype(float).round(args.round_sigma)

        # rounding for band factors
        out["x95"] = out["x95"].astype(float).round(args.round_band)
        out["x95_total"] = out["x95_total"].astype(float).round(args.round_band)

    # ---- order/round for presentation (after computations) ----
    out = out.sort_values("beta_r", ascending=False).reset_index(drop=True)
    out["beta_r"]   = out["beta_r"].astype(float).round(args.round_beta)
    out["lambda_r"] = out["lambda_r"].astype(float).round(args.round_beta)
    out["gamma_r"]  = out["gamma_r"].astype(float).round(args.round_beta)
    for c in [c for c in out.columns if c.startswith("fr_if_fM_")]:
        out[c] = out[c].astype(float).round(args.round_growth)
    out["Doubling time"] = out["Doubling time"].astype(float).round(args.round_time)

    # ---- choose columns for export ----
    growth_cols = [c for c in out.columns if c.startswith("fr_if_fM_")]
    export_cols = ["Region", "beta_r", "lambda_r", "gamma_r"] + growth_cols + ["Doubling time"]
    if args.include_uncertainty:
        export_cols += ["x95", "x95_total"]
        if args.include_components:
            export_cols += ["lam_sigma_PS","gam_sigma_L","sigma_epsilon"]

    # ---- save CSV & LaTeX ----
    if len(args.f_m) == 1:
        tag = f"{args.f_m[0]:.2f}"
        csv_default = os.path.join(OUT, f"growth_mapping_fM{tag}.csv")
        tex_default = os.path.join(OUT, f"table_growth_mapping_fM{tag}.tex")
    else:
        csv_default = os.path.join(OUT, "growth_mapping_multi_fM.csv")
        tex_default = os.path.join(OUT, "table_growth_mapping_multi_fM.tex")
    csv_out = args.csv_out or csv_default
    tex_out = args.tex_out or tex_default

    out[export_cols].to_csv(csv_out, index=False, float_format="%.2f")
    print(f"[ok] Wrote CSV: {csv_out}")

    # LaTeX (ASCII column names)
    colfmt = "l" + "r" * (len(export_cols) - 1)
    formatters = {
        "beta_r":        lambda x: f"{x:.{args.round_beta}f}",
        "lambda_r":      lambda x: f"{x:.{args.round_beta}f}",
        "gamma_r":       lambda x: f"{x:.{args.round_beta}f}",
        "Doubling time": lambda x: f"{x:.{args.round_time}f}",
    }
    for c in growth_cols:
        if c in export_cols:
            nd = args.round_growth
            formatters[c] = (lambda nd=nd: (lambda x: f"{x:.{nd}f}"))()
    if args.include_uncertainty:
        if "x95" in export_cols:
            nd = args.round_band
            formatters["x95"] = (lambda nd=nd: (lambda x: f"{x:.{nd}f}"))()
        if "x95_total" in export_cols:
            nd = args.round_band
            formatters["x95_total"] = (lambda nd=nd: (lambda x: f"{x:.{nd}f}"))()
        if args.include_components:
            nds = args.round_sigma
            for c in ["lam_sigma_PS","gam_sigma_L","sigma_epsilon"]:
                if c in export_cols:
                    formatters[c] = (lambda nds=nds: (lambda x: f"{x:.{nds}f}"))()

    latex_df = out[export_cols]
    latex = latex_df.to_latex(index=False, escape=False, column_format=colfmt, formatters=formatters)
    with open(tex_out, "w") as f:
        f.write(latex)
    print(f"[ok] Wrote LaTeX: {tex_out}")
    print("\n" + latex + "\n")

    # Optional: sanity check beta from slopes
    if args.do_sanity_check:
        city = pd.read_csv(os.path.join(DATA, "city_indexes.csv"))
        if "month_date" not in city:
            raise ValueError("city_indexes.csv must have 'month_date'")
        city["month_date"] = pd.to_datetime(city["month_date"])
        city = city.sort_values("month_date")
        if args.tm_start:
            city = city[city["month_date"] >= pd.to_datetime(args.tm_start + "-01")]
        if args.tm_end:
            city = city[city["month_date"] <= pd.to_datetime(args.tm_end + "-01")]

        fac = pd.read_csv(args.factor_csv)
        fac["month_date"] = pd.to_datetime(fac["month_date"])
        fac = fac.sort_values("month_date")
        if args.tm_start:
            fac = fac[fac["month_date"] >= pd.to_datetime(args.tm_start + "-01")]
        if args.tm_end:
            fac = fac[fac["month_date"] <= pd.to_datetime(args.tm_end + "-01")]
        slope_m = ols_slope_per_month(fac["month_date"], fac["market"]) if args.tm_method=="ols" \
                  else endpoint_slope_per_month(fac["month_date"], fac["market"])

        raw_cols = [c for c in city.columns if c != "month_date"]
        pretty_map = {c: region2label(c, remove_greater=True, max_components=2, max_len=30) for c in raw_cols}
        inv = {}
        for raw, pretty in pretty_map.items():
            inv.setdefault(pretty, raw)

        rows = []
        for _, r in out.iterrows():
            pretty = r["Region"]
            raw_col = inv.get(pretty, None)
            if raw_col is None or raw_col not in city.columns:
                continue
            y = city[raw_col].dropna()
            dt = city.loc[y.index, "month_date"]
            slope_r = ols_slope_per_month(dt, y) if args.tm_method=="ols" else endpoint_slope_per_month(dt, y)
            beta_hat = float(slope_r / slope_m) if slope_m not in (0, np.nan) else np.nan
            T_r_obs_years = float(np.log(2.0) / slope_r / 12.0) if slope_r not in (0, np.nan) else np.nan
            rows.append({
                "Region": pretty,
                "beta_median": float(r["beta_r"]),
                "beta_hat_slope_ratio": beta_hat,
                "T_M_years": T_M_years,
                "T_r_obs_years": T_r_obs_years,
                "delta_beta": beta_hat - float(r["beta_r"]) if beta_hat==beta_hat else np.nan,
            })

        chk = pd.DataFrame(rows)
        for c in ["beta_median","beta_hat_slope_ratio","T_M_years","T_r_obs_years","delta_beta"]:
            chk[c] = chk[c].astype(float).round(3)
        chk_path = os.path.join(OUT, "beta_sanity_check.csv")
        chk.to_csv(chk_path, index=False)
        print(f"[ok] Wrote sanity-check CSV: {chk_path}")

if __name__ == "__main__":
    main()
