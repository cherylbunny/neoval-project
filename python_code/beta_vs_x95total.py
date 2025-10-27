#!/usr/bin/env python
import os, sys, re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Paths (env-aware; defaults match your repo)
# -----------------------
def resolve_env_path(var, default):
    p = os.environ.get(var, "").strip()
    return Path(os.path.expanduser(p)) if p else Path(default).expanduser()

HOME = Path.home()
PAPER_PATH = resolve_env_path("PAPER_PATH", HOME / "Documents/github/neoval-project")
DATA_PATH  = resolve_env_path("DATA_PATH",  PAPER_PATH / "data")
OUT        = resolve_env_path("OUT",        PAPER_PATH / "data_out")
OUT.mkdir(parents=True, exist_ok=True)

# Inputs
LOADINGS_CSV  = OUT / "expanding_loadings_median_2014-01_2024-12_step3.csv"
GROWTHMAP_CSV = OUT / "growth_mapping_fM2.00.csv"
FUNNELS_CSV   = OUT / "factor_funnels_h120.csv"
SIGEPS_CSV    = OUT / "sigma2_epsilon_h120.csv"

# Outputs
FIG_PDF  = OUT / "fig_beta_vs_x95total.pdf"
FIG_PNG  = OUT / "fig_beta_vs_x95total.png"
TABLE_CSV = OUT / "beta_vs_x95total.csv"

# -----------------------
# Header + region normalization
# -----------------------
def norm_header(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip()
    t = t.replace("$", "")                             # strip TeX math $
    t = re.sub(r"\\(beta|lambda|gamma)", r"\1", t)     # \beta_r -> beta_r
    t = t.replace("\\", "")                            # remove any remaining backslashes
    t = t.replace("–", "-").replace("—", "-")
    t = t.replace("{", "").replace("}", "")
    t = t.lower()
    t = t.replace("β", "beta").replace("λ", "lambda").replace("γ", "gamma")
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: norm_header(c) for c in df.columns}
    # Map to canonical names
    inv = {}
    # City / Region label
    city_col = next((k for k,v in cols.items() if v in ("city","region","name","sa4_name","region_label","pretty_region")), None)
    if city_col is None:
        raise ValueError(f"Could not find region label column in headers: {list(df.columns)}")
    inv[city_col] = "City"
    # betas / lambdas / gammas
    beta_col   = next((k for k,v in cols.items() if v in ("beta_r","beta","beta_market")), None)
    lambda_col = next((k for k,v in cols.items() if v in ("lambda_r","lambda","lambda_mining")), None)
    gamma_col  = next((k for k,v in cols.items() if v in ("gamma_r","gamma","gamma_lifestyle")), None)
    missing = [n for (n,c) in [("beta_r",beta_col),("lambda_r",lambda_col),("gamma_r",gamma_col)] if c is None]
    if missing:
        raise ValueError(f"Could not find required columns {missing} in headers: {list(df.columns)}")
    inv[beta_col]   = "beta_r"
    inv[lambda_col] = "lambda_r"
    inv[gamma_col]  = "gamma_r"
    return df.rename(columns=inv)

def normalize_region(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip()
    if t.upper().startswith("GREATER "):
        t = t[8:].strip()
    t = {"AUSTRALIAN CAPITAL TERRITORY":"ACT","Australian Capital Territory":"ACT"}.get(t, t)
    t = " ".join(t.replace("–","-").replace("—","-").split())
    return t

# -----------------------
# Core
# -----------------------
def read_required_csv(path: Path, purpose: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file for {purpose}: {path}")
    return pd.read_csv(path)

def compute_table(loadings_renamed: pd.DataFrame) -> pd.DataFrame:
    """Return Region, beta_r, x95_total; prefer growth_map if available."""
    if GROWTHMAP_CSV.exists():
        gm = pd.read_csv(GROWTHMAP_CSV)
        # be tolerant with column names
        cols = {c: norm_header(c) for c in gm.columns}
        region_col = next((c for c,n in cols.items() if n in ("region","city","name")), None)
        beta_col   = next((c for c,n in cols.items() if n in ("beta_r","beta","beta_market")), None)
        x95_col    = next((c for c,n in cols.items() if n in ("x95_total","x95total")), None)
        if region_col and beta_col and x95_col:
            out = gm[[region_col, beta_col, x95_col]].copy()
            out.columns = ["Region","beta_r","x95_total"]
            return out
        else:
            print(f"[info] {GROWTHMAP_CSV.name} present but missing Region/beta_r/x95_total; computing from funnels+epsilon.", file=sys.stderr)

    funnels = read_required_csv(FUNNELS_CSV, "factor funnels (sigma2_PS, sigma2_L @ 10y)")
    # tolerant column pick
    fcols = {c: norm_header(c) for c in funnels.columns}
    ps_col = next((c for c,n in fcols.items() if n == "sigma2_ps"), None)
    l_col  = next((c for c,n in fcols.items() if n == "sigma2_l"), None)
    if not (ps_col and l_col):
        raise ValueError(f"{FUNNELS_CSV} must have sigma2_PS and sigma2_L. Found: {list(funnels.columns)}")
    s2_PS = float(funnels.iloc[0][ps_col]); s2_L = float(funnels.iloc[0][l_col])

    eps = pd.read_csv(SIGEPS_CSV) if SIGEPS_CSV.exists() else None
    if eps is None:
        print(f"[warn] {SIGEPS_CSV.name} not found; PS+L only for all regions.", file=sys.stderr)

    df = loadings_renamed.copy()
    df["Region"]   = df["City"].apply(normalize_region)
    df["beta_r"]   = pd.to_numeric(df["beta_r"], errors="coerce")
    df["lambda_r"] = pd.to_numeric(df["lambda_r"], errors="coerce")
    df["gamma_r"]  = pd.to_numeric(df["gamma_r"], errors="coerce")
    df = df.dropna(subset=["beta_r","lambda_r","gamma_r"])

    var_partial = (df["lambda_r"]**2) * s2_PS + (df["gamma_r"]**2) * s2_L

    if eps is not None and {"Region","sigma2_epsilon"}.issubset(eps.columns):
        eps = eps.copy()
        eps["Region_norm"] = eps["Region"].apply(normalize_region)
        df["Region_norm"]  = df["Region"].apply(normalize_region)
        df = df.merge(eps[["Region_norm","sigma2_epsilon"]], how="left", on="Region_norm")
        missing = df.loc[df["sigma2_epsilon"].isna(), "Region"].tolist()
        if missing:
            print("[warn] Missing sigma2_epsilon for these regions; PS+L only:", *missing, sep="\n  - ")
        s2_eps = df["sigma2_epsilon"].fillna(0.0).astype(float)
    else:
        s2_eps = 0.0

    var_total = var_partial + s2_eps
    x95_total = np.exp(1.96 * np.sqrt(var_total))

    out = pd.DataFrame({"Region": df["Region"], "beta_r": df["beta_r"], "x95_total": x95_total})
    return out

# -----------------------
# Plot
# -----------------------
def minimal_axes(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_linewidth(0.6)
    ax.grid(False)

def stable_jitter(labels, scale=0.006):
    vals = []
    for s in labels:
        h = hashlib.sha1(str(s).encode("utf-8")).hexdigest()
        v = int(h[:8], 16) / 0xFFFFFFFF
        vals.append((v - 0.5) * 2 * scale)
    return np.array(vals)

def make_scatter(df_rr: pd.DataFrame):
    fig = plt.figure(figsize=(9, 4.5), dpi=300)
    ax = plt.gca()
    x = df_rr["beta_r"].astype(float).to_numpy()
    y = df_rr["x95_total"].astype(float).to_numpy()
    xj = x + stable_jitter(df_rr["Region"])
    yj = y + stable_jitter(df_rr["Region"])

    ax.scatter(xj, yj, s=16)
    ax.axvline(1.0, linewidth=0.8)
    ax.axhline(float(np.median(y)), linewidth=0.8)
    ax.set_xlabel(r"$\beta_r$ (Market sensitivity)")
    ax.set_ylabel(r"$x95_{\mathrm{total}}$ (10-year 95% band)")

    for xi, yi, lbl in zip(xj, yj, df_rr["Region"]):
        ax.annotate(lbl, (xi, yi), xytext=(2, 2), textcoords="offset points", fontsize=7)

    minimal_axes(ax)
    plt.tight_layout()
    return fig

# -----------------------
# Main
# -----------------------
def main():
    if not LOADINGS_CSV.exists():
        raise FileNotFoundError(f"Required loadings file not found: {LOADINGS_CSV}")

    loadings = pd.read_csv(LOADINGS_CSV)
    loadings = normalize_headers(loadings)

    rr = compute_table(loadings)
    rr = rr[["Region","beta_r","x95_total"]].copy().sort_values("beta_r", ascending=False)
    rr.to_csv(TABLE_CSV, index=False)

    fig = make_scatter(rr)
    fig.savefig(FIG_PDF, bbox_inches="tight")
    fig.savefig(FIG_PNG, bbox_inches="tight")

    print(f"[ok] Figure saved:\n  - {FIG_PDF}\n  - {FIG_PNG}")
    print(f"[ok] Table saved:\n  - {TABLE_CSV}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        print("Expected inputs:", file=sys.stderr)
        print(f" - {LOADINGS_CSV}", file=sys.stderr)
        print(f" - {GROWTHMAP_CSV} (optional; will compute if absent)", file=sys.stderr)
        print(f" - {FUNNELS_CSV}", file=sys.stderr)
        print(f" - {SIGEPS_CSV} (optional; falls back to PS+L only when absent)", file=sys.stderr)
        sys.exit(1)
