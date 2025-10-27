import os, re, math, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.lines import Line2D

# Optional imports for ARIMA; only needed if we have to recompute funnels
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: F401 (kept for parity with build script)
    _HAS_SM = True
except Exception:
    _HAS_SM = False

# ---------------- Paths ----------------
def resolve_env_path(var, default):
    p = os.environ.get(var, '').strip()
    return Path(os.path.expanduser(p)) if p else Path(default).expanduser()

HOME = Path.home()
PAPER_PATH = resolve_env_path('PAPER_PATH', HOME / 'Documents/github/neoval-project')
DATA_PATH  = resolve_env_path('DATA_PATH',  PAPER_PATH / 'data')
OUT        = resolve_env_path('OUT',        PAPER_PATH / 'data_out')
OUT.mkdir(parents=True, exist_ok=True)

# Canonical files
LOADINGS_CSV  = OUT / 'expanding_loadings_median_2014-01_2024-12_step3.csv'
GROWTHMAP_CSV = OUT / 'growth_mapping_fM2.00.csv'
FUNNELS_CSV   = OUT / 'factor_funnels_h120.csv'
SIGEPS_CSV    = OUT / 'sigma2_epsilon_h120.csv'
FACTORS_CSV   = DATA_PATH / 'df_factor_trends.csv'   # for market/mining/lifestyle series

# Outputs (figure)
FIG_PDF = OUT / 'fig_scenario_bands_fM2.pdf'
FIG_PNG = OUT / 'fig_scenario_bands_fM2.png'

# ---------------- Utils ----------------
def norm_header(s: str) -> str:
    if not isinstance(s, str):
        return ''
    t = s.strip().replace('$', '')
    t = re.sub(r'\\(beta|lambda|gamma)', r'\1', t)  # \beta_r -> beta_r
    t = t.replace('\\', '').replace('–', '-').replace('—', '-').replace('{', '').replace('}', '').lower()
    t = t.replace('β', 'beta').replace('λ', 'lambda').replace('γ', 'gamma')
    t = re.sub(r'\s+', '_', t)
    t = re.sub(r'_+', '_', t).strip('_')
    return t

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: norm_header(c) for c in df.columns}
    inv = {}
    city = next((k for k,v in cols.items() if v in ('city','region','name','sa4_name','region_label','pretty_region')), None)
    if city is None:
        raise ValueError(f'Could not find region label column. Headers: {list(df.columns)}')
    inv[city] = 'City'
    beta = next((k for k,v in cols.items() if v in ('beta_r','beta','beta_market')), None)
    lam  = next((k for k,v in cols.items() if v in ('lambda_r','lambda','lambda_mining')), None)
    gam  = next((k for k,v in cols.items() if v in ('gamma_r','gamma','gamma_lifestyle')), None)
    missing = [n for (n,c) in [('beta_r',beta),('lambda_r',lam),('gamma_r',gam)] if c is None]
    if missing:
        raise ValueError(f'Missing columns in loadings CSV for {missing}. Headers: {list(df.columns)}')
    inv[beta] = 'beta_r'; inv[lam] = 'lambda_r'; inv[gam] = 'gamma_r'
    return df.rename(columns=inv)

def normalize_region(s: str) -> str:
    if not isinstance(s, str):
        return ''
    t = s.strip()
    if t.upper().startswith('GREATER '):
        t = t[8:].strip()
    t = {'AUSTRALIAN CAPITAL TERRITORY':'ACT','Australian Capital Territory':'ACT'}.get(t, t)
    return ' '.join(t.replace('–','-').replace('—','-').split())

def minimal_axes(ax):
    for s in ('top','right'):
        ax.spines[s].set_visible(False)
    for s in ('left','bottom'):
        ax.spines[s].set_linewidth(0.6)
    ax.grid(False)

def fmt_plain(x, pos):
    s = f'{x:.2f}'.rstrip('0').rstrip('.')
    return s

def regular_ticks(xmin, xmax, step):
    start = math.floor(xmin / step) * step
    end   = math.ceil(xmax / step) * step
    ticks = np.arange(start, end + step*0.5, step)
    return ticks[ticks > 0].tolist()

# -------------- Funnels (re)compute --------------
def _aicc(llf: float, k: int, n: int) -> float:
    aic = 2.0 * k - 2.0 * llf
    return aic if (n - k - 1) <= 0 else aic + (2.0 * k * (k + 1)) / (n - k - 1)

def _ensure_ms(df: pd.DataFrame, date_col='month_date') -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df.asfreq('MS')

def _fit_arima_parsimony(y: pd.Series, pmax=3, qmax=2, parsimony_delta=2.0):
    y_fit = y.dropna()
    n = y_fit.shape[0]
    cands = []
    for p in range(pmax+1):
        for q in range(qmax+1):
            try:
                mod = SARIMAX(endog=y, order=(p,0,q), seasonal_order=(0,0,0,0),
                              trend='c', enforce_stationarity=True, enforce_invertibility=True)
                res = mod.fit(disp=False, maxiter=500)
                k = int(res.params.shape[0]); llf = float(res.llf)
                aicc_val = _aicc(llf, k, n)
                cands.append({'order': (p,0,q), 'aicc': aicc_val, 'aic': float(res.aic),
                              'bic': float(res.bic), 'k': k, 'res': res})
            except Exception:
                continue
    if not cands:
        raise RuntimeError('No ARIMA candidate converged for funnels.')
    best = min(cands, key=lambda c: c['aicc'])
    shortlist = [c for c in cands if c['aicc'] <= best['aicc'] + parsimony_delta]
    shortlist.sort(key=lambda c: ((c['order'][0]+c['order'][2]), c['order'][2], c['order'][0], c['aicc']))
    chosen = shortlist[0]
    return chosen['order'], chosen['res']

def _forecast_var_at_h(res, h: int) -> float:
    pred = res.get_forecast(steps=h, exog=None)
    try:
        se = pred.se_mean
    except Exception:
        ci = pred.conf_int(alpha=0.05)
        se = (ci.iloc[:,1] - ci.iloc[:,0]) / (2 * 1.96)
    var = float((se.iloc[-1] if hasattr(se, 'iloc') else se[-1])**2)
    return var

def ensure_funnels(h: int = 120, recompute: bool = False) -> tuple:
    """Return (sigma2_PS, sigma2_L). If CSV missing or recompute=True, fit from df_factor_trends.csv."""
    if Path(FUNNELS_CSV).exists() and not recompute:
        fn = pd.read_csv(FUNNELS_CSV)
        cols = {c: norm_header(c) for c in fn.columns}
        s2ps = next((c for c,n in cols.items() if n == 'sigma2_ps'), None)
        s2l  = next((c for c,n in cols.items() if n == 'sigma2_l'), None)
        if s2ps and s2l:
            return float(fn.iloc[0][s2ps]), float(fn.iloc[0][s2l])

    if not _HAS_SM:
        raise RuntimeError('statsmodels is required to recompute funnels (pip install statsmodels).')
    if not Path(FACTORS_CSV).exists():
        raise FileNotFoundError(f'Missing factor trends CSV for funnels: {FACTORS_CSV}')

    df = pd.read_csv(FACTORS_CSV)
    if 'month_date' not in df.columns:
        raise ValueError(f"{FACTORS_CSV} must include 'month_date'.")
    df = _ensure_ms(df, 'month_date')
    for col in ('mining','lifestyle'):
        if col not in df.columns:
            raise ValueError(f"{FACTORS_CSV} is missing required column '{col}'.")

    # Mining
    y_m = df['mining'].astype(float).dropna()
    order_m, res_m = _fit_arima_parsimony(y_m)
    s2_PS = _forecast_var_at_h(res_m, h)

    # Lifestyle
    y_l = df['lifestyle'].astype(float).dropna()
    order_l, res_l = _fit_arima_parsimony(y_l)
    s2_L = _forecast_var_at_h(res_l, h)

    pd.DataFrame([{'h_months': h, 'sigma2_PS': s2_PS, 'sigma2_L': s2_L}]).to_csv(FUNNELS_CSV, index=False)
    print(f'[ok] Wrote {FUNNELS_CSV}')
    return s2_PS, s2_L

# -------------- Epsilon ensure --------------
def ensure_sigma2_epsilon(h: int = 120):
    """Ensure epsilon CSV exists; if missing, create zeros per region (PS+L-only fallback)."""
    if Path(SIGEPS_CSV).exists():
        return pd.read_csv(SIGEPS_CSV)
    if not Path(LOADINGS_CSV).exists():
        raise FileNotFoundError(f'To fabricate {SIGEPS_CSV}, need {LOADINGS_CSV} for region list.')
    ld = pd.read_csv(LOADINGS_CSV)
    ld = normalize_headers(ld)
    df = pd.DataFrame({'Region': ld['City'].apply(normalize_region).unique()})
    df = df.sort_values('Region').reset_index(drop=True)
    df['sigma2_epsilon'] = 0.0
    df['h_months'] = h
    df.to_csv(SIGEPS_CSV, index=False)
    print(f'[warn] Created {SIGEPS_CSV} with zeros (PS+L-only fallback).')
    return df

# -------------- Growth map ensure --------------
def ensure_growth_mapping(s2_PS: float, s2_L: float, recompute: bool = False):
    """Ensure growth_mapping_fM2.00.csv exists; else create from loadings + funnels + epsilon (+ doubling time)."""
    if Path(GROWTHMAP_CSV).exists() and not recompute:
        return pd.read_csv(GROWTHMAP_CSV)

    if not Path(LOADINGS_CSV).exists():
        raise FileNotFoundError(f'Missing loadings CSV: {LOADINGS_CSV}')
    ld = pd.read_csv(LOADINGS_CSV)
    ld = normalize_headers(ld)
    ld['Region']   = ld['City'].apply(normalize_region)
    ld['beta_r']   = pd.to_numeric(ld['beta_r'], errors='coerce')
    ld['lambda_r'] = pd.to_numeric(ld['lambda_r'], errors='coerce')
    ld['gamma_r']  = pd.to_numeric(ld['gamma_r'], errors='coerce')
    ld = ld.dropna(subset=['beta_r','lambda_r','gamma_r'])

    # epsilon
    eps = ensure_sigma2_epsilon()
    eps['Region_norm'] = eps['Region'].apply(normalize_region)
    ld['Region_norm']  = ld['Region'].apply(normalize_region)
    df = ld.merge(eps[['Region_norm','sigma2_epsilon']], how='left', on='Region_norm')
    s2_eps = df['sigma2_epsilon'].fillna(0.0).astype(float)

    # x95 components
    var_partial = (df['lambda_r']**2) * s2_PS + (df['gamma_r']**2) * s2_L
    x95 = np.exp(1.96 * np.sqrt(var_partial))
    x95_total = np.exp(1.96 * np.sqrt(var_partial + s2_eps))

    # Scenario under fM=2 and doubling time
    fr_if_fM_2 = np.power(2.0, df['beta_r'])

    # Market doubling time from factors CSV
    doubling_time = np.nan
    if Path(FACTORS_CSV).exists():
        try:
            f = pd.read_csv(FACTORS_CSV)
            f = _ensure_ms(f, 'month_date')
            if 'market' in f.columns:
                t = np.arange(f.shape[0], dtype=float)
                y = f['market'].astype(float).to_numpy()
                t_mean = t.mean(); y_mean = np.nanmean(y)
                cov_ty = np.nanmean((t - t_mean) * (y - y_mean))
                var_t  = np.nanmean((t - t_mean)**2)
                slope_m = cov_ty / var_t if var_t > 0 else np.nan
                if slope_m and slope_m > 0:
                    T_M_years = (np.log(2.0) / slope_m) / 12.0
                    doubling_time = T_M_years
        except Exception:
            pass

    if not np.isnan(doubling_time):
        doubling = doubling_time / df['beta_r']
    else:
        doubling = np.nan

    gm = pd.DataFrame({
        'Region': df['Region'],
        'beta_r': df['beta_r'].round(2),
        'fr_if_fM_2': fr_if_fM_2.round(2),
        'Doubling time': np.round(doubling, 2),
        'x95': np.round(x95, 2),
        'x95_total': np.round(x95_total, 2)
    })
    gm.to_csv(GROWTHMAP_CSV, index=False)
    print(f'[ok] Wrote {GROWTHMAP_CSV}')
    return gm

# -------------- Plot (segmented bar) --------------
def make_segmented_plot(df: pd.DataFrame, s2_PS: float, s2_L: float, tick_step: float = 0.25, linear_x: bool=False, bar_lw: float=6.0):
    var_PS  = (df['lambda_r']**2) * s2_PS
    var_L   = (df['gamma_r']**2)  * s2_L
    var_eps = df['sigma2_epsilon'].fillna(0.0)
    var_tot = var_PS + var_L + var_eps
    w_tot   = 1.96 * np.sqrt(var_tot.clip(lower=0))

    denom = var_tot.replace(0, np.nan)
    sh_PS  = (var_PS  / denom).fillna(0.0).to_numpy()
    sh_L   = (var_L   / denom).fillna(0.0).to_numpy()
    sh_eps = (var_eps / denom).fillna(0.0).to_numpy()

    df_plot = df.copy().assign(var_tot=var_tot, w_tot=w_tot, sh_PS=sh_PS, sh_L=sh_L, sh_eps=sh_eps)
    df_plot = df_plot.sort_values('beta_r', ascending=False).reset_index(drop=True)

    mu = np.log(df_plot['fr_if_fM_2'].astype(float))
    L = mu - df_plot['w_tot']; R = mu + df_plot['w_tot']
    xmin = float(np.exp(L.min())); xmax = float(np.exp(R.max()))

    C_MINING, C_LIFE, C_EPS, C_EDGE = '#8c564b', '#2ca02c', '#7f7f7f', '#000000'

    fig = plt.figure(figsize=(9, 4.5), dpi=300); ax = plt.gca()
    y = np.arange(len(df_plot))

    for i, r in df_plot.iterrows():
        mu_i = float(np.log(r['fr_if_fM_2'])); w_i = float(r['w_tot'])
        if not np.isfinite(w_i) or w_i <= 0:
            ax.scatter(np.exp(mu_i), i, s=16, color='black', zorder=3)
            continue
        sh = np.array([r['sh_PS'], r['sh_L'], r['sh_eps']], dtype=float)
        if sh.sum() <= 0: sh = np.array([1.0, 0.0, 0.0])
        sh = sh / sh.sum()
        cum = np.cumsum(np.concatenate(([0.0], sh)))
        L_i, R_i = mu_i - w_i, mu_i + w_i
        seg_edges = np.exp(L_i + (R_i - L_i) * cum)

        colors = [C_MINING, C_LIFE, C_EPS]
        for k in range(3):
            ax.plot([seg_edges[k], seg_edges[k+1]], [i, i], color=colors[k], linewidth=bar_lw, solid_capstyle='butt', zorder=2)

        ax.plot([np.exp(L_i), np.exp(R_i)], [i, i], color=C_EDGE, linewidth=0.8, zorder=1)
        ax.scatter(np.exp(mu_i), i, s=16, color='black', zorder=3)

    ax.set_xscale('linear' if linear_x else 'log')
    ticks = regular_ticks(xmin, xmax, step=tick_step)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_plain))

    ax.set_xlabel(r'Scenario under $f_M=2$ ($f_r$), with 95% band (segmented by variance share)')
    ax.set_yticks(y)
    ax.set_yticklabels(df_plot['Region'], fontsize=8)

    ax.axvline(2.0, linewidth=0.8, color='#444444', alpha=0.9)

    legend_elems = [
        Line2D([0],[0], color=C_MINING, lw=bar_lw, label='Mining'),
        Line2D([0],[0], color=C_LIFE,   lw=bar_lw, label='Lifestyle'),
        Line2D([0],[0], color=C_EPS,    lw=bar_lw, label='Idiosyncratic ($\epsilon$)'),
        Line2D([0],[0], color=C_EDGE,   lw=0.8,    label='Total band'),
        Line2D([0],[0], marker='o', color='black', lw=0, label=r'Scenario $f_r$'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', frameon=False, fontsize=8)

    minimal_axes(ax); plt.tight_layout()
    fig.savefig(FIG_PDF, bbox_inches='tight')
    fig.savefig(FIG_PNG, bbox_inches='tight')
    print(f'[ok] Saved figure: {FIG_PDF} / {FIG_PNG}')

def main():
    warnings.filterwarnings('ignore')

    ap = argparse.ArgumentParser()
    ap.add_argument('--tick-step', type=float, default=0.25, help='Numeric step for major xticks')
    ap.add_argument('--linear-x', action='store_true', help='Use a linear x-axis instead of log')
    ap.add_argument('--bar-lw', type=float, default=6.0, help='Line width for segmented bar')
    ap.add_argument('--recompute-funnels', action='store_true', help='Recompute sigma2_PS/sigma2_L from factor trends')
    ap.add_argument('--recompute-growthmap', action='store_true', help='Rebuild growth_mapping_fM2.00.csv from components')
    ap.add_argument('--h', type=int, default=120, help='Forecast horizon in months for funnels and epsilon (default 120)')
    args = ap.parse_args()

    if not Path(LOADINGS_CSV).exists():
        raise FileNotFoundError(f'Required loadings file not found: {LOADINGS_CSV}')

    s2_PS, s2_L = ensure_funnels(h=args.h, recompute=args.recompute_funnels)
    eps_df = ensure_sigma2_epsilon(h=args.h)
    gm = ensure_growth_mapping(s2_PS, s2_L, recompute=args.recompute_growthmap)

    ld = pd.read_csv(LOADINGS_CSV); ld = normalize_headers(ld)
    ld['Region'] = ld['City'].apply(normalize_region)
    # Avoid 'beta_r' column collision: take beta from growth map, lambda/gamma from loadings
    ld_nob = ld.drop(columns=[c for c in ['beta_r'] if c in ld.columns])
    df = ld_nob.merge(gm[['Region','beta_r','fr_if_fM_2']], on='Region', how='inner')
    df = df.merge(eps_df[['Region','sigma2_epsilon']], on='Region', how='left')

    make_segmented_plot(df=df, s2_PS=s2_PS, s2_L=s2_L,
                        tick_step=args.tick_step, linear_x=args.linear_x, bar_lw=args.bar_lw)

if __name__ == '__main__':
    raise SystemExit(main())
