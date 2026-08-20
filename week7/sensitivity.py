"""
Week 7 Extended Sensitivity Analysis
====================================
Two-part analysis that answers the Week 7 deliverable:

  Part A -- SHAP-based impact quantification of the "new" features
           (sentiment, VIX-linked) on the *chooser price*.
           Method: chain-rule decomposition
              marginal price impact of feature j = SHAP_j(sigma) x dPrice/dsigma
           computed on the exact Week-6 held-out test set, plus a direct
           univariate perturbation (price vs feature value) as an independent
           cross-check.

  Part B -- Extreme-scenario testing on the live market state:
              - volatility spike +50%
              - rate hike +2% (200 bps)
              - combined spike
              - VIX shock +50% (re-engineered features fed through the models)
           Prices are reported for the BSM(vol_21d) baseline and the best ML
           models (XGBoost = live pick, VIX-proxy = IV-aligned pick).
"""

import json
import pickle
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import w7config as cfg
import tool_engine as te

from data_preparation import load_dataset
from targets import add_fwd_vol_targets
from chooser_option_pricer import bs_call, bs_put

# ═══════════════════════════════════════════════════════════════════════════════
# Part A -- SHAP price-impact decomposition
# ═══════════════════════════════════════════════════════════════════════════════

# the "new" features whose price impact the task asks us to quantify
NEW_FEATURES = ['sentiment_score', 'vix_ratio', 'vix_change_1d',
                'vix_jpm_corr_21d', 'vix_jpm_cross_1d']


def load_test_frame() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Return the exact Week-6 held-out test frame (from vol_search_results.pkl),
    its model-aligned feature matrix, the market state, and the feature list.
    """
    with open(cfg.WEEK6_DIR / 'output' / 'vol_search_results.pkl', 'rb') as f:
        res = pickle.load(f)
    meta = res['_meta']
    mod = meta['mod'].reset_index(drop=True)
    feats = list(meta['feats'])
    test = mod.iloc[meta['te']].reset_index(drop=True)
    X_test = test[feats].astype(float)
    market = pd.DataFrame({
        'spot': test[cfg.SPOT_COL].values,
        'rate': test[cfg.RATE_COL].values / 100.0,
        'q': cfg.Q_YIELD,
        'vol_21d': test['vol_21d'].values,
        'sigma_actual': test[cfg.FWD_VOL_TARGET].values,
    })
    return test, X_test, market, feats


def _pipeline(payload):
    return payload['pipeline'] if isinstance(payload, dict) else payload


def shap_price_impact(family: str, sample: int = 500,
                      verbose: bool = False) -> pd.DataFrame:
    """
    Chain-rule SHAP: SHAP_j on the ML vol prediction, multiplied by the
    chooser vega at the predicted sigma. Returns one row per feature with the
    aggregate price impact (mean |.|) and its share of the mean price.
    """
    import shap as _shap
    test, X_test, market, feats = load_test_frame()
    payload, _feats, _ = te.load_vol_model(family)
    pipe = _pipeline(payload)
    inner = pipe.named_steps['model']
    scaler = pipe.named_steps.get('scale')

    idx = np.arange(len(X_test))
    if len(X_test) > sample:
        idx = np.random.RandomState(0).choice(len(X_test), sample, replace=False)
    X = X_test.values[idx]
    market = market.iloc[idx].reset_index(drop=True)

    X_scaled = scaler.transform(X) if scaler is not None else X
    explainer = _shap.TreeExplainer(inner)
    sv = np.asarray(explainer.shap_values(X_scaled))

    X_df = pd.DataFrame(X, columns=feats)
    sigma = te.predict_sigma_from_pipe(payload, X_df)

    # SHAP is expressed in the model's own target units. For the VIX-proxy the
    # target is the raw VIX level (points), so a 1-point contribution moves
    # sigma by only 0.01; for direct vol models 1 unit of SHAP == 1 unit of
    # (decimal) sigma. dPrice/dtarget = vega * sigma_scale.
    sigma_scale = 0.01 if te._is_vix_proxy(payload) else 1.0

    K, t1, T2 = cfg.CHOOSER_PARAMS['K'], cfg.CHOOSER_PARAMS['t1'], cfg.CHOOSER_PARAMS['T2']
    vega = np.array([te.price_vega(float(market['spot'][i]), K, t1, T2,
                                   float(market['rate'][i]), float(market['q'][i]),
                                   float(sigma[i])) for i in range(len(sigma))])
    price = np.array([te.price_chooser(float(market['spot'][i]), K, t1, T2,
                                       float(market['rate'][i]), float(market['q'][i]),
                                       float(sigma[i])) for i in range(len(sigma))])

    impact = sv * sigma_scale * vega[:, None]        # per-row price impact
    mean_price = float(np.mean(price))
    rows = pd.DataFrame({
        'feature': feats,
        'mean_abs_SHAP_sigma': np.abs(sv).mean(axis=0),
        'mean_vega_$': np.abs(vega).mean(),
        'mean_abs_price_impact_$': np.abs(impact).mean(axis=0),
        'price_impact_%of_mean_price': np.abs(impact).mean(axis=0) / mean_price * 100,
        'is_new_feature': [c in NEW_FEATURES for c in feats],
    }).sort_values('mean_abs_price_impact_$', ascending=False).reset_index(drop=True)
    if verbose:
        print(f'  [{family}] SHAP price-impact decomposition on test set (n={len(X)})')
        print(rows.to_string(index=False))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Part A2 -- univariate perturbation: price vs feature value
# ═══════════════════════════════════════════════════════════════════════════════

def build_base_state() -> Tuple[pd.Series, float, float, float, float]:
    """
    Live market state used as the scenario / perturbation baseline:
        (feature_row_with_regime, spot, rate, q, vol_21d)
    Built from the newest market history (yfinance or cached snapshot).
    """
    market = te.fetch_market_history(period='2y')
    feat = te.build_features(market)
    feat = pd.concat([feat, te.regime_dummies(feat)], axis=1)
    feat = feat.dropna()
    row = feat.iloc[-1]
    spot = float(row[cfg.SPOT_COL])
    rate = float(row[cfg.RATE_COL]) / 100.0
    vol21 = float(row['vol_21d'])
    return row, spot, rate, cfg.Q_YIELD, vol21


def vix_shocked_features(base_row: pd.Series, vix_factor: float = 1.5) -> pd.Series:
    """
    A +50% VIX jump re-engineered through the feature set: the raw VIX level is
    scaled, which moves sentiment_score (VIX position in its 252d range),
    vix_ratio, vix_change_1d and vix_jpm_cross_1d. Correlation and rate
    features are untouched (a level shock, not a correlation shock).
    """
    out = base_row.copy()
    vix_new = float(base_row[cfg.VIX_COL]) * vix_factor
    out[cfg.VIX_COL] = vix_new

    train = load_dataset()
    lo, hi = float(train[cfg.VIX_COL].min()), float(train[cfg.VIX_COL].max())
    span = max(hi - lo, 1e-6)
    sent = float(np.clip(1 - (vix_new - lo) / span, 0.0, 1.0))
    out['sentiment_score'] = sent
    out['vix_ratio'] = float(base_row['vix_ratio']) * vix_factor
    out['vix_change_1d'] = float(base_row['vix_change_1d']) * vix_factor
    out['vix_jpm_cross_1d'] = float(base_row['vix_jpm_cross_1d']) * vix_factor
    return out


def perturb_price_curve(feature: str, base_row: pd.Series,
                        model_family: str = 'vol_vix_proxy',
                        step: Optional[float] = None,
                        n_steps: int = 3,
                        spot_override: Optional[float] = None) -> pd.DataFrame:
    """
    Price the chooser as a single feature is moved away from its base value.
    Uses the given model's predicted sigma; the BSM(vol_21d) baseline line is
    kept fixed at the base vol_21d (except for the vol_21d feature itself,
    where the baseline is re-anchored to the perturbed vol). Spot defaults to
    the base feature row's spot; pass spot_override to price a reference
    moneyness (e.g. the ATM canonical base S=K).
    """
    payload, feats, _ = te.load_vol_model(model_family)
    step = step or 0.0
    base = base_row.copy()
    x0 = float(base[feature])
    span = abs(x0) * 0.1 + 1e-6
    step = step or span

    K, t1, T2 = cfg.CHOOSER_PARAMS['K'], cfg.CHOOSER_PARAMS['t1'], cfg.CHOOSER_PARAMS['T2']
    spot = spot_override if spot_override is not None else float(base[cfg.SPOT_COL])
    rate, q = float(base[cfg.RATE_COL]) / 100.0, cfg.Q_YIELD
    vol21_base = float(base['vol_21d'])

    rows = []
    for k in range(-n_steps, n_steps + 1):
        row = base.copy()
        row[feature] = x0 + k * step
        X = pd.DataFrame([row[feats].values], columns=feats).astype(float)
        sig_ml = float(te.predict_sigma_from_pipe(payload, X)[0])
        # baseline: vol_21d anchor (perturbed for the vol_21d feature)
        sig_base = float(row['vol_21d']) if feature == 'vol_21d' else vol21_base
        p_ml = te.price_chooser(spot, K, t1, T2, rate, q, sig_ml)
        p_base = te.price_chooser(spot, K, t1, T2, rate, q, sig_base)
        rows.append({'feature': feature, 'feature_value': x0 + k * step,
                     'sigma_ML': sig_ml, 'sigma_base': sig_base,
                     'price_ML_$': p_ml, 'price_BSM_$': p_base,
                     'spread_$': p_ml - p_base})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Part B -- extreme-scenario testing
# ═══════════════════════════════════════════════════════════════════════════════

def extreme_scenarios(base_row: pd.Series, spot: float, rate: float, q: float,
                      vol21: float, verbose: bool = True) -> pd.DataFrame:
    """
    Price the chooser under each extreme scenario for the BSM(vol_21d)
    baseline, XGBoost (live pick) and the VIX-proxy model.

    Scenario mechanics:
      * vol_factor  scales every volatility input (BSM and ML) by 1.5;
      * rate_shift  adds 200 bps to the risk-free rate;
      * vix_factor  re-engineers the VIX-linked features and re-predicts the
                    ML volatilities (so the ML models "feel" the VIX jump).
    """
    K, t1, T2 = cfg.CHOOSER_PARAMS['K'], cfg.CHOOSER_PARAMS['t1'], cfg.CHOOSER_PARAMS['T2']

    payload_xgb, feats_xgb, _ = te.load_vol_model('vol_xgb')
    payload_vp, feats_vp, _ = te.load_vol_model('vol_vix_proxy')

    def _predict(payload, feats, row):
        X = pd.DataFrame([row[feats].values], columns=feats).astype(float)
        return float(te.predict_sigma_from_pipe(payload, X)[0])

    base_sig_xgb = _predict(payload_xgb, feats_xgb, base_row)
    base_sig_vp = _predict(payload_vp, feats_vp, base_row)

    rows = []
    for key, sc in cfg.SCENARIOS.items():
        row = base_row
        vix_shocked = sc['vix_factor'] > 1.0
        if vix_shocked:
            row = vix_shocked_features(base_row, sc['vix_factor'])
            sig_xgb = _predict(payload_xgb, feats_xgb, row) * sc['vol_factor']
            sig_vp = _predict(payload_vp, feats_vp, row) * sc['vol_factor']
        else:
            sig_xgb = base_sig_xgb * sc['vol_factor']
            sig_vp = base_sig_vp * sc['vol_factor']
        sig_base = vol21 * sc['vol_factor']
        r_eff = rate + sc['rate_shift']

        p_bsm = te.price_chooser(spot, K, t1, T2, r_eff, q, sig_base)
        p_xgb = te.price_chooser(spot, K, t1, T2, r_eff, q, sig_xgb)
        p_vp = te.price_chooser(spot, K, t1, T2, r_eff, q, sig_vp)
        rows.append({
            'scenario': key, 'label': sc['label'],
            'r_%': round(r_eff * 100, 3),
            'sigma_base_%': round(sig_base * 100, 2),
            'sigma_xgb_%': round(sig_xgb * 100, 2),
            'sigma_vixproxy_%': round(sig_vp * 100, 2),
            'price_BSM_$': p_bsm, 'price_XGB_$': p_xgb, 'price_VIXproxy_$': p_vp,
        })
    out = pd.DataFrame(rows)
    base_row_out = out.iloc[0]
    for col in ('price_BSM_$', 'price_XGB_$', 'price_VIXproxy_$'):
        out[f'{col}_delta'] = out[col] - base_row_out[col]
        out[f'{col}_pct'] = (out[col] - base_row_out[col]) / base_row_out[col] * 100
    if verbose:
        print(f'\n  Extreme-scenario table (base S={spot:.2f}, r={rate*100:.2f}%, q={q*100:.2f}%)')
        print(out.round(3).to_string(index=False))
    return out


def t1_sweep(base_row: pd.Series, spot: float, rate: float, q: float,
             vol21: float) -> pd.DataFrame:
    """Price vs choice date t1 under the base and the +50% vol-spike scenario."""
    payload_xgb, feats_xgb, _ = te.load_vol_model('vol_xgb')
    X = pd.DataFrame([base_row[feats_xgb].values], columns=feats_xgb).astype(float)
    sig_ml = float(np.clip(payload_xgb.predict(X)[0], 1e-4, 1.0))
    K, T2 = cfg.CHOOSER_PARAMS['K'], cfg.CHOOSER_PARAMS['T2']
    rows = []
    for t1 in np.arange(0.1, 1.01, 0.1):
        rows.append({'t1_years': round(float(t1), 2),
                     'price_BSM_base': te.price_chooser(spot, K, t1, T2, rate, q, vol21),
                     'price_ML_base': te.price_chooser(spot, K, t1, T2, rate, q, sig_ml),
                     'price_BSM_spike': te.price_chooser(spot, K, t1, T2, rate, q, vol21 * 1.5),
                     'price_ML_spike': te.price_chooser(spot, K, t1, T2, rate, q, sig_ml * 1.5)})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure generation
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_fonts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def plot_price_impact(impact: pd.DataFrame, out_path):
    plt = _setup_fonts()
    d = impact.head(12).iloc[::-1]
    colors = ['#d73027' if r['is_new_feature'] else '#74add1'
              for _, r in d.iterrows()]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(d['feature'], d['mean_abs_price_impact_$'], color=colors)
    ax.set_xlabel('平均 |价格影响| ($)')
    ax.set_title('SHAP×Vega：特征对 Chooser 价格的边际影响\n(红色=情感/VIX 新特征, 蓝色=其余特征)')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_perturb_curves(curves: Dict[str, pd.DataFrame], out_path):
    plt = _setup_fonts()
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (feature, df) in zip(axes, curves.items()):
        ax.plot(df['feature_value'], df['price_ML_$'], '-o', color='#d73027', label='ML (VIX-proxy)')
        ax.plot(df['feature_value'], df['price_BSM_$'], '-s', color='#2166ac', label='BSM(vol21d)')
        ax.axvline(df['feature_value'].iloc[len(df)//2], color='grey', ls=':', lw=1)
        ax.set_title(feature)
        ax.set_xlabel('特征值'); ax.set_ylabel('Chooser 价格 ($)')
        ax.legend(fontsize=7)
    fig.suptitle('单变量边际影响：特征变动 → Chooser 价格', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_scenarios(scenarios: pd.DataFrame, out_path):
    plt = _setup_fonts()
    sc = scenarios.iloc[1:]                      # drop base
    labels = sc['label'].tolist()
    x = np.arange(len(labels)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w, sc['price_BSM_$_pct'], w, label='BSM(vol21d)', color='#2166ac')
    ax.bar(x, sc['price_XGB_$_pct'], w, label='XGBoost', color='#d73027')
    ax.bar(x + w, sc['price_VIXproxy_$_pct'], w, label='VIX-proxy', color='#fdae61')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, fontsize=8)
    ax.axhline(0, color='grey', lw=0.8)
    ax.set_ylabel('相对基准价格变动 (%)')
    ax.set_title('极端场景：Chooser 价格相对基准的变动 (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_t1_sweep(sweep: pd.DataFrame, out_path):
    plt = _setup_fonts()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(sweep['t1_years'], sweep['price_BSM_base'], '-s', label='BSM 基准', color='#2166ac')
    ax.plot(sweep['t1_years'], sweep['price_ML_base'], '-o', label='ML 基准', color='#d73027')
    ax.plot(sweep['t1_years'], sweep['price_BSM_spike'], '--s', label='BSM 波动率+50%', color='#2166ac')
    ax.plot(sweep['t1_years'], sweep['price_ML_spike'], '--o', label='ML 波动率+50%', color='#d73027')
    ax.set_xlabel('选择日 t1 (年)'); ax.set_ylabel('Chooser 价格 ($)')
    ax.set_title('极端场景随选择日 t1 的变化')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

def run_sensitivity(verbose: bool = True) -> Dict:
    """Run Part A + Part B, write CSVs and figures, return a results dict."""
    if verbose:
        print('=' * 72)
        print('  WEEK 7 -- EXTENDED SENSITIVITY ANALYSIS')
        print('=' * 72)

    out = {}

    # --- Part A: SHAP x vega price-impact decomposition ----------------------
    if verbose:
        print('\n[Part A] SHAP-based price-impact of sentiment/VIX features...')
    impact_xgb = shap_price_impact('vol_xgb', verbose=verbose)
    impact_vp = shap_price_impact('vol_vix_proxy', verbose=verbose)
    impact_xgb.to_csv(cfg.OUTPUT_DIR / 'sensitivity_price_impact_xgb.csv', index=False)
    impact_vp.to_csv(cfg.OUTPUT_DIR / 'sensitivity_price_impact_vixproxy.csv', index=False)
    plot_price_impact(impact_xgb, cfg.ASSETS_DIR / 'fig_w7_shap_price_impact_xgb.png')
    plot_price_impact(impact_vp, cfg.ASSETS_DIR / 'fig_w7_shap_price_impact_vixproxy.png')
    out['impact_xgb'] = impact_xgb
    out['impact_vp'] = impact_vp

    # --- Part A2: univariate perturbation curves ------------------------------
    # Curves are priced at the ATM canonical base (S=K=150) where the chooser's
    # vol/feature sensitivity is maximal and the feature->price link is visible.
    if verbose:
        print('\n[Part A2] Univariate price-perturbation curves (ATM base S=K)...')
    base_row, spot, rate, q, vol21 = build_base_state()
    atm_spot = cfg.CHOOSER_PARAMS['K']
    curves = {}
    for feat, (label, step, n_steps) in cfg.PERTURB_GRID.items():
        curves[feat] = perturb_price_curve(feat, base_row, step=step, n_steps=n_steps,
                                           spot_override=atm_spot)
    pd.concat(curves, names=['feature_group', 'k']).reset_index().to_csv(
        cfg.OUTPUT_DIR / 'perturb_curves.csv', index=False)
    plot_perturb_curves(curves, cfg.ASSETS_DIR / 'fig_w7_perturb_curves.png')
    out['base_state'] = (base_row, spot, rate, q, vol21)
    out['curves'] = curves
    if verbose:
        print(f'  base state: S={spot:.2f}, r={rate*100:.2f}%, vol_21d={vol21*100:.2f}%')

    # --- Part B: extreme scenarios --------------------------------------------
    # Reported at the live market base (deep-ITM chooser, intrinsic-dominated)
    # and at the ATM canonical base (option-like, vol/rate-sensitive).
    if verbose:
        print('\n[Part B] Extreme-scenario testing (live base)...')
    scenarios = extreme_scenarios(base_row, spot, rate, q, vol21, verbose=verbose)
    scenarios.to_csv(cfg.OUTPUT_DIR / 'extreme_scenarios_live.csv', index=False)
    plot_scenarios(scenarios, cfg.ASSETS_DIR / 'fig_w7_scenarios_live.png')
    out['scenarios_live'] = scenarios

    if verbose:
        print('\n[Part B2] Extreme-scenario testing (ATM base S=K=150)...')
    scenarios_atm = extreme_scenarios(base_row, atm_spot, rate, q, vol21, verbose=verbose)
    scenarios_atm.to_csv(cfg.OUTPUT_DIR / 'extreme_scenarios_atm.csv', index=False)
    plot_scenarios(scenarios_atm, cfg.ASSETS_DIR / 'fig_w7_scenarios_atm.png')
    out['scenarios_atm'] = scenarios_atm

    sweep = t1_sweep(base_row, atm_spot, rate, q, vol21)
    sweep.to_csv(cfg.OUTPUT_DIR / 't1_sweep.csv', index=False)
    plot_t1_sweep(sweep, cfg.ASSETS_DIR / 'fig_w7_t1_sweep.png')
    out['t1_sweep'] = sweep

    if verbose:
        print('\nArtifacts -> output/sensitivity_*.csv, assets/fig_w7_*.png')
    return out


if __name__ == '__main__':
    run_sensitivity()
