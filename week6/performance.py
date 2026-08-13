"""
Performance Comparison & Consolidated Metrics -- Week 6
=======================================================
Builds the quantitative core of the Week 6 comparative-analysis report:

  1. Volatility forecast comparison (test) -- every tuned model vs the
     BSM(vol_21d) persistence baseline, with Week-5-default rows for a
     before/after-tuning view.
  2. Chooser price comparison (test) -- ML volatility inputs vs the forward-vol
     fair price and the static-vol baseline.
  3. Approach-2 end-to-end pricing comparison.
  4. Regime-stratified error decomposition (BULL / BEAR / HIGH_VOL / CALM /
     NORMAL) for the best volatility model.
  5. Before / after hyper-parameter tuning table (Week 5 defaults vs Week 6
     tuned) and the Week 4->5->6 improvement ladder.

All tables are written to output/*.csv for the report.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

import w6config
import hp_search as hp
from evaluation import evaluate_predictions

WEEK5_OUT = w6config.WEEK5_DIR / 'output'


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Before / after tuning comparison
# ═══════════════════════════════════════════════════════════════════════════════

def _read_week5(name):
    path = WEEK5_OUT / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def before_after_vol_table(a1: Dict) -> pd.DataFrame:
    """Volatility MAE before (Week 5 defaults) vs after (Week 6 tuned)."""
    w5 = _read_week5('approach1_vol_comparison_enhanced.csv')
    w5_map = dict(zip(w5['model'], w5['MAE'] * 100))

    rows = []
    for _, r in a1['metrics'].iterrows():
        fam = r['family']
        before = w5_map.get(_before_name(fam))
        rows.append({
            'model': _display_name(fam),
            'week5_default_MAE_%': before if before is not None else np.nan,
            'week6_tuned_MAE_%': r['MAE'] * 100,
            'week6_RMSE_%': r['RMSE'] * 100,
            'week6_R2': r['R2'],
        })
    df = pd.DataFrame(rows)
    # persistence baseline row
    base = w5_map.get('BSM(vol_21d) persistence', 7.32)
    df.loc[len(df)] = ['BSM(vol_21d) persistence', base, base, np.nan, np.nan]
    df['MAE_change_%'] = df['week6_tuned_MAE_%'] - df['week5_default_MAE_%']
    df = df.sort_values('week6_tuned_MAE_%').reset_index(drop=True)
    df.to_csv(w6config.OUTPUT_DIR / 'before_after_vol_tuning.csv', index=False)
    return df


def _before_name(fam: str) -> str:
    return {'rf': 'rf', 'gbdt': 'gbdt', 'xgb': 'xgb',
            'gbdt_anchored': 'gbdt_anchored', 'lstm': 'lstm'}.get(fam, fam)


def _display_name(fam: str) -> str:
    return {'rf': 'RandomForest', 'gbdt': 'GBDT', 'xgb': 'XGBoost',
            'gbdt_anchored': 'GBDT-anchored (vol_ratio)', 'lstm': 'LSTM',
            'vix_proxy': 'XGB-VIX proxy (IV)'}.get(fam, fam)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Chooser pricing comparison (already computed by train_final)
# ═══════════════════════════════════════════════════════════════════════════════

def chooser_price_table(a1: Dict) -> pd.DataFrame:
    df = a1['price_comparison'].copy()
    df.to_csv(w6config.OUTPUT_DIR / 'chooser_price_comparison_w6.csv', index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Approach-2 pricing comparison
# ═══════════════════════════════════════════════════════════════════════════════

def approach2_price_table(a2: Dict) -> pd.DataFrame:
    w5 = _read_week5('approach2_price_comparison.csv')
    w5_map = dict(zip(w5['model'], w5['MAE']))
    cmp = a2['comparison'].copy()
    cmp['week5_default_MAE_$'] = cmp['model'].map(w5_map)
    cmp.to_csv(w6config.OUTPUT_DIR / 'approach2_price_comparison_w6.csv', index=False)
    return cmp


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Regime-stratified error decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def regime_error_analysis(a1: Dict, mod_regime: pd.Series, best_key: str) -> pd.DataFrame:
    """
    Error of the best volatility forecast split by market regime on the test set.
    `mod_regime` is the regime label series aligned to the full frame rows; the
    test rows are the trailing ones.
    """
    te = a1['_te'] if '_te' in a1 else None
    if te is None:
        return pd.DataFrame()
    labels = mod_regime.iloc[te].reset_index(drop=True).values
    y_true = a1['y_true']
    preds = a1['preds']

    rows = []
    for label in w6config.REGIME_LABELS:
        m = labels == label
        if m.sum() < 5:
            continue
        met = evaluate_predictions(y_true[m], preds[best_key][m])
        rows.append({'regime': label, 'n': int(m.sum()),
                     'vol_MAE_%': met['MAE'] * 100, 'vol_R2': met['R2'],
                     'mean_vol_21d_%': np.nan})
    df = pd.DataFrame(rows).sort_values('regime')
    df.to_csv(w6config.OUTPUT_DIR / 'regime_error_analysis.csv', index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Consolidated summary (for the report's headline tables)
# ═══════════════════════════════════════════════════════════════════════════════

def consolidated_summary(a1: Dict, a2: Dict) -> pd.DataFrame:
    rows = []
    for _, r in a1['metrics'].iterrows():
        rows.append({'track': 'A1_vol', 'model': _display_name(r['family']),
                     'MAE': r['MAE'], 'RMSE': r['RMSE'], 'R2': r['R2'],
                     'unit': 'vol_frac'})
    for _, r in a1['price_comparison'].iterrows():
        rows.append({'track': 'A1_chooser', 'model': r['model'],
                     'MAE': r['MAE'], 'RMSE': r['RMSE'], 'R2': r['R2'],
                     'unit': 'usd'})
    for _, r in a2['comparison'].iterrows():
        rows.append({'track': 'A2_price', 'model': r['model'],
                     'MAE': r['MAE'], 'RMSE': r['RMSE'], 'R2': r['R2'],
                     'unit': 'usd'})
    df = pd.DataFrame(rows)
    df.to_csv(w6config.OUTPUT_DIR / 'consolidated_metrics.csv', index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Figures
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparisons(a1: Dict, a2: Dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 110

    # --- vol MAE: week5 default vs week6 tuned ---
    bf = before_after_vol_table(a1).dropna(subset=['week5_default_MAE_%'])
    bf = bf[~bf['model'].isin(['BSM(vol_21d) persistence', 'XGB-VIX proxy (IV)'])]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(bf))
    w = 0.38
    ax.bar(x - w/2, bf['week5_default_MAE_%'], w, label='Week 5 (default)', color='#b0b0b0')
    ax.bar(x + w/2, bf['week6_tuned_MAE_%'], w, label='Week 6 (tuned)', color='#2166ac')
    ax.axhline(7.32, color='#b2182b', ls='--', lw=1.2, label='BSM persistence (7.32%)')
    ax.set_xticks(x); ax.set_xticklabels(bf['model'], rotation=20, fontsize=8)
    ax.set_ylabel('Volatility MAE (%)')
    ax.set_title('Hyper-parameter tuning: volatility MAE (test)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(w6config.ASSETS_DIR / 'fig_w6_tuning_vol_mae.png', bbox_inches='tight')
    plt.close(fig)

    # --- chooser price MAE ---
    pc = a1['price_comparison'].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    cols = ['#b2182b' if 'BSM' in m else '#2166ac' for m in pc['model']]
    ax.bar(range(len(pc)), pc['MAE'], color=cols)
    ax.set_xticks(range(len(pc))); ax.set_xticklabels(pc['model'], rotation=22, fontsize=8)
    ax.set_ylabel('Chooser price MAE ($)')
    ax.set_title('Chooser option pricing MAE (test, vs forward-vol fair price)')
    ax.axhline(1.46, color='grey', ls=':', lw=1, label='Week 5 best ($1.46)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(w6config.ASSETS_DIR / 'fig_w6_chooser_price_mae.png', bbox_inches='tight')
    plt.close(fig)

    # --- approach2 price MAE ---
    cmp2 = a2['comparison'].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    cols = ['#b2182b' if 'BSM' in m else '#2166ac' for m in cmp2['model']]
    ax.bar(range(len(cmp2)), cmp2['MAE'], color=cols)
    ax.set_xticks(range(len(cmp2))); ax.set_xticklabels(cmp2['model'], rotation=20, fontsize=8)
    ax.set_ylabel('Option price MAE ($)')
    ax.set_title('End-to-end pricing MAE (test contract grid)')
    fig.tight_layout()
    fig.savefig(w6config.ASSETS_DIR / 'fig_w6_approach2_price_mae.png', bbox_inches='tight')
    plt.close(fig)


def plot_regime_errors(regime_df: pd.DataFrame):
    """Bar chart of the best volatility model's error by market regime."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    if regime_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(regime_df['regime'], regime_df['vol_MAE_%'],
           color=['#2166ac'] * len(regime_df))
    ax.axhline(7.32, color='#b2182b', ls='--', lw=1.2, label='persistence (7.32%)')
    for i, (_, r) in enumerate(regime_df.iterrows()):
        ax.text(i, r['vol_MAE_%'] + 0.05, f"n={int(r['n'])}", ha='center', fontsize=8)
    ax.set_xlabel('market regime (Week 4 classification)')
    ax.set_ylabel('volatility MAE (%)')
    ax.set_title('Volatility MAE by market regime (test)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(w6config.ASSETS_DIR / 'fig_w6_regime_error.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    pass
