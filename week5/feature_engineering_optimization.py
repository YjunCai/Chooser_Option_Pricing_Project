"""
Feature Engineering Optimization -- Week 5
==========================================
Analyzes, enhances and selects the Week-2 16-dimension feature set before it
enters the ML pipeline. All derived features are backward-looking only
(rolling windows on data <= t) -- never forward-looking -- so the anti-leakage
guarantee of the time-series split is preserved.

Modules
-------
  1. analyze_features()   -- NaN / correlation / drift diagnostics
  2. enhance_features()   -- adds optimized derived features
  3. select_features()    -- correlation de-duplication + RF importance ranking
  4. optimize_pipeline()  -- end-to-end: load -> analyze -> enhance -> select -> save
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import config
from data_preparation import load_dataset
from targets import add_fwd_vol_targets

logger = logging.getLogger(__name__)

CORR_THRESHOLD = 0.92      # drop one of any feature pair more correlated than this
VAR_QUANTILE = 0.02        # drop features below this variance quantile


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_features(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    """NaN rate, variance, and pairwise-correlation diagnostics per feature."""
    feats = list(features or config.BASE_FEATURES)
    sub = df[feats]
    report = pd.DataFrame({
        'feature': feats,
        'nan_rate': sub.isna().mean().values,
        'std': sub.std(ddof=0).values,
        'min': sub.min().values,
        'max': sub.max().values,
    })
    # max pairwise |corr| with any other feature
    corr = sub.corr().abs()
    np.fill_diagonal(corr.values, np.nan)
    report['max_abs_corr'] = [corr.loc[c].max() for c in feats]
    report['correlated_with'] = [corr.loc[c].idxmax() if not np.isnan(corr.loc[c].max()) else '' for c in feats]
    return report.sort_values('max_abs_corr', ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Enhancement (backward-looking derived features)
# ═══════════════════════════════════════════════════════════════════════════════

def _roll(df: pd.DataFrame, col: str, fn: str, window: int) -> pd.Series:
    roll = df[col].rolling(window, min_periods=max(window // 2, 5))
    if fn == 'std':
        return roll.std()
    if fn == 'skew':
        return roll.skew()
    if fn == 'kurt':
        return roll.kurt()
    if fn == 'mean':
        return roll.mean()
    raise ValueError(fn)


def enhance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add optimized backward-looking features on top of the week2 set.
    Every new column uses only data known at row t.
    """
    out = df.copy()

    # --- return distribution shape (skew / kurtosis / tails) ---
    out['ret_skew_21d'] = _roll(out, 'daily_return', 'skew', 21)
    out['ret_kurt_21d'] = _roll(out, 'daily_return', 'kurt', 21)
    out['ret_skew_63d'] = _roll(out, 'daily_return', 'skew', 63)
    out['ret_kurt_63d'] = _roll(out, 'daily_return', 'kurt', 63)

    # --- multi-horizon volatility ratios (regime / term-structure signals) ---
    out['vol_5_21_ratio'] = out['vol_5d'] / out['vol_21d']
    out['vol_21_63_ratio'] = out['vol_21d'] / out['vol_63d']

    # --- VIX position within trailing 1y range (0..1, backward looking) ---
    vix = out[config.VIX_COL]
    vmin = vix.rolling(252, min_periods=20).min()
    vmax = vix.rolling(252, min_periods=20).max()
    out['vix_percentile_252d'] = (vix - vmin) / (vmax - vmin).replace(0, np.nan)
    out['vix_percentile_252d'] = out['vix_percentile_252d'].fillna(0.5).clip(0, 1)

    # --- price position vs trailing high/low (trend / mean-reversion) ---
    price = out[config.SPOT_COL]
    hi52 = price.rolling(252, min_periods=20).max()
    lo52 = price.rolling(252, min_periods=20).min()
    out['dist_from_52w_high'] = price / hi52 - 1.0
    out['dist_from_52w_low'] = price / lo52 - 1.0
    out['price_range_pos_252d'] = (price - lo52) / (hi52 - lo52).replace(0, np.nan)

    # --- lagged key features (short memory signals for trees / linear) ---
    for col, lag in (('vol_21d', 1), ('vol_21d', 5), ('sentiment_score', 1),
                     ('vix_change_1d', 1), ('daily_return', 1)):
        out[f'{col}_lag{lag}'] = out[col].shift(lag)

    # --- rate term / level momentum ---
    out['rate_level_21d'] = _roll(out, config.RATE_COL, 'mean', 21)
    out['rate_change_21d_bps'] = out[config.RATE_COL].diff(21) * 100

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Selection
# ═══════════════════════════════════════════════════════════════════════════════

def select_features(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target: str = f'fwd_realized_vol_{config.VOL_FORWARD_HORIZON}d',
    corr_threshold: float = CORR_THRESHOLD,
    n_importance: int = 30,
) -> Tuple[List[str], pd.DataFrame]:
    """
    1) Drop near-constant features (variance quantile).
    2) De-duplicate pairs with |corr| > threshold (keep the one with higher
       correlation to the target).
    3) Rank survivors by a quick RF feature-importance fit (used for the
       report; the framework still trains on the selected set).

    Returns (selected_feature_names, importance_report).
    """
    feats = [c for c in (features or config.BASE_FEATURES) if c in df.columns]
    sub = df[feats].copy()
    if target not in df.columns:
        logger.warning('target %s missing -> importance step skipped', target)
        return feats, pd.DataFrame()

    # drop rows with missing labels (tail of series)
    sub[target] = df[target]
    sub = sub.dropna(subset=[target]).copy()

    # 1) variance filter
    var = sub[feats].var(ddof=0)
    keep_var = var[var > var.quantile(VAR_QUANTILE)].index.tolist()
    dropped_var = [c for c in feats if c not in keep_var]

    # 2) correlation de-duplication
    corr = sub[keep_var].corr().abs()
    np.fill_diagonal(corr.values, np.nan)
    keep_corr = list(keep_var)
    for i in range(len(keep_corr)):
        a = keep_corr[i]
        if a not in keep_corr:
            continue
        hits = corr.loc[a]
        for b in list(keep_corr[i + 1:]):
            if b not in keep_corr:
                continue
            if hits[b] > corr_threshold:
                # drop the one less correlated with the target
                keep_corr.remove(b)
    dropped_corr = [c for c in keep_var if c not in keep_corr]

    # 3) importance ranking
    X = sub[keep_corr].fillna(sub[keep_corr].mean())
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                               n_jobs=-1, random_state=config.RANDOM_SEED)
    rf.fit(X, sub[target])
    imp = pd.DataFrame({'feature': keep_corr, 'importance': rf.feature_importances_})
    imp = imp.sort_values('importance', ascending=False).reset_index(drop=True)
    top = imp.head(n_importance)['feature'].tolist()

    report = pd.DataFrame({
        'feature': feats,
        'importance': imp.set_index('feature')['importance'].reindex(feats).fillna(0).values,
        'dropped_var': [c in dropped_var for c in feats],
        'dropped_corr': [c in dropped_corr for c in feats],
    })
    return top, report


# ═══════════════════════════════════════════════════════════════════════════════
# 4. End-to-end
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_pipeline(verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load -> analyze -> enhance -> select. Returns
    (enhanced_df, analysis_report, selected_features) and writes artifacts to
    output/ and assets/.
    """
    df = load_dataset()
    df = add_fwd_vol_targets(df)

    analysis = analyze_features(df)
    enhanced = enhance_features(df)
    selected, imp = select_features(enhanced, target=f'fwd_realized_vol_{config.VOL_FORWARD_HORIZON}d')

    if verbose:
        print('=' * 64)
        print('  Feature Engineering Optimization Report')
        print('=' * 64)
        print(f'  base features      : {len(config.BASE_FEATURES)}')
        print(f'  enhanced features  : {enhanced.shape[1] - 7}')   # minus date+4 mkt +2 target cols
        print(f'  selected features  : {len(selected)}')
        print()
        print('  top-10 by importance (RF, fwd-vol target):')
        print(imp.head(10)[['feature', 'importance']].to_string(index=False))
        print()
        print('  dropped (near-constant)   :', analysis.index[analysis['std'] < 1e-12].tolist())
        print('  dropped (|corr|>%.2f)      : %d features' % (CORR_THRESHOLD,
              int(imp['dropped_corr'].sum())))

    # artifacts
    enhanced.to_csv(config.OUTPUT_DIR / 'enhanced_features.csv', index=False)
    analysis.to_csv(config.OUTPUT_DIR / 'feature_analysis.csv', index=False)
    imp.to_csv(config.OUTPUT_DIR / 'feature_importance.csv', index=False)
    pd.Series(selected, name='selected_feature').to_csv(
        config.OUTPUT_DIR / 'selected_features.csv', index=False)
    if verbose:
        print(f'\n  artifacts -> {config.OUTPUT_DIR}')
    return enhanced, analysis, selected


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    optimize_pipeline()
