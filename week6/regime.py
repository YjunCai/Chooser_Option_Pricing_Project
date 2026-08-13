"""
Regime-Adaptive Features -- Week 6
==================================
Reuses the Week 4 five-state market-regime classification (BULL / BEAR /
HIGH_VOL / CALM / NORMAL) and turns it into leak-free ML features:

  * one-hot regime dummies (NORMAL dropped as the reference level), appended
    to the Week 5 feature set;
  * a `regime` column for stratified error analysis and per-regime modelling.

Every input to the classifier (SMA_252 of close, vol_21d, sentiment_score) is
backward-looking, so the regime label at row t uses only information known at t.
Note (as in Week 4): the vol_21d quantile thresholds are computed over the full
sample -- a mild compromise kept for consistency with the Week 4 report.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

import w6config
from data_preparation import load_dataset
from targets import add_fwd_vol_targets


def classify_market_regime(df: pd.DataFrame) -> pd.Series:
    """
    Five-state regime label per trading day (mirrors week4/bsm_evaluation.py).

    Regimes (first match wins, NORMAL as the fallback):
      - BULL     : S > SMA_252 and vol_21d <= median(vol_21d)
      - BEAR     : S < SMA_252 and vol_21d >  median(vol_21d)
      - HIGH_VOL : vol_21d > 75th percentile
      - CALM     : vol_21d < 25th percentile and sentiment_score > 0.5
      - NORMAL   : everything else
    """
    sma_252 = df[w6config.SPOT_COL].rolling(252, min_periods=60).mean()
    vol_median = df['vol_21d'].median()
    vol_q75 = df['vol_21d'].quantile(0.75)
    vol_q25 = df['vol_21d'].quantile(0.25)

    conditions = [
        (df[w6config.SPOT_COL] > sma_252) & (df['vol_21d'] <= vol_median),
        (df[w6config.SPOT_COL] < sma_252) & (df['vol_21d'] > vol_median),
        (df['vol_21d'] > vol_q75),
        (df['vol_21d'] < vol_q25) & (df['sentiment_score'] > 0.5),
    ]
    labels = ['BULL', 'BEAR', 'HIGH_VOL', 'CALM']

    regime = pd.Series('NORMAL', index=df.index)
    for cond, label in zip(conditions, labels):
        regime[cond] = label
    return regime


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the `regime` column plus one-hot dummies (reference level NORMAL)
    to the frame. Returns a copy; the dummies are stored in
    w6config.REGIME_DUMMY_FEATURES.
    """
    out = df.copy()
    out['regime'] = classify_market_regime(out)
    dummies = pd.get_dummies(out['regime'], prefix='regime').astype(float)
    for col in w6config.REGIME_DUMMY_FEATURES:
        out[col] = dummies.get(col, 0.0)
    return out


def build_regime_feature_set(features: Optional[List[str]] = None) -> List[str]:
    """Base features + regime one-hot dummies."""
    base = list(features or w6config.BASE_FEATURES)
    return base + list(w6config.REGIME_DUMMY_FEATURES)


def regime_distribution(df: pd.DataFrame) -> pd.Series:
    return df['regime'].value_counts(normalize=True)


if __name__ == '__main__':
    d = add_fwd_vol_targets(load_dataset())
    d = add_regime_features(d)
    print(d['regime'].value_counts().to_string())
    print(f'\ndummy features: {w6config.REGIME_DUMMY_FEATURES}')
    print(d[w6config.REGIME_DUMMY_FEATURES].mean().round(3).to_string())
