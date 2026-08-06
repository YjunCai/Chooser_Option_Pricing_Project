"""
BSM Hybrid Pricing Engine -- Week 5
===================================
Converts any volatility forecast (historical, ML-predicted, or forward-looking
"fair" vol) into a Chooser Option price via the validated Week-3 Rubinstein
formula. Used by Approach 1 to compare BSM(ML-sigma) against the BSM(vol_21d)
baseline.

The engine is vectorized over dates: `simple_chooser` already broadcasts over
(S, sigma) arrays.
"""

from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path

import config
from data_preparation import load_dataset
from targets import market_state

# Reuse the Week-3 validated pricer (same approach as week4/bsm_evaluation.py)
if str(config.WEEK3_DIR) not in sys.path:
    sys.path.insert(0, str(config.WEEK3_DIR))

from chooser_option_pricer import bs_call, bs_put, simple_chooser


def price_chooser_with_vol(
    spot: float,
    rate: float,
    q: float,
    vol,
    K: float = config.CHOOSER_PARAMS['K'],
    t1: float = config.CHOOSER_PARAMS['t1'],
    T2: float = config.CHOOSER_PARAMS['T2'],
):
    """Price a simple chooser with a scalar/array volatility input."""
    return simple_chooser(spot, K, t1, T2, rate, q, vol)


def chooser_price_series(
    market_df: pd.DataFrame,
    vol_series: pd.Series,
    K: float = config.CHOOSER_PARAMS['K'],
    t1: float = config.CHOOSER_PARAMS['t1'],
    T2: float = config.CHOOSER_PARAMS['T2'],
) -> pd.Series:
    """
    Price a chooser at every row of `market_df` using `vol_series` (aligned by
    position, same order). Returns a Series of prices.
    """
    spot = market_df['spot'].values
    rate = market_df['rate'].values
    q = market_df['q'].values
    vol = np.asarray(vol_series, dtype=float)

    prices = simple_chooser(spot, K, t1, T2, rate, q, vol)
    return pd.Series(prices, index=market_df.index, name='chooser_price')


def build_market_frame(feature_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Load the daily market-state frame used for chooser pricing evaluation."""
    df = feature_df if feature_df is not None else load_dataset()
    from targets import add_fwd_vol_targets
    df = add_fwd_vol_targets(df)
    return market_state(df)
