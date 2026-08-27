"""
Target Construction
===================
Forward-looking supervised labels for volatility prediction (Approach 1).
All targets are strictly forward-looking relative to the feature row, so no
look-ahead bias enters:

  - fwd_realized_vol[h] : annualized std of JPM returns over [t+1, t+h]
  - vix_target          : VIX level at t+1 (market implied-vol proxy)

Dependencies: numpy, pandas.
"""

from typing import Optional

import numpy as np
import pandas as pd

import config


def forward_realized_vol(
    returns: pd.Series,
    horizon: int = config.VOL_FORWARD_HORIZON,
    annualize: bool = True,
) -> pd.Series:
    """
    Realized volatility of the forward window [t+1, t+h] anchored at row t.
    Rows whose window extends past the data end get NaN.
    """
    fwd = returns.shift(-1)                        # fwd[t] = ret[t+1]
    rv = fwd.rolling(horizon).std().shift(-(horizon - 1))
    if annualize:
        rv = rv * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    rv.name = f'fwd_realized_vol_{horizon}d'
    return rv


def add_fwd_vol_targets(df: pd.DataFrame, horizons: Optional[list] = None) -> pd.DataFrame:
    """Attach forward realized-vol columns (one per horizon) to the frame."""
    out = df.copy()
    horizons = horizons or config.TENOR_GRID_DAYS + [config.VOL_FORWARD_HORIZON]
    for h in sorted(set(horizons)):
        out[f'fwd_realized_vol_{h}d'] = forward_realized_vol(df['daily_return'], h)
    return out


def add_vix_target(df: pd.DataFrame, shift: int = config.VIX_TARGET_SHIFT) -> pd.DataFrame:
    """Target: VIX level `shift` rows ahead (market forward implied vol)."""
    out = df.copy()
    out['vix_target'] = out[config.VIX_COL].shift(-shift)
    return out


def market_state(df: pd.DataFrame) -> pd.DataFrame:
    """Daily (S, r, q, sigma_actual, sigma_21d) used by the BSM hybrid engine to
    price a chooser under any candidate volatility forecast."""
    return pd.DataFrame({
        config.DATE_COL: df[config.DATE_COL],
        'spot': df[config.SPOT_COL],
        'rate': df[config.RATE_COL].values / 100.0,
        'q': config.Q_YIELD,
        'sigma_actual': df[config.FWD_VOL_TARGET],
        'sigma_21d': df['vol_21d'],
    })
