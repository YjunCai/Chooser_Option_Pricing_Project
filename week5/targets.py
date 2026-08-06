"""
Target Construction -- Week 5
=============================
Builds the supervised labels for both tracks. All targets are strictly
forward-looking relative to the feature row, so no look-ahead bias enters.

Approach 1 (volatility prediction):
  - fwd_realized_vol[h] : annualized std of JPM returns over [t+1, t+h]
  - vix_target          : VIX level at t+1  (market implied-vol proxy)

Approach 2 (end-to-end pricing):
  - contract grid labels : for each date t and each contract (moneyness,
    tenor, type), the BSM price obtained using the *future* realized
    volatility over [t+1, t+tenor] -- i.e. the 'fair price' the model must
    learn to predict from info known at t. Benchmark is BSM priced with the
    static vol_21d input.
"""

import sys
from typing import Optional

import numpy as np
import pandas as pd

import config

# Week-3 validated pricer lives in a sibling week directory.
if str(config.WEEK3_DIR) not in sys.path:
    sys.path.insert(0, str(config.WEEK3_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Forward realized volatility
# ═══════════════════════════════════════════════════════════════════════════════

def forward_realized_vol(
    returns: pd.Series,
    horizon: int = config.VOL_FORWARD_HORIZON,
    annualize: bool = True,
) -> pd.Series:
    """
    Realized volatility of the window [t+1, t+h] anchored at row t.

    Implementation: shift returns by -1 so row t holds ret[t+1]; a rolling
    std of length h is then positioned back at row t via shift(-(h-1)).
    Rows whose window extends past the data end get NaN and are dropped by
    the caller.
    """
    fwd = returns.shift(-1)                       # fwd[t] = ret[t+1]
    rv = fwd.rolling(horizon).std().shift(-(horizon - 1))
    if annualize:
        rv = rv * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    rv.name = f'fwd_realized_vol_{horizon}d'
    return rv


def add_fwd_vol_targets(df: pd.DataFrame, horizons=None) -> pd.DataFrame:
    """Attach forward realized-vol columns (one per horizon) to the frame."""
    out = df.copy()
    horizons = horizons or config.TENOR_GRID_DAYS + [config.VOL_FORWARD_HORIZON]
    horizons = sorted(set(horizons))
    for h in horizons:
        col = f'fwd_realized_vol_{h}d'
        out[col] = forward_realized_vol(df['daily_return'], h)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VIX implied-vol proxy target
# ═══════════════════════════════════════════════════════════════════════════════

def add_vix_target(df: pd.DataFrame, shift: int = config.VIX_TARGET_SHIFT) -> pd.DataFrame:
    """Target: VIX level `shift` rows ahead (market forward implied vol)."""
    out = df.copy()
    out['vix_target'] = out[config.VIX_COL].shift(-shift)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Approach-2 contract labels
# ═══════════════════════════════════════════════════════════════════════════════

def build_contract_frame(
    df: pd.DataFrame,
    features: Optional[list] = None,
) -> pd.DataFrame:
    """
    Expand the daily feature frame into a (date x contract) long frame with a
    forward-vol-priced BSM label and a vol_21d baseline price.

    Returns columns:
      date, spot, rate, log_moneyness, tenor_years, is_call, S_K,
      <market features...>, label_price, bsm_vol21_price
    """
    feats = list(features or config.BASE_FEATURES)
    rows = []
    r = df[config.RATE_COL].values / 100.0          # bps pct -> decimal
    S = df[config.SPOT_COL].values
    tenors = sorted(config.TENOR_GRID_DAYS)

    from chooser_option_pricer import bs_call, bs_put

    for i, row in df.iterrows():
        base = {c: row[c] for c in feats}
        base.update({config.DATE_COL: row[config.DATE_COL], 'spot': row[config.SPOT_COL],
                     'rate': r[i]})
        s0 = S[i]
        for m in config.MONEYNESS_GRID:
            K = m * s0
            for tau in tenors:
                sig_col = f'fwd_realized_vol_{tau}d'
                sig_fwd = row.get(sig_col)
                if not np.isfinite(sig_fwd) or sig_fwd <= 0:
                    continue
                T = tau / config.TRADING_DAYS_PER_YEAR
                sig_21 = row.get('vol_21d')
                sig_base = sig_21 if np.isfinite(sig_21) and sig_21 > 0 else sig_fwd
                for typ in config.OPT_TYPES:
                    if typ == 'call':
                        label = bs_call(s0, K, T, r[i], config.Q_YIELD, sig_fwd)
                        base_p = bs_call(s0, K, T, r[i], config.Q_YIELD, sig_base)
                    else:
                        label = bs_put(s0, K, T, r[i], config.Q_YIELD, sig_fwd)
                        base_p = bs_put(s0, K, T, r[i], config.Q_YIELD, sig_base)
                    rows.append({
                        **base,
                        'log_moneyness': float(np.log(m)),
                        'tenor_years': float(T),
                        'is_call': 1.0 if typ == 'call' else 0.0,
                        'label_price': float(label),
                        'bsm_vol21_price': float(base_p),
                    })

    out = pd.DataFrame(rows)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Market state frame for the BSM hybrid engine
# ═══════════════════════════════════════════════════════════════════════════════

def market_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily (S_t, r_t, q, sigma_actual, sigma_21d) used by the hybrid engine to
    price a chooser with any candidate volatility forecast.
    """
    out = pd.DataFrame({
        config.DATE_COL: df[config.DATE_COL],
        'spot': df[config.SPOT_COL],
        'rate': df[config.RATE_COL].values / 100.0,
        'q': config.Q_YIELD,
        'sigma_actual': df[f'fwd_realized_vol_{config.VOL_FORWARD_HORIZON}d'],
        'sigma_21d': df['vol_21d'],
    })
    return out


if __name__ == '__main__':
    from data_preparation import load_dataset
    d = add_fwd_vol_targets(load_dataset())
    d = add_vix_target(d)
    print(d[['date', 'vol_21d', 'fwd_realized_vol_21d', 'vix_target']].head(8).to_string())
    cf = build_contract_frame(d)
    print(f'\ncontract frame: {cf.shape} rows')
    print(cf[['date', 'log_moneyness', 'tenor_years', 'is_call',
              'label_price', 'bsm_vol21_price']].head(5).to_string())
