"""
Feature Engineering Module -- Week 2

Derives traditional and advanced features from cleaned, aligned market data.

Feature Categories:
  - Price-based:    returns, volatility, spread ratios
  - Dividend:       DPS growth rate (from real JPM dividend data, api/web-sourced)
  - Volume:         volume change
  - Cross-asset:    VIX-JPM correlation, rate momentum
  - Sentiment:      VIX-derived sentiment score (0-1), min-max rolling norm

All features required by the project spec:
  Traditional: rolling volatility, daily returns, dividend growth rate
  Advanced:    VIX-JPM correlation, interest rate momentum, sentiment score (0-1)
"""

import logging

import numpy as np
import pandas as pd

import dividend_api
from config import ROLLING_WINDOWS, TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)

# Column name mappings (produced by data_cleaner.align_by_date)
COL_JPM_CLOSE = "close_jpm"
COL_JPM_HIGH = "high_jpm"
COL_JPM_LOW = "low_jpm"
COL_JPM_VOLUME = "volume_jpm"
COL_VIX_CLOSE = "close_vix"
COL_RATE = "value_treasury_3mo"


# =============================================================================
# 1. Price-Based Features (Traditional)
# =============================================================================

def compute_returns(df: pd.DataFrame) -> pd.Series:
    """Close-to-close log return."""
    if COL_JPM_CLOSE not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    ret = np.log(df[COL_JPM_CLOSE] / df[COL_JPM_CLOSE].shift(1))
    ret.name = "daily_return"
    return ret


def compute_rolling_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    """Annualized rolling volatility from daily returns."""
    if "daily_return" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    vol = df["daily_return"].rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    vol.name = f"vol_{window}d"
    return vol


def compute_high_low_spread(df: pd.DataFrame) -> pd.Series:
    """Daily (high - low) / close as a measure of intraday volatility."""
    if not all(c in df.columns for c in [COL_JPM_HIGH, COL_JPM_LOW, COL_JPM_CLOSE]):
        return pd.Series(index=df.index, dtype=float)
    spread = (df[COL_JPM_HIGH] - df[COL_JPM_LOW]) / df[COL_JPM_CLOSE]
    spread.name = "high_low_spread"
    return spread


# =============================================================================
# 2. Volume Features
# =============================================================================

def compute_volume_change(df: pd.DataFrame) -> pd.Series:
    """1-day volume change ratio (log)."""
    if COL_JPM_VOLUME not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    vol_chg = np.log(df[COL_JPM_VOLUME] / df[COL_JPM_VOLUME].shift(1)).replace(
        [np.inf, -np.inf], 0
    )
    vol_chg.name = "volume_change_1d"
    return vol_chg


# =============================================================================
# 3. Cross-Asset / VIX Features (Advanced)
# =============================================================================

def compute_vix_change(df: pd.DataFrame) -> pd.Series:
    """VIX daily absolute change (points)."""
    if COL_VIX_CLOSE not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    chg = df[COL_VIX_CLOSE].diff()
    chg.name = "vix_change_1d"
    return chg


def compute_vix_jpm_corr(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Rolling correlation between JPM daily return and VIX change.

    Uses min_periods to handle edge cases where one series has temporarily
    constant values (e.g., IQR-capped prices at historic highs). This follows
    the Week 1 methodology of computing Pearson correlation between the two
    series, adapted to a rolling window context.

    Note: The 16 leading NaN rows (window warmup) are expected; they are
    removed by the dropna step in build_feature_set().
    """
    needed = {"daily_return", "vix_change_1d"}
    if not needed.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)
    # Allow correlation with 75% of window size to avoid NaN during
    # temporary flat-price periods (e.g. IQR-capped prices at trend highs)
    min_p = max(int(window * 0.75), 2)
    corr = (
        df["daily_return"]
        .rolling(window, min_periods=min_p)
        .corr(df["vix_change_1d"])
    )
    # Replace any remaining extreme/undefined values with NaN
    corr = corr.replace([np.inf, -np.inf], np.nan)
    corr.name = f"vix_jpm_corr_{window}d"
    return corr


def compute_vix_jpm_cross_1d(df: pd.DataFrame) -> pd.Series:
    """1-day VIX-JPM co-movement indicator (cross-moment).

    Since a 1-day rolling correlation is mathematically undefined
    (requires >= 2 data points), this computes the product of
    JPM return and VIX change as a daily co-movement measure:
        cross_1d = -daily_return * vix_change_1d

    Positive values = VIX up / JPM down (risk-off / flight-to-safety).
    Negative values = VIX down / JPM up (risk-on).
    Zero = no significant co-movement.

    This captures the well-known equity-volatility negative relationship.
    """
    needed = {"daily_return", "vix_change_1d"}
    if not needed.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)
    cross = -df["daily_return"] * df["vix_change_1d"]
    cross.name = "vix_jpm_cross_1d"
    return cross


# =============================================================================
# 4. Interest Rate Features (Advanced)
# =============================================================================

def compute_rate_change(df: pd.DataFrame) -> pd.Series:
    """3Mo Treasury daily change in basis points."""
    if COL_RATE not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    chg = df[COL_RATE].diff() * 100
    chg.name = "rate_change_1d_bps"
    return chg


def compute_rate_momentum(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """N-day change in 3Mo Treasury rate (interest rate momentum)."""
    if COL_RATE not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    mom = df[COL_RATE].diff(window) * 100
    mom.name = f"rate_momentum_{window}d_bps"
    return mom


# =============================================================================
# 5. Sentiment Score -- Advanced (0-1)
#    Formula: Score_t = 1 - (VIX_t - min(VIX_t-w:t)) / (max(VIX_t-w:t) - min(VIX_t-w:t))
#    Reference: Week 1 experiment report eq. (Sentiment Score)
# =============================================================================

def compute_sentiment_score(df: pd.DataFrame, window: int = 252) -> pd.Series:
    """Market sentiment score in [0,1] using VIX min-max normalization.

    Formula (from Week 1 report):
        Score_t = 1 - (VIX_t - min(VIX_{t-w:t})) / (max(VIX_{t-w:t}) - min(VIX_{t-w:t}))

    where w = 252 trading days (~1 year).
    High score = optimism (low VIX), low score = fear (high VIX).
    """
    if COL_VIX_CLOSE not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    vix = df[COL_VIX_CLOSE]

    # Rolling min and max
    roll_min = vix.rolling(window, min_periods=1).min()
    roll_max = vix.rolling(window, min_periods=1).max()

    # Handle edge case: if min == max (constant VIX), score = 0.5 (neutral)
    denominator = roll_max - roll_min
    score = 1.0 - (vix - roll_min) / denominator.replace(0, np.nan)
    score = score.fillna(0.5)

    # Ensure strict [0,1] bounds
    score = score.clip(0, 1)
    score.name = "sentiment_score"

    logger.info(
        "Sentiment score: range=[%.4f, %.4f], window=%d",
        score.min(), score.max(), window,
    )
    return score


# =============================================================================
# 6. Combined / Regime Indicators
# =============================================================================

def compute_sma_ratio(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Close / Simple Moving Average ratio -- mean reversion indicator."""
    if COL_JPM_CLOSE not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    sma = df[COL_JPM_CLOSE].rolling(window).mean()
    ratio = df[COL_JPM_CLOSE] / sma
    ratio.name = f"sma_ratio_{window}"
    return ratio


def compute_vol_ratio(df: pd.DataFrame) -> pd.Series:
    """Short-term vol / medium-term vol -- regime indicator (>1 = rising vol)."""
    needed = {"vol_5d", "vol_21d"}
    if not needed.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)
    ratio = df["vol_5d"] / df["vol_21d"]
    ratio.name = "jpm_vol_ratio"
    return ratio


def compute_vix_ratio(df: pd.DataFrame) -> pd.Series:
    """VIX / (21d ATM vol proxy) -- VIX premium indicator."""
    needed = {COL_VIX_CLOSE, "vol_21d"}
    if not needed.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)
    ratio = df[COL_VIX_CLOSE] / (df["vol_21d"] * 100)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio.name = "vix_ratio"
    return ratio


# =============================================================================
# Orchestration
# =============================================================================

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all features from cleaned, aligned dataframe.

    Note: dividend features (dps_growth_rate) are expected to be merged
    by build_feature_set() before this function is called.
    """
    logger.info("Computing features...")
    features = pd.DataFrame(index=df.index)

    # --- Price-based (Traditional) ---
    features["daily_return"] = compute_returns(df)

    for label, w in ROLLING_WINDOWS.items():
        features[f"vol_{w}d"] = compute_rolling_volatility(
            df.assign(daily_return=features["daily_return"]), w
        )

    features["high_low_spread"] = compute_high_low_spread(df)

    # --- Volume ---
    features["volume_change_1d"] = compute_volume_change(df)

    # SMA ratio (mean reversion)
    features["sma_ratio_21"] = compute_sma_ratio(df)

    # --- Cross-asset (Advanced) ---
    features["vix_change_1d"] = compute_vix_change(df)

    features["vix_jpm_corr_21d"] = compute_vix_jpm_corr(
        df.assign(daily_return=features["daily_return"],
                  vix_change_1d=features["vix_change_1d"])
    )

    features["vix_jpm_cross_1d"] = compute_vix_jpm_cross_1d(
        df.assign(daily_return=features["daily_return"],
                  vix_change_1d=features["vix_change_1d"])
    )

    # --- Interest rate (Advanced) ---
    features["rate_change_1d_bps"] = compute_rate_change(df)
    features["rate_momentum_5d_bps"] = compute_rate_momentum(df)

    # --- Sentiment (Advanced: 0-1) ---
    features["sentiment_score"] = compute_sentiment_score(df)

    # --- Combined indicators ---
    features["jpm_vol_ratio"] = compute_vol_ratio(
        features.assign(vol_5d=features["vol_5d"], vol_21d=features["vol_21d"])
    )
    features["vix_ratio"] = compute_vix_ratio(
        df.assign(vol_21d=features["vol_21d"])
    )

    n_feats = len([c for c in features.columns if not c.startswith("_")])
    logger.info("Generated %d features", n_feats)
    return features


def build_feature_set(aligned: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline: merge dividend -> derive -> join -> clean.

    Follows Week 1 inner-join methodology: only compute features on dates where
    all core sources (JPM, VIX) have actual data. Forward-filled-only dates at
    edges are dropped before feature computation to avoid constant-value windows
    causing undefined rolling correlations.

    Dividend data from JPM official records (via dividend_api) is merged first,
    then all technical features are computed.
    """
    anchor_cols = [COL_JPM_CLOSE, COL_VIX_CLOSE, COL_RATE]
    keep = [c for c in anchor_cols if c in aligned.columns]

    # Inner join: keep only rows where ALL core sources have data (Week 1 methodology)
    clean = aligned.dropna(subset=[c for c in keep if c != COL_RATE
                           and c in aligned.columns]).copy()
    n_dropped = len(aligned) - len(clean)
    if n_dropped > 0:
        logger.info("Dropped %d rows without full JPM/VIX data (inner join)", n_dropped)
    logger.info("Working dataset: %d rows", len(clean))
    result = clean[["date"] + keep].copy()

    # Step 1: Merge real dividend data (DPS growth rate from JPM official records)
    div_features = dividend_api.build_daily_dividend_features(result["date"])
    result = pd.concat([result, div_features], axis=1)
    # Ensure no remaining NaN in dps_growth_rate (edge dates before first dividend)
    if "dps_growth_rate" in result.columns:
        result["dps_growth_rate"] = result["dps_growth_rate"].fillna(0)

    # Step 2: Compute all technical features (using the inner-joined subset)
    tech_features = compute_features(clean)
    result = pd.concat([result, tech_features], axis=1)

    # Step 3: Drop leading NaN rows (rolling window warmup)
    core_feats = ["daily_return", "vol_5d"]
    valid_core = [c for c in core_feats if c in result.columns]
    before = len(result)
    result.dropna(subset=valid_core, inplace=True)
    if len(result) < before:
        logger.info("Dropped %d leading NaN rows (rolling window warmup)", before - len(result))

    result.reset_index(drop=True, inplace=True)
    return result
