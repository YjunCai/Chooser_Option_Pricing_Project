"""
Data Cleaning Module — Week 2

Handles:
  1. Loading raw data from Week 1 output files
  2. Missing value detection & interpolation
  3. Outlier detection via IQR
  4. Cross-source time alignment (outer join on date)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_FILES,
    IQR_MULTIPLIER,
    MAX_MISSING_FRAC,
    INTERPOLATION_METHOD,
)

logger = logging.getLogger(__name__)


def load_raw_data() -> dict[str, pd.DataFrame]:
    """Load all raw data files from Week 1 output."""
    raw = {}
    for name, path in DATA_FILES.items():
        if not path.exists():
            logger.warning("File not found, skipping: %s", path)
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        logger.info("Loaded %s: %d rows × %d cols", name, len(df), len(df.columns))
        raw[name] = df
    return raw


def detect_missing(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Log missing value summary for a dataframe."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        logger.info("Missing values in %s:\n%s", name, missing.to_string())
    else:
        logger.info("No missing values in %s", name)
    return missing


def interpolate_missing(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Interpolate missing values in numeric columns using time-weighted method."""
    before = df[numeric_cols].isnull().sum().sum()
    if before == 0:
        return df

    df = df.copy()
    # Ensure index is datetime for time-based interpolation
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("date")

    df[numeric_cols] = df[numeric_cols].interpolate(method=INTERPOLATION_METHOD)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()  # fill any remaining gaps

    after = df[numeric_cols].isnull().sum().sum()
    logger.info("Interpolated %d → %d missing values", before, after)

    if "date" in df.columns or df.index.name == "date":
        if df.index.name == "date":
            df = df.reset_index()
    return df


def detect_outliers_iqr(series: pd.Series, name: str) -> pd.Series:
    """Return boolean mask where values are IQR outliers."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr
    outliers = (series < lower) | (series > upper)
    n = outliers.sum()
    if n > 0:
        logger.info("  %s: %d IQR outliers (bounds: [%.4f, %.4f])", name, n, lower, upper)
    return outliers


def cap_outliers_iqr(series: pd.Series, name: str) -> pd.Series:
    """Cap outliers at IQR bounds (winsorize)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr

    n_before = ((series < lower) | (series > upper)).sum()
    capped = series.clip(lower, upper)
    n_after = ((capped < lower) | (capped > upper)).sum()
    if n_before > 0:
        logger.info("  %s: capped %d outliers (bounds: [%.4f, %.4f])", name, n_before, lower, upper)
    return capped


def align_by_date(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align all dataframes on date via outer join.

    Each dataframe must have a 'date' column (datetime).
    Returns a single wide-form dataframe indexed by date.
    """
    if not data:
        return pd.DataFrame()

    # Start with JPM as anchor
    anchor_name = "jpm" if "jpm" in data else next(iter(data))
    anchor = data[anchor_name][["date"]].drop_duplicates().sort_values("date").copy()
    anchor.rename(columns={"date": "date"}, inplace=True)

    merged = anchor.set_index("date")
    for name, df in data.items():
        if "date" not in df.columns:
            continue
        # Use only date and the value columns (skip ticker metadata)
        suffix = f"_{name}"
        df_use = df.set_index("date")
        # Keep only numeric columns + rename with source prefix
        numeric = df_use.select_dtypes(include=[np.number])
        numeric = numeric.add_suffix(suffix)
        merged = merged.join(numeric, how="outer")

    merged = merged.sort_index()
    logger.info("Aligned dataset: %d rows × %d cols", len(merged), len(merged.columns))
    return merged.reset_index()


def clean_all(
    raw: dict[str, pd.DataFrame],
    skip_iqr_sources: set[str] | None = None,
    skip_iqr_cols: set[str] | None = None,
) -> pd.DataFrame:
    """Full cleaning pipeline: load → clean → align.

    Args:
        raw: Dictionary of source name → raw DataFrame.
        skip_iqr_sources: Set of source names to skip IQR capping entirely.
        skip_iqr_cols: Set of column name patterns to skip IQR capping.
                       Price columns (open, high, low, close, adjusted_close)
                       are skipped because IQR on trending level data flags
                       legitimate bull-market highs as false positives.
    """
    if skip_iqr_sources is None:
        skip_iqr_sources = set()
    if skip_iqr_cols is None:
        skip_iqr_cols = set()

    # 1. Detect & interpolate missing values per source
    cleaned = {}
    for name, df in raw.items():
        logger.info("--- Cleaning: %s ---", name)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = detect_missing(df, name)
        # Reload and interpolate
        df = pd.read_csv(DATA_FILES[name], parse_dates=["date"])
        df = interpolate_missing(df, numeric_cols)

        # Cap outliers: skip VIX entirely, skip price-level columns
        for col in numeric_cols:
            if col in df.columns:
                should_skip = (
                    name in skip_iqr_sources
                    or col in skip_iqr_cols
                )
                if should_skip:
                    logger.debug("  Skipping IQR capping for %s.%s", name, col)
                else:
                    df[col] = cap_outliers_iqr(df[col], f"{name}.{col}")
        cleaned[name] = df

    # 2. Cross-source time alignment
    aligned = align_by_date(cleaned)
    logger.info("Cleaned & aligned: %d rows, %d cols", *aligned.shape)

    # 3. Post-alignment fill: only fill treasury gaps on dates where JPM/VIX exist
    #    (equity markets open on Columbus Day/Veterans Day but bond market closed).
    #    JPM/VIX columns intentionally left NaN on non-overlapping dates so that
    #    downstream feature engineering can detect and exclude them (following the
    #    Week 1 inner-join methodology for correlation calculations).
    treasury_cols = [c for c in aligned.columns if "treasury" in c.lower()]
    if treasury_cols:
        before = aligned[treasury_cols].isnull().sum().sum()
        for col in treasury_cols:
            aligned[col] = aligned[col].ffill().bfill()
        after = aligned[treasury_cols].isnull().sum().sum()
        filled = before - after
        if filled > 0:
            logger.info("Treasury post-alignment fill: %d values filled on holiday dates", filled)
    elif "date" in aligned.columns:
        # Fallback: fill all numeric gaps (treasury col not found by name)
        numeric_cols = aligned.select_dtypes(include=[np.number]).columns
        before = aligned[numeric_cols].isnull().sum().sum()
        aligned[numeric_cols] = aligned[numeric_cols].ffill().bfill()
        after = aligned[numeric_cols].isnull().sum().sum()
        if before > after:
            logger.info("Post-alignment fill (fallback): %d -> %d", before, after)

    return aligned
