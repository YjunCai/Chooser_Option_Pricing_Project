"""
Configuration file for Week 2 — Data Preprocessing & Feature Engineering.

Controls input/output paths, cleaning parameters, and feature flags.
"""

import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
# Project root is two levels up from this config file:
#   config.py → week2/ → root (e:\实习交付\week2\)
#   root/week1/data/  contains raw data
_WEEK2_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = _WEEK2_DIR.parent 

WEEK1_DATA_DIR = _PROJECT_ROOT / "week1" / "data"
WEEK2_OUTPUT_DIR = _WEEK2_DIR / "output"

# =============================================================================
# Data sources (Week 1 output files)
# =============================================================================
DATA_FILES = {
    "jpm": WEEK1_DATA_DIR / "yahoo_jpm.csv",
    "vix": WEEK1_DATA_DIR / "yahoo_vix.csv",
    "treasury_3mo": WEEK1_DATA_DIR / "fred_3mo_treasury.csv",
}

# =============================================================================
# Cleaning Parameters
# =============================================================================
# IQR outlier multiplier
IQR_MULTIPLIER = 1.5

# Maximum fraction of missing values allowed per column before dropping
MAX_MISSING_FRAC = 0.3

# Interpolation method for missing values
INTERPOLATION_METHOD = "time"  # 'time' = time-weighted linear interpolation

# =============================================================================
# Feature Engineering Parameters
# =============================================================================
# Rolling windows (trading days ≈ 21 per month, 63 per quarter, 252 per year)
ROLLING_WINDOWS = {
    "short": 5,    # 1 week
    "medium": 21,  # 1 month
    "long": 63,    # 1 quarter
}

# Annualisation factor for volatility
TRADING_DAYS_PER_YEAR = 252

# Correlation rolling window
CORRELATION_WINDOW = 21

# =============================================================================
# Output Settings
# =============================================================================
OUTPUT_FORMAT = "csv"  # "csv" or "parquet"
