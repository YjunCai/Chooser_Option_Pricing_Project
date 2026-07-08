"""
Configuration file for Chooser Option Pricing Model — Week 1 Data Collection.

API keys are loaded from environment variables. Set them before running:
  $env:ALPHA_VANTAGE_API_KEY = "your_key"
  $env:FRED_API_KEY = "your_key"
"""

import os
from datetime import datetime

# =============================================================================
# Project Parameters
# =============================================================================
PROJECT_NAME = "chooser_option_pricing"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# =============================================================================
# Collection Period
# =============================================================================
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# =============================================================================
# Tickers & Symbols
# =============================================================================
YAHOO_TICKERS = {
    "JPM": "JPM",                   # JPMorgan Chase stock
    "VIX": "^VIX",                  # CBOE Volatility Index
    # "SP500": "^GSPC",               # S&P 500 Index
    # "TNX": "^TNX",                  # 10-Year Treasury Yield (symbol)
}

# Alpha Vantage symbols (may differ from Yahoo)
ALPHA_VANTAGE_SYMBOLS = {
    "JPM": "JPM",
    "VIX": "^VIX",
}

# FRED Series IDs
FRED_SERIES = {
    # "10Y_TREASURY": "DGS10",         # 10-Year Treasury Constant Maturity Rate
    "3MO_TREASURY": "DGS3MO",        # 3-Month Treasury Bill
    # "FED_FUNDS": "FEDFUNDS",         # Federal Funds Effective Rate
    "VIX": "VIXCLS",                 # CBOE Volatility Index (FRED version)
    # "CPI": "CPIAUCSL",               # Consumer Price Index (monthly)
}

# =============================================================================
# API Configuration
# =============================================================================
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Rate limiting
ALPHA_VANTAGE_CALLS_PER_MIN = 5
ALPHA_VANTAGE_SLEEP_SEC = 15  # conservative: 60s / 5 calls = 12s, use 15s

# =============================================================================
# Output Settings
# =============================================================================
OUTPUT_FORMAT = "csv"  # "parquet" or "csv"
COMPRESSION = "snappy"

# =============================================================================
# Update tracking
# =============================================================================
LAST_UPDATE_FILE = os.path.join(OUTPUT_DIR, ".last_update")

# =============================================================================
# Ensure output directory exists
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
