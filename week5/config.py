"""
Global Configuration -- Week 5 ML Framework
===========================================
Central config for the dual-track ML framework:
  - Approach 1: ML volatility prediction (LSTM/RF/GBDT/XGBoost) + BSM pricing
  - Approach 2: End-to-end supervised pricing (Linear/GBDT/NN)

All paths, split ratios, target settings and model hyper-parameters live here.
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Paths
# ═══════════════════════════════════════════════════════════════════════════════

WEEK5_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEEK5_DIR.parent                 # E:\实习交付\week2 (weeks live side-by-side)

# Prior-week artifacts reused by this framework
WEEK2_DIR = PROJECT_ROOT / 'week2'
WEEK3_DIR = PROJECT_ROOT / 'week3'
WEEK4_DIR = PROJECT_ROOT / 'week4'

FEATURE_DATASET = WEEK2_DIR / 'data' / 'feature_dataset.csv'
PRICER_MODULE = WEEK3_DIR / 'chooser_option_pricer.py'

# Outputs of this week
OUTPUT_DIR = WEEK5_DIR / 'output'
ASSETS_DIR = WEEK5_DIR / 'assets'
for _d in (OUTPUT_DIR, ASSETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Feature / data columns (match week2 feature_dataset.csv)
# ═══════════════════════════════════════════════════════════════════════════════

DATE_COL = 'date'
SPOT_COL = 'close_jpm'
VIX_COL = 'close_vix'
RATE_COL = 'value_treasury_3mo'

# Base feature set delivered by week 2 (excluding date / market-state cols)
BASE_FEATURES = [
    'dps_growth_rate',
    'daily_return',
    'vol_5d',
    'vol_21d',
    'vol_63d',
    'high_low_spread',
    'volume_change_1d',
    'sma_ratio_21',
    'vix_change_1d',
    'vix_jpm_corr_21d',
    'vix_jpm_cross_1d',
    'rate_change_1d_bps',
    'rate_momentum_5d_bps',
    'sentiment_score',
    'jpm_vol_ratio',
    'vix_ratio',
]

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Time-series split (70% / 15% / 15%), no look-ahead bias
# ═══════════════════════════════════════════════════════════════════════════════

SPLIT = {
    'train': 0.70,
    'val': 0.15,
    'test': 0.15,
}
# Number of days "embargoed" between train and val (and val and test) so that
# target windows [t, t+h] straddling the boundary do not leak into the next set.
PURGE_GAP_DAYS = 21

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Targets
# ═══════════════════════════════════════════════════════════════════════════════

# Approach 1 volatility target: forward realized volatility horizon (trading days)
VOL_FORWARD_HORIZON = 21
# Approach 1 secondary "implied-vol" target: VIX level (market IV index)
VIX_TARGET_SHIFT = 1          # predict VIX at t+1

# Approach 2 contract grid (trading days)
MONEYNESS_GRID = [0.90, 1.00, 1.10]
TENOR_GRID_DAYS = [30, 60, 120]
OPT_TYPES = ['call', 'put']

# Chooser contract used by the BSM hybrid engine (matches project paper & week4)
CHOOSER_PARAMS = {
    'K': 150.0,       # strike (paper value)
    't1': 0.5,        # choice date (years)
    'T2': 1.0,        # final maturity (years)
}
# Dividend yield estimate (week4 reported q=1.70%); dataset only has dps growth
Q_YIELD = 0.017

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Models
# ═══════════════════════════════════════════════════════════════════════════════

VOL_MODELS = {   # Approach 1 registry
    'rf': dict(
        name='RandomForest',
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    ),
    'gbdt': dict(
        name='GBDT',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=10,
        random_state=42,
    ),
    'xgb': dict(   # used when xgboost is installed
        name='XGBoost',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    ),
    'lstm': dict(
        name='LSTM',
        seq_len=20,           # lookback window (trading days)
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        epochs=30,
        batch_size=32,
        lr=1e-3,
        seed=42,
    ),
}

PRICING_MODELS = {   # Approach 2 registry
    'linear': dict(
        name='LinearRegression',
        fit_intercept=True,
    ),
    'gbdt': dict(
        name='GBDT',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=10,
        random_state=42,
    ),
    'nn': dict(
        name='MLP-NeuralNet',
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=300,
        learning_rate_init=1e-3,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=42,
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Miscellaneous
# ═══════════════════════════════════════════════════════════════════════════════

TRADING_DAYS_PER_YEAR = 252
RANDOM_SEED = 42
VERBOSE = True
