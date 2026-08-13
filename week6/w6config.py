"""
Week 6 Configuration -- Hyperparameter Optimization, Final Training & SHAP
==========================================================================
Extends the Week 5 dual-track ML framework config with everything Week 6 needs:

  1. Hyper-parameter search spaces for grid / random search (regularization-
     heavy, to address the Week 5 over-fitting diagnosis: train R2 ~ 0.97,
     test R2 < 0 on the noisy realized-vol target).
  2. Purged time-series cross-validation settings (expanding window + purge).
  3. Regime-adaptive features (Week 4 five-state classification: BULL / BEAR /
     HIGH_VOL / CALM / NORMAL).
  4. IV-target switch: real JPM single-name option IV history is unavailable
     (only a one-day snapshot), so the market-implied-vol proxy stays VIX
     (predicting VIX at t+1 and feeding VIX/100 into the BSM engine).

All week6 modules import this module first; it puts the week5 directory at the
front of sys.path so that `import config`, `data_preparation`, `models.*`, ...
all resolve to the Week 5 artifacts we reuse unchanged.
"""

import sys
from pathlib import Path

# ── path wiring: week5 first, so `import config` inside week5 modules hits week5 ──
WEEK6_DIR = Path(__file__).resolve().parent
WEEK5_DIR = WEEK6_DIR.parent / 'week5'
if str(WEEK5_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK5_DIR))
if str(WEEK6_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK6_DIR))

import config as w5  # noqa: E402  (week5 config)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Re-export the week5 constants so week6 code can refer to them via w6config
# ═══════════════════════════════════════════════════════════════════════════════

WEEK5_DIR = w5.WEEK5_DIR
PROJECT_ROOT = w5.PROJECT_ROOT
WEEK2_DIR = w5.WEEK2_DIR
WEEK3_DIR = w5.WEEK3_DIR
WEEK4_DIR = w5.WEEK4_DIR
FEATURE_DATASET = w5.FEATURE_DATASET

OUTPUT_DIR = WEEK6_DIR / 'output'
ASSETS_DIR = WEEK6_DIR / 'assets'
MODELS_DIR = WEEK6_DIR / 'models'
for _d in (OUTPUT_DIR, ASSETS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATE_COL = w5.DATE_COL
SPOT_COL = w5.SPOT_COL
VIX_COL = w5.VIX_COL
RATE_COL = w5.RATE_COL
BASE_FEATURES = list(w5.BASE_FEATURES)
SPLIT = dict(w5.SPLIT)
PURGE_GAP_DAYS = w5.PURGE_GAP_DAYS
VOL_FORWARD_HORIZON = w5.VOL_FORWARD_HORIZON
VIX_TARGET_SHIFT = w5.VIX_TARGET_SHIFT
MONEYNESS_GRID = list(w5.MONEYNESS_GRID)
TENOR_GRID_DAYS = list(w5.TENOR_GRID_DAYS)
OPT_TYPES = list(w5.OPT_TYPES)
CHOOSER_PARAMS = dict(w5.CHOOSER_PARAMS)
Q_YIELD = w5.Q_YIELD
TRADING_DAYS_PER_YEAR = w5.TRADING_DAYS_PER_YEAR
RANDOM_SEED = w5.RANDOM_SEED

# The forward-vol target column name (identical to week5)
FWD_VOL_TARGET = f'fwd_realized_vol_{VOL_FORWARD_HORIZON}d'

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Purged time-series CV (used by hyper-parameter search)
# ═══════════════════════════════════════════════════════════════════════════════

CV = {
    'n_splits': 4,            # expanding-window folds inside the searchable region
    'val_frac': 0.12,         # per-fold validation block (fraction of search region)
    'purge_gap': PURGE_GAP_DAYS,   # embargo rows between train and val
    'min_train_frac': 0.35,   # first fold's training block must be at least this
    'scoring': 'neg_mean_absolute_error',
}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Regime-adaptive features (Week 4 five-state classification)
# ═══════════════════════════════════════════════════════════════════════════════

REGIME_LABELS = ['BULL', 'BEAR', 'HIGH_VOL', 'CALM', 'NORMAL']
# one-hot columns added to the feature set (NORMAL dropped as reference level)
REGIME_DUMMY_FEATURES = ['regime_BULL', 'regime_BEAR', 'regime_HIGH_VOL', 'regime_CALM']

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Hyper-parameter search spaces
#
# Regularization emphasis per the Week 5 over-fitting diagnosis:
#   * cap tree depth, raise min_samples_leaf / min_child_weight
#   * add subsample / colsample_bytree / L1/L2 (xgb)
#   * small learning rate with more estimators
# ═══════════════════════════════════════════════════════════════════════════════

SEARCH_SPACES = {
    'rf': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, None],
        'min_samples_leaf': [5, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': [0.3, 0.5, 0.7, 'sqrt'],
    },
    'gbdt': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [2, 3, 4, 5],
        'min_samples_leaf': [10, 20, 30],
        'subsample': [0.7, 0.85, 1.0],
        'max_features': [0.5, 0.7, 1.0],
    },
    'xgb': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [2, 3, 4, 5],
        'subsample': [0.7, 0.85, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0.0, 0.1, 1.0],
        'reg_lambda': [1.0, 5.0],
    },
    'pricing_gbdt': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.02, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'min_samples_leaf': [5, 10, 20],
        'subsample': [0.7, 0.85, 1.0],
    },
    'pricing_nn': {
        'hidden_layer_sizes': [(32,), (64,), (64, 32), (128, 64)],
        'alpha': [1e-4, 1e-3, 1e-2],
        'learning_rate_init': [1e-3, 5e-3, 1e-2],
        'max_iter': [300, 500],
    },
}

# How many random draws per model family (random search budget)
SEARCH_ITERS = {
    'rf': 20,
    'gbdt': 25,
    'xgb': 25,
    'pricing_gbdt': 20,
    'pricing_nn': 10,
}

# LSTM search is intentionally tiny (CPU training is slow on 1.6k rows)
LSTM_SEARCH = {
    'hidden_size': [24, 32],
    'dropout': [0.0, 0.1, 0.2],
    'lr': [5e-4, 1e-3],
    'n_iter': 4,
    'epochs': 25,
}

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Final-model training
# ═══════════════════════════════════════════════════════════════════════════════

# Fit the final model on train+val (searchable region) after hyper-parameter
# selection on the inner purged CV, then evaluate ONCE on the held-out test set.
FINAL_FIT_ON = 'train_val'        # 'train_val' | 'train'

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Live snapshot (identical to week5/week4)
# ═══════════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = '2026-07-27'
WEEK6_TARGETS = {
    'live_MAE_$': 1.00,      # 实盘 MAE target < $1.00
    'atm_iv_premium_%': 2.0, # ATM IV premium target < 2%
    'otm_put_bias_%': -30.0, # OTM put bias target > -30%
}
