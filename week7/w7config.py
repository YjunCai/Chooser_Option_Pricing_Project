"""
Week 7 Configuration -- Advanced Sensitivity Analysis & Tool Development
========================================================================
Builds the Week 7 deliverable stack on top of the Week 6 trained models:

  1. Extended sensitivity analysis:
       - SHAP-based impact quantification of the "new" features (sentiment,
         VIX-linked) on the *chooser price* (not just the vol forecast);
       - Extreme-scenario testing (50% volatility spike, +2% rate hike,
         combined, VIX shock) under both the BSM(vol_21d) baseline and the
         best ML pricing models.
  2. Pricing tool framework (Streamlit prototype) integrating models/*.pkl
     dual-track pricing (BSM + best ML model).
  3. Real-time data integration: an auto-update module that refreshes the
     market feature dataset on a schedule (GitHub Actions) or on demand.

All week7 modules import this module first; it puts the week6 / week5 / week3
directories at the front of sys.path so `import w6config`, `import config`,
`from data_preparation import ...`, `from chooser_option_pricer import ...`
all resolve to the artifacts we reuse unchanged.
"""

import sys
from pathlib import Path

# ── path wiring: week6 -> week5 -> week3 first ─────────────────────────────────
W7_DIR = Path(__file__).resolve().parent
WEEK6_DIR = W7_DIR.parent / 'week6'
WEEK5_DIR = W7_DIR.parent / 'week5'
WEEK4_DIR = W7_DIR.parent / 'week4'
WEEK3_DIR = W7_DIR.parent / 'week3'
WEEK2_DIR = W7_DIR.parent / 'week2'

# Insertion order matters: each insert(0) lands at the front, so the LAST
# inserted dir wins name resolution. week5 must win `import config` over
# week2/week2/config.py, so week5/week6 are inserted last (front).
for _d in (WEEK2_DIR, WEEK4_DIR, WEEK3_DIR, WEEK5_DIR, WEEK6_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
if str(W7_DIR) not in sys.path:
    sys.path.insert(0, str(W7_DIR))

import w6config as w6   # noqa: E402
import config as w5     # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Re-export paths & constants
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = W7_DIR / 'output'
ASSETS_DIR = W7_DIR / 'assets'
DATA_DIR = W7_DIR / 'data'
for _d in (OUTPUT_DIR, ASSETS_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MODELS_DIR = w6.MODELS_DIR          # reuse the Week 6 trained models/*.pkl

# re-export the most-used week5/week6 constants (so week7 code is terse)
DATE_COL = w5.DATE_COL
SPOT_COL = w5.SPOT_COL
VIX_COL = w5.VIX_COL
RATE_COL = w5.RATE_COL
BASE_FEATURES = list(w5.BASE_FEATURES)
REGIME_DUMMY_FEATURES = list(w6.REGIME_DUMMY_FEATURES)
CHOOSER_PARAMS = dict(w5.CHOOSER_PARAMS)
Q_YIELD = w5.Q_YIELD
TRADING_DAYS_PER_YEAR = w5.TRADING_DAYS_PER_YEAR
RANDOM_SEED = w5.RANDOM_SEED
SNAPSHOT_DATE = w6.SNAPSHOT_DATE
WEEK2_FEATURE_DATASET = w5.FEATURE_DATASET
FWD_VOL_TARGET = w6.FWD_VOL_TARGET

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Which trained models the tool integrates (dual-track pricing)
#
#   * approach-1 vol models  -> predict the BSM volatility input, then price
#                               with the Week-3 Rubinstein chooser formula.
#   * approach-2 price model -> end-to-end supervised price (contract grid).
#
#   "best ML" on the live snapshot (Week 6 report): vol_xgb  (XGBoost)
#   "best ML" on the synthetic test set:             vol_vix_proxy (VIX proxy)
#   approach-2 best:                                 price_pricing_gbdt
# ═══════════════════════════════════════════════════════════════════════════════

VOL_MODEL_ARTIFACTS = {
    'vol_xgb': 'vol_xgb',
    'vol_vix_proxy': 'vol_vix_proxy',
    'vol_gbdt_anchored': 'vol_gbdt_anchored',
    'vol_rf': 'vol_rf',
    'vol_gbdt': 'vol_gbdt',
    'vol_lstm': 'vol_lstm',
}
PRICE_MODEL_ARTIFACTS = {
    'price_pricing_gbdt': 'price_pricing_gbdt',
    'price_pricing_nn': 'price_pricing_nn',
}

# human labels used across the tool / report
MODEL_LABELS = {
    'vol_xgb': 'XGBoost (实盘首选)',
    'vol_vix_proxy': 'VIX-proxy (隐含波动率对齐)',
    'vol_gbdt_anchored': 'GBDT-anchored (波动率比率)',
    'vol_rf': 'RandomForest',
    'vol_gbdt': 'GBDT',
    'vol_lstm': 'LSTM',
    'price_pricing_gbdt': 'GBDT 端到端',
    'price_pricing_nn': 'MLP 端到端',
}

# Default best-model picks (Week 6 conclusion: XGBoost live / VIX-proxy synthetic)
TOOL_LIVE_MODEL = 'vol_xgb'
TOOL_SYNTH_MODEL = 'vol_vix_proxy'
TOOL_BASELINE = 'vol_21d'           # BSM(vol_21d) persistence baseline

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Extreme-scenario definitions (Week 7 task: "极端场景测试")
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    'base':      dict(label='基准 (Base)',       vol_factor=1.0,  rate_shift=0.0,  vix_factor=1.0),
    'vol_spike_50': dict(label='波动率激增 +50%',  vol_factor=1.5, rate_shift=0.0,  vix_factor=1.5),
    'rate_hike_2': dict(label='利率上调 +2%',     vol_factor=1.0, rate_shift=0.02, vix_factor=1.0),
    'combined':  dict(label='波动率 +50% & 利率 +2%', vol_factor=1.5, rate_shift=0.02, vix_factor=1.5),
    'vix_shock': dict(label='VIX 冲击 +50%',      vol_factor=1.0, rate_shift=0.0,  vix_factor=1.5),
}

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Sensitivity-perturbation grid (univariate marginal price impact)
# ═══════════════════════════════════════════════════════════════════════════════

# (feature, human label, step, n_steps) -> value +/- n_steps*step
PERTURB_GRID = {
    'sentiment_score':  ('情感得分 sentiment', 0.10, 3),
    'vix_ratio':        ('VIX/已实现波动率比 vix_ratio', 0.20, 3),
    'vix_change_1d':    ('VIX 日变动 vix_change_1d', 1.0, 3),
    'vol_21d':          ('21 日已实现波动率 vol_21d', 0.02, 3),
    'rate_change_1d_bps': ('利率日变动 rate_change_1d_bps', 2.0, 3),
}

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Report / demo paths
# ═══════════════════════════════════════════════════════════════════════════════

REPORT_TEX = W7_DIR / 'Week_7_实验报告.tex'
REPORT_PDF = W7_DIR / 'Week_7_实验报告.pdf'

# Real-time data module
RAW_DATA_SOURCES = {
    'jpm': {'ticker': 'JPM',  'col': 'j_Close'},
    'vix': {'ticker': '^VIX', 'col': 'v_Close'},
    'irx': {'ticker': '^IRX', 'col': 'r_Close'},
}
UPDATER_STATUS = OUTPUT_DIR / 'updater_status.json'
