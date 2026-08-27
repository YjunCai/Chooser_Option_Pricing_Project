"""
Unified Configuration -- Final Project
======================================
Single source of truth for the chooser-option pricing tool. Consolidates the
constants that were previously split across the week-by-week configs (week5
config, week6 w6config, week7 w7config, week8 w8config) into one clean module
with no cross-directory sys.path wiring.

Everything the tool / dashboard / sensitivity / data-updater / training need
lives here: data columns, feature list, chooser contract, model artifact
registry, and the Week-6 held-out test metrics used as error margins.
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Paths
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'output'
ASSETS_DIR = PROJECT_ROOT / 'assets'
for _d in (DATA_DIR, MODELS_DIR, OUTPUT_DIR, ASSETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FEATURE_DATASET = DATA_DIR / 'feature_dataset.csv'

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Data columns (match the feature dataset / market frame)
# ═══════════════════════════════════════════════════════════════════════════════

DATE_COL = 'date'
SPOT_COL = 'close_jpm'
VIX_COL = 'close_vix'
RATE_COL = 'value_treasury_3mo'

# 16-dim feature set built by feature engineering (identical at train & inference)
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
# 3. Chooser contract (paper parameters) & market conventions
# ═══════════════════════════════════════════════════════════════════════════════

CHOOSER_PARAMS = {
    'K': 150.0,       # strike (reference-paper value)
    't1': 0.5,        # choice date (years)
    'T2': 1.0,        # final maturity (years)
}
Q_YIELD = 0.017                    # dividend yield (Week-4 estimate q = 1.70%)
TRADING_DAYS_PER_YEAR = 252
RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Targets (forward-looking labels, no look-ahead)
# ═══════════════════════════════════════════════════════════════════════════════

VOL_FORWARD_HORIZON = 21           # forward realized-vol horizon (trading days)
VIX_TARGET_SHIFT = 1               # predict VIX at t+1 (market IV proxy)
FWD_VOL_TARGET = f'fwd_realized_vol_{VOL_FORWARD_HORIZON}d'
VIX_TARGET = 'vix_target'

# Approach-2 contract grid (trading days / moneyness / type)
MONEYNESS_GRID = [0.90, 1.00, 1.10]
TENOR_GRID_DAYS = [30, 60, 120]
OPT_TYPES = ['call', 'put']

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Regime-adaptive features (Week-4 five-state classification)
# ═══════════════════════════════════════════════════════════════════════════════

REGIME_LABELS = ['BULL', 'BEAR', 'HIGH_VOL', 'CALM', 'NORMAL']
REGIME_DUMMY_FEATURES = ['regime_BULL', 'regime_BEAR', 'regime_HIGH_VOL', 'regime_CALM']

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Trained model artifacts (Week-6 output, consumed by the tool)
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

# Best-model picks (Week-6 conclusion): XGBoost live / VIX-proxy synthetic
TOOL_LIVE_MODEL = 'vol_xgb'
TOOL_SYNTH_MODEL = 'vol_vix_proxy'
TOOL_BASELINE = 'vol_21d'            # BSM(vol_21d) persistence baseline

# Output convention of each vol model's raw prediction:
#   'vol'   -> annualized volatility directly (rf / gbdt / xgb / lstm)
#   'vix'   -> VIX level at t+1, convert to sigma = VIX / 100
#   'ratio' -> anchored model predicts sigma_fwd / sigma_21d, so sigma = ratio * vol_21d
VOL_MODEL_CONVENTIONS = {
    'vol_xgb': 'vol',
    'vol_vix_proxy': 'vix',
    'vol_gbdt_anchored': 'ratio',
    'vol_rf': 'vol',
    'vol_gbdt': 'vol',
    'vol_lstm': 'vol',
}

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Week-6 held-out test-set error margins (2024, 208 days)
#    Used by the tool's error-bar display. Units: chooser price in USD.
# ═══════════════════════════════════════════════════════════════════════════════

ERROR_MARGINS = {
    'vol_vix_proxy':     dict(label='vix_proxy',     MAE=1.174456, RMSE=1.693299, R2=0.981482),
    'vol_21d':           dict(label='BSM(vol_21d)',  MAE=1.517406, RMSE=2.171197, R2=0.969554),
    'vol_gbdt_anchored': dict(label='gbdt_anchored', MAE=1.553905, RMSE=2.711794, R2=0.952505),
    'vol_rf':            dict(label='rf',            MAE=1.603219, RMSE=2.918453, R2=0.944990),
    'vol_xgb':           dict(label='xgb',           MAE=1.707077, RMSE=3.323377, R2=0.928667),
    'vol_gbdt':          dict(label='gbdt',          MAE=1.718958, RMSE=2.753733, R2=0.951025),
    'vol_lstm':          dict(label='lstm',          MAE=2.627795, RMSE=4.171924, R2=0.887590),
}

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Live snapshot & dashboards
# ═══════════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = '2026-07-27'         # live-market snapshot used for validation
PRICE_SERIES_SAMPLE = 500            # trend-chart downsampling cap

# ═══════════════════════════════════════════════════════════════════════════════
# 8b. Week-6 performance metrics (test set + live snapshot), used by the
#     dashboard's performance-metrics charts and tables.
# ═══════════════════════════════════════════════════════════════════════════════

# Approach-1 chooser pricing test metrics (2024, 208 days), USD
CHOOSER_TEST_METRICS = [
    dict(model='vix_proxy',       MAE=1.174456, RMSE=1.693299, MAPE=2.259251, R2=0.981482, bias=-0.579981),
    dict(model='BSM(vol_21d)',    MAE=1.517406, RMSE=2.171197, MAPE=2.890400, R2=0.969554, bias=0.071838),
    dict(model='gbdt_anchored',   MAE=1.553905, RMSE=2.711794, MAPE=3.109113, R2=0.952505, bias=0.214844),
    dict(model='rf',              MAE=1.603219, RMSE=2.918453, MAPE=3.180489, R2=0.944990, bias=0.365116),
    dict(model='xgb',             MAE=1.707077, RMSE=3.323377, MAPE=3.546475, R2=0.928667, bias=0.909954),
    dict(model='gbdt',            MAE=1.718958, RMSE=2.753733, MAPE=3.588821, R2=0.951025, bias=0.884184),
    dict(model='lstm',            MAE=2.627795, RMSE=4.171924, MAPE=4.884665, R2=0.887590, bias=1.320113),
]

# Approach-1 volatility-prediction test metrics (2024), annualized fraction
VOL_TEST_METRICS = [
    dict(model='gbdt_anchored',            MAE=0.071600, RMSE=0.092867, MAPE=86.284, R2=-0.110103, bias=0.020969, improve=2.240310),
    dict(model='BSM(vol_21d) persistence', MAE=0.073241, RMSE=0.089459, MAPE=79.335, R2=-0.030121, bias=0.012773, improve=0.0),
    dict(model='vix_proxy',               MAE=0.074575, RMSE=0.092437, MAPE=77.410, R2=-0.099853, bias=0.003512, improve=-1.821495),
    dict(model='gbdt',                    MAE=0.083578, RMSE=0.112655, MAPE=99.333, R2=-0.633577, bias=0.059883, improve=-14.113988),
    dict(model='rf',                      MAE=0.083593, RMSE=0.112498, MAPE=96.615, R2=-0.629032, bias=0.038462, improve=-14.133792),
    dict(model='xgb',                     MAE=0.084169, RMSE=0.120639, MAPE=100.965, R2=-0.873335, bias=0.060484, improve=-14.920947),
    dict(model='historical mean vol',     MAE=0.092407, RMSE=0.123458, MAPE=110.385, R2=-0.961924, bias=0.086447, improve=-26.168863),
    dict(model='lstm',                    MAE=0.109151, RMSE=0.155650, MAPE=109.397, R2=-2.118447, bias=0.053051, improve=-49.029547),
]

# Live-snapshot metrics (2026-07-27, 595 contracts)
LIVE_METRICS = [
    dict(model='xgb',           sigma=24.4,  MAE=0.793,  RMSE=1.097, atm_iv_gap=0.76,  otm_put_bias=-74.7, group_bias=-29.5),
    dict(model='gbdt',          sigma=24.0,  MAE=0.799,  RMSE=1.149, atm_iv_gap=1.16,  otm_put_bias=-75.9, group_bias=-34.0),
    dict(model='rf',            sigma=22.88, MAE=0.897,  RMSE=1.365, atm_iv_gap=2.28,  otm_put_bias=-78.9, group_bias=-45.0),
    dict(model='gbdt_anchored', sigma=22.02, MAE=1.056,  RMSE=1.583, atm_iv_gap=3.14,  otm_put_bias=-81.1, group_bias=-52.6),
    dict(model='vol_21d',       sigma=20.29, MAE=1.421,  RMSE=2.078, atm_iv_gap=4.87,  otm_put_bias=-84.9, group_bias=-65.2),
    dict(model='vol_21d_snapshot', sigma=20.21, MAE=1.438, RMSE=2.101, atm_iv_gap=4.95, otm_put_bias=-85.1, group_bias=-65.7),
    dict(model='vix_proxy',     sigma=18.61, MAE=1.768,  RMSE=2.582, atm_iv_gap=6.55,  otm_put_bias=-88.1, group_bias=-74.5),
]

# Approach-2 end-to-end pricing test metrics (contract grid, 2024), USD
END2END_METRICS = [
    dict(model='BSM(vol_21d) static-vol benchmark', MAE=1.772819, RMSE=2.244794, R2=0.929579),
    dict(model='pricing_gbdt',                      MAE=3.692113, RMSE=4.857797, R2=0.670215),
    dict(model='pricing_nn',                        MAE=4.607888, RMSE=5.919290, R2=0.510344),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Extreme-scenario definitions (Week-7 task: 极端场景测试)
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    'base':       dict(label='基准 (Base)',          vol_factor=1.0, rate_shift=0.0,  vix_factor=1.0),
    'vol_spike_50': dict(label='波动率激增 +50%',      vol_factor=1.5, rate_shift=0.0,  vix_factor=1.5),
    'rate_hike_2': dict(label='利率上调 +2%',         vol_factor=1.0, rate_shift=0.02, vix_factor=1.0),
    'combined':   dict(label='波动率 +50% & 利率 +2%', vol_factor=1.5, rate_shift=0.02, vix_factor=1.5),
    'vix_shock':  dict(label='VIX 冲击 +50%',        vol_factor=1.0, rate_shift=0.0,  vix_factor=1.5),
}

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Sensitivity-perturbation grid (univariate marginal price impact)
# ═══════════════════════════════════════════════════════════════════════════════

PERTURB_GRID = {
    'sentiment_score':    ('情感得分 sentiment', 0.10, 3),
    'vix_ratio':          ('VIX/已实现波动率比 vix_ratio', 0.20, 3),
    'vix_change_1d':      ('VIX 日变动 vix_change_1d', 1.0, 3),
    'vol_21d':            ('21 日已实现波动率 vol_21d', 0.02, 3),
    'rate_change_1d_bps': ('利率日变动 rate_change_1d_bps', 2.0, 3),
}
