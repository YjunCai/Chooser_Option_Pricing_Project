"""
Week 8 Configuration -- Tool Finalization & Project Delivery
============================================================
Finalizes the chooser-option pricing tool and packages the 8-week project:

  1. Tool feature completion:
       - dual pricing (BSM + best ML model) with Week-6 error margins;
       - complete visualization dashboard (historical price trends,
         sensitivity charts, performance metrics);
       - real-time data panel (data_updater + GitHub Actions).
  2. Final report (10-15 pages) synthesizing all 8 weeks.
  3. Final presentation deck + tool demo video script.
  4. Fully deployable pricing tool (GitHub repo with README).

All week8 modules import this module first. It puts the week7 directory at the
front of sys.path so `import w7config / tool_engine / sensitivity / data_updater`
resolve to the Week-7 artifacts we reuse unchanged, and re-exports the paths and
constants the final dashboard needs.
"""

import sys
from pathlib import Path

# ── path wiring: week7 does the full week6/week5/... insert in the correct ────
# order (week7 at the front), so importing it wires every reuse target.
W8_DIR = Path(__file__).resolve().parent
W7_DIR = W8_DIR.parent / 'week7'
sys.path.insert(0, str(W7_DIR))   # week7 wins `import tool_engine / sensitivity / ...`

import w7config as w7            # noqa: E402  (wires week6/week5/week3 paths)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Re-export paths & constants
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = W8_DIR / 'output'
ASSETS_DIR = W8_DIR / 'assets'
DATA_DIR = W8_DIR / 'data'
PRESENTATION_DIR = W8_DIR / 'presentation'
for _d in (OUTPUT_DIR, ASSETS_DIR, DATA_DIR, PRESENTATION_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MODELS_DIR = w7.MODELS_DIR          # Week-6 trained models/*.pkl
WEEK2_FEATURE_DATASET = w7.WEEK2_FEATURE_DATASET

# re-export the constants the dashboard / report use
DATE_COL = w7.DATE_COL
SPOT_COL = w7.SPOT_COL
VIX_COL = w7.VIX_COL
RATE_COL = w7.RATE_COL
BASE_FEATURES = list(w7.BASE_FEATURES)
REGIME_DUMMY_FEATURES = list(w7.REGIME_DUMMY_FEATURES)
CHOOSER_PARAMS = dict(w7.CHOOSER_PARAMS)
Q_YIELD = w7.Q_YIELD
TRADING_DAYS_PER_YEAR = w7.TRADING_DAYS_PER_YEAR
SNAPSHOT_DATE = w7.SNAPSHOT_DATE
FWD_VOL_TARGET = w7.FWD_VOL_TARGET

VOL_MODEL_ARTIFACTS = dict(w7.VOL_MODEL_ARTIFACTS)
PRICE_MODEL_ARTIFACTS = dict(w7.PRICE_MODEL_ARTIFACTS)
MODEL_LABELS = dict(w7.MODEL_LABELS)

TOOL_LIVE_MODEL = w7.TOOL_LIVE_MODEL       # vol_xgb
TOOL_SYNTH_MODEL = w7.TOOL_SYNTH_MODEL     # vol_vix_proxy
TOOL_BASELINE = w7.TOOL_BASELINE           # vol_21d (BSM persistence)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Week-6 metric artifacts the dashboard displays
# ═══════════════════════════════════════════════════════════════════════════════

W6_OUTPUT = W7_DIR.parent / 'week6' / 'output'
CHOOSER_TEST_CSV = W6_OUTPUT / 'chooser_test_comparison.csv'   # approach-1 chooser
VOL_TEST_CSV = W6_OUTPUT / 'vol_test_comparison.csv'           # approach-1 vol
LIVE_METRICS_CSV = W6_OUTPUT / 'live_metrics_w6.csv'           # live snapshot metrics
CONSOLIDATED_CSV = W6_OUTPUT / 'consolidated_metrics.csv'      # all tracks

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Report / presentation / repo paths
# ═══════════════════════════════════════════════════════════════════════════════

REPORT_TEX = W8_DIR / 'Week_8_实验报告.tex'
REPORT_PDF = Path(r'E:\实习交付\实验报告\Week_8_实验报告.pdf')
PRESENTATION_PPTX = PRESENTATION_DIR / 'Week_8_Final_Presentation.pptx'
DEMO_SCRIPT = W8_DIR / 'demo_script.md'

# price-trend chart window: whole training history is too dense; downsample
PRICE_SERIES_SAMPLE = 500          # ~20/day-equivalent across 2018-2024
