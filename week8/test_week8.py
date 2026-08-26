"""
Week 8 Unit Tests
=================
Validates the Week-8 additions (dashboard price-trend series, performance
metrics) and the finalized tool's imports / Streamlit render.

Run:
    python test_week8.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import w8config as cfg
import tool_engine as te
import dashboard as db


# ═══════════════════════════════════════════════════════════════════════════════
# 1. dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def test_historical_price_series():
    s = db.historical_price_series(max_rows=200, extend_live=False)
    assert len(s) <= 200 and len(s) > 50, f'rows={len(s)}'
    for col in ('date', 'spot', 'vol_21d', 'sigma_xgb', 'sigma_vix_proxy',
                'price_BSM', 'price_xgb', 'price_vix_proxy'):
        assert col in s.columns, f'missing {col}'
    assert s['price_BSM'].notna().all()
    assert (s['price_BSM'] > 0).all()
    # ML sigma should differ from the raw vol_21d at least somewhere
    assert np.abs(s['sigma_xgb'] - s['vol_21d']).max() > 1e-6
    print('  [ok] historical_price_series: rows=', len(s))


def test_price_trend_extend_live():
    s = db.historical_price_series(max_rows=250, extend_live=True)
    assert len(s) >= 250
    print('  [ok] trend extends to live date:', str(s['date'].iloc[-1].date()))


def test_performance_metrics():
    m = db.performance_metrics()
    for key in ('chooser', 'vol', 'live', 'end2end'):
        assert key in m and len(m[key]) > 0, f'missing {key}'
    assert 'MAE' in m['chooser'].columns
    # best synthetic chooser model is the VIX-proxy
    best = m['chooser'].sort_values('MAE').iloc[0]['model']
    assert best == 'vix_proxy', f'expected vix_proxy best, got {best}'
    print('  [ok] performance_metrics:', {k: len(v) for k, v in m.items()})


# ═══════════════════════════════════════════════════════════════════════════════
# 2. reused tool engine still healthy
# ═══════════════════════════════════════════════════════════════════════════════

def test_dual_pricing():
    payload_xgb, feats, _ = te.load_vol_model('vol_xgb')
    assert len(feats) > 10
    p = te.price_dual(150, 150, 0.5, 1.0, 0.05, 0.017, 0.24, 0.20)
    assert p['price_ML'] > 0 and p['price_BSM'] > 0
    assert abs(p['spread_$'] - (p['price_ML'] - p['price_BSM'])) < 1e-6
    print(f'  [ok] dual pricing: BSM=${p["price_BSM"]:.2f}, ML=${p["price_ML"]:.2f}')


def test_error_margins():
    em = te.error_margins()
    assert len(em) >= 7
    assert 'price_MAE_$' in em.columns
    print('  [ok] error margins rows=', len(em))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Streamlit render
# ═══════════════════════════════════════════════════════════════════════════════

def test_streamlit_app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(cfg.W8_DIR / 'streamlit_app.py'), default_timeout=180)
    at.run()
    assert len(at.exception) == 0, [e.message for e in at.exception]
    labels = [t.label for t in at.tabs]
    assert any('价格趋势' in l for l in labels)
    assert len(at.dataframe) >= 5
    print('  [ok] streamlit app renders: tabs=', len(at.tabs),
          'charts=', len(at.get('plotly_chart')))


def main():
    print('=' * 60)
    print('  WEEK 8 -- TOOL FINALIZATION & PROJECT DELIVERY TESTS')
    print('=' * 60)
    test_historical_price_series()
    test_price_trend_extend_live()
    test_performance_metrics()
    test_dual_pricing()
    test_error_margins()
    test_streamlit_app()
    print('=' * 60)
    print('  All Week-8 tests passed.')
    print('=' * 60)


if __name__ == '__main__':
    main()
