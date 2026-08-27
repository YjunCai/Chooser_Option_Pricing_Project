"""
Smoke Tests
===========
Lightweight end-to-end checks for the consolidated project.

Run:
    python test_smoke.py        # or: python run.py --tests
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import config
import pricer
import tool_engine as te
import dashboard as db
from data_preparation import load_dataset, make_chronological_split


def test_pricer():
    p = pricer.simple_chooser(156.70, 150.0, 0.5, 1.0, 0.0015, 0.0233, 0.282)
    assert abs(p - 29.13) < 0.01, f'expected ~29.13, got {p}'
    # boundary conditions
    v0 = pricer.simple_chooser(100, 100, 0.0, 1.0, 0.08, 0.0, 0.20)
    assert abs(v0 - max(pricer.bs_call(100, 100, 1, 0.08, 0, 0.20),
                        pricer.bs_put(100, 100, 1, 0.08, 0, 0.20))) < 1e-6
    print('  [ok] pricer matches paper value + boundary condition')


def test_data_prep():
    df = load_dataset()
    tr, va, te_ = make_chronological_split(len(df))
    assert tr.stop < va.start < te_.start, 'split must be chronological: train < val < test'
    print(f'  [ok] split: train {tr.stop}, val {va.stop - va.start}, test {te_.stop - te_.start}')


def test_engine():
    payload_xgb, feats, _ = te.load_vol_model('vol_xgb')
    assert len(feats) >= 15
    p = te.price_dual(150, 150, 0.5, 1.0, 0.05, 0.017, 0.24, 0.20)
    assert p['price_ML'] > 0 and p['price_BSM'] > 0
    em = te.error_margins()
    assert len(em) == len(config.ERROR_MARGINS)
    print(f'  [ok] engine: dual price BSM=${p["price_BSM"]:.2f} ML=${p["price_ML"]:.2f}, '
          f'{len(em)} error-margin rows')


def test_dashboard():
    s = db.historical_price_series(max_rows=300, extend_live=True)
    assert s['price_BSM'].notna().all() and (s['price_BSM'] > 0).all()
    m = db.performance_metrics()
    for key in ('chooser', 'vol', 'live', 'end2end'):
        assert len(m[key]) > 0
    print(f'  [ok] dashboard: trend rows={len(s)}, metrics={ {k: len(v) for k, v in m.items()} }')


def test_streamlit():
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    at = AppTest.from_file(str(Path(__file__).parent / 'streamlit_app.py'), default_timeout=180)
    at.run()
    assert len(at.exception) == 0, [e.message for e in at.exception]
    assert len(at.tabs) == 6
    print('  [ok] streamlit app renders 6 tabs')


def main():
    print('=' * 56)
    print('  FINAL PROJECT -- SMOKE TESTS')
    print('=' * 56)
    test_pricer()
    test_data_prep()
    test_engine()
    test_dashboard()
    test_streamlit()
    print('=' * 56)
    print('  All smoke tests passed.')
    print('=' * 56)


if __name__ == '__main__':
    main()
