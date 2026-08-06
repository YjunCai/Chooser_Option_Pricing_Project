"""
Week 5 Framework Tests
======================
Run with:  python test_ml_framework.py   (or pytest test_ml_framework.py)

Covers the anti-look-ahead guarantees, split correctness, target alignment,
model contracts, LSTM sequence shapes and the BSM hybrid engine.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# week3 pricer must be importable
if str(Path('E:/实习交付/week2/week3')) not in sys.path:
    sys.path.insert(0, 'E:/实习交付/week2/week3')

import config
from data_preparation import (load_dataset, make_chronological_split, build_splits,
                              make_sequences, report_split)
from targets import forward_realized_vol, add_fwd_vol_targets, build_contract_frame, market_state
from feature_engineering_optimization import enhance_features, select_features
from models.base import SklearnModel
from models.volatility_models import build_volatility_models, build_lstm_datasets, LSTMVolatilityPredictor
from models.pricing_models import build_pricing_models
from models.bsm_engine import price_chooser_with_vol, chooser_price_series

_PASS = []


def check(name, cond):
    assert cond, f'FAILED: {name}'
    _PASS.append(name)
    print(f'  [ok] {name}')


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Time-series split correctness
# ═══════════════════════════════════════════════════════════════════════════════

def test_split():
    n = 1000
    tr, va, te = make_chronological_split(n, purge_gap=5)
    ti = set(range(tr.start, tr.stop))
    vi = set(range(va.start, va.stop))
    ei = set(range(te.start, te.stop))

    check('split: three disjoint sets', ti.isdisjoint(vi) and ti.isdisjoint(ei) and vi.isdisjoint(ei))
    check('split: chronological ordering', max(ti) < min(vi) and max(vi) < min(ei))
    check('split: train ratio ~70%', abs(len(ti) / n - 0.70) < 0.02)
    check('split: val ratio ~15%', abs(len(vi) / n - 0.15) < 0.02)
    check('split: test ratio ~15%', abs(len(ei) / n - 0.15) < 0.02)
    # purge gap removed (no overlap with gap)
    gap1 = set(range(tr.stop, va.start))
    check('split: purge gap excluded', gap1.isdisjoint(ti) and gap1.isdisjoint(vi))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. No-look-ahead target alignment
# ═══════════════════════════════════════════════════════════════════════════════

def test_forward_vol_no_lookahead():
    rng = np.random.RandomState(7)
    returns = pd.Series(rng.randn(120))
    rv = forward_realized_vol(returns, horizon=5, annualize=False)

    # rv[t] must equal std of returns[t+1 .. t+5] -- strictly future data
    # (pandas rolling.std uses ddof=1; matches the week-2 realized-vol convention)
    expected = np.std(returns[1:6].values, ddof=1)
    check('fwd vol: aligned to future window', np.isclose(rv.iloc[0], expected))
    check('fwd vol: does not use today', not np.isclose(rv.iloc[0], np.std(returns[0:5].values, ddof=1)))
    # last rows unavailable (target window extends past data)
    check('fwd vol: NaN at tail', np.isnan(rv.iloc[-5]))
    # default annualizes to a yearly scale
    rv_ann = forward_realized_vol(returns, horizon=5, annualize=True)
    check('fwd vol: annualized', np.isclose(rv_ann.iloc[0], expected * np.sqrt(config.TRADING_DAYS_PER_YEAR)))


def test_build_splits_scaler():
    df = load_dataset()
    df = add_fwd_vol_targets(df)
    target = f'fwd_realized_vol_{config.VOL_FORWARD_HORIZON}d'
    mod = df.dropna(subset=config.BASE_FEATURES + [target]).reset_index(drop=True)
    split = build_splits(mod, config.BASE_FEATURES, target, scale=True)

    check('splits: scaler fit on train only',
          np.allclose(split['scaler'].mean_, mod.iloc[split['train']['idx']][config.BASE_FEATURES].mean()))
    # every set shares the same train-fitted scaler
    check('splits: X_full has no NaN', not split['X_full'].isna().any().any())
    check('splits: train/val/test lengths',
          len(split['train']['X']) + len(split['val']['X']) + len(split['test']['X'])
          == len(split['X_full']) - 2 * config.PURGE_GAP_DAYS)


def test_sequences():
    df = load_dataset()
    df = add_fwd_vol_targets(df)
    target = f'fwd_realized_vol_{config.VOL_FORWARD_HORIZON}d'
    mod = df.dropna(subset=config.BASE_FEATURES + [target]).reset_index(drop=True)
    split = build_splits(mod, config.BASE_FEATURES, target, scale=True)

    seq = build_lstm_datasets(split, seq_len=12)
    Xs, ys = seq['train']['X'], seq['train']['y']
    check('lstm seq: 3D input', Xs.ndim == 3 and Xs.shape[1] == 12 and Xs.shape[2] == len(config.BASE_FEATURES))
    check('lstm seq: y aligned', ys.shape[0] == Xs.shape[0])
    # window strictly backward: last timestep equals anchor row features
    anchor = split['train']['idx'][-1]
    check('lstm seq: last step == anchor features',
          np.allclose(Xs[-1, -1], split['X_full'].iloc[anchor].values))
    check('lstm seq: val/test present', seq['val']['X'].shape[0] > 0 and seq['test']['X'].shape[0] > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Model contracts
# ═══════════════════════════════════════════════════════════════════════════════

def test_volatility_models():
    df = load_dataset()
    df = add_fwd_vol_targets(df)
    target = f'fwd_realized_vol_{config.VOL_FORWARD_HORIZON}d'
    mod = df.dropna(subset=config.BASE_FEATURES + [target]).iloc[:600].reset_index(drop=True)
    split = build_splits(mod, config.BASE_FEATURES, target, scale=True)

    for key, model in build_volatility_models().items():
        model.fit(split['train']['X'], split['train']['y'])
        p = model.predict(split['test']['X'])
        check(f'model {key}: shape', p.shape == split['test']['y'].shape)
        check(f'model {key}: finite output', np.isfinite(p).all())


def test_lstm_model():
    df = load_dataset()
    df = add_fwd_vol_targets(df)
    target = f'fwd_realized_vol_{config.VOL_FORWARD_HORIZON}d'
    mod = df.dropna(subset=config.BASE_FEATURES + [target]).iloc[:500].reset_index(drop=True)
    split = build_splits(mod, config.BASE_FEATURES, target, scale=True)
    seq = build_lstm_datasets(split, seq_len=10)

    lstm = LSTMVolatilityPredictor(seq_len=10, hidden_size=8, num_layers=1,
                                   epochs=3, batch_size=32, lr=1e-2)
    lstm.fit(seq['train']['X'], seq['train']['y'])
    p = lstm.predict(seq['test']['X'])
    check('lstm: backend resolved', lstm.backend in ('torch', 'mlp-fallback'))
    check('lstm: predict shape', p.shape == seq['test']['y'].shape)
    check('lstm: finite output', np.isfinite(p).all())


def test_pricing_models():
    df = load_dataset()
    df = add_fwd_vol_targets(df)
    cf = build_contract_frame(df).dropna(subset=['label_price']).iloc[:2000].reset_index(drop=True)
    feats = config.BASE_FEATURES + ['log_moneyness', 'tenor_years', 'is_call']
    X = cf[feats].fillna(cf[feats].mean()).astype(float)
    y = cf['label_price'].values
    tr = slice(0, 1400)
    for key, model in build_pricing_models().items():
        model.fit(X.iloc[tr], y[tr])
        p = model.predict(X.iloc[tr])
        check(f'pricing {key}: fit+predict', p.shape == y[tr].shape and np.isfinite(p).all())


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BSM hybrid engine
# ═══════════════════════════════════════════════════════════════════════════════

def test_bsm_engine():
    p_low = price_chooser_with_vol(156.7, 0.02, 0.017, 0.10)
    p_high = price_chooser_with_vol(156.7, 0.02, 0.017, 0.50)
    check('bsm: finite chooser price', np.isfinite(p_low) and p_low > 0)
    check('bsm: monotonic in vol (vega+ direction)', p_high > p_low)

    mkt = market_state(add_fwd_vol_targets(load_dataset()))
    mkt = mkt.dropna().iloc[:50]
    ser = chooser_price_series(mkt, mkt['sigma_actual'])
    check('bsm: vectorized series', ser.shape == (mkt.shape[0],) and np.isfinite(ser).all())


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Feature engineering optimization
# ═══════════════════════════════════════════════════════════════════════════════

def test_feature_engineering():
    df = load_dataset()
    df = add_fwd_vol_targets(df)
    enhanced = enhance_features(df)
    check('feat: enhanced superset', set(config.BASE_FEATURES).issubset(set(enhanced.columns)))
    check('feat: added derived features', len(enhanced.columns) > len(df.columns) + 1)

    # lags are backward: lag columns are NaN at the head (shift left empty)
    check('feat: lag feature backward-looking', pd.isna(enhanced['vol_21d_lag1'].iloc[0]))

    sel, imp = select_features(enhanced, target=f'fwd_realized_vol_{config.VOL_FORWARD_HORIZON}d')
    check('feat: selection returns features', len(sel) > 0)
    check('feat: selection is subset of enhanced', set(sel).issubset(set(enhanced.columns)))
    check('feat: importance report covers base features', len(imp) == len(config.BASE_FEATURES))


# ═══════════════════════════════════════════════════════════════════════════════

def run_all():
    print('Running Week 5 framework tests...\n')
    test_split()
    test_forward_vol_no_lookahead()
    test_build_splits_scaler()
    test_sequences()
    test_volatility_models()
    test_lstm_model()
    test_pricing_models()
    test_bsm_engine()
    test_feature_engineering()
    print(f'\nAll {len(_PASS)} checks passed.')


if __name__ == '__main__':
    run_all()
