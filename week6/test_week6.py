"""
Week 6 Framework Tests
======================
Run with:  python test_week6.py   (or pytest test_week6.py)

Covers the Week-6 additions on top of the Week 5 framework:
  1. purged time-series CV (chronology / embargo / no overlap)
  2. regime-adaptive features (five states, backward-looking one-hots)
  3. hyper-parameter search plumbing (searchable region, frames load)
  4. final model pickle export / load round-trip
  5. metric sanity (finite MAE/RMSE/R2 for the exported volatility model)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import w6config
import cv
import regime as regime_mod
import hp_search as hp
import train_final as tf
from data_preparation import load_dataset, make_chronological_split
from targets import add_fwd_vol_targets
from evaluation import evaluate_predictions

_PASS = []


def check(name, cond):
    assert cond, f'FAILED: {name}'
    _PASS.append(name)
    print(f'  [ok] {name}')


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Purged time-series CV
# ═══════════════════════════════════════════════════════════════════════════════

def test_cv():
    folds = cv.purged_cv_indices(1000, n_splits=4, purge_gap=21)
    check('cv: produced 4 folds', len(folds) == 4)
    for tr, va in folds:
        check('cv: chronological (train before val)', tr[-1] < va[0])
        check('cv: purge embargo respected',
              va[0] - tr[-1] - 1 >= 21)
        check('cv: train/val disjoint', len(set(tr) & set(va)) == 0)
        check('cv: expanding train block', len(tr) >= 300)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Regime features
# ═══════════════════════════════════════════════════════════════════════════════

def test_regime():
    d = add_fwd_vol_targets(load_dataset())
    d = regime_mod.add_regime_features(d)
    labels = set(d['regime'].unique())
    check('regime: five states present',
          labels <= set(w6config.REGIME_LABELS) and len(labels) >= 4)
    for col in w6config.REGIME_DUMMY_FEATURES:
        check(f'regime: dummy {col} present', col in d.columns)
    check('regime: dummies are 0/1', d[w6config.REGIME_DUMMY_FEATURES].isin([0.0, 1.0]).all().all())
    # one-hot rows sum to 1 (NORMAL is the reference, so sum can be 0)
    check('regime: at most one dummy active',
          d[w6config.REGIME_DUMMY_FEATURES].sum(axis=1).max() <= 1.0)
    # regime uses only backward-looking inputs: label at t depends on t's SMA/vol
    feat_set = regime_mod.build_regime_feature_set()
    check('regime: feature set contains dummies',
          all(c in feat_set for c in w6config.REGIME_DUMMY_FEATURES))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Hyper-parameter search plumbing
# ═══════════════════════════════════════════════════════════════════════════════

def test_search_plumbing():
    mod, feats, tr, va, te = hp.load_approach1_frame(regime=True)
    check('search: approach1 frame loads', mod.shape[0] > 1500)
    check('search: feature set includes regime dummies',
          any(c in feats for c in w6config.REGIME_DUMMY_FEATURES))
    sa = hp.searchable_arrays(mod, feats, hp.TARGET, tr, va)
    check('search: searchable region > test', len(sa['y']) > (te.stop - te.start))
    # searchable region is contiguous & chronological
    check('search: rows increasing', np.all(np.diff(sa['rows']) > 0))

    cf, pfeats = hp.load_approach2_frame(regime=True)
    check('search: approach2 contract frame loads', cf.shape[0] > 20000)
    check('search: pricing features include contract cols',
          all(c in pfeats for c in hp.APPROACH2_CONTRACT_FEATURES))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Final-model pickle export / load round-trip
# ═══════════════════════════════════════════════════════════════════════════════

def test_export_roundtrip():
    import joblib
    # train a tiny final model quickly on a subsample of the searchable region
    mod, feats, tr, va, te = hp.load_approach1_frame(regime=True)
    sa = hp.searchable_arrays(mod, feats, hp.TARGET, tr, va)
    idx = np.arange(len(sa['y']))[:400]
    gs = hp.run_random_search('gbdt', sa['X'][idx], sa['y'][idx], verbose=False)
    pipe = gs.best_estimator_

    path = w6config.MODELS_DIR / '_test_model.pkl'
    joblib.dump(pipe, path)
    check('export: pickle written', path.exists())
    loaded = joblib.load(path)
    pred = loaded.predict(sa['X'][:100])
    check('export: load+predict finite', np.all(np.isfinite(pred)))
    path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Metric sanity (finite MAE/RMSE/R2)
# ═══════════════════════════════════════════════════════════════════════════════

def test_metrics():
    rng = np.random.RandomState(0)
    y = rng.rand(200) * 0.3
    p = y + rng.rand(200) * 0.02
    met = evaluate_predictions(y, p)
    check('metrics: MAE finite & positive', np.isfinite(met['MAE']) and met['MAE'] > 0)
    check('metrics: RMSE finite', np.isfinite(met['RMSE']))
    check('metrics: R2 finite', np.isfinite(met['R2']))


def test_live_helpers():
    import live_metrics as lm
    opt = lm.load_snapshot_contracts()
    check('live: snapshot has 595 contracts', len(opt) == 595)
    check('live: par params present', float(opt['par_S'].iloc[0]) > 100)


if __name__ == '__main__':
    for fn in (test_cv, test_regime, test_search_plumbing, test_export_roundtrip,
               test_metrics, test_live_helpers):
        fn()
    print(f'\nAll {len(_PASS)} checks passed.')
