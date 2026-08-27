"""
Train Models
============
Reproduce the Week-6 Approach-1 volatility models (the winning path) and export
them in the format the pricing tool consumes (models/<family>.pkl + .json).

The hyper-parameters below are the Week-6 randomized-search optima (regularized
per the over-fitting diagnosis). The model is fit on train+val and evaluated once
on the held-out 2024 test set; the test MAE is recorded in the JSON sidecar.

Note: the shipped models/ already contain the validated artifacts; run this only
to retrain (e.g., after extending the feature dataset with data_updater --commit).

Usage:
    python train_models.py            # train all Approach-1 vol models
    python train_models.py --quick    # train only the 3 tool models (xgb / vix_proxy / gbdt_anchored)
"""

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

import config
import tool_engine as te
from data_preparation import load_dataset, build_splits, make_sequences
from targets import add_fwd_vol_targets, add_vix_target


# Week-6 best hyper-parameters (from the purged-CV random search)
BEST_PARAMS = {
    'xgb': dict(n_estimators=100, learning_rate=0.01, max_depth=4, min_child_weight=1,
                subsample=0.7, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=5.0),
    'rf': dict(n_estimators=100, max_depth=None, min_samples_leaf=20, max_features='sqrt'),
    'gbdt': dict(n_estimators=100, learning_rate=0.01, max_depth=3, min_samples_leaf=30, subsample=1.0),
    'gbdt_anchored': dict(n_estimators=100, learning_rate=0.01, max_depth=2, min_samples_leaf=20, subsample=1.0),
    'vix_proxy': dict(n_estimators=100, learning_rate=0.01, max_depth=4, min_child_weight=1,
                      subsample=0.7, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=5.0),
}


def prepare_frame() -> pd.DataFrame:
    """Feature dataset + forward-vol / VIX targets + regime dummies."""
    df = add_fwd_vol_targets(load_dataset())
    df = add_vix_target(df)
    df = pd.concat([df, te.regime_dummies(df)], axis=1)
    return df.dropna(subset=['vol_21d']).reset_index(drop=True)


def build_x_y(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Select model features + target from the full frame."""
    feats = list(config.BASE_FEATURES) + list(config.REGIME_DUMMY_FEATURES)
    return df[feats + [target]].dropna().reset_index(drop=True)


def make_model(family: str, target: str):
    """Return (estimator, pipeline) for a vol-model family."""
    p = dict(BEST_PARAMS[family])
    if family == 'xgb':
        from xgboost import XGBRegressor
        est = XGBRegressor(objective='reg:squarederror', random_state=config.RANDOM_SEED, **p)
    elif family == 'rf':
        est = RandomForestRegressor(n_jobs=-1, random_state=config.RANDOM_SEED, **p)
    elif family in ('gbdt', 'gbdt_anchored'):
        est = GradientBoostingRegressor(random_state=config.RANDOM_SEED, **p)
    elif family == 'vix_proxy':
        from xgboost import XGBRegressor
        est = XGBRegressor(objective='reg:squarederror', random_state=config.RANDOM_SEED, **p)
    else:
        raise ValueError(family)
    return est, Pipeline([('scale', StandardScaler()), ('model', est)])


def train_family(family: str, df: pd.DataFrame):
    """Train one vol-model family, export models/<family>.pkl + .json."""
    target = config.VIX_TARGET if family == 'vix_proxy' else config.FWD_VOL_TARGET
    if family == 'gbdt_anchored':
        # anchored model predicts sigma_fwd / sigma_21d
        df = df.copy()
        df['ratio_target'] = df[config.FWD_VOL_TARGET] / df['vol_21d']
        target = 'ratio_target'

    d = build_x_y(df, target)
    feats = [c for c in d.columns if c != target]
    split = build_splits(d, features=feats, target=target, scale=True)
    est, pipe = make_model(family, target)

    X_tr = pd.concat([split['train']['X'], split['val']['X']], ignore_index=True)
    y_tr = np.concatenate([split['train']['y'], split['val']['y']])
    pipe.fit(X_tr, y_tr)

    # one-shot held-out evaluation
    y_hat = pipe.predict(split['test']['X'])
    test_mae = float(np.mean(np.abs(y_hat - split['test']['y'])))

    payload = {'pipeline': pipe, 'params': BEST_PARAMS[family]} if family == 'vix_proxy' else pipe
    name = config.VOL_MODEL_ARTIFACTS[f'vol_{family}']
    joblib.dump(payload, config.MODELS_DIR / f'{name}.pkl')
    meta = {
        'track': 'approach1_vol', 'family': family, 'features': feats,
        'target': 'vix_target' if family == 'vix_proxy' else config.FWD_VOL_TARGET,
        'fit_region': 'train+val', 'test_MAE': test_mae,
        'params': BEST_PARAMS[family],
        'artifact': str(config.MODELS_DIR / f'{name}.pkl'),
    }
    with open(config.MODELS_DIR / f'{name}.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'  {family:<16} test_MAE={test_mae:.4f}  -> models/{name}.pkl')
    return test_mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='train only the 3 tool models (xgb / vix_proxy / gbdt_anchored)')
    args = ap.parse_args()

    families = ['xgb', 'vix_proxy', 'gbdt_anchored'] if args.quick else list(BEST_PARAMS)
    print('Preparing features + targets...')
    df = prepare_frame()
    print('Training Approach-1 volatility models:')
    for fam in families:
        train_family(fam, df)
    print('Done. Models written to models/*.pkl + .json')


if __name__ == '__main__':
    main()
