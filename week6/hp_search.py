"""
Hyper-Parameter Optimization -- Week 6
======================================
Random / grid search with purged time-series cross-validation for every model
family in both tracks:

  Approach 1 (volatility prediction + BSM hybrid):
      rf / gbdt / xgb   -> flat feature matrix -> forward realized vol
      gbdt_anchored     -> predicts vol_ratio = fwd_vol / vol_21d (anchored)
      lstm              -> sliding-window sequence (small manual search)
  Approach 2 (end-to-end pricing):
      pricing_gbdt / pricing_nn -> contract-grid rows -> option price

Design
------
  * The outer 70/15/15 chronological split (purged) is built first; the held-out
    TEST set is NEVER touched during the search.
  * The searchable region = train + val rows. An inner PurgedTimeSeriesSplit
    (expanding window + 21-day embargo) selects hyper-parameters on that region.
  * Scaling is done inside a sklearn Pipeline, so the StandardScaler is refit on
    each fold's TRAINING block only (no look-ahead via the scaler).
  * refit=True -> RandomizedSearchCV refits the best model on the full searchable
    region; that best_estimator_ is reused directly for final test evaluation.
"""

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import w6config
from data_preparation import (load_dataset, build_splits, make_chronological_split,
                              report_split)
from targets import add_fwd_vol_targets
from feature_engineering_optimization import enhance_features, select_features
from regime import add_regime_features, build_regime_feature_set
from cv import PurgedTimeSeriesSplit
from evaluation import evaluate_predictions
from models.base import SklearnModel

TARGET = w6config.FWD_VOL_TARGET


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Shared data preparation (Approach 1)
# ═══════════════════════════════════════════════════════════════════════════════

def load_approach1_frame(regime: bool = True):
    """Feature frame with fwd-vol target (+ regime features) and outer split."""
    df = load_dataset()
    df = add_fwd_vol_targets(df)
    if regime:
        df = add_regime_features(df)

    # Feature selection mirrors Week 5 (variance filter + |corr|>0.92
    # de-duplication + RF importance). Selection runs on the full sample
    # distribution -- a documented compromise kept for consistency with the
    # Week 5 pipeline; the model-fit / evaluation split remains strictly
    # chronological and leak-free.
    selected, _ = select_features(df, target=TARGET)
    feats = build_regime_feature_set(selected) if regime else selected
    feats = [c for c in feats if c in df.columns]

    mod = df.dropna(subset=feats + [TARGET, 'vol_21d']).reset_index(drop=True)
    tr, va, te = make_chronological_split(len(mod), purge_gap=w6config.PURGE_GAP_DAYS)
    return mod, feats, tr, va, te


def searchable_arrays(frame: pd.DataFrame, feats: List[str], target: str,
                      tr, va) -> Dict[str, np.ndarray]:
    """(X, y) for the train+val searchable region; rows contiguous + chronological."""
    rows = np.r_[np.arange(tr.start, tr.stop), np.arange(va.start, va.stop)]
    sub = frame.iloc[rows].reset_index(drop=True)
    return {'X': sub[feats].astype(float).values, 'y': sub[target].astype(float).values,
            'frame': sub, 'rows': rows}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RandomizedSearchCV helper
# ═══════════════════════════════════════════════════════════════════════════════

def _estimator(family: str):
    """Raw sklearn estimator constructor for a search family."""
    if family == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=w6config.RANDOM_SEED)
    if family == 'gbdt':
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=w6config.RANDOM_SEED)
    if family == 'xgb':
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(random_state=w6config.RANDOM_SEED, verbosity=0)
        except Exception:
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(random_state=w6config.RANDOM_SEED)
    if family == 'pricing_gbdt':
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=w6config.RANDOM_SEED)
    if family == 'pricing_nn':
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(random_state=w6config.RANDOM_SEED, early_stopping=True,
                            validation_fraction=0.12, n_iter_no_change=15)
    raise KeyError(family)


def run_random_search(family: str, X, y, verbose: bool = True):
    """RandomizedSearchCV over the searchable region with purged time-series CV."""
    from sklearn.model_selection import RandomizedSearchCV

    space = dict(w6config.SEARCH_SPACES[family])
    n_iter = w6config.SEARCH_ITERS[family]
    pipe = Pipeline([('scale', StandardScaler()), ('model', _estimator(family))])
    cv = PurgedTimeSeriesSplit()
    gs = RandomizedSearchCV(
        pipe, {f'model__{k}': v for k, v in space.items()},
        n_iter=n_iter, cv=cv, scoring=w6config.CV['scoring'],
        random_state=w6config.RANDOM_SEED, n_jobs=2, refit=True, verbose=0)
    gs.fit(X, y)
    if verbose:
        print(f'      {family:<12} CV-MAE={-gs.best_score_:.4f}  '
              f'best_params={gs.best_params_}')
    return gs


def _flatten_params(params: Dict[str, object]) -> Dict[str, object]:
    return {k.replace('model__', ''): (None if v is None else v)
            for k, v in params.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Anchored vol-ratio model
# ═══════════════════════════════════════════════════════════════════════════════

def search_anchored(frame: pd.DataFrame, feats: List[str], tr, va, verbose=True):
    """Search the persistence-anchored GBDT: predict vol_ratio, rescale by vol_21d."""
    df = frame.copy()
    df['vol_ratio'] = (df[TARGET] / df['vol_21d']).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feats + ['vol_ratio']).reset_index(drop=True)

    rows = np.r_[np.arange(tr.start, tr.stop), np.arange(va.start, va.stop)]
    sub = df.iloc[rows].reset_index(drop=True)
    X, y = sub[feats].astype(float).values, sub['vol_ratio'].astype(float).values
    gs = run_random_search('gbdt', X, y, verbose=verbose)
    gs.best_estimator_.fit(X, y)          # ensure refit on full searchable region
    return {'gs': gs, 'frame': df, 'params': _flatten_params(gs.best_params_)}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LSTM (manual small search over 3D sequences)
# ═══════════════════════════════════════════════════════════════════════════════

def search_lstm(frame: pd.DataFrame, feats: List[str], verbose=True):
    """Small random search over the LSTM hyper-parameters on sequence data."""
    from models.volatility_models import (LSTMVolatilityPredictor,
                                          build_lstm_datasets)
    cfg = w6config.LSTM_SEARCH
    rng = np.random.RandomState(w6config.RANDOM_SEED)

    # build the sequence split over the FULL frame (scaled by train-fit scaler)
    split = build_splits(frame.dropna(subset=feats + [TARGET]).reset_index(drop=True),
                         feats, TARGET, scale=True, purge_gap=w6config.PURGE_GAP_DAYS)
    seq = build_lstm_datasets(split, seq_len=20)
    Xt, yt = seq['train']['X'], seq['train']['y']
    Xv, yv = seq['val']['X'], seq['val']['y']
    if Xt.shape[0] < 64 or Xv.shape[0] < 32:
        if verbose:
            print('      lstm  skipped: too few sequence samples')
        return None

    best = {'mae': np.inf}
    for i in range(cfg['n_iter']):
        h = int(rng.choice(cfg['hidden_size']))
        d = float(rng.choice(cfg['dropout']))
        lr = float(rng.choice(cfg['lr']))
        m = LSTMVolatilityPredictor(seq_len=20, hidden_size=h, num_layers=1,
                                    dropout=d, epochs=cfg['epochs'],
                                    batch_size=32, lr=lr, seed=w6config.RANDOM_SEED)
        m.fit(Xt, yt)
        mae = evaluate_predictions(yv, m.predict(Xv))['MAE']
        if mae < best['mae']:
            best = {'mae': mae, 'hidden_size': h, 'dropout': d, 'lr': lr}
        if verbose:
            print(f'      lstm  iter {i+1}/{cfg["n_iter"]} h={h} d={d} lr={lr:.0e} '
                  f'val-MAE={mae*100:.2f}%')
    if verbose:
        print(f'      lstm  best val-MAE={best["mae"]*100:.2f}%  params={best}')
    best.pop('mae')
    return {'params': best, 'seq': seq}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Approach 2 (end-to-end pricing) search
# ═══════════════════════════════════════════════════════════════════════════════

APPROACH2_CONTRACT_FEATURES = ['log_moneyness', 'tenor_years', 'is_call']


def load_approach2_frame(regime: bool = True):
    """Contract-grid long frame with market + regime + contract features."""
    from targets import build_contract_frame
    df = load_dataset()
    df = add_fwd_vol_targets(df)
    if regime:
        df = add_regime_features(df)
    base = list(w6config.BASE_FEATURES)
    if regime:
        base = base + list(w6config.REGIME_DUMMY_FEATURES)
    # pass the regime-augmented feature list into build_contract_frame so the
    # dummies are carried into every contract row.
    cf = build_contract_frame(df, features=base)
    feats = [c for c in base + APPROACH2_CONTRACT_FEATURES if c in cf.columns]
    cf = cf.dropna(subset=feats + ['label_price']).reset_index(drop=True)
    return cf, feats


def search_price_models(cf: pd.DataFrame, feats: List[str], verbose=True) -> Dict:
    """RandomizedSearchCV for pricing_gbdt / pricing_nn over train+val dates."""
    dates = np.array(sorted(cf['date'].unique()))
    tr, va, te = make_chronological_split(len(dates), purge_gap=w6config.PURGE_GAP_DAYS)
    date_sets = {'train': set(dates[tr]), 'val': set(dates[va]), 'test': set(dates[te])}
    search_dates = np.r_[dates[tr], dates[va]]

    masks = {k: cf['date'].isin(v) for k, v in date_sets.items()}
    X = cf[feats].astype(float).values
    y = cf['label_price'].values

    s_mask = masks['train'] | masks['val']
    Xs, ys = X[s_mask], y[s_mask]

    results = {}
    for fam in ('pricing_gbdt', 'pricing_nn'):
        gs = run_random_search(fam, Xs, ys, verbose=verbose)
        results[fam] = {'gs': gs, 'params': _flatten_params(gs.best_params_)}

    # keep the test masks & test arrays for later final evaluation
    results['_meta'] = {'feats': feats, 'X': X, 'y': y, 'masks': masks,
                        'search_dates': search_dates,
                        'base_price': cf['bsm_vol21_price'].values}
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Orchestrators
# ═══════════════════════════════════════════════════════════════════════════════

def run_vol_search(regime: bool = True, verbose: bool = True) -> Dict:
    """Full Approach-1 hyper-parameter search; returns per-model best fit objects."""
    mod, feats, tr, va, te = load_approach1_frame(regime=regime)
    sa = searchable_arrays(mod, feats, TARGET, tr, va)
    if verbose:
        print(f'  searchable region: {len(sa["y"])} rows, {len(feats)} features')

    results = {}
    for fam in ('rf', 'gbdt', 'xgb'):
        gs = run_random_search(fam, sa['X'], sa['y'], verbose=verbose)
        results[fam] = {'gs': gs, 'params': _flatten_params(gs.best_params_)}

    anch = search_anchored(mod, feats, tr, va, verbose=verbose)
    results['gbdt_anchored'] = anch

    lstm = search_lstm(mod, feats, verbose=verbose)
    results['lstm'] = lstm

    results['_meta'] = {'mod': mod, 'feats': feats, 'tr': tr, 'va': va, 'te': te,
                        'search': sa}
    return results


def summarize_search(results: Dict) -> pd.DataFrame:
    """Best-params summary table (CV MAE in raw vol units)."""
    rows = []
    for key in ('rf', 'gbdt', 'xgb', 'gbdt_anchored'):
        if key not in results or not results[key]:
            continue
        r = results[key]
        cv_mae = -r['gs'].best_score_
        rows.append({'model': key, 'cv_MAE': cv_mae, 'best_params': r['params']})
    if results.get('lstm'):
        rows.append({'model': 'lstm', 'cv_MAE': np.nan,
                     'best_params': results['lstm']['params']})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    res = run_vol_search(verbose=True)
    print(summarize_search(res).to_string(index=False))
