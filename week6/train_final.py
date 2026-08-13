"""
Final Model Training & Pickle Export -- Week 6
==============================================
After hyper-parameter selection (hp_search.py), this module:

  1. Refits every Approach-1 volatility model on the searchable region
     (train + val) and evaluates ONCE on the held-out test set;
  2. Adds a VIX-proxy "market-implied vol" model (predicts VIX at t+1, feeds
     VIX/100 into the BSM engine) -- the closest available stand-in for a real
     JPM historical-IV target;
  3. Converts every volatility forecast into a Chooser price (Week 3 Rubinstein
     formula) and scores it against the forward-vol fair price;
  4. Refits the Approach-2 end-to-end pricing models on train+val dates and
     evaluates on test dates;
  5. Exports every final model as a pickle (joblib) file + a JSON metadata
     sidecar into models/.
"""

import json
import sys
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import joblib

import w6config
from data_preparation import make_chronological_split
from targets import add_vix_target, market_state
from evaluation import evaluate_predictions, compare_vol_models, summarize_pricing_comparison
from models.bsm_engine import chooser_price_series
from models.volatility_models import LSTMVolatilityPredictor, build_lstm_datasets
import hp_search as hp
import regime as regime_mod

TARGET = hp.TARGET


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Test-set prediction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def predict_pipeline(best_estimator, X_test: pd.DataFrame):
    """Predict from a (scale, model) Pipeline -- scaling handled internally."""
    return np.asarray(best_estimator.predict(X_test), dtype=float)


def anchored_vol_forecast(best_estimator, X_test: pd.DataFrame, vol_21d_test: np.ndarray):
    """predict vol_ratio then rescale by the current vol level (anchored model)."""
    ratio = np.clip(predict_pipeline(best_estimator, X_test), 0.1, 5.0)
    return np.asarray(vol_21d_test) * ratio


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Approach 1 -- final fit + test evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_approach1_final(search_results: Optional[Dict] = None, verbose: bool = True) -> Dict:
    """Train final vol models, evaluate on test, return fitted artifacts + metrics."""
    t0 = time.time()
    if search_results is None:
        search_results = hp.run_vol_search(regime=True, verbose=False)

    meta = search_results['_meta']
    mod, feats, tr, va, te = meta['mod'], meta['feats'], meta['tr'], meta['va'], meta['te']
    sa = meta['search']

    # --- test data (raw features; pipelines scale internally) ---
    test_df = mod.iloc[te].reset_index(drop=True)
    X_test = test_df[feats].astype(float)
    y_true = test_df[TARGET].values
    test_vol21 = test_df['vol_21d'].values          # BSM persistence baseline
    baseline_vol = test_vol21

    # --- searchable region arrays (for refit / VIX model) ---
    X_search, y_search = sa['X'], sa['y']

    # --- train+val sequence data for LSTM ---
    lstm_meta = search_results['lstm']
    lstm_params = lstm_meta['params'] if lstm_meta else None
    seq = lstm_meta['seq'] if lstm_meta else None

    preds: Dict[str, np.ndarray] = {}
    fitted: Dict[str, object] = {}
    metrics_rows: list = []

    for fam in ('rf', 'gbdt', 'xgb'):
        r = search_results[fam]
        model = r['gs'].best_estimator_                       # refit on train+val
        pred = predict_pipeline(model, X_test)
        preds[fam] = pred
        fitted[fam] = model
        met = evaluate_predictions(y_true, pred)
        metrics_rows.append({'model': r['gs'].best_estimator_.__class__.__name__,
                             'family': fam, 'target': 'fwd_vol', **met,
                             'params': json.dumps(r['params'])})
        if verbose:
            print(f'      {fam:<14} test MAE={met["MAE"]*100:.2f}%  R2={met["R2"]:.3f}')

    # --- anchored model (predict vol_ratio) ---
    if 'gbdt_anchored' in search_results and search_results['gbdt_anchored']:
        anch = search_results['gbdt_anchored']['gs'].best_estimator_
        pred = anchored_vol_forecast(anch, X_test, test_vol21)
        preds['gbdt_anchored'] = pred
        fitted['gbdt_anchored'] = anch
        met = evaluate_predictions(y_true, pred)
        metrics_rows.append({'model': 'GBDT-anchored (vol_ratio)', 'family': 'gbdt_anchored',
                             'target': 'fwd_vol', **met,
                             'params': json.dumps(search_results['gbdt_anchored']['params'])})
        if verbose:
            print(f'      gbdt_anchored test MAE={met["MAE"]*100:.2f}%  R2={met["R2"]:.3f}')

    # --- LSTM (refit on train+val sequences) ---
    if lstm_params is not None and seq is not None:
        Xc = np.concatenate([seq['train']['X'], seq['val']['X']], axis=0)
        yc = np.concatenate([seq['train']['y'], seq['val']['y']], axis=0)
        lstm_model = LSTMVolatilityPredictor(
            seq_len=20, hidden_size=lstm_params['hidden_size'],
            num_layers=1, dropout=lstm_params['dropout'], epochs=w6config.LSTM_SEARCH['epochs'],
            batch_size=32, lr=lstm_params['lr'], seed=w6config.RANDOM_SEED)
        lstm_model.fit(Xc, yc)
        pred = np.asarray(lstm_model.predict(seq['test']['X']), dtype=float)
        # align to full test length (sequences drop the first seq_len-1 anchors)
        n_full = len(y_true)
        n_seq = len(pred)
        if n_seq < n_full:
            pred = np.concatenate([np.full(n_full - n_seq, np.nan), pred])
        preds['lstm'] = pred
        fitted['lstm'] = lstm_model
        mask = np.isfinite(pred)
        if mask.sum() > 0:
            met = evaluate_predictions(y_true[mask], pred[mask])
            metrics_rows.append({'model': 'LSTM', 'family': 'lstm', 'target': 'fwd_vol',
                                 'n': int(mask.sum()), **met,
                                 'params': json.dumps(lstm_params)})
            if verbose:
                print(f'      lstm          test MAE={met["MAE"]*100:.2f}%  R2={met["R2"]:.3f}')

    # --- VIX-proxy market-implied vol model (best GBDT params on VIX target) ---
    vix_model, vix_pred = _fit_vix_proxy(mod, feats, search_results)
    if vix_pred is not None:
        preds['vix_proxy'] = vix_pred
        fitted['vix_proxy'] = vix_model
        met = evaluate_predictions(y_true, vix_pred)
        metrics_rows.append({'model': 'XGB-VIX proxy (IV)', 'family': 'vix_proxy',
                             'target': 'vix', **met, 'params': json.dumps(vix_model.get('params', {}))})
        if verbose:
            print(f'      vix_proxy    test MAE={met["MAE"]*100:.2f}%  R2={met["R2"]:.3f}')

    # --- volatility comparison vs persistence + historical-mean baselines ---
    train_mean_vol = float(mod.iloc[tr]['vol_21d'].mean())
    baselines = {
        'BSM(vol_21d) persistence': baseline_vol,
        'historical mean vol': np.full_like(y_true, train_mean_vol),
    }
    vol_cmp = compare_vol_models(y_true, {k: v for k, v in preds.items() if np.all(np.isfinite(v))},
                                 baselines=baselines)
    if verbose:
        print('\n      Volatility forecast error (test):')
        print(vol_cmp[['model', 'MAE', 'RMSE', 'R2', 'MAE_improve_vs_base_%']]
              .round(5).to_string(index=False))
    vol_cmp.to_csv(w6config.OUTPUT_DIR / 'vol_test_comparison.csv', index=False)

    # --- Chooser pricing (vs forward-vol fair price) ---
    test_market = market_state(mod).iloc[te].reset_index(drop=True)
    price_artifacts, price_cmp = _hybrid_pricing(test_market, preds)
    if verbose:
        print('\n      Chooser price error (test, vs forward-vol fair price):')
        print(price_cmp.round(4).to_string(index=False))
    price_cmp.to_csv(w6config.OUTPUT_DIR / 'chooser_test_comparison.csv', index=False)

    return {
        'metrics': pd.DataFrame(metrics_rows),
        'vol_comparison': vol_cmp,
        'price_comparison': price_cmp,
        'preds': preds,
        'fitted': fitted,
        'y_true': y_true,
        'test_market': test_market,
        'X_test': X_test,
        'mod': mod,
        'te': te,
        'feats': feats,
        'runtime_s': time.time() - t0,
    }


def _fit_vix_proxy(mod, feats, search_results):
    """Train a tuned XGBoost on the VIX-at-t+1 target; return (artifact, sigma_pred).

    The VIX label (shifted by -1) is absent only on the very last row; the model
    is fit on the train+val rows that *have* a finite label, then it predicts on
    ALL test rows (features only) so the forecast aligns 1:1 with `y_true`.
    """
    df = add_vix_target(mod.copy(), shift=w6config.VIX_TARGET_SHIFT)

    tr, va, te = make_chronological_split(len(df), purge_gap=w6config.PURGE_GAP_DAYS)
    rows = np.r_[np.arange(tr.start, tr.stop), np.arange(va.start, va.stop)]
    search_df = df.iloc[rows].dropna(subset=feats + ['vix_target']).reset_index(drop=True)
    X_s, y_s = search_df[feats].astype(float).values, search_df['vix_target'].values

    try:
        xgb_params = dict(search_results['xgb']['params']) if 'xgb' in search_results else {}
    except Exception:
        xgb_params = dict(search_results['gbdt']['params']) if 'gbdt' in search_results else {}

    xgb_params.pop('name', None)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    model = Pipeline([('scale', StandardScaler()),
                      ('model', _xgb_estimator(xgb_params))])
    model.fit(X_s, y_s)

    test_df = df.iloc[te].reset_index(drop=True)          # all 208 test rows
    X_t = test_df[feats].astype(float)
    vix_pred = np.clip(model.predict(X_t), 5.0, 60.0) / 100.0      # VIX/100 as sigma
    return {'pipeline': model, 'params': xgb_params}, vix_pred


def _xgb_estimator(params):
    from xgboost import XGBRegressor
    return XGBRegressor(random_state=w6config.RANDOM_SEED, verbosity=0, **params)


def _hybrid_pricing(test_rows: pd.DataFrame, preds: Dict[str, np.ndarray]):
    fair = chooser_price_series(test_rows, test_rows['sigma_actual'])
    baseline = chooser_price_series(test_rows, test_rows['sigma_21d'])
    price_dict = {k: chooser_price_series(test_rows, v) for k, v in preds.items()
                  if np.all(np.isfinite(v))}
    cmp = summarize_pricing_comparison(fair, price_dict,
                                       baseline_name='BSM(vol_21d)', baseline=baseline)
    return {'fair': fair, 'baseline': baseline, 'models': price_dict}, cmp


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Approach 2 -- final fit + test evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_approach2_final(search_results: Optional[Dict] = None, verbose: bool = True) -> Dict:
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    if search_results is None:
        search_results = hp.search_price_models(*hp.load_approach2_frame(), verbose=False)

    meta = search_results['_meta']
    feats = meta['feats']
    X, y = meta['X'], meta['y']
    masks = meta['masks']

    fitted, preds = {}, {}
    rows = []
    from models.base import SklearnModel
    for fam in ('pricing_gbdt', 'pricing_nn'):
        gs = search_results[fam]['gs']
        model = gs.best_estimator_                       # Pipeline(scaler+model), fit train+val
        pred = model.predict(X[masks['test']])
        preds[fam] = pred
        fitted[fam] = model
        met = evaluate_predictions(y[masks['test']], pred, mape_min_abs=0.25)
        rows.append({'model': f'{fam}', 'family': fam, **met,
                     'params': json.dumps(search_results[fam]['params'])})
        if verbose:
            print(f'      {fam:<14} test MAE=${met["MAE"]:.4f}  R2={met["R2"]:.3f}')

    # static-vol BSM benchmark
    base_price = meta['base_price']
    rows.insert(0, {'model': 'BSM(vol_21d) static-vol benchmark', 'family': 'bsm',
                    **evaluate_predictions(y[masks['test']], base_price[masks['test']],
                                           mape_min_abs=0.25)})
    cmp = pd.DataFrame(rows).sort_values('MAE').reset_index(drop=True)
    base_mae = cmp['MAE'].iloc[0]
    cmp['MAE_improve_vs_base_%'] = (base_mae - cmp['MAE']) / base_mae * 100
    if verbose:
        print('\n      Option price error (test, vs forward-vol fair price):')
        print(cmp.round(4).to_string(index=False))
    cmp.to_csv(w6config.OUTPUT_DIR / 'price_test_comparison.csv', index=False)

    X_test = X[masks['test']]
    return {'comparison': cmp, 'preds': preds, 'fitted': fitted,
            'y_true': y[masks['test']], 'X_test': X_test, 'feats': feats,
            'runtime_s': time.time() - t0}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Pickle export
# ═══════════════════════════════════════════════════════════════════════════════

def export_models(a1: Dict, a2: Optional[Dict] = None):
    """Serialize every final fitted model to models/*.pkl (+ JSON sidecar).

    The PyTorch LSTM holds a function-local nn.Module class that pickle cannot
    resolve, so it is stored as a reconstruction dict (hyper-parameters +
    learned state_dict) that `load_model` can rebuild.
    """
    exported = []
    for key, model in a1['fitted'].items():
        name = f'vol_{key}'
        if key == 'lstm' and isinstance(model, LSTMVolatilityPredictor):
            payload = _lstm_artifact(model)
        else:
            payload = model
        _dump(name, payload, {
            'track': 'approach1_vol', 'family': key, 'features': a1['feats'],
            'target': TARGET, 'fit_region': 'train+val',
            'test_MAE': _lookup(a1, key), 'params': _params_of(model),
        })
        exported.append(name)
    if a2 is not None:
        for key, model in a2['fitted'].items():
            name = f'price_{key}'
            _dump(name, model, {
                'track': 'approach2_price', 'family': key, 'features': a2['feats'],
                'target': 'label_price', 'fit_region': 'train+val dates',
                'test_MAE': _lookup(a2, key),
            })
            exported.append(name)
    return exported


def _lookup(artifacts: Dict, key: str) -> Optional[float]:
    table = artifacts.get('metrics')
    if table is None or len(table) == 0:
        return None
    hit = table.loc[table['family'] == key] if 'family' in table.columns else None
    if hit is None or len(hit) == 0:
        return None
    return float(hit['MAE'].iloc[0])


def _params_of(model) -> dict:
    if isinstance(model, dict):
        return model.get('params', {})
    try:
        inner = model.named_steps['model'] if hasattr(model, 'named_steps') else model
        return inner.get_params()
    except Exception:
        return {}


def _lstm_artifact(model: 'LSTMVolatilityPredictor') -> dict:
    """Pickle-safe LSTM payload: config + learned weights (state_dict)."""
    state = {}
    if model.model is not None and model.backend == 'torch':
        state = {k: v.cpu().numpy() for k, v in model.model.state_dict().items()}
    return {
        'kind': 'lstm',
        'params': {'seq_len': model.seq_len, 'hidden_size': model.hidden_size,
                   'num_layers': model.num_layers, 'dropout': model.dropout,
                   'epochs': model.epochs, 'batch_size': model.batch_size,
                   'lr': model.lr, 'seed': model.seed},
        'state_dict': state,
    }


def load_model(name: str):
    """Load a pickled model artifact (rebuilding the LSTM from its state_dict)."""
    import joblib
    payload = joblib.load(w6config.MODELS_DIR / f'{name}.pkl')
    if isinstance(payload, dict) and payload.get('kind') == 'lstm':
        p = payload['params']
        m = LSTMVolatilityPredictor(seq_len=p['seq_len'], hidden_size=p['hidden_size'],
                                    num_layers=p['num_layers'], dropout=p['dropout'],
                                    epochs=p['epochs'], batch_size=p['batch_size'],
                                    lr=p['lr'], seed=p['seed'])
        sd = payload.get('state_dict') or {}
        if sd:
            import torch
            torch.manual_seed(m.seed)
            n_feat = sd['lstm.weight_ih_l0'].shape[1]   # (4*hidden, n_feat)
            torch_module = m._build_torch_model(n_feat)
            torch_module.load_state_dict({k: torch.tensor(v) for k, v in sd.items()})
            m.model = torch_module
        return m
    return payload


def _dump(name: str, model, metadata: dict):
    path = w6config.MODELS_DIR / f'{name}.pkl'
    joblib.dump(model, path)
    with open(w6config.MODELS_DIR / f'{name}.json', 'w', encoding='utf-8') as f:
        json.dump({**metadata, 'artifact': str(path)}, f, ensure_ascii=False, indent=2)
    print(f'    exported -> {path.name}')


if __name__ == '__main__':
    print('Approach 1 final training...')
    a1 = run_approach1_final()
    print('\nApproach 2 final training...')
    a2 = run_approach2_final()
    print('\nExporting models...')
    export_models(a1, a2)
