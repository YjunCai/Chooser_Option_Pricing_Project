"""
Week 5 ML Framework -- Dual-Track Orchestration
===============================================
Runs the complete initial ML pipeline for both approaches:

  Approach 1 (volatility prediction + BSM hybrid)
    base features vs enhanced/selected features -> RF / GBDT / XGBoost / LSTM
    -> volatility MAE/RMSE/R2 vs BSM(vol_21d) persistence baseline
    -> Chooser price error under each volatility forecast vs the "fair"
       forward-vol price (Week-3 Rubinstein formula, vectorized)

  Approach 2 (end-to-end supervised pricing)
    (market + contract features) -> Linear / GBDT / MLP-NN
    -> price MAE/RMSE/R2 vs the static-vol BSM benchmark

All splitting is chronological 70/15/15 with a purged gap (see
data_preparation.py) and all labels are strictly forward-looking.
"""

import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import config
from data_preparation import (load_dataset, build_splits, make_chronological_split,
                              make_sequences, report_split)
from targets import add_fwd_vol_targets, add_vix_target, build_contract_frame
from sklearn.ensemble import GradientBoostingRegressor
from feature_engineering_optimization import enhance_features, select_features
from evaluation import (evaluate_predictions, compare_vol_models,
                        summarize_pricing_comparison, format_metric_table)
from models.volatility_models import build_volatility_models, build_lstm_datasets, LSTMVolatilityPredictor
from models.pricing_models import build_pricing_models
from models.bsm_engine import chooser_price_series
from targets import market_state

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.bbox'] = 'tight'

APPROACH2_CONTRACT_FEATURES = ['log_moneyness', 'tenor_years', 'is_call']
PRICING_MAPE_MIN_ABS = 0.25   # % error only meaningful for prices >= $0.25


# ═══════════════════════════════════════════════════════════════════════════════
# Shared data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_weekly_data() -> pd.DataFrame:
    """Daily frame: base features + forward-vol & VIX targets + enhanced set."""
    df = load_dataset()
    df = add_fwd_vol_targets(df)
    df = add_vix_target(df)
    return df


def _drop_incomplete(frame: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    n = len(frame)
    out = frame.dropna(subset=cols).reset_index(drop=True)
    if len(out) < n:
        print(f'      dropped {n - len(out)} rows with NaN features/labels')
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Approach 1
# ═══════════════════════════════════════════════════════════════════════════════

def run_approach1(df: pd.DataFrame, feature_sets: Optional[Dict[str, List[str]]] = None,
                  verbose: bool = True) -> Dict:
    """Volatility prediction (RF/GBDT/XGB/LSTM) + BSM hybrid pricing."""
    print('\n' + '=' * 70)
    print('  APPROACH 1 -- ML Volatility Prediction + BSM Pricing')
    print('=' * 70)

    if feature_sets is None:
        enhanced = enhance_features(df)
        selected, imp = select_features(enhanced,
                                        target=f'fwd_realized_vol_{config.VOL_FORWARD_HORIZON}d')
        feature_sets = {
            'base': config.BASE_FEATURES,
            'enhanced': selected,
        }
        print(f'  enhanced features -> {len(selected)} selected')

    target = f'fwd_realized_vol_{config.VOL_FORWARD_HORIZON}d'
    results = {}

    for set_name, feats in feature_sets.items():
        print(f'\n  --- feature set: {set_name} ({len(feats)} features) ---')
        mod = _drop_incomplete(df.copy(), feats + [target, config.SPOT_COL, config.RATE_COL, 'vol_21d'])
        tr, va, te = make_chronological_split(len(mod), purge_gap=config.PURGE_GAP_DAYS)
        if verbose:
            print(report_split(mod, tr, va, te))

        split = build_splits(mod, feats, target, scale=True)
        y_true = split['test']['y'].values

        # market-state aliases for the BSM hybrid engine (raw, unscaled values)
        market = market_state(mod)
        test_market = market.iloc[split['test']['idx']].reset_index(drop=True)
        baseline_vol = test_market['sigma_21d'].values    # BSM(vol_21d) persistence

        models = build_volatility_models()
        preds = {}
        for key, model in models.items():
            t0 = time.time()
            model.fit(split['train']['X'], split['train']['y'])
            preds[key] = model.predict(split['test']['X'])
            val_mae = evaluate_predictions(split['val']['y'], model.predict(split['val']['X']))['MAE']
            test_mae = evaluate_predictions(y_true, preds[key])['MAE']
            print(f'      {model.name:<28} test MAE={test_mae*100:6.2f}%  '
                  f'val MAE={val_mae*100:6.2f}%  ({time.time()-t0:.1f}s)')

        # LSTM (sequence backend, torch -> windowed-MLP fallback)
        lstm_cfg = config.VOL_MODELS['lstm']
        seq = build_lstm_datasets(split, seq_len=lstm_cfg['seq_len'])
        if seq['train']['X'].shape[0] > 32:
            lstm_model = LSTMVolatilityPredictor(
                seq_len=lstm_cfg['seq_len'], hidden_size=lstm_cfg['hidden_size'],
                num_layers=lstm_cfg['num_layers'], dropout=lstm_cfg['dropout'],
                epochs=lstm_cfg['epochs'], batch_size=lstm_cfg['batch_size'],
                lr=lstm_cfg['lr'], seed=lstm_cfg['seed'])
            print(f'      LSTM backend: {lstm_model.backend}')
            lstm_model.fit(seq['train']['X'], seq['train']['y'])
            preds['lstm'] = lstm_model.predict(seq['test']['X'])
            val_mae = evaluate_predictions(split['val']['y'],
                                           lstm_model.predict(seq['val']['X']))['MAE']
            print(f'      {lstm_model.name:<28} test MAE='
                  f'{evaluate_predictions(y_true, preds["lstm"])["MAE"]*100:6.2f}%  '
                  f'val MAE={val_mae*100:6.2f}%')
        else:
            print('      LSTM skipped: not enough sequence samples')

        # ---- persistence-anchored GBDT (industry-standard vol forecasting) ----
        # predict the RATIO fwd_vol/vol_21d so the model stays anchored to the
        # current vol level and only learns the *change*.
        try:
            ratio_df = mod.copy()
            ratio_df['vol_ratio'] = ratio_df[target] / ratio_df['vol_21d']
            ratio_split = build_splits(ratio_df, feats, 'vol_ratio', scale=True)
            anch = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                             max_depth=3, random_state=config.RANDOM_SEED)
            anch.fit(ratio_split['train']['X'], ratio_split['train']['y'])
            ratio_pred = anch.predict(ratio_split['test']['X'])
            preds['gbdt_anchored'] = test_market['sigma_21d'].values * np.clip(ratio_pred, 0.1, 5.0)
            print('      GBDT-anchored (vol_ratio) test MAE='
                  f'{evaluate_predictions(y_true, preds["gbdt_anchored"])["MAE"]*100:6.2f}%')
        except Exception as e:
            print(f'      GBDT-anchored skipped: {e}')

        # volatility comparison vs persistence + historical-mean baselines
        train_mean_vol = float(mod.iloc[tr]['vol_21d'].mean())
        baselines = {
            'BSM(vol_21d) persistence': baseline_vol,
            'historical mean vol': np.full_like(y_true, train_mean_vol),
        }
        vol_cmp = compare_vol_models(y_true, preds, baselines=baselines)
        results[set_name] = {'vol_comparison': vol_cmp, 'predictions': preds,
                             'y_true': y_true, 'baseline_vol': baseline_vol,
                             'test_rows': test_market}

        # regime diagnostic: chronological split forces extrapolation to a
        # different-vol regime in the test window
        print(f'\n      [regime] train vol_21d mean={mod.iloc[tr]["vol_21d"].mean():.3f} | '
              f'test vol_21d mean={mod.iloc[te]["vol_21d"].mean():.3f}, '
              f'test fwd_vol mean={mod.iloc[te][target].mean():.3f}')

        print(f'\n      Volatility forecast error (test) -- feature set [{set_name}]:')
        print(vol_cmp[['model', 'MAE', 'RMSE', 'R2', 'MAE_improve_vs_base_%']]
              .round(5).to_string(index=False))

        _save(vol_cmp, f'approach1_vol_comparison_{set_name}.csv')

        # ---- BSM hybrid: chooser price error ----
        _, price_cmp = _hybrid_pricing(test_market, preds)
        results[set_name]['price_comparison'] = price_cmp
        _save(price_cmp, f'approach1_price_comparison_{set_name}.csv')

        print(f'\n      Chooser price error (test, vs forward-vol fair price):')
        print(price_cmp[['model', 'MAE', 'RMSE', 'R2', 'MAPE%']].round(4).to_string(index=False))

        # ---- plots ----
        _plot_approach1(set_name, test_market, y_true, baseline_vol, preds, price_cmp)

    return results


def _hybrid_pricing(test_rows: pd.DataFrame, preds: Dict[str, np.ndarray]) -> tuple:
    """Chooser price under each vol forecast vs the forward-vol 'fair' price."""
    fair = chooser_price_series(test_rows, test_rows['sigma_actual'])
    baseline = chooser_price_series(test_rows, test_rows['sigma_21d'])
    price_dict = {k: chooser_price_series(test_rows, v) for k, v in preds.items()}
    cmp = summarize_pricing_comparison(fair, price_dict, baseline_name='BSM(vol_21d)', baseline=baseline)
    return {'fair': fair, 'baseline': baseline, 'models': price_dict}, cmp


# ═══════════════════════════════════════════════════════════════════════════════
# Approach 2
# ═══════════════════════════════════════════════════════════════════════════════

def run_approach2(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """End-to-end supervised option pricing (Linear / GBDT / MLP-NN)."""
    print('\n' + '=' * 70)
    print('  APPROACH 2 -- End-to-End Supervised Option Pricing')
    print('=' * 70)

    print('  building contract grid (moneyness x tenor x type)...')
    cf = build_contract_frame(df)
    feats = config.BASE_FEATURES + APPROACH2_CONTRACT_FEATURES
    cf = _drop_incomplete(cf, feats + ['label_price', 'bsm_vol21_price'])

    # chronological date split over unique dates
    dates = np.array(sorted(cf['date'].unique()))
    tr, va, te = make_chronological_split(len(dates), purge_gap=config.PURGE_GAP_DAYS)
    date_sets = {
        'train': set(dates[tr]),
        'val': set(dates[va]),
        'test': set(dates[te]),
    }
    if verbose:
        print(f'  dates: train {len(date_sets["train"])} / val {len(date_sets["val"])} '
              f'/ test {len(date_sets["test"])}')

    masks = {k: cf['date'].isin(v) for k, v in date_sets.items()}
    X = cf[feats].astype(float)
    y = cf['label_price'].values
    base_price = cf['bsm_vol21_price'].values

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X[masks['train']])
    Xs = pd.DataFrame(scaler.transform(X), columns=feats)

    models = build_pricing_models()
    preds = {}
    for key, model in models.items():
        t0 = time.time()
        model.fit(Xs[masks['train']], y[masks['train']])
        preds[key] = model.predict(Xs[masks['test']])
        print(f'      {model.name:<20} test MAE=${evaluate_predictions(y[masks["test"]], preds[key])["MAE"]:.4f}'
              f'  ({time.time()-t0:.1f}s)')

    # benchmark = static-vol BSM (MAPE restricted to prices >= $0.25)
    rows = []
    rows.append({**{'model': 'BSM(vol_21d) static-vol benchmark'},
                 **evaluate_predictions(y[masks['test']], base_price[masks['test']],
                                        mape_min_abs=PRICING_MAPE_MIN_ABS)})
    for key, m in models.items():
        rows.append({**{'model': m.name},
                     **evaluate_predictions(y[masks['test']], preds[key],
                                            mape_min_abs=PRICING_MAPE_MIN_ABS)})
    cmp = pd.DataFrame(rows).sort_values('MAE').reset_index(drop=True)
    base_mae = rows[0]['MAE']
    cmp['MAE_improve_vs_base_%'] = (base_mae - cmp['MAE']) / base_mae * 100

    print('\n      Option price error (test, vs forward-vol fair price):')
    print(cmp.round(4).to_string(index=False))
    _save(cmp, 'approach2_price_comparison.csv')

    _plot_approach2(cf, masks, y, preds, cmp)

    return {'comparison': cmp, 'predictions': preds, 'y_true': y[masks['test']],
            'cf': cf, 'masks': masks}


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestration + helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save(df: pd.DataFrame, name: str):
    df.to_csv(config.OUTPUT_DIR / name, index=False)


_VOL_NAME_TO_KEY = {
    'RandomForest': 'rf', 'GBDT': 'gbdt', 'XGBoost': 'xgb', 'LSTM': 'lstm',
    'GBDT-anchored': 'gbdt_anchored',
}


def _plot_approach1(set_name, test_rows, y_true, baseline_vol, preds, price_cmp):
    dates = pd.to_datetime(test_rows['date'])
    best = None
    for name in price_cmp['model']:
        if name in _VOL_NAME_TO_KEY and _VOL_NAME_TO_KEY[name] in preds:
            best = _VOL_NAME_TO_KEY[name]
            break

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(dates, y_true * 100, 'k-', lw=1.2, label='actual (fwd vol)')
    ax.plot(dates, baseline_vol * 100, '--', lw=1.0, color='grey', label='BSM vol_21d (baseline)')
    if best:
        ax.plot(dates, np.asarray(preds[best]) * 100, lw=1.0, alpha=0.85,
                label=f'ML forecast ({price_cmp.iloc[0]["model"]})')
    ax.set_title(f'Volatility forecast vs actual -- [{set_name}]')
    ax.set_ylabel('annualized vol (%)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    mae = price_cmp.set_index('model')['MAE']
    cols = ['#b2182b' if 'BSM' in i else '#2166ac' for i in mae.index]
    mae.plot(kind='bar', ax=ax, color=cols)
    ax.set_title('Chooser price MAE (vs forward-vol fair price)')
    ax.set_ylabel('MAE ($)'); ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    fig.savefig(config.ASSETS_DIR / f'fig_w5_approach1_{set_name}.png')
    plt.close(fig)


def _plot_approach2(cf, masks, y, preds, cmp):
    best_row = cmp.iloc[0]
    best_name = best_row['model']
    key = None
    for k, m in build_pricing_models().items():
        if m.name == best_name:
            key = k
            break
    if key is None:
        # best row is the static-vol benchmark -> use the best ML model
        for row_name in cmp['model']:
            for k, m in build_pricing_models().items():
                if m.name == row_name:
                    key = k
                    break
            if key is not None:
                break

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    yt = y[masks['test']]
    ax.scatter(yt, preds[key], s=6, alpha=0.5, c='#2166ac')
    lim = [yt.min(), yt.max()]
    ax.plot(lim, lim, 'r--', lw=1)
    ax.set_title(f'End-to-end pricing: predicted vs fair price (test, {best_name})')
    ax.set_xlabel('fair price ($)'); ax.set_ylabel('predicted price ($)')

    ax = axes[1]
    mae = cmp.set_index('model')['MAE']
    cols = ['#b2182b' if 'BSM' in i else '#2166ac' for i in mae.index]
    mae.plot(kind='bar', ax=ax, color=cols)
    ax.set_title('Option price MAE (vs forward-vol fair price)')
    ax.set_ylabel('MAE ($)'); ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    fig.savefig(config.ASSETS_DIR / 'fig_w5_approach2_pricing.png')
    plt.close(fig)


def run_framework(verbose: bool = True) -> Dict:
    t0 = time.time()
    print('Loading weekly data...')
    df = load_weekly_data()
    res1 = run_approach1(df, verbose=verbose)
    res2 = run_approach2(df, verbose=verbose)
    print(f'\nTotal runtime: {time.time()-t0:.1f}s')
    return {'approach1': res1, 'approach2': res2}


if __name__ == '__main__':
    run_framework()
