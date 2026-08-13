"""
Live-Market Metrics vs Week-6 Targets -- Week 6
===============================================
Re-prices the SAME 2026-07-27 option snapshot (595 contracts) used by Week 4/5
using the Week-6 *tuned* volatility models, and recomputes the three live
metrics that the Week 5 report set as Week 6 targets:

    实盘 MAE (ATM+OTM)          target < $1.00
    ATM IV 溢价                  target < 2%
    OTM 看跌定价偏差             target > -30%

Method (mirrors week5/compute_live_metrics.py):
  1. Reconstruct the 16-dim feature vector for the snapshot date from freshly
     fetched JPM / VIX / ^IRX daily data.
  2. Predict the snapshot volatility input with each tuned model (pipelines
     already refit on train+val during hyper-parameter selection).
  3. Reprice all 595 contracts with that volatility (same S, r, q, T, K).
  4. Recompute the three metrics and compare to Week 6 targets.
"""

import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import w6config
from data_preparation import load_dataset
from targets import add_fwd_vol_targets
from chooser_option_pricer import bs_call, bs_put
from scipy.stats import norm

SNAPSHOT_DATE = w6config.SNAPSHOT_DATE
FEATS = list(w6config.BASE_FEATURES)


def _load_clean(path):
    d = pd.read_csv(path, index_col=0, parse_dates=False)
    d.index = pd.to_datetime(d.index.str.split(' ').str[0])
    return d


def snapshot_features() -> pd.DataFrame:
    """Reconstruct the week-2 feature frame; the snapshot is its last row."""
    jpm = _load_clean('d:/tmp/jpm_daily.csv')
    vix = _load_clean('d:/tmp/vix_daily.csv')
    irx = _load_clean('d:/tmp/irx_daily.csv')

    a = pd.DataFrame(index=jpm.index)
    a['close'] = jpm['j_Close']; a['high'] = jpm['j_High']; a['low'] = jpm['j_Low']
    a['volume'] = jpm['j_Volume']
    a['vix'] = vix['v_Close'].reindex(jpm.index)
    a['rate'] = irx['r_Close'].reindex(jpm.index)
    a = a.dropna()

    r = np.log(a['close'] / a['close'].shift(1))
    v = (a['high'] - a['low']) / a['close']

    f = pd.DataFrame(index=a.index)
    f[w6config.SPOT_COL] = a['close']               # close_jpm (mirror week2 dataset)
    f[w6config.VIX_COL] = a['vix']                  # close_vix
    f[w6config.RATE_COL] = a['rate']                # value_treasury_3mo
    f['daily_return'] = r
    for w in (5, 21, 63):
        f[f'vol_{w}d'] = r.rolling(w).std() * np.sqrt(w6config.TRADING_DAYS_PER_YEAR)
    f['high_low_spread'] = v
    f['volume_change_1d'] = np.log(a['volume'] / a['volume'].shift(1)).replace([np.inf, -np.inf], 0)
    f['sma_ratio_21'] = a['close'] / a['close'].rolling(21).mean()
    f['vix_change_1d'] = a['vix'].diff()
    f['vix_jpm_corr_21d'] = r.rolling(21, min_periods=16).corr(a['vix'].diff())
    f['vix_jpm_cross_1d'] = -r * a['vix'].diff()
    f['rate_change_1d_bps'] = a['rate'].diff() * 100
    f['rate_momentum_5d_bps'] = a['rate'].diff(5) * 100
    roll_min = a['vix'].rolling(252, min_periods=20).min()
    roll_max = a['vix'].rolling(252, min_periods=20).max()
    f['sentiment_score'] = (1 - (a['vix'] - roll_min) / (roll_max - roll_min).replace(0, np.nan)).fillna(0.5).clip(0, 1)
    f['jpm_vol_ratio'] = f['vol_5d'] / f['vol_21d']
    f['vix_ratio'] = a['vix'] / (f['vol_21d'] * 100)

    try:
        import yfinance as yf
        div = yf.Ticker('JPM').dividends
        if hasattr(div.index, 'tz'):
            div.index = div.index.tz_localize(None)
        div = div[div.index.normalize() <= pd.Timestamp(SNAPSHOT_DATE)]
        if len(div) >= 8:
            yoy = div.groupby(div.index.year).sum().pct_change().iloc[-1]
            dps = float(yoy) if np.isfinite(yoy) else 0.0
        else:
            dps = 0.0
    except Exception:
        dps = 0.0
    f['dps_growth_rate'] = dps

    return f.loc[:SNAPSHOT_DATE]


def load_snapshot_contracts() -> pd.DataFrame:
    COL = ['expiry', 'T', 'strike', 'otype', 'mkt_price', 'bid', 'ask', 'YahooIV',
           'volume', 'open_int', 'bsm_price', 'mid', 'abs_err', 'sq_err', 'bias',
           'bias_pct', 'mape_pct', 'moneyness', 'log_moneyness', 'tenor_bucket',
           'implied_iv', 'iv_vol21_gap', 'par_S', 'par_r', 'par_q', 'par_vol_21d']
    opt = pd.read_csv(w6config.WEEK4_DIR / 'data' / 'jpm_options_595contracts.csv',
                      encoding='utf-8-sig', header=0)
    opt.columns = COL[:len(opt.columns)]
    return opt


def price_all(opt: pd.DataFrame, sigma: float, S, r, q) -> np.ndarray:
    T = opt['T'].values; K = opt['strike'].values; t = opt['otype'].values
    s = np.maximum(sigma, 1e-10)
    d1 = (np.log(np.maximum(S / K, 1e-10)) + (r - q + 0.5 * s ** 2) * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    out = np.empty(len(opt))
    c = t == 'call'
    out[c] = S * np.exp(-q * T[c]) * norm.cdf(d1[c]) - K[c] * np.exp(-r * T[c]) * norm.cdf(d2[c])
    out[~c] = K[~c] * np.exp(-r * T[~c]) * norm.cdf(-d2[~c]) - S * np.exp(-q * T[~c]) * norm.cdf(-d1[~c])
    return out


def snapshot_regime_dummies(snap: pd.Series, snap_frame: pd.DataFrame) -> dict:
    """One-hot regime for the snapshot using the SAME thresholds as training.

    Thresholds (vol_median / q25 / q75) come from the full 2018-2024 feature
    dataset, matching regime.py; the SMA_252 is computed from the trailing
    history available at the snapshot date (backward-looking).
    """
    df = load_dataset()
    vol_median = df['vol_21d'].median()
    vol_q75 = df['vol_21d'].quantile(0.75)
    vol_q25 = df['vol_21d'].quantile(0.25)
    sma = snap_frame['close_jpm'].rolling(252, min_periods=60).mean().iloc[-1]

    s, v = float(snap['close_jpm']), float(snap['vol_21d'])
    label = 'NORMAL'
    if s > sma and v <= vol_median:
        label = 'BULL'
    elif s < sma and v > vol_median:
        label = 'BEAR'
    elif v > vol_q75:
        label = 'HIGH_VOL'
    elif v < vol_q25 and float(snap['sentiment_score']) > 0.5:
        label = 'CALM'
    return {c: (1.0 if c == f'regime_{label}' else 0.0)
            for c in w6config.REGIME_DUMMY_FEATURES}


def predict_snapshot_vols(search_results: dict, snap: pd.Series,
                          snap_frame: pd.DataFrame) -> dict:
    """Predict the snapshot volatility input with every tuned model.

    The snapshot row is built from the exact 19 features the tuned models were
    trained on (15 selected market features + 4 regime one-hots).
    """
    feats = search_results['_meta']['feats']
    base_feats = [c for c in feats if c in snap.index]
    row_vals = {c: float(snap[c]) for c in base_feats}
    row_vals.update(snapshot_regime_dummies(snap, snap_frame))
    row = pd.DataFrame([row_vals], columns=feats)

    preds = {}
    for key in ('rf', 'gbdt', 'xgb', 'gbdt_anchored'):
        if key not in search_results:
            continue
        pipe = search_results[key]['gs'].best_estimator_
        if key == 'gbdt_anchored':
            ratio = np.clip(pipe.predict(row)[0], 0.1, 5.0)
            preds[key] = float(snap['vol_21d']) * ratio
        else:
            preds[key] = float(pipe.predict(row)[0])
    preds['vol_21d'] = float(snap['vol_21d'])
    preds['vol_21d_snapshot'] = 0.202097      # week4 official snapshot input
    return preds


def predict_snapshot_vix(snap: pd.Series, feats: List[str]) -> float:
    """VIX-proxy model: tuned XGBoost on VIX-at-t+1 target -> VIX/100."""
    df = add_fwd_vol_targets(load_dataset())
    df = df.dropna(subset=feats + ['vol_21d']).reset_index(drop=True)
    df = df.copy()
    df['vix_target'] = df[w6config.VIX_COL].shift(-w6config.VIX_TARGET_SHIFT)
    df = df.dropna(subset=feats + ['vix_target']).reset_index(drop=True)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    try:
        from xgboost import XGBRegressor
        base = dict(w6config.SEARCH_SPACES['xgb'])
        m = XGBRegressor(random_state=w6config.RANDOM_SEED, verbosity=0,
                         **{k: base[k][0] for k in base})
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        m = GradientBoostingRegressor(random_state=w6config.RANDOM_SEED)
    model = Pipeline([('scale', StandardScaler()), ('model', m)])
    model.fit(df[feats].astype(float).values, df['vix_target'].values)
    vix = float(np.clip(model.predict([snap[feats]])[0], 5.0, 60.0))
    return vix / 100.0


def compute_live_metrics(verbose: bool = True) -> pd.DataFrame:
    snap_frame = snapshot_features()
    snap = snap_frame.iloc[-1]
    if verbose:
        print('snapshot vol_21d =', round(float(snap['vol_21d']), 4))
        print('snapshot VIX     =', round(float(snap[w6config.VIX_COL]), 2))

    opt = load_snapshot_contracts()
    S = float(opt['par_S'].iloc[0]); r = float(opt['par_r'].iloc[0]); q = float(opt['par_q'].iloc[0])

    import pickle
    with open(w6config.OUTPUT_DIR / 'vol_search_results.pkl', 'rb') as f:
        search_results = pickle.load(f)
    preds = predict_snapshot_vols(search_results, snap, snap_frame)
    preds['vix_proxy'] = predict_snapshot_vix(snap, FEATS)
    if verbose:
        print('\n=== ML volatility forecasts for snapshot date (tuned) ===')
        for k, v in preds.items():
            print(f'  {k:<18} sigma = {v*100:.2f}%')

    opt['atm'] = opt['log_moneyness'].abs() < 0.05
    opt['itm'] = ((opt['otype'] == 'call') & (opt['log_moneyness'] < 0)) | \
                 ((opt['otype'] == 'put') & (opt['log_moneyness'] > 0))
    atm_otm = opt[~opt['itm']]
    otm_puts = opt[(opt['otype'] == 'put') & (~opt['itm'])]
    atm_iv = float(opt.loc[opt['atm'], 'implied_iv'].mean())

    rows = []
    for name, sigma in preds.items():
        px = price_all(opt, sigma, S, r, q)
        mae = float(np.mean(np.abs(px[atm_otm.index] - atm_otm['mkt_price'].values)))
        rmse = float(np.sqrt(np.mean((px[atm_otm.index] - atm_otm['mkt_price'].values) ** 2)))
        atm_gap = (atm_iv - sigma) * 100
        put_bias = float(np.mean((px[otm_puts.index] - otm_puts['mkt_price'].values)
                                 / otm_puts['mkt_price'].values) * 100)
        # group_bias = mean % bias over the ATM+OTM group (the metric the
        # Week 4/5 reports call "OTM 看跌定价偏差")
        group_bias = float(np.mean((px[atm_otm.index] - atm_otm['mkt_price'].values)
                                   / atm_otm['mkt_price'].values) * 100)
        rows.append({'model': name, 'sigma_%': round(sigma * 100, 2),
                     'MAE_$': round(mae, 3), 'RMSE_$': round(rmse, 3),
                     'ATM_IV_gap_%': round(atm_gap, 2),
                     'OTM_put_bias_%': round(put_bias, 1),
                     'group_bias_%': round(group_bias, 1)})
    res = pd.DataFrame(rows).sort_values('MAE_$')
    res.to_csv(w6config.OUTPUT_DIR / 'live_metrics_w6.csv', index=False)

    if verbose:
        print('\n=== Live-market metrics (Week-6 tuned, 2026-07-27 snapshot) ===')
        print(f'  Week 6 targets : MAE<${w6config.WEEK6_TARGETS["live_MAE_$"]} | '
              f'ATM<{w6config.WEEK6_TARGETS["atm_iv_premium_%"]}% | '
              f'OTM>{w6config.WEEK6_TARGETS["otm_put_bias_%"]}%')
        print(f'  Week 4 baseline: MAE=$1.44 | ATM=4.9% | OTM=-65.7%')
        print(res.to_string(index=False))
    return res


def plot_live_metrics(res: pd.DataFrame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    names = res['model'].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    bar_col = ['#b2182b' if 'vol_21d' in n else '#2166ac' for n in names]

    axes[0].bar(range(len(names)), res['MAE_$'], color=bar_col)
    axes[0].set_title('实盘定价 MAE ($)')
    axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels(names, rotation=25, fontsize=7)
    axes[0].axhline(1.00, color='grey', ls='--', lw=1, label='Week6 目标 $1.00')
    axes[0].legend(fontsize=7)

    axes[1].bar(range(len(names)), res['ATM_IV_gap_%'], color=bar_col)
    axes[1].set_title('ATM IV 溢价 (%)')
    axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, rotation=25, fontsize=7)
    axes[1].axhline(2.0, color='grey', ls='--', lw=1, label='Week6 目标 2%')
    axes[1].legend(fontsize=7)

    axes[2].bar(range(len(names)), res['OTM_put_bias_%'], color=bar_col)
    axes[2].set_title('OTM 看跌定价偏差 (%)')
    axes[2].set_xticks(range(len(names))); axes[2].set_xticklabels(names, rotation=25, fontsize=7)
    axes[2].axhline(-30, color='grey', ls='--', lw=1, label='Week6 目标 -30%')
    axes[2].legend(fontsize=7)

    fig.suptitle('Week 6 实盘指标: 调优模型 vs Week 6 目标 (2026-07-27 快照, 595 合约)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(w6config.ASSETS_DIR / 'fig_w6_live_metrics.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    res = compute_live_metrics()
    plot_live_metrics(res)
