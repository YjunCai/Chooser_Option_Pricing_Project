# -*- coding: utf-8 -*-
"""
Compute Week-5 ML-enhanced live-market metrics on the SAME 595-contract option
snapshot used by Week 4, so Table 8 of the report can be filled with real values.

Method
------
1. Reconstruct the 16-dim feature vector for the snapshot date (2026-07-27)
   from freshly fetched JPM / VIX / ^IRX daily data (formulas mirror week2).
2. Train ML volatility models on the full 2018-2024 feature dataset.
3. Predict the volatility input for the snapshot date.
4. Reprice all option contracts with the ML volatility (same S, r, q, T, K).
5. Recompute the three live metrics vs Week-4 baseline:
      - 实盘 MAE     (BSM(ML sigma) vs market price, ATM+OTM)
      - ATM IV 溢价  (market ATM implied vol - ML sigma)
      - OTM 看跌偏差 (mean % price bias on OTM puts)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

WEEK5 = Path(__file__).resolve().parent
WEEK3 = WEEK5.parent / 'week3'
WEEK4 = WEEK5.parent / 'week4'
sys.path.insert(0, str(WEEK3))
sys.path.insert(0, str(WEEK5))

import config  # noqa: E402
from data_preparation import load_dataset, build_splits  # noqa: E402
from models.volatility_models import build_volatility_models  # noqa: E402
from chooser_option_pricer import bs_call, bs_put  # noqa: E402

SNAPSHOT_DATE = '2026-07-27'
FEATS = config.BASE_FEATURES


# ── 1. load recent daily data, align, compute week2-style features ────────────
def _load_clean(path):
    d = pd.read_csv(path, index_col=0, parse_dates=False)
    d.index = pd.to_datetime(d.index.str.split(' ').str[0])
    return d


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
f['daily_return'] = r
for w in (5, 21, 63):
    f[f'vol_{w}d'] = r.rolling(w).std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
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

# dps growth rate from JPM dividends (year-over-year sum)
try:
    import yfinance as yf
    div = yf.Ticker('JPM').dividends
    div = div[div.index.tz_localize(None).normalize() <= SNAPSHOT_DATE] if hasattr(div.index, 'tz') else div
    if len(div) >= 8:
        yoy = div.groupby(div.index.year).sum().pct_change().iloc[-1]
        dps = float(yoy) if np.isfinite(yoy) else 0.0
    else:
        dps = 0.0
except Exception:
    dps = 0.0
f['dps_growth_rate'] = dps

snap = f.loc[:SNAPSHOT_DATE].iloc[-1]
print('snapshot feature row:')
print(snap.round(4).to_string())


# ── 2. train ML volatility models on full 2018-2024 data ──────────────────────
df = load_dataset()
target = f'fwd_realized_vol_{config.VOL_FORWARD_HORIZON}d'
fwd = df['daily_return'].shift(-1)
df[target] = (fwd.rolling(config.VOL_FORWARD_HORIZON).std()
              .shift(-(config.VOL_FORWARD_HORIZON - 1))
              .mul(np.sqrt(config.TRADING_DAYS_PER_YEAR)))

mod = df.dropna(subset=FEATS + [target]).reset_index(drop=True)
split = build_splits(mod, FEATS, target, scale=True)

X_snap = split['scaler'].transform(pd.DataFrame([snap[FEATS]], columns=FEATS))

preds = {}
for key, model in build_volatility_models().items():
    model.fit(split['train']['X'], split['train']['y'])
    preds[key] = float(model.predict(X_snap)[0])

# persistence-anchored GBDT (predict vol ratio anchored to current vol_21d)
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
ratio_mod = mod.copy()
ratio_mod['vol_ratio'] = ratio_mod[target] / ratio_mod['vol_21d']
rsplit = build_splits(ratio_mod, FEATS, 'vol_ratio', scale=True)
anch = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                 random_state=config.RANDOM_SEED)
anch.fit(rsplit['train']['X'], rsplit['train']['y'])
ratio_snap = float(anch.predict(X_snap)[0])
vol21_snap = float(snap['vol_21d'])
preds['gbdt_anchored'] = vol21_snap * np.clip(ratio_snap, 0.1, 5.0)
preds['vol_21d'] = vol21_snap          # my-fetched rolling vol
preds['vol_21d_snapshot'] = 0.202097   # week4 official snapshot input (reproduces $1.44)

print('\n=== ML volatility forecasts for snapshot date ===')
for k, v in preds.items():
    print(f'  {k:<16} sigma = {v*100:.2f}%')


# ── 3. reprice the 595 contracts with each volatility ─────────────────────────
COL = ['expiry', 'T', 'strike', 'otype', 'mkt_price', 'bid', 'ask', 'YahooIV',
       'volume', 'open_int', 'bsm_price', 'mid', 'abs_err', 'sq_err', 'bias',
       'bias_pct', 'mape_pct', 'moneyness', 'log_moneyness', 'tenor_bucket',
       'implied_iv', 'iv_vol21_gap', 'par_S', 'par_r', 'par_q', 'par_vol_21d']
opt = pd.read_csv(WEEK4 / 'data' / 'jpm_options_595contracts.csv',
                  encoding='utf-8-sig', header=0)
opt.columns = COL[:len(opt.columns)]

from scipy.stats import norm  # noqa: E402

S = float(opt['par_S'].iloc[0]); r = float(opt['par_r'].iloc[0]); q = float(opt['par_q'].iloc[0])


def price_all(sigma):
    """Vectorized BSM price (T always > 0 in this snapshot)."""
    T = opt['T'].values; K = opt['strike'].values; t = opt['otype'].values
    s = np.maximum(sigma, 1e-10)
    d1 = (np.log(np.maximum(S / K, 1e-10)) + (r - q + 0.5 * s ** 2) * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    out = np.empty(len(opt))
    c = t == 'call'
    out[c] = S * np.exp(-q * T[c]) * norm.cdf(d1[c]) - K[c] * np.exp(-r * T[c]) * norm.cdf(d2[c])
    out[~c] = K[~c] * np.exp(-r * T[~c]) * norm.cdf(-d2[~c]) - S * np.exp(-q * T[~c]) * norm.cdf(-d1[~c])
    return out


opt['atm'] = opt['log_moneyness'].abs() < 0.05
opt['itm'] = ((opt['otype'] == 'call') & (opt['log_moneyness'] < 0)) | \
             ((opt['otype'] == 'put') & (opt['log_moneyness'] > 0))
atm_otm = opt[~opt['itm']]
otm_puts = opt[(opt['otype'] == 'put') & (~opt['itm'])]
atm_iv = float(opt.loc[opt['atm'], 'implied_iv'].mean())

rows = []
for name, sigma in preds.items():
    px = price_all(sigma)
    mae = float(np.mean(np.abs(px[atm_otm.index] - atm_otm['mkt_price'].values)))
    rmse = float(np.sqrt(np.mean((px[atm_otm.index] - atm_otm['mkt_price'].values) ** 2)))
    atm_gap = (atm_iv - sigma) * 100
    put_bias = float(np.mean((px[otm_puts.index] - otm_puts['mkt_price'].values)
                             / otm_puts['mkt_price'].values) * 100)
    group_bias = float(np.mean((px[atm_otm.index] - atm_otm['mkt_price'].values)
                               / atm_otm['mkt_price'].values) * 100)
    rows.append({'model': name, 'sigma_%': round(sigma * 100, 2),
                 'MAE_$': round(mae, 3), 'RMSE_$': round(rmse, 3),
                 'ATM_IV_gap_%': round(atm_gap, 2),
                 'OTM_put_bias_%': round(put_bias, 1),
                 'group_bias_%': round(group_bias, 1)})

res = pd.DataFrame(rows).sort_values('MAE_$')
print('\n=== Live-market metrics (ML-enhanced vs Week-4 baseline) ===')
print('  Week-4 baseline (BSM vol_21d=20.21%): MAE=$1.44, ATM IV premium=4.9%, OTM put bias=-65.7%')
print(res.to_string(index=False))
res.to_csv(WEEK5 / 'output' / 'live_metrics_ml_vs_baseline.csv', index=False)
print('\nsaved -> output/live_metrics_ml_vs_baseline.csv')

# ── 4. figure: live metrics comparison ────────────────────────────────────────
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

names = res['model'].tolist()
mae = res['MAE_$'].values
atm_gap = res['ATM_IV_gap_%'].values
bias = res['group_bias_%'].values

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
bar_col = ['#b2182b' if 'vol_21d' in n else '#2166ac' for n in names]

axes[0].bar(range(len(names)), mae, color=bar_col)
axes[0].set_title('实盘定价 MAE ($)')
axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels(names, rotation=25, fontsize=7)
axes[0].axhline(1.00, color='grey', ls='--', lw=1, label='Week6 目标 $1.00')
axes[0].legend(fontsize=7)

axes[1].bar(range(len(names)), atm_gap, color=bar_col)
axes[1].set_title('ATM IV 溢价 (%)')
axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, rotation=25, fontsize=7)
axes[1].axhline(2.0, color='grey', ls='--', lw=1, label='Week6 目标 2%')
axes[1].legend(fontsize=7)

axes[2].bar(range(len(names)), bias, color=bar_col)
axes[2].set_title('OTM 看跌定价偏差 (%)')
axes[2].set_xticks(range(len(names))); axes[2].set_xticklabels(names, rotation=25, fontsize=7)
axes[2].axhline(-30, color='grey', ls='--', lw=1, label='Week6 目标 -30%')
axes[2].legend(fontsize=7)

fig.suptitle('Week 5 实盘指标: ML 增强 vs Week 4 BSM 基线 (2026-07-27 快照, 595 合约)',
             fontsize=13)
fig.tight_layout()
fig.savefig(WEEK5 / 'assets' / 'fig_w5_live_metrics.png', bbox_inches='tight')
plt.close(fig)
print('saved -> assets/fig_w5_live_metrics.png')
