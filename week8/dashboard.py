"""
Week 8 Dashboard -- Price Trends & Performance Metrics
======================================================
Adds the two blocks the Week-7 prototype was still missing, completing the
visualization dashboard deliverable:

  1. Historical price-trend series: the chooser price priced over the full
     2018-2024 feature dataset under the BSM(vol_21d) baseline and the best ML
     volatility models (XGBoost = live pick, VIX-proxy = IV-aligned pick), so
     the tool can show "price trends over time" instead of only point prices.
  2. Performance-metric charts: the Week-6 held-out test metrics and live
     snapshot metrics rendered as bar charts (approach-1 chooser pricing,
     approach-1 vol prediction, live snapshot targets, approach-2 end-to-end).

Both blocks read the same Week-6 artifacts the reports quote, so the dashboard
numbers always match the final report.
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import w8config as cfg
import tool_engine as te


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Historical price-trend series
# ═══════════════════════════════════════════════════════════════════════════════

def historical_price_series(model_families=('vol_xgb', 'vol_vix_proxy'),
                            max_rows: int = None,
                            spot: float = None, t1: float = None,
                            T2: float = None,
                            extend_live: bool = True) -> pd.DataFrame:
    """
    Price the chooser over the week-2 feature dataset (2018-2024) with the
    BSM(vol_21d) baseline vs each ML-vol model, returning a per-day frame:

        date, spot, vol_21d, sigma_<fam>..., price_<fam>..., price_BSM

    `max_rows` downsamples the dense daily grid for chart readability.
    If `spot` is given, all rows are priced at a fixed reference spot instead
    of the historical one (used for the ATM canonical trend).
    If `extend_live`, ALL fetched live feature rows after the training dataset's
    last date (2024-12-30) are appended (not just the single latest row), so the
    trend chart extends continuously to today instead of drawing a straight line
    across a data gap. Rows after 2024-12-30 are out-of-sample for the trained
    models; callers can shade the extension region via the 'oos' column.
    """
    from data_preparation import load_dataset
    train = load_dataset().reset_index(drop=True)
    feat = pd.concat([train, te.regime_dummies(train)], axis=1).copy()
    feat = feat.dropna(subset=['vol_21d']).reset_index(drop=True)
    feat['oos'] = 0.0

    if extend_live:
        feat = _extend_live_history(feat)

    if max_rows is not None and len(feat) > max_rows:
        idx = np.linspace(0, len(feat) - 1, max_rows).astype(int)
        feat = feat.iloc[idx].reset_index(drop=True)

    K = cfg.CHOOSER_PARAMS['K']
    t1 = cfg.CHOOSER_PARAMS['t1'] if t1 is None else t1
    T2 = cfg.CHOOSER_PARAMS['T2'] if T2 is None else T2
    spot_arr = np.full(len(feat), float(spot)) if spot is not None else feat[cfg.SPOT_COL].values.astype(float)
    rate_arr = feat[cfg.RATE_COL].values.astype(float) / 100.0
    q = cfg.Q_YIELD
    vol21 = feat['vol_21d'].values.astype(float)

    out = pd.DataFrame({
        'date': pd.to_datetime(feat[cfg.DATE_COL]),
        'spot': spot_arr,
        'vol_21d': vol21,
        'oos': feat['oos'].astype(float).values if 'oos' in feat.columns
               else np.zeros(len(feat)),
    })

    for fam in model_families:
        key = fam.replace('vol_', '')
        payload, feats, _ = te.load_vol_model(fam)
        X = feat[feats].astype(float)
        sigma = te.predict_sigma_from_pipe(payload, X)
        out[f'sigma_{key}'] = sigma
        out[f'price_{key}'] = te.price_chooser_vec(spot_arr, K, t1, T2, rate_arr, q, sigma)

    out['price_BSM'] = te.price_chooser_vec(spot_arr, K, t1, T2, rate_arr, q, vol21)
    return out


def _extend_live_history(feat: pd.DataFrame,
                         last_train_date: str = '2024-12-30') -> pd.DataFrame:
    """
    Append every fetched live feature row AFTER `last_train_date` to `feat`, so
    the trend series is continuous through 2025-2026 (out-of-sample) instead of
    ending with a single point. Marks extension rows with `oos = 1`.

    Uses a 5-year fetch window so rolling features (63d vol, 252d sentiment) are
    fully warmed by the time the extension begins. Falls back to the cached
    snapshot if yfinance is unavailable.
    """
    try:
        market = te.fetch_market_history(period='5y')
        ext = te.build_features(market)
        ext = pd.concat([ext, te.regime_dummies(ext)], axis=1).dropna()
        ext.index.name = cfg.DATE_COL      # rename index BEFORE reset_index
        ext = ext.reset_index()
        if len(ext) == 0:
            return feat
        cut = pd.Timestamp(last_train_date)
        add = ext[pd.to_datetime(ext[cfg.DATE_COL]) > cut].copy()
        if len(add) == 0:
            return feat
        add['oos'] = 1.0
        return pd.concat([feat, add], ignore_index=True)
    except Exception:
        return feat


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Performance metrics (Week-6 artifacts)
# ═══════════════════════════════════════════════════════════════════════════════

def performance_metrics() -> dict:
    """Return the Week-6 metric frames the dashboard charts, keyed by track:
    'chooser' (approach-1 chooser test), 'vol' (approach-1 vol test),
    'live' (2026-07-27 snapshot), 'end2end' (approach-2 price)."""
    chooser = pd.read_csv(cfg.CHOOSER_TEST_CSV)
    vol = pd.read_csv(cfg.VOL_TEST_CSV)
    live = pd.read_csv(cfg.LIVE_METRICS_CSV)
    cons = pd.read_csv(cfg.CONSOLIDATED_CSV)
    end2end = cons[cons['track'] == 'A2_price'][['model', 'MAE', 'RMSE', 'R2']].copy()
    return {'chooser': chooser, 'vol': vol, 'live': live, 'end2end': end2end}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Figures
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_fonts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def _shade_oos(ax, series: pd.DataFrame, ylabel: str):
    """Shade the out-of-sample extension region (2025+) and mark the boundary."""
    if 'oos' not in series.columns or not series['oos'].astype(float).any():
        return
    boundary = pd.Timestamp('2024-12-30')
    ax.axvline(boundary, color='grey', ls='--', lw=1.1, alpha=0.7)
    ax.axvspan(boundary, series['date'].max(), color='gold', alpha=0.08)
    ax.text(boundary, 0.97, '样本外延伸 (2025–)', transform=ax.get_xaxis_transform(),
            ha='left', va='top', fontsize=9, color='#8a6d1a')


def plot_price_trend(series: pd.DataFrame, out_path):
    """Chooser price over time: BSM baseline vs ML-vol models."""
    plt = _setup_fonts()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    cols = [c for c in series.columns if c.startswith('price_')]
    order = ['price_BSM'] + [c for c in cols if c != 'price_BSM']
    labels = {'price_BSM': 'BSM(vol_21d) 基线',
              'price_xgb': 'ML · XGBoost (实盘首选)',
              'price_vix_proxy': 'ML · VIX-proxy (IV 对齐)'}
    colors = {'price_BSM': '#2166ac', 'price_xgb': '#d73027', 'price_vix_proxy': '#fdae61'}
    for c in order:
        if c in series.columns:
            ax.plot(series['date'], series[c], lw=1.6, color=colors.get(c, 'grey'),
                    label=labels.get(c, c))
    _shade_oos(ax, series, 'Chooser 价格 ($)')
    ax.set_xlabel('日期')
    ax.set_ylabel('Chooser 价格 ($)')
    ax.set_title('双轨定价：BSM(vol_21d) 与最优 ML 模型的 Chooser 价格趋势 (2018–2026，2025 起样本外)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_sigma_trend(series: pd.DataFrame, out_path):
    """Volatility inputs over time: BSM(vol_21d) vs ML predicted sigma."""
    plt = _setup_fonts()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = {'vol_21d': '#2166ac', 'sigma_xgb': '#d73027', 'sigma_vix_proxy': '#fdae61'}
    labels = {'vol_21d': 'vol_21d (BSM 基线)',
              'sigma_xgb': 'XGBoost 预测 σ',
              'sigma_vix_proxy': 'VIX-proxy 预测 σ'}
    for c, label in labels.items():
        if c in series.columns:
            ax.plot(series['date'], series[c] * 100, lw=1.4, color=colors.get(c, 'grey'),
                    label=label)
    _shade_oos(ax, series, '年化波动率 (%)')
    ax.set_xlabel('日期')
    ax.set_ylabel('年化波动率 (%)')
    ax.set_title('波动率输入：BSM(vol_21d) vs ML 预测 (2018–2026，2025 起样本外)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_chooser_metrics(metrics: dict, out_path):
    """Approach-1 chooser pricing MAE/RMSE bar chart (synthetic test set)."""
    plt = _setup_fonts()
    df = metrics['chooser'].copy()
    df['label'] = [cfg.MODEL_LABELS.get(m, m) if m != 'BSM(vol_21d)'
                   else 'BSM(vol_21d) 基线' for m in df['model']]
    order = df.sort_values('MAE')['label'].tolist()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    y = np.arange(len(order))
    mae = df.set_index('label').loc[order, 'MAE']
    rmse = df.set_index('label').loc[order, 'RMSE']
    h = 0.38
    ax.barh(y + h / 2, mae, h, color='#d73027', label='MAE ($)')
    ax.barh(y - h / 2, rmse, h, color='#fdae61', label='RMSE ($)')
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('误差 ($)')
    ax.set_title('方法一：Chooser 定价测试集误差 (MAE / RMSE, 2024 共 208 日)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_vol_metrics(metrics: dict, out_path):
    """Approach-1 vol MAE bar chart, highlighting the persistence threshold."""
    plt = _setup_fonts()
    df = metrics['vol'].copy()
    df['label'] = [('GBDT-anchored (波动率比率)' if m.startswith('gbdt_anchored') else
                    ('BSM(vol_21d) persistence (基线)' if 'persistence' in m else
                     ('XGB-VIX proxy (IV)' if 'vix_proxy' in m else m)))
                   for m in df['model']]
    df = df.sort_values('MAE')
    fig, ax = plt.subplots(figsize=(11, 5.2))
    colors = ['#d73027' if 'persistence' in lab else
              ('#2c7bb6' if 'anchored' in lab else '#74add1')
              for lab in df['label']]
    bars = ax.barh(df['label'], df['MAE'] * 100, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel('波动率 MAE (%)')
    ax.set_title('方法一：波动率预测测试集 MAE (%, 2024 共 208 日)')
    # persistence threshold line
    base = float(df.loc[df['label'].str.contains('persistence'), 'MAE'].iloc[0] * 100)
    ax.axvline(base, color='#d73027', ls='--', lw=1.2)
    ax.text(base + 0.02, len(df) - 0.5, f'persistence 基线 {base:.2f}%',
            color='#d73027', fontsize=9, va='center')
    for b, lab in zip(bars, df['label']):
        v = b.get_width()
        if 'anchored' in lab:
            ax.text(v + 0.03, b.get_y() + b.get_height() / 2,
                    f'{v:.2f}%', va='center', fontsize=9, color='#2c7bb6', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_live_metrics(metrics: dict, out_path):
    """Live-snapshot targets: dual-track models vs Week-6 target lines."""
    plt = _setup_fonts()
    df = metrics['live'].set_index('model')
    picks = {'xgb': 'XGBoost', 'vol_21d': 'BSM(vol_21d)'}
    targets = {'MAE_$': 1.00, 'ATM_IV_gap_%': 2.0, 'OTM_put_bias_%': -30.0}
    titles = {'MAE_$': '实盘 MAE ($)', 'ATM_IV_gap_%': 'ATM IV 溢价 (%)',
              'OTM_put_bias_%': 'OTM 看跌定价偏差 (%)'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (col, tgt) in zip(axes, targets.items()):
        vals = {k: float(df.loc[k, col]) for k in picks if k in df.index}
        names = list(vals.keys()); vv = list(vals.values())
        colors = ['#d73027' if n == 'xgb' else '#2166ac' for n in names]
        ax.bar([cfg.MODEL_LABELS.get(f'vol_{n}', n) for n in names], vv, color=colors)
        ax.axhline(tgt, color='grey', ls='--', lw=1.2)
        ax.text(1, tgt, f'目标 {tgt}', color='grey', fontsize=8, va='bottom', ha='right')
        ax.set_title(titles[col]); ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(vv):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    fig.suptitle('实盘快照指标 (2026-07-27)：XGBoost 与 BSM(vol_21d) vs Week-6 目标', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_end2end(metrics: dict, out_path):
    """Approach-2 end-to-end pricing vs static-BSM benchmark."""
    plt = _setup_fonts()
    df = metrics['end2end'].copy()
    df['label'] = df['model'].map(lambda m: {'BSM(vol_21d) static-vol benchmark': 'BSM(vol_21d) 静态基准',
                                             'pricing_gbdt': 'GBDT 端到端 (调优)',
                                             'pricing_nn': 'MLP-NN 端到端 (调优)'}.get(m, m))
    df = df.sort_values('MAE')
    fig, ax = plt.subplots(figsize=(10, 4.4))
    y = np.arange(len(df)); h = 0.36
    ax.barh(y + h / 2, df['MAE'], h, color='#d73027', label='MAE ($)')
    ax.barh(y - h / 2, df['RMSE'], h, color='#fdae61', label='RMSE ($)')
    ax.set_yticks(y); ax.set_yticklabels(df['label']); ax.invert_yaxis()
    ax.set_xlabel('误差 ($)')
    ax.set_title('方法二：端到端定价测试集误差 (合约网格, 2024)')
    ax.legend(); ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

def run_dashboard(verbose: bool = True) -> dict:
    """Generate all dashboard figures + CSVs; return the price-trend series."""
    if verbose:
        print('=' * 72)
        print('  WEEK 8 -- DASHBOARD: PRICE TRENDS & PERFORMANCE METRICS')
        print('=' * 72)

    series = historical_price_series(max_rows=cfg.PRICE_SERIES_SAMPLE)
    series.to_csv(cfg.OUTPUT_DIR / 'price_trend_series.csv', index=False)
    metrics = performance_metrics()

    # latest live market state (base for pricing self-checks / scenarios)
    import sensitivity as sens
    base_state = sens.build_base_state()

    plot_price_trend(series, cfg.ASSETS_DIR / 'fig_w8_price_trend.png')
    plot_sigma_trend(series, cfg.ASSETS_DIR / 'fig_w8_sigma_trend.png')
    plot_chooser_metrics(metrics, cfg.ASSETS_DIR / 'fig_w8_chooser_metrics.png')
    plot_vol_metrics(metrics, cfg.ASSETS_DIR / 'fig_w8_vol_metrics.png')
    plot_live_metrics(metrics, cfg.ASSETS_DIR / 'fig_w8_live_metrics.png')
    plot_end2end(metrics, cfg.ASSETS_DIR / 'fig_w8_end2end.png')

    if verbose:
        print('  figures -> assets/fig_w8_*.png, series -> output/price_trend_series.csv')
        last = series.iloc[-1]
        print(f"  latest date: {last['date'].date()}, spot ${last['spot']:.2f}")
        print(f"    BSM(vol_21d) = ${last['price_BSM']:.2f}, "
              f"XGB = ${last.get('price_xgb', float('nan')):.2f}, "
              f"VIX-proxy = ${last.get('price_vix_proxy', float('nan')):.2f}")
    return {'series': series, 'metrics': metrics, 'base_state': base_state}


if __name__ == '__main__':
    run_dashboard()
