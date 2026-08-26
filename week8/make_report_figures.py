"""
Generate the Week-1/Week-2 summary figures used by the project completion report
(结项报告). Reads the week-2 feature dataset so the figures match the numbers
the weekly reports quote.

Figures written to report_assets/:
  fig_w1_series.png       3-panel JPM / VIX / 3MO-Treasury time series
  fig_w2_corr.png         feature correlation heatmap
  fig_w2_sentiment.png    sentiment score + rolling vol over time

Run:
    python make_report_figures.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w8config as cfg

OUT = cfg.W8_DIR / 'report_assets'
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(cfg.WEEK2_FEATURE_DATASET, parse_dates=[cfg.DATE_COL])
df = df.sort_values(cfg.DATE_COL).reset_index(drop=True)


def fig_w1_series():
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(df['date'], df['close_jpm'], color='#2166ac', lw=0.9)
    axes[0].fill_between(df['date'], df['close_jpm'], alpha=0.12, color='#2166ac')
    axes[0].set_ylabel('JPM 股价 ($)')
    axes[0].set_title('Week 1 数据采集：JPM / VIX / 3M 国债 (2018–2024)')
    axes[1].plot(df['date'], df['close_vix'], color='#d73027', lw=0.9)
    axes[1].fill_between(df['date'], df['close_vix'], alpha=0.12, color='#d73027')
    axes[1].set_ylabel('VIX')
    axes[2].plot(df['date'], df['value_treasury_3mo'], color='#fdae61', lw=0.9)
    axes[2].fill_between(df['date'], df['value_treasury_3mo'], alpha=0.2, color='#fdae61')
    axes[2].set_ylabel('3M 国债收益率 (%)')
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout()
    fig.savefig(OUT / 'fig_w1_series.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('  fig_w1_series.png')


def fig_w2_corr():
    cols = ['daily_return', 'vol_5d', 'vol_21d', 'vol_63d', 'vix_change_1d',
            'vix_jpm_corr_21d', 'vix_jpm_cross_1d', 'rate_change_1d_bps',
            'rate_momentum_5d_bps', 'sentiment_score', 'high_low_spread',
            'sma_ratio_21']
    c = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(c, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f'{c.iloc[i, j]:.2f}', ha='center', va='center',
                    fontsize=6.5, color='black')
    ax.set_title('Week 2 特征相关性矩阵（关键特征）')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_w2_corr.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('  fig_w2_corr.png')


def fig_w2_sentiment():
    fig, ax1 = plt.subplots(figsize=(11, 4.6))
    ax1.fill_between(df['date'], df['sentiment_score'], alpha=0.25, color='#2c7bb6')
    ax1.plot(df['date'], df['sentiment_score'], color='#2c7bb6', lw=0.8)
    ax1.axhline(0.5, color='grey', ls='--', lw=0.8)
    ax1.set_ylabel('情绪评分 [0,1]', color='#2c7bb6')
    ax1.set_xlabel('日期')
    ax1.set_title('Week 2 情绪评分（VIX 位置构造）与 21 日已实现波动率')
    ax2 = ax1.twinx()
    ax2.plot(df['date'], df['vol_21d'] * 100, color='#d73027', lw=0.9, alpha=0.85)
    ax2.set_ylabel('vol_21d 年化波动率 (%)', color='#d73027')
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout()
    fig.savefig(OUT / 'fig_w2_sentiment.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('  fig_w2_sentiment.png')


if __name__ == '__main__':
    print('Generating Week-1/Week-2 summary figures...')
    fig_w1_series()
    fig_w2_corr()
    fig_w2_sentiment()
    print('done ->', OUT)
