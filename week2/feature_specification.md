# Week 2 — 特征工程规格说明

## 概述

本文档定义了 Week 2 数据预处理 Pipeline 生成的所有特征。该特征集旨在通过提供超越基础 Black-Scholes 参数的多维信号，丰富 Chooser Option 定价模型。

**覆盖的项目要求：**
- 传统特征：滚动波动率、日收益率、**股息增长率**
- 高级特征：VIX-JPM 相关性、利率动量、**情绪评分（0-1）**

## 数据源

| 数据源 | 文件 | 内容 |
|---|---|---|
| Yahoo Finance | `yahoo_jpm.csv` | JPM 日线 OHLCV 数据（2018-2024） |
| Yahoo Finance | `yahoo_vix.csv` | VIX 日度指数水平 |
| FRED | `fred_3mo_treasury.csv` | 3 个月期国债固定期限利率 |

## 特征分类

### 1. 基于价格的特征（传统）

| # | 特征 | 公式 | 含义 |
|---|---|---|---|
| 1 | `daily_return` | ln(closeₜ / closeₜ₋₁) | 对数收益率；所有波动率计算的基础 |
| 2 | `vol_5d` | σ(returnₜ₋₄…ₜ) × √252 | 短期年化波动率（1 周） |
| 3 | `vol_21d` | σ(returnₜ₋₂₀…ₜ) × √252 | 中期年化波动率（1 个月） |
| 4 | `vol_63d` | σ(returnₜ₋₆₂…ₜ) × √252 | 长期年化波动率（1 个季度） |
| 5 | `high_low_spread` | (highₜ - lowₜ) / closeₜ | 日内波动率比例 |
| 6 | `dps_growth_rate` | (DPS_quarter / DPS_quarter_same_quarter_last_year - 1) × 100 | 每股股息年化增长率（%），来源：JPM 官方股息记录 |

### 2. 成交量特征

| # | 特征 | 公式 | 含义 |
|---|---|---|---|
| 8 | `volume_change_1d` | ln(volumeₜ / volumeₜ₋₁) | 成交量激增/萎缩指示器 |

### 3. 跨资产 / VIX 特征（高级）

| # | 特征 | 公式 | 含义 |
|---|---|---|---|
| 9 | `vix_change_1d` | VIXₜ - VIXₜ₋₁ | 市场恐慌指数的绝对变动值 |
| 10 | `vix_jpm_corr_21d` | 21 天内 ρ(return, ΔVIX) | 股票-波动率相关性机制（VIX-JPM 相关性） |

### 4. 利率特征（高级）

| # | 特征 | 公式 | 含义 |
|---|---|---|---|
| 11 | `rate_change_1d_bps` | (rₜ - rₜ₋₁) × 100 | 日度利率冲击（基点） |
| 12 | `rate_momentum_5d_bps` | (rₜ - rₜ₋₅) × 100 | 短期利率趋势（基点）——利率动量 |

### 5. 情绪评分（高级）

| # | 特征 | 公式 | 含义 |
|---|---|---|---|
| 13 | `sentiment_score` | 1 - (VIX_t - min(VIX_t-w:t)) / (max(VIX_t-w:t) - min(VIX_t-w:t)), w=252 | 市场情绪评分 [0,1]，1=乐观，0=恐慌 |

**公式说明：** 采用 VIX 滚动窗口 min-max 标准化后取反。$Score_t = 1 - \frac{VIX_t - \min(VIX_{t-w:t})}{\max(VIX_{t-w:t}) - \min(VIX_{t-w:t})}$，其中 w=252 个交易日。低 VIX → 分母小 → 分数接近 1.0（乐观）；高 VIX → 分数接近 0.0（恐慌）。

### 6. 组合 / 市场机制指标

| # | 特征 | 公式 | 含义 |
|---|---|---|---|
| 14 | `jpm_vol_ratio` | vol_5d / vol_21d | >1 = 短期波动率相对于中期上升 |
| 15 | `sma_ratio_21` | closeₜ / SMA(close, 21)ₜ | 均值回归：>1 = 高于均值，<1 = 低于均值 |
| 16 | `vix_ratio` | VIX / (vol_21d × 100) | VIX 溢价：>1 = 期权相对已实现波动率偏贵 |

## 输出结构

最终生成的 `feature_dataset.csv` 包含：

- **输入列**：`close_jpm`、`close_vix`、`value_treasury_3mo`（原始对齐值）
- **16 个衍生特征**（见上表）
- **日期列**：ISO 8601 日期索引

## 方法论

- **缺失值处理**：基于时间的线性插值，对剩余缺口进行前向/后向填充
- **异常值处理**：基于 IQR 的 Winsorization 缩尾处理（1.5 倍 IQR）
- **时间对齐**：对全部三个数据源按日期进行外连接
- **滚动窗口**：最少 5 天预热期；最终数据集中移除前 5 行
