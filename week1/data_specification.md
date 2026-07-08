# 数据需求规格说明 — Chooser Option 定价模型

## 1. 概述

本文档定义了高级 Chooser Option 定价模型项目的数据需求。目标是采集涵盖金融市场、宏观经济指标和情绪信号的多维度数据，以增强传统的 Black-Scholes (BSM) chooser option 定价模型。

## 2. 数据分类

### 2.1 金融市场数据

| 数据字段 | 来源 | 代码/符号 | 频率 | 用途 |
|---|---|---|---|---|
| JPM 股价（收盘价） | Yahoo Finance | JPM | 日 | 期权定价的基础资产价格 |
| JPM 股价（开/高/低/量） | Yahoo Finance | JPM | 日 | 特征工程（波动率、收益率） |
| S&P 500 指数 | Yahoo Finance | ^GSPC | 日 | 大盘走势 |

### 2.2 宏观经济数据

| 数据字段 | 来源 | 系列 ID | 频率 | 用途 |
|---|---|---|---|---|
| 10 年期国债收益率 | FRED | DGS10 | 日 | 无风险利率代理 |
| 3 个月国库券收益率 | FRED | DGS3MO | 日 | 短期利率 |
| 联邦基金有效利率 | FRED | FEDFUNDS | 日 | 货币政策立场 |
| CPI（消费者物价指数） | FRED | CPIAUCSL | 月 | 通胀调整 |
| GDP 增长率 | FRED | GDP | 季 | 经济增长背景 |

### 2.3 情绪/另类数据

| 数据字段 | 来源 | 频率 | 用途 |
|---|---|---|---|
| VIX（CBOE 波动率指数） | Yahoo Finance/FRED  | 日 | 市场恐慌指标、波动率代理 |
| 新闻情绪评分 | NewsAPI / Alpha Vantage | 日 | 量化市场情绪（0-1 分） |
| 看跌/看涨期权比率 | CME（通过 Alpha Vantage） | 日 | 期权市场情绪 |
| 交易量（JPM） | Yahoo Finance | 日 | 流动性指标 |

## 3. 采集周期

- **开始日期**：2018-01-01
- **结束日期**：2024-12-31
- **选择理由**：7 年窗口覆盖多个市场体制（疫情前、COVID 暴跌、复苏、高通胀、加息环境）。

## 4. 技术规范

- **存储格式**：Parquet（辅助）/ CSV（主要）
- **编码**：UTF-8
- **时间戳**：ISO 8601 (UTC)

## 5. 所需 API Key

| API | 基础 URL | 免费版限制 | 环境变量 |
|---|---|---|---|
| Alpha Vantage | https://www.alphavantage.co/query | 5 次/分钟，500 次/天 | ALPHA_VANTAGE_API_KEY |
| FRED | https://api.stlouisfed.org/fred | 无速率限制（免费 Key） | FRED_API_KEY |
| Yahoo Finance | https://query1.finance.yahoo.com/v8 | 无需 Key（有频率限制） | — |
