# Week 2 交付成果 — 数据预处理与特征工程

**Advanced Chooser Option Pricing Model with Real-World Data & Machine Learning**

---

## 项目概述

Week 2 对 Week 1 采集的原始数据（JPM 日线、VIX、3 个月国债收益率）进行清洗和特征工程，构建可直接用于 BSM 模型和 ML 模型的**结构化特征数据集**。

### 核心任务

1. **数据清洗** — 缺失值插值、IQR 异常值检测（价格列和 VIX 跳过）、时间对齐
2. **特征工程** — 19 个衍生特征（传统 + 高级）
3. **Pipeline 自动化** — 一键执行完整预处理流程

---

## 文件结构

```
week2/
├── data/                          # 输出数据集
│   ├── inner_join_aligned.csv     # 清洗对齐后的宽表 (1760×14)
│   └── feature_dataset.csv        # 最终特征数据集 (1755×20)
├── output/                        # 旧版输出（已弃用，以 data/ 为准）
├── config.py                      # 配置（路径、滚动窗口参数）
├── data_cleaner.py                # 数据清洗模块
├── feature_engineering.py         # 特征工程模块
├── dividend_api.py                # 股息数据获取（yfinance API）
├── preprocessing_pipeline.py      # 主调度 Pipeline 脚本
├── feature_specification.md       # 特征工程文档
├── requirements.txt               # Python 依赖
├── week2_experiment_report.ipynb  # 实验报告（含图表分析）
└── README.md                      # 本文件
```

---

## 环境配置

### 依赖安装

```powershell
pip install -r requirements.txt
```

已安装的核心依赖：

| 包 | 用途 |
|---|---|
| `pandas>=2.0.0` | 数据处理 |
| `numpy>=1.24.0` | 数值计算 |
| `yfinance>=0.2.30` | Yahoo Finance 数据（股息） |
| `requests>=2.31.0` | HTTP 请求 |
| `pyarrow>=14.0.0` | Parquet 格式支持 |
| `matplotlib`, `seaborn` | 可视化（notebook） |

### API Key（可选）

```powershell
$env:ALPHA_VANTAGE_API_KEY = "your_key"    # 仅 Alpha Vantage 需要
$env:FRED_API_KEY = "your_key"              # 仅 FRED 需要
```

**注意**：Yahoo Finance 无需 API Key。在数据中心 IP 下可能被限流，可通过 VPN 解决。

---

## 执行步骤

### 方式一：完整 Pipeline（一键运行）

```powershell
python data\..\preprocessing_pipeline.py
```

**内部流程**：
1. 读取 Week 1 原始数据（`week1/data/yahoo_jpm.csv`, `yahoo_vix.csv`, `fred_3mo_treasury.csv`）
2. 数据清洗：IQR 截断（所有数值列）、缺失值插值、outer join 对齐
3. 特征工程：19 个衍生特征
4. 输出到 `output/` 目录

### 方式二：从清洗后的内连接数据开始（推荐）

```powershell
# Step 1: 生成内连接数据（仅需运行一次）
python -c "import sys; sys.path.insert(0,'.'); import data_cleaner; raw=data_cleaner.load_raw_data(); aligned=data_cleaner.clean_all(raw, skip_iqr_sources={'vix'}, skip_iqr_cols={'open','high','low','close','adjusted_close'}); inner=aligned.dropna(subset=['close_jpm','close_vix']); inner.to_csv('data/inner_join_aligned.csv', index=False)"

# Step 2: 特征工程
python -c "import sys, pandas as pd; sys.path.insert(0,'.'); import feature_engineering; df=pd.read_csv('data/inner_join_aligned.csv', parse_dates=['date']); ds=feature_engineering.build_feature_set(df); ds.to_csv('data/feature_dataset.csv', index=False)"
```

或使用 Notebook：

### 方式三：通过 Jupyter Notebook

直接打开 `week2_experiment_report.ipynb`，依次运行所有单元格。

### 方式四：GitHub Actions 自动化调度

工作流文件位于 `.github/workflows/data_pipeline.yml`。

#### 触发方式

| 触发 | 说明 |
|------|------|
| **定时调度** | 每个交易日 06:00 UTC（美国收盘后）自动运行 |
| **手动触发** | GitHub UI → Actions → Data Preprocessing Pipeline → Run workflow |
| **代码变更** | 推送 week1/week2 代码或 workflow 配置时自动触发 |

#### 流程

```
Yahoo Finance (JPM/VIX)    FRED (Treasury)
         ↓                       ↓
    data_collection.py  ←  Week 1 Pipeline
         ↓
    preprocessing_pipeline.py  ←  Week 2 Pipeline
         ↓
    data/feature_dataset.csv
         ↓
    git commit & push  ←  自动提交更新
```

---

## Pipeline 处理流程

### Step 1: 数据清洗 (`data_cleaner.py`)

| 操作 | 说明 |
|------|------|
| **缺失值插值** | 时间加权线性插值，剩余缺口 ffill/bfill |
| **IQR 异常值截断** | 1.5×IQR Winsorization，应用于所有数值列（含价格 OHLC 和 VIX）。注意：趋势数据的高价可能被误标为异常值 |
| **时间对齐** | outer join on date → 1763 行 |
| **内连接** | 只保留 JPM + VIX 都有数据的日期 → **1760 行** |
| **Treasury 假日填充** | 债市假日（Columbus Day, Veterans Day）的国债利率用 ffill 填充 |

### Step 2: 特征工程 (`feature_engineering.py`)

#### 传统特征（6 个）

| 特征 | 公式 | 含义 |
|------|------|------|
| `daily_return` | ln(closeₜ/closeₜ₋₁) | 日收益率 |
| `vol_5d` | σ(ret₅d) × √252 | 短期年化波动率 |
| `vol_21d` | σ(ret₂₁d) × √252 | 中期年化波动率 |
| `vol_63d` | σ(ret₆₃d) × √252 | 长期年化波动率 |
| `high_low_spread` | (highₜ-lowₜ)/closeₜ | 日内波动率 |
| `dps_growth_rate` | (DPSₜ/DPSₜ₋₄ -1) × 100 | 股息增长率（yfinance 真实数据） |

#### 高级特征（6 个）

| 特征 | 公式 | 含义 |
|------|------|------|
| `vix_change_1d` | VIXₜ - VIXₜ₋₁ | VIX 日变动 |
| `vix_jpm_corr_21d` | ρ(ret₂₁d, ΔVIX₂₁d) | VIX-JPM 滚动相关性 |
| `vix_jpm_cross_1d` | -return × ΔVIX | 1 日交叉动量（风险指标） |
| `rate_change_1d_bps` | (rₜ - rₜ₋₁) × 100 | 利率日变动 |
| `rate_momentum_5d_bps` | (rₜ - rₜ₋₅) × 100 | 利率动量 |
| `sentiment_score` | 1 - minmax(VIX, 252d) | 市场情绪 [0,1] |

#### 辅助特征（7 个）

`volume_change_1d`, `sma_ratio_21`, `jpm_vol_ratio`, `vix_ratio`, 以及 3 个原始列

---

## 输出数据

### `data/inner_join_aligned.csv`

清洗对齐后的原始宽表，包含 JPM OHLCV、VIX OHLCV、3Mo 国债利率，共 **1760 行 × 14 列**，**0 空值**。

**IQR 截断影响**：
- JPM 最高价：$221.04（原始 $250.29，被截断）
- VIX 最高价：35.51（原始 82.69，COVID 峰值被截断）

### `data/feature_dataset.csv`

最终特征数据集，**1755 行 × 20 列**：

| 列 | 类型 | 空值 | 说明 |
|----|------|------|------|
| `date` | date | 0 | 交易日 |
| `close_jpm` | float | 0 | JPM 收盘价 |
| `close_vix` | float | 0 | VIX 指数 |
| `value_treasury_3mo` | float | 0 | 3 个月国债收益率 |
| `dps_growth_rate` | float | 0 | 股息增长率 |
| `daily_return` | float | 0 | 日收益率 |
| `vol_5d/21d/63d` | float | 0/16/58 | 滚动波动率 |
| `sentiment_score` | float | 0 | 情绪评分 [0,1] |
| `vix_jpm_corr_21d` | float | 10 | VIX-JPM 相关性 |
| `rate_momentum_5d_bps` | float | 0 | 利率动量 |
| ... | | | 其余 7 个特征 |

**空值说明**：空值来自滚动窗口预热期以及 IQR 截断导致的短暂常数列（VIX 卡在 35.51 时相关性不可计算），属预期行为。

---

## 数据源

| 数据源 | 数据内容 | 原始行数 | 时间范围 |
|--------|---------|---------|---------|
| Yahoo Finance (JPM) | JPM 日线 OHLCV | 1760 | 2018-01-02 ~ 2024-12-30 |
| Yahoo Finance (VIX) | VIX 日度指数 | 1760 | 2018-01-02 ~ 2024-12-30 |
| FRED (DGS3MO) | 3 个月国债收益率 | 1750 | 2018-01-02 ~ 2024-12-31 |
| yfinance (股息) | JPM 季度股息 | 28 次 | 2018-01-04 ~ 2024-10-04 |

---

## 关键发现

1. **sentiment_score 与 volatility 强负相关**（r=-0.48）：市场恐慌时价差扩大
2. **股息增长为阶梯函数**：42.7% 的交易日增长率为 0（股息政策通常每年调整 1-2 次）
3. **VIX-JPM 相关性**：均值 -0.50，COVID 期间最负
4. **IQR 截断影响**：VIX 最高被截断至 35.51（原始 82.69），导致 COVID 恐慌期情绪信号失真；JPM 最高被截断至 $221.04（原始 $250.29）

---

## 里程碑状态

| 里程碑 | 预计 | 状态 |
|--------|------|------|
| 里程碑 1（第 2 周末）：完整数据处理流程 | Week 2 | ✅ 完成 |

---

## Notebook

打开 `week2_experiment_report.ipynb` 浏览完整的数据分析报告，包含所有特征的分布图、时间序列图和相关性热力图。
