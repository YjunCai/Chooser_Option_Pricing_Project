# Week 1 交付成果总结 — Data Source Design & Initial Collection

## 项目书内容

**Advanced Chooser Option Pricing Model with Real-World Data & Machine Learning**
### Week 1 核心任务

1. **Data Requirement Specification** — 定义金融、宏观、情绪三类数据范围
2. **API Setup & Testing** — 配置 Yahoo Finance、Alpha Vantage、FRED 接口
3. **Initial Data Pull** — 采集 JPM 股票日线、国债收益率、VIX (2018–2024)

---

## 文件结构

```
week1/
├── data_specification.md    # 数据需求说明文档
├── config.py                # 配置中心（ticker、API Key、输出设置）
├── yahoo_finance.py         # Yahoo Finance 采集器（无需 API Key）
├── alpha_vantage.py         # Alpha Vantage 采集器（需 API Key）
├── fred_api.py              # FRED 宏观数据采集器（需 API Key）
├── data_collection.py       # 主调度脚本
├── requirements.txt         # Python 依赖清单
└── README.md                # 本文件
```

### 各文件说明

| 文件 | 说明 |
|---|---|
| `data_specification.md` | 定义了三类数据（金融市场、宏观经济、情绪/另类数据）、字段规格、来源、更新频率及采集周期 |
| `config.py` | 集中管理股票代码、FRED 系列 ID、API Key 环境变量名、输出格式等参数 |
| `yahoo_finance.py` | 使用 `yfinance` 拉取 JPM / VIX / S&P 500 / TNX 日线 OHLCV 数据，无需 API Key |
| `alpha_vantage.py` | 通过 Alpha Vantage API 获取 JPM 日线行情和新闻情绪评分（0-1），内置频率限制保护 |
| `fred_api.py` | 通过 FRED API 获取国债利率、联邦基金利率、CPI 等宏观指标，支持合并为宽表 |
| `data_collection.py` | 主调度器，按顺序执行各数据源采集并输出汇总报告 |

---

## 虚拟环境

```
依赖：requirement.txt
Python: 3.11
```

**已安装的核心依赖：** pandas, numpy, yfinance, requests, pyarrow, fastparquet

---

## 使用方法

### 设置 API Key（可选）

Alpha Vantage 和 FRED 需要 API Key 才能使用。Yahoo Finance 无需 Key。

```powershell
$env:ALPHA_VANTAGE_API_KEY = ""
$env:FRED_API_KEY = ""
```

- Alpha Vantage 免费申请：https://www.alphavantage.co/support/#api-key
- FRED 免费申请：https://fred.stlouisfed.org/docs/api/

### 运行数据采集

```powershell
# 运行全部三个数据源
python week1\data_collection.py

# 仅运行 Yahoo Finance
python week1\data_collection.py --sources yahoo

# 仅运行 FRED
python week1\data_collection.py --sources fred

# 输出 parquet 格式（默认 CSV格式）
python week1\data_collection.py --format parquet

# 指定输出目录
python week1\data_collection.py --output data\raw
```

### 验证环境

```powershell
D:\anaconda\envs\chooser_option_pricing\python.exe -c "import pandas, numpy, yfinance; print('OK')"
```

---

## Week 1 交付物清单

- [x] **数据需求文档** — `data_specification.md`
- [x] **API 连接脚本** — `yahoo_finance.py`, `alpha_vantage.py`, `fred_api.py`
- [x] **主调度脚本** — `data_collection.py`
- [x] **配置文件** — `config.py`
- [x] **依赖管理** — `requirements.txt`

---

## 采集结果

| 数据源 | 状态 | 数据内容 | 行数 |
|---|---|---|---|
| Yahoo Finance | ✅ 成功 | JPM / VIX / SP500 / TNX 日线 (2018-2024) | 各 1760 行 |
| Alpha Vantage | ⚠️ 免费版限制 | TIME_SERIES_DAILY_ADJUSTED 为付费端点；NEWS_SENTIMENT 免费版频率受限 仅返回最近 100 个交易日，不满足 7 年窗口期 |
| FRED | ✅ 成功 | 国债利率 / 联邦基金利率 / VIX / CPI (2018-2024) | 各 84~1780 行 |

