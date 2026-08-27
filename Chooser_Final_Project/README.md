# Chooser Option Pricing Tool

基于真实数据与机器学习的高级选择权（Chooser Option）定价工具 —— 8 周量化实习项目

## 项目结构

```
Final_Project/
├── config.py            # 统一配置：路径、特征列、合约参数、模型注册表、误差指标、场景定义
├── pricer.py            # BSM 选择权定价（Rubinstein 1991）：bs_call / bs_put / simple_chooser
├── data_preparation.py  # 时序 70/15/15 分割（反前瞻）+ LSTM 序列构建
├── targets.py           # 前向波动率 / VIX 目标构造
├── tool_engine.py       # 定价引擎：模型加载、特征构建、波动率预测、双轨定价、误差条
├── sensitivity.py       # 敏感性分析：极端场景测试、单变量扰动、t1 扫描
├── data_updater.py      # 实时数据：yfinance 抓取 + 增量合并进特征数据集
├── dashboard.py         # 仪表板：价格趋势序列 + 性能指标图表（matplotlib）
├── streamlit_app.py     # 定价工具 UI（Streamlit，6 个标签页）
├── train_models.py      # 复现训练 Approach-1 波动率模型并导出 models/*.pkl
├── run.py               # 一键编排：仪表板 + 引擎自检 + 数据新鲜度
├── test_smoke.py        # 冒烟测试（含 Streamlit AppTest）
├── requirements.txt
├── data/
│   └── feature_dataset.csv   # 训练特征数据集（2018–2024，1755 行）
├── models/                   # 已训练模型（Week-6，models/*.pkl + JSON 元数据）
│   ├── vol_xgb / vol_vix_proxy / vol_gbdt_anchored / vol_rf / vol_gbdt / vol_lstm
│   └── price_pricing_gbdt / price_pricing_nn
├── assets/                   # 生成的报告图表
└── output/                   # 生成的趋势序列 / 状态文件
```

## 快速开始

```bash
pip install -r requirements.txt

python test_smoke.py            # 冒烟测试（含 Streamlit AppTest）
python run.py                   # 一键流程：仪表板 + 引擎自检 + 数据新鲜度

streamlit run streamlit_app.py  # 启动定价工具（http://localhost:8511）

python data_updater.py --check      # 实时数据新鲜度（只读）
python data_updater.py --commit     # 增量合并最新数据到 data/feature_dataset.csv

python train_models.py --quick      # 复现训练 3 个核心模型（可选）
```

## 工具功能

- **双轨定价**：BSM(vol_21d) 基线 vs 最优 ML 波动率模型（XGBoost 实盘首选 / VIX-proxy 隐含波动率对齐 / GBDT-anchored 波动率比率），基于 Rubinstein 公式。
- **误差范围**：每项定价附带 Week-6 测试集 MAE（`价格 ± MAE`）。
- **可视化仪表板**（6 个标签页）：
  - 📈 价格趋势：2018–2024 训练区间 + 2025 起样本外实时延伸（灰虚线分界 + 阴影标注）
  - 📊 敏感性分析：价格 vs S / σ / t1 / r 交互图
  - 🌪 极端场景：波动率 +50% / 利率 +2% / 组合 / VIX 冲击
  - 🏆 性能指标：Week-6 测试集与实盘快照指标图表
  - 🔬 误差范围：各模型测试集 MAE/RMSE/R²
  - 📡 实时数据：JPM/VIX/3M 最新状态 + 自动更新
- **实时数据**：`data_updater.py` 抓取 JPM/VIX/IRX，重建与训练一致的特征，增量合并。

## 关键结论（8 周）

1. 以市场隐含波动率为目标是定价改善最大来源：VIX-proxy 合成测试集 Chooser 定价 MAE \$1.17，优于 BSM 基线 \$1.52（改善 23%）。
2. 超参调优越过 persistence 门槛：GBDT-anchored 波动率 MAE 7.16% < 基线 7.32%。
3. 调优后 XGBoost 实盘三目标全达成：MAE \$0.79 / ATM IV 溢价 0.76% / OTM 看跌偏差 -29.5%。
4. 将 ML 能力集中于 BSM 唯一不确定的输入（波动率）优于端到端重建定价函数。

## 依赖

见 `requirements.txt`（numpy / pandas / scipy / scikit-learn / xgboost / joblib / yfinance / matplotlib / streamlit / plotly）。

> 数据说明：`data/feature_dataset.csv` 为训练数据集（2018–2024）；实时延伸由 `data_updater` 抓取。模型在 GPU/联网不可用时离线亦可加载（模型是 pickle 文件）。
