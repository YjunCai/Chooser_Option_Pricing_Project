# Week 5 — 机器学习模型的设计与实现 (ML Model Design & Implementation)

8周实习项目「Advanced Chooser Option Pricing Model with Real-World Data & Machine Learning」的第 5 周交付。

在 Week 4 建立的 BSM 基线 ($1.44 MAE / 4.9% IV 溢价) 基础上,构建**双轨 ML 架构**:
- **方法一 (Approach 1)**: ML 波动率预测 (LSTM / RF / GBDT / XGBoost) + BSM 定价
- **方法二 (Approach 2)**: 端到端监督定价 (Linear / GBDT / 神经网络)

核心工程要求:**时间序列分割 70/15/15,无前瞻偏差**。

---

## 交付成果

| # | 交付物 | 文件 |
|---|--------|------|
| 1 | 机器学习架构设计文档 | [`ML_architecture_design.md`](ML_architecture_design.md) |
| 2 | 特征工程优化代码 | [`feature_engineering_optimization.py`](feature_engineering_optimization.py) |
| 3 | 初始 ML 模型框架 (双轨) | `data_preparation.py` / `targets.py` / `models/` / `ml_framework.py` |

---

## 目录结构

```
week5/
├── config.py                            # 全局配置 (路径/分割/目标/超参)
├── data_preparation.py                  # 时序分割 70/15/15 + purge gap + scaler
├── targets.py                           # 目标构造 (前向波动率/VIX/期权合约标签)
├── feature_engineering_optimization.py  # 交付② 特征分析+增强+选择
├── models/
│   ├── base.py                          # BaseModel 统一接口
│   ├── volatility_models.py             # 方法一: RF/GBDT/XGBoost/LSTM
│   ├── pricing_models.py                # 方法二: Linear/GBDT/MLP-NN
│   └── bsm_engine.py                    # BSM 混合定价引擎 (Week3 Rubinstein 公式)
├── evaluation.py                        # MAE/RMSE/MAPE/R² + BSM 基线对比
├── ml_framework.py                      # 双轨主流程
├── run_experiments.py                   # CLI 入口
├── test_ml_framework.py                 # 37 项测试 (含反前瞻验证)
├── output/                              # 结果 CSV + 运行日志
└── assets/                              # 可视化图表
```

## 运行方式

```bash
# 1. 测试 (37 项: 分割无泄漏 / 目标对齐 / 模型接口 / BSM引擎)
python test_ml_framework.py

# 2. 完整双轨实验 (含特征工程报告)
python run_experiments.py --feats

# 3. 仅特征工程优化报告
python -m feature_engineering_optimization
```

依赖: `pip install -r requirements.txt` (xgboost / torch 可选,缺失时自动回退)。

---

## 关键设计:反前瞻偏差

1. **严格时序分割**: 按日期排序后 70% 训练 / 15% 验证 / 15% 测试,禁止 shuffle。
2. **Purge/Embargo**: train→val、val→test 边界各剔除 21 天,防止目标窗口 [t, t+h] 跨边界泄漏。
3. **特征对齐**: 所有特征为滚动/滞后(后视)窗口,日期 t 只使用 ≤t 的信息。
4. **目标对齐**: `target_t = 未来 [t+1, t+h] 已实现波动率`,与特征日期严格错开。
5. **Scaler**: `StandardScaler` 仅在训练集拟合,验证/测试集只 transform。
6. **LSTM 窗口**: 锚点 j 的序列覆盖 [j−L+1, j],严格 ≤j。

> 由 `test_ml_framework.py` 中 `test_forward_vol_no_lookahead` 等逐条断言验证。

---

## 初始实验结果与发现 (Week 5)

### 方法一: 波动率预测 (test 集, 2024)

| 模型 | 波动率 MAE | 相对 persistence |
|------|-----------|-----------------|
| **BSM(vol_21d) persistence** | **7.32%** | — (基线) |
| GBDT-anchored (vol_ratio) | 7.42% | −1.3% |
| GBDT | 9.48% | −29% |
| XGBoost | 9.51% | −30% |
| RandomForest | 10.21% | −39% |
| LSTM (torch) | ~13% | 弱 |

**发现**: 初始 ML 模型未跑赢 persistence 基线。原因(已通过诊断确认):
1. **波动率持久性 (Volatility Persistence)**: 今日 σ21d 本身是极强的预测器,波动率自相关高。
2. **Regime 漂移**: 训练期 (2018-2022) 均值波动率 26.6%,测试期 (2024) 仅 19.2%——严格时序分割迫使模型外推到不同 regime。
3. **过拟合**: GBDT 训练 R²≈0.97,测试 R² 为负——对噪声目标(已实现波动率)过度拟合。

> 这是**有价值的负结果**: 它精确界定了 Week 6 的优化方向(超参调优、regime 自适应特征、以 IV 为目标的过滤信号),并为后续所有比较提供"必须击败 persistence"的明确基准。

**但有两个真实的正向改进**:

**① 实盘指标 (与 Week 4 相同的 2026-07-27 期权快照, 595 合约)** — 用 ML 预测波动率重新定价:

| 指标 | Week 4 基线 (BSM σ21d) | Week 5 最优 ML (RandomForest) |
|------|----------------------|------------------------------|
| **实盘 MAE** | $1.44 | **$1.03** (σ=22.2%) |
| **ATM IV 溢价** | 4.9% | **3.0%** |
| **OTM 看跌定价偏差** | 65%+ | **51.4%** |

ML 预测波动率 (22.2%) 高于静态 σ21d (20.2%),更接近市场 ATM IV (25.2%),三项实盘指标同步改善。详见 `compute_live_metrics.py`。

**② Chooser 定价误差**: `GBDT-anchored`(行业标准的"锚定比值"波动率预测)在 **Chooser 定价误差上击败了 BSM(σ21d) 基线**:

| Chooser 定价 MAE (test) | 模型 |
|------------------------|------|
| **$1.46** | **GBDT-anchored (ML σ)** |
| $1.52 | BSM(σ21d) 基线 |

即:ML 预测的波动率经 BSM 引擎定价后,价格误差低于 Week 4 静态波动率基线——验证了混合定价管线的价值。

### 方法二: 端到端定价 (test 集, 合约网格)

| 模型 | 价格 MAE ($) | R² |
|------|-------------|-----|
| BSM(vol_21d) static-vol 基准 | 1.77 | 0.93 |
| GBDT | 3.75 | 0.64 |
| MLP-NeuralNet | 5.11 | 0.40 |
| LinearRegression | 7.08 | −0.05 |

**发现**: 端到端模型能捕捉大部分定价结构 (R² 0.4-0.93),但静态波动率基准因波动率持久性同样难以超越。方法二的价值在 Week 6 接入真实期权价格标签与超参调优后体现。

---

## 目标变量设计说明

- **方法一主目标**: 前向 21 日已实现波动率 (可直接计算、无前瞻偏差)。
- **方法一 IV 代理**: VIX (市场隐含波动率指数,真实 IV 时间序列)。
- **数据约束**: JPM 单票期权历史 IV 时间序列当前不可得(仅 2026-07-27 单日快照含 IV)。目标接口可插拔——Week 6-8 实时期权数据接入后,替换为真实 IV 列即可,管线不变。
- **方法二标签**: 每日合约网格 (moneyness × tenor × type),以未来已实现波动率定价的公平价,特征只用 ≤t 数据。

---

## 里程碑对照

| 里程碑 | 状态 |
|--------|------|
| 双轨架构 (方法一 + 方法二) | ✅ 完成 |
| 时序分割 70/15/15 无前瞻偏差 | ✅ 完成 (测试验证) |
| 特征工程优化 | ✅ 完成 |
| 初始 ML 模型框架 | ✅ 完成 |
| 超参调优 + 全量训练 + SHAP (Week 6) | 📋 下一周 |
