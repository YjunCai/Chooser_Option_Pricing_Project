# Week 5 — 机器学习架构设计文档 (ML Architecture Design)

**项目**: Advanced Chooser Option Pricing Model with Real-World Data & Machine Learning
**作者**: Yujun Cai
**周期**: 8周实习项目 · Week 5 (机器学习模型的设计与实现)

---

## 1. 背景与 Week 4 基线

Week 4 用真实 JPM 期权链数据验证了 BSM 模型的性能边界,建立了供后续 ML 模型比较的基线:

| 指标 | Week 4 基线值 | 含义 |
|------|--------------|------|
| BSM vs 市场 MAE (ATM+OTM) | **$1.44** | 静态 σ21d 定价的系统性偏差 |
| ATM 隐含波动率溢价 | **4.9%** | 市场 IV 高于历史 σ21d |
| OTM 看跌 IV 溢价 | 最高 47.5% | 波动率微笑,深虚值低估 65%+ |
| 波动率 CV | **0.64** | 波动率时变性,恒定假设严重违背 |
| 假设诊断 | 3 项严重违背 | 恒定σ、正态性、历史σ滞后 |

**核心结论**: BSM 误差主要来自**波动率输入的滞后与恒定**(`σ21d` 无法反映市场预期),Week 5-6 引入 ML 波动率预测是消除系统性偏差的最有效方向。

---

## 2. 总体架构:双轨 (Dual-Track Architecture)

```
                         ┌─────────────────────────────────────────────┐
                         │          真实市场数据 (2018-2024)            │
                         │  JPM日线 · VIX · 3Mo国债 · 股息 · 期权链     │
                         └────────────────────┬────────────────────────┘
                                              │
                         ┌────────────────────▼────────────────────────┐
                         │     特征工程 (Week2 16维 + Week5 增强)      │
                         │  滚动波动率/收益形态/VIX联动/利率/情绪/滞后    │
                         └────────────────────┬────────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
     ┌────────▼─────────┐          ┌──────────▼──────────┐      ┌───────────────┴──────┐
     │  时序分割 70/15/15 │          │   方法一 (Approach 1) │      │   方法二 (Approach 2)  │
     │  无前瞻偏差 + purge │          │   ML 波动率预测      │      │   端到端监督定价       │
     └────────────────────┘          └──────────┬──────────┘      └───────────────┬──────┘
                                                │                                  │
             ┌──────────────────────────────────▼────────────┐       ┌─────────────▼─────┐
             │  波动率模型: RF / GBDT / XGBoost / LSTM       │       │ 定价模型:         │
             │  target = 前向已实现波动率 (VIX作为IV代理)     │       │ Linear/GBDT/NN    │
             └──────────────────────────────────┬────────────┘       │  target = 期权价格 │
                                                │                    └─────────────┬─────┘
                                                ▼                                  │
             ┌────────────────────────────────────────────┐                       │
             │  BSM 混合定价引擎 (Week3 Rubinstein公式)     │                       │
             │  Chooser Price = f(S, K, t1, T2, r, q, σ̂)  │                       │
             └────────────────────────────────────────────┘                       │
                                                │                                  │
                             ┌──────────────────▼──────────────────────────────────┘
                             │   评估与对比 (MAE / RMSE / MAPE / R²)                │
                             │   ML 模型 vs BSM(σ21d) 基线                         │
                             └─────────────────────────────────────────────────────┘
```

**双轨定位**:
- **方法一 (两阶段、可解释)**: 先预测波动率 σ̂,再代入 BSM 公式定价。波动率预测本身可解释、可审计,且能直接替换 BSM 的静态 σ21d 输入,保留金融理论的透明度。
- **方法二 (单阶段、数据驱动)**: 从 (市场特征 + 合约参数) 直接学习定价映射,不依赖 BSM 结构假设,更能捕捉特征间的非线性交互,但可解释性较弱(需 SHAP/LIME,Week 6)。

---

## 3. 数据流与特征工程

### 3.1 输入数据 (复用 Week 1-2 交付)

`feature_dataset.csv` — 1755 个交易日 (2018-01-09 ~ 2024-12-30),20 列:

| 类别 | 特征 |
|------|------|
| 价格/收益 | daily_return, vol_5d, vol_21d, vol_63d, high_low_spread |
| 成交量 | volume_change_1d |
| 均线 | sma_ratio_21 |
| 跨资产 | vix_change_1d, vix_jpm_corr_21d, vix_jpm_cross_1d |
| 利率 | rate_change_1d_bps, rate_momentum_5d_bps |
| 情绪 | sentiment_score (0-1) |
| 股息 | dps_growth_rate |
| 组合信号 | jpm_vol_ratio, vix_ratio |
| 市场状态 | close_jpm (S), close_vix, value_treasury_3mo (r) |

### 3.2 特征工程优化 (Week 5 交付②)

`feature_engineering_optimization.py` 实现三步:

1. **分析 (analyze_features)**: NaN 率、方差、两两相关矩阵 → `output/feature_analysis.csv`
2. **增强 (enhance_features)**: 新增**纯后视**衍生特征:
   - 收益形态: 滚动偏度/峰度 (21/63 日) — 捕捉厚尾(Week 4 诊断: 超额峰度 13.7)
   - 多周期波动率比: vol_5/21, vol_21/63 — 波动率期限结构/regime 信号
   - VIX 百分位 (252日) — 市场情绪位置
   - 距 52 周高/低距离、价格区间位置 — 趋势/均值回归
   - 关键特征滞后项 (vol_21d_lag1/5, sentiment_lag1 等)
   - 利率水平与 21 日动量
3. **选择 (select_features)**: 方差阈值过滤 + 相关系数去冗余 (|ρ|>0.92 保留与目标更相关者) + RF 重要性排序 → `output/selected_features.csv`、`output/feature_importance.csv`

### 3.3 反前瞻偏差 (核心设计)

| 环节 | 措施 |
|------|------|
| 分割 | 严格**按时间顺序** 70% 训练 / 15% 验证 / 15% 测试,禁止 shuffle |
| Purge/Embargo | train→val、val→test 边界各剔除 `PURGE_GAP_DAYS=21` 天,防止目标窗口跨边界泄漏 |
| 特征对齐 | 所有特征为滚动/滞后(后视)窗口,日期 t 只使用 ≤t 的信息 |
| 目标对齐 | `target_t = 未来[t+1, t+h] 已实现波动率`,与特征日期严格错开 |
| Scaler | `StandardScaler` 仅在**训练集**拟合,验证/测试集只 transform |
| LSTM 窗口 | 锚点 j 的序列覆盖 [j−L+1, j],严格 ≤j |

测试验证:`test_ml_framework.py` 中的 `test_forward_vol_no_lookahead` 逐条断言目标窗口只含未来数据。

---

## 4. 方法一:ML 波动率预测 + BSM 混合定价

### 4.1 目标变量

| 目标 | 说明 |
|------|------|
| **前向已实现波动率** (主) | `σ_fwd[21d] = std(returns[t+1..t+21]) × √252`,可直接计算、无前瞻偏差 |
| **VIX** (IV 代理, 副) | 市场隐含波动率指数(真实 IV 时间序列),体现"以隐含波动率为目标"的设计意图 |

> **数据约束说明**: JPM 单票期权历史 IV 时间序列在当前数据管道中不可得(仅 2026-07-27 单日快照含 IV)。框架的 `TargetEngine` 接口可插拔——当 Week 6-8 实时期权数据接入后,将预测目标替换为真实 IV 列即可,管线不变。

### 4.2 模型注册表 (config.VOL_MODELS)

| 模型 | 类别 | 定位 |
|------|------|------|
| RandomForest | 非线性树 | 特征交互、稳健性 |
| GBDT | 梯度提升树 | 强基线 |
| XGBoost | 梯度提升树 | 若安装成功;否则回退 sklearn HistGradientBoosting |
| LSTM | 循环神经网络 | 序列记忆;torch 可用→真 LSTM,否则回退滑动窗口 MLP |

所有模型实现 `BaseModel` 接口 (`fit(X,y) / predict(X)`),统一被 pipeline、评估与未来的超参搜索调用。

### 4.3 混合定价引擎 (models/bsm_engine.py)

预测的 σ̂ 代入 Week 3 验证过的 Rubinstein (1991) 公式:

```
V = S·e^(−q·T2)·[N(d1) − N(−d3)] − K·e^(−r·T2)·[N(d2) − N(−d4)]
```

引擎按日期向量化,对比三种波动率输入的 Chooser 价格:
- **BSM 基线**: σ = σ21d (Week 4 静态输入)
- **ML 预测**: σ = σ̂_模型
- **公平参照**: σ = 前向已实现波动率 (作为 σ_true)

---

## 5. 方法二:端到端监督定价

### 5.1 目标变量与合约网格

无市场 Chooser 价格,框架在每日生成**合约网格**作为有监督标签:

```
每日 t × {moneyness: 0.90/1.00/1.10} × {tenor: 30/60/120 交易日} × {call/put}
label_price = BSM( S_t, K=m·S_t, T=τ/252, r_t, q, σ_fwd[τ] )
```

`σ_fwd[τ]` 使用日期 t 之后的 [t+1, t+τ] 已实现波动率定价——标签是模型**必须从过去特征预测出的公平价格**;特征只用 ≤t 数据,无前瞻泄漏。当真实期权价格数据接入后,同一管线可直接替换标签来源。

### 5.2 模型注册表 (config.PRICING_MODELS)

| 模型 | 类别 |
|------|------|
| LinearRegression | 线性 |
| GBDT | 梯度提升树 |
| MLP-NeuralNet | 多层感知机 (sklearn MLPRegressor) |

特征 = 16 维市场特征 + 合约特征 (log_moneyness, tenor_years, is_call) + 市场状态 (spot, rate)。

---

## 6. 评估框架与基线对比

`evaluation.py` 提供统一指标: **MAE / RMSE / MAPE / R² / Bias**,并输出与 BSM 基线的**改进百分比**。

| 对比 | 内容 |
|------|------|
| 波动率 | 各模型预测 σ̂ vs σ_true,对照 BSM(σ21d) persistence 基线 |
| Chooser 价格 | 各模型 σ̂ 代入 BSM 的 Chooser 价 vs 前向波动率公平价 |
| 期权价格 (方法二) | 端到端预测价 vs 公平价,对照 BSM(σ21d) 静态基准 |

**Week 5 改善目标** (来自 Week 4 报告附):
- 波动率预测 MAE 显著低于 persistence 基线
- Chooser 定价误差低于 BSM(σ21d)
- 远期目标 (Week 6): 实盘 MAE < $1.00, ATM IV 溢价 < 2%, OTM 看跌偏差 < 30%

---

## 7. 目录结构与运行

```
week5/
├── config.py                           # 全局配置
├── data_preparation.py                 # 时序分割 70/15/15 + 无前瞻设计
├── targets.py                          # 目标构造 (前向波动率/VIX/合约标签)
├── feature_engineering_optimization.py # 交付② 特征工程优化
├── models/
│   ├── base.py                         # BaseModel 接口
│   ├── volatility_models.py            # 方法一: RF/GBDT/XGB/LSTM
│   ├── pricing_models.py               # 方法二: Linear/GBDT/NN
│   └── bsm_engine.py                   # BSM 混合定价引擎
├── evaluation.py                       # 指标 + 基线对比
├── ml_framework.py                     # 双轨主流程
├── run_experiments.py                  # CLI 入口
├── test_ml_framework.py                # 测试
├── output/                             # 结果 CSV + 运行日志
└── assets/                             # 可视化图表
```

**运行**:
```bash
python test_ml_framework.py     # 37 项测试
python run_experiments.py --feats   # 特征报告 + 双轨全流程
```

---

## 8. Week 5 初始实验结果与关键发现

### 8.1 方法一: 波动率预测 (test 集 2024, 相对 persistence 基线的改进%)

| 模型 | 波动率 MAE | 相对 BSM(σ21d) |
|------|-----------|----------------|
| **BSM(σ21d) persistence** | **7.32%** | 基线 |
| GBDT-anchored (vol_ratio) | 7.42% | −1.3% |
| GBDT | 9.48% | −29% |
| XGBoost | 9.51% | −30% |
| RandomForest | 10.21% | −39% |
| LSTM (torch) | ~13% | 弱 |

### 8.1b 实盘指标 (与 Week 4 相同的 2026-07-27 期权快照, 595 合约)

用 ML 预测波动率重新定价,重算三项实盘指标 (详见 `compute_live_metrics.py`):

| 模型 | σ 输入 | 实盘 MAE ($) | ATM IV 溢价 (%) | OTM 偏差 (%) |
|------|--------|-------------|----------------|--------------|
| **RandomForest** | 22.16% | **1.03** | **3.0** | **−51.4** |
| XGBoost | 21.39% | 1.19 | 3.8 | −57.6 |
| GBDT | 21.28% | 1.21 | 3.9 | −58.4 |
| GBDT-anchored | 20.89% | 1.29 | 4.3 | −61.1 |
| BSM(σ21d) 基线 | 20.21% | 1.44 | 4.9 | −65.7 |

ML 预测波动率 (22.2%) 高于静态 σ21d (20.2%),部分捕捉市场 IV 溢价 (ATM IV=25.2%),三项实盘指标同步改善。

### 8.1c Chooser 定价 (test 集, 混合引擎对比)

| 模型 | 定价 MAE ($) | 结论 |
|------|-------------|------|
| **GBDT-anchored (ML σ)** | **$1.46** | **击败 BSM 基线** |
| BSM(σ21d) 基线 | $1.52 | Week 4 基准 |
| GBDT / XGBoost / RF | $2.14–$2.58 | 未及基线 |
| LSTM | $5.19–$5.59 | 弱 |

### 8.2 方法二: 端到端定价 (test 集)

| 模型 | 价格 MAE ($) | R² |
|------|-------------|-----|
| BSM(σ21d) static-vol 基准 | 1.77 | 0.93 |
| GBDT | 3.75 | 0.64 |
| MLP-NeuralNet | 5.11 | 0.40 |
| LinearRegression | 7.08 | −0.05 |

### 8.3 关键发现 (诊断已确认)

1. **波动率持久性是极强基线**: 今日 σ21d 自相关高,初始模型难以超越,这是波动率预测文献中的已知事实。
2. **Regime 漂移**: 训练期 (2018-2022) 均值波动率 26.6% vs 测试期 (2024) 19.2%。严格时序分割迫使模型外推到不同 regime;persistence 天然自适应。
3. **过拟合噪声目标**: GBDT 训练 R²≈0.97、测试 R² 为负——已实现波动率作为目标噪声大。
4. **行业标准缓解 + 正向结果**: `GBDT-anchored` (预测 vol_ratio 而非水平) 把模型锚定在当前波动率水平,波动率误差收敛至接近 persistence,**并在 Chooser 定价上击败 BSM 基线 ($1.46 vs $1.52)**——证明混合定价管线已产生真实价值。
5. **方法二**: 端到端模型捕捉大部分定价结构 (R² 0.4-0.93),但静态波动率基准因持久性仍难超越。

> **价值**: 初始框架精确界定了 Week 6 的优化方向,并把"必须击败 persistence"确立为一切比较的硬性门槛。真实 JPM IV 接入后,以市场过滤后的 IV 为目标(而非噪声的已实现波动率)是预期收益最大的方向。

## 9. Week 6+ 扩展方向

1. **超参优化**: 网格/随机搜索 + 时序交叉验证 (config 已集中超参);针对过拟合加大正则
2. **真实 IV 接入**: 实时期权数据 → 替换方法一目标为 JPM 历史 IV (市场过滤信号,预期可击败 persistence)
3. **Regime 自适应**: 以 regime (Week 4 分类) 为条件特征或分模型
4. **可解释性**: SHAP/LIME 特征重要性(与 `feature_importance.csv` 对比)
5. **工具集成**: 最优模型导出 pickle,Week 7-8 接入 Streamlit/FastAPI 定价工具
