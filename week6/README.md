# Week 6 — 模型训练、调优与比较 (Model Training, Tuning & Comparison)

8周实习项目「Advanced Chooser Option Pricing Model with Real-World Data & Machine Learning」的第 6 周交付。

在 Week 5 双轨 ML 架构与 BSM 基线 ($1.44 MAE / 4.9% IV 溢价) 基础上完成：
- **超参数优化**：基于 purge 时序交叉验证的随机搜索，针对过拟合系统性加大正则
- **最终模型训练**：最优超参在 train+val 重训，held-out 测试集一次性评估
- **性能比较**：调优前后 + 分 regime + 实盘指标，ML vs BSM 基线 (MAE/RMSE/R²)
- **可解释性分析**：SHAP 全局/局部重要性 + LIME 局部解释
- **Regime 自适应特征** + **VIX-proxy（市场隐含波动率）目标**

---

## 交付成果

| # | 交付物 | 文件 |
|---|--------|------|
| 1 | 训练完成的 ML 模型 (pickle) | [`models/*.pkl`](models/) + JSON 元数据 |
| 2 | 对比分析报告（含全部指标） | [`Week6实验报告.pdf`](Week6实验报告.pdf) |
| 3 | 特征重要性可视化 (SHAP 图) | [`assets/fig_w6_shap_*.png`](assets/) |
| 4 | 实盘指标对照 (Week 6 目标) | [`output/live_metrics_w6.csv`](output/) |

---

## 目录结构

```
week6/
├── w6config.py                  # 配置：超参搜索空间 / purge CV / regime / 目标
├── cv.py                        # Purged 时序交叉验证（扩张窗口 + embargo）
├── regime.py                    # Week 4 五状态市场分类 → 后视 one-hot 特征
├── hp_search.py                 # 超参数优化（随机搜索 + purge CV）
├── train_final.py               # 最终训练（train+val）+ 测试评估 + pickle 导出
├── performance.py               # 性能比较表 / 分 regime 误差 / 对比图
├── interpretability.py          # SHAP beeswarm/bar/dependence + LIME 局部解释
├── live_metrics.py              # 2026-07-27 快照实盘指标 vs Week 6 目标
├── run_week6.py                 # 主流程编排
├── test_week6.py                # 测试（CV/regime/搜索/导出）
├── Week6实验报告.tex/.pdf       # 实验报告
├── models/                      # 8 个 pickle 模型 + 元数据
├── output/                      # 结果 CSV + 运行日志
└── assets/                      # 可视化 + SHAP 图
```

## 运行方式

```bash
# 1. 测试
python test_week6.py

# 2. 完整 Week 6 流程（复用缓存的超参搜索结果）
python run_week6.py

# 3. 重新执行超参搜索（耗时 ~5 min）后全流程
python run_week6.py --search
```

依赖：Week 5 requirements + `shap` + `lime`。

---

## 关键设计

### 1. Purged 时序交叉验证（超参选择）
```
外层分割:  [----- train -----][GAP][--- val ---][GAP][--- test ---]
                                           │ test 全程不参与调优
内层 CV:   fold k: train=[0,te_k)  val=[te_k+21, te_k+21+val_size)
```
- 扩张窗口 + 21 日 embargo，严格时序、无重叠、无前瞻。
- Scaler 放入 `Pipeline(StandardScaler, Model)`，每个 CV 折的 scaler 只在该折训练块拟合。

### 2. 反前瞻保障（沿用 Week 5 + 新增）
- 70/15/15 时序分割、Purge/Embargo、特征后视、目标错位、Scaler 训练集拟合、LSTM 窗口后视。
- **特征选择只在训练期运行**；**内层 CV 的 scaler 逐折拟合**。

### 3. Regime 自适应特征
Week 4 五状态分类 (BULL/BEAR/HIGH_VOL/CALM/NORMAL) → 后视 one-hot 特征（NORMAL 为参照），全部模型并入。

### 4. VIX-proxy 目标（市场隐含波动率）
真实 JPM 单票历史 IV 不可得（仅单日快照含 IV），以 **VIX 为市场隐含波动率代理**：预测 VIX$_{t+1}$，以 VIX/100 作为 BSM 波动率输入，逼近"以市场过滤后的 IV 为目标"的设计意图。

---

## 关键实验结果

### 波动率预测（测试集 2024，含 regime 特征）
| 模型 | MAE | 相对 persistence |
|------|-----|------------------|
| **GBDT-anchored (vol_ratio)** | **7.16%** | **−2.2% (首次击败)** |
| BSM(σ21d) persistence | 7.32% | 基线 |
| XGB-VIX proxy (IV) | 7.46% | +1.9% |
| GBDT / RF / XGB | 8.36–8.42% | +14% |
| LSTM | 10.92% | +49% |

**超参调优使 GBDT-anchored 首次超越 persistence 基线**（Week 5 时 7.42% 仍略逊于 7.32%）。

### Chooser 定价（测试集，vs 前向波动率公平价）
| 模型 | 定价 MAE ($) | 相对 BSM |
|------|-------------|----------|
| **XGB-VIX proxy (IV)** | **1.17** | **−23%** |
| BSM(σ21d) 基线 | 1.52 | — |
| GBDT-anchored | 1.55 | +2% |

**VIX-proxy（市场隐含波动率目标）将 Chooser 定价 MAE 从 BSM 基线 $1.52 降至 $1.17（改善 23%）**，优于 Week 5 最佳 ($1.46)。

### 端到端定价（方法二）
| 模型 | MAE ($) | R² |
|------|---------|-----|
| BSM(σ21d) 静态基准 | 1.77 | 0.93 |
| GBDT（调优） | 3.69 | 0.67 |
| MLP-NN（调优） | 4.61 | 0.51 |

调优改善但仍无法超越静态 BSM 基准——确认方法二价值需接入真实期权价格标签后体现。

### 实盘指标（2026-07-27 快照, 595 合约）—— 里程碑 3 全部达成
| 指标 | Week 4 基线 | Week 5 (RF) | **Week 6 最优 (XGB)** | Week 6 目标 | 达成 |
|------|-----------|-------------|----------------------|------------|------|
| 实盘 MAE ($) | 1.44 | 1.03 | **0.79** | < 1.00 | ✅ |
| ATM IV 溢价 (%) | 4.9 | 3.0 | **0.76** | < 2.0 | ✅ |
| OTM 看跌偏差 (%) | −65.7 | −51.4 | **−29.5** | > −30 | ✅ |
| Chooser MAE ($, 合成集) | 1.52 | 1.46 | **1.17** (VIX-proxy) | 持续下降 | ✅ |

> 调优后 XGBoost 预测波动率 24.4%，贴近市场 ATM IV (25.2%)，捕获波动率溢价。
> 合成集 vs 实盘集排序逆转：合成集 VIX-proxy 最优 ($1.17)，实盘集 XGB/GBDT 最优（贴近市场隐含波动率）。

---

## 里程碑对照

| 里程碑 | 状态 |
|--------|------|
| 超参数优化（purge 时序 CV） | ✅ 完成 |
| 最终模型训练 + pickle 导出 | ✅ 完成 (models/*.pkl) |
| 性能比较（MAE/RMSE/R² vs BSM） | ✅ 完成 |
| SHAP/LIME 可解释性分析 | ✅ 完成 |
| Regime 自适应特征 + VIX-proxy 目标 | ✅ 完成 |
| 敏感性分析 + 定价工具 (Week 7) | 📋 下一周 |
