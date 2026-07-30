# Week 4 — BSM 基线模型性能评估 (BSM Baseline Evaluation)

对 Week 3 实现的 BSM Chooser Option 定价模型进行全面性能评估，
包括收敛性分析、Greeks 风险度量、参数敏感性量化、市场 Regime 误差分析
以及 BSM 假设违背诊断。

---

## 目录结构

```
week4/
├── bsm_evaluation.py             # 核心评估模块
├── test_bsm_evaluation.py        # 测试与验证脚本
├── week4_experiment_report.ipynb # 完整实验报告（含可视化）
├── README.md                     # 本文件
└── assets/                       # 生成的可视化图表
    ├── fig_4_1_convergence.png           # 收敛性分析
    ├── fig_4_2_accuracy_heatmap.png      # 参数空间精度热力图
    ├── fig_4_3_greeks_surface.png        # Greeks 曲面
    ├── fig_4_4_greeks_vs_t1.png          # Greeks vs 选择日
    ├── fig_4_5_tornado.png               # 参数敏感性 tornado 图
    ├── fig_4_6_full_sensitivity.png      # 全参数敏感性曲线
    ├── fig_4_7_regime_classification.png # 市场 Regime 分类
    ├── fig_4_8_regime_error.png          # Regime 误差分布
    └── fig_4_9_bsm_diagnostics.png       # BSM 假设诊断
```

---

## 文件说明

### 1. `bsm_evaluation.py` — 核心评估模块

| 功能 | 函数 | 说明 |
|------|------|------|
| 收敛性分析 | `convergence_analysis()` | 解析解 vs MC 多 n_sims 对比 (MAE/RMSE) |
| Grid 精度扫描 | `convergence_grid()` | 参数空间中的精度分布 |
| Greeks | `compute_all_greeks()` | Delta, Gamma, Vega, Rho, Theta |
| Greeks 扫描 | `greeks_surface_scan()` | Greeks 随参数变化曲面 |
| 参数敏感性 | `parameter_impact_analysis()` | Tornado 冲击分析 |
| Regime 分类 | `classify_market_regime()` | Bull/Bear/High-Vol/Calm 分类 |
| BSM 假设诊断 | `full_bsm_assessment()` | 正态性、波动率、跳跃检验 |

### 2. `test_bsm_evaluation.py` — 测试与验证脚本

```bash
python test_bsm_evaluation.py
```

| 测试项 | 验证内容 |
|--------|---------|
| 收敛性分析 | RMSE 随 n_sims 增加递减 |
| Grid 覆盖 | 所有参数组合的误差计算 |
| Greeks 边界 | Vega 始终为正 |
| 参数敏感性 | 所有参数影响范围非负 |
| Regime 分类 | 完整分类覆盖 |
| BSM 假设 | 正态性、波动率、跳跃诊断 |

### 3. `week4_experiment_report.ipynb` — 实验报告

Jupyter Notebook 格式的完整报告，包含 15+ 个 Cell：

| Cell | 内容 |
|------|------|
| 1–2 | 依赖导入与配置 |
| 3–4 | 收敛性分析 (MAE/RMSE 表 + 可视化) |
| 5–6 | Grid 精度扫描 + 热力图 |
| 7–8 | Greeks 基准值与曲面分析 |
| 9 | Greeks vs 选择日变化 |
| 10–11 | 参数敏感性 tornado + 全曲线 |
| 12–13 | 市场数据加载与 Regime 分类 |
| 14–15 | Regime 误差分析 + 箱线图 |
| 16–17 | BSM 假设诊断 + 总结表 |
| 18 | 局限性总结与改进方向 |

---

## 评估维度

### 1. 数值精度 (Numerical Accuracy)
- **方法**: 解析解 vs Monte Carlo (n_sims: 1K–500K)
- **指标**: MAE, RMSE, 偏差百分比, 收敛速率 β
- **预期**: RMSE ∝ 1/√n_sims, β ≈ -0.5

### 2. Greeks 风险度量
- **Delta**: 平价敏感性, ∂V/∂S
- **Gamma**: Delta 凸性, ∂²V/∂S²
- **Vega**: 波动率敏感性, ∂V/∂σ
- **Rho**: 利率敏感性, ∂V/∂r
- **Theta**: 时间敏感性, ∂V/∂t₁

### 3. 参数敏感性
- Tornado 图显示 ±10% 冲击的价格影响范围
- 排序关键风险因子

### 4. 市场 Regime 误差
- Bull/Bear/High-Vol/Calm/Normal 五状态分类
- 各 Regime 下解析解 vs MC 的误差统计

### 5. BSM 假设诊断
- Jarque-Bera 正态性检验
- 波动率变异系数 (CV)
- 跳跃频率检测
- 综合适用性评分 (0-100)

---

## 核心发现

| 维度 | 结果 |
|------|------|
| 数值精度 | MAE < $0.01 at 200K MC paths |
| 收敛速率 | β ≈ -0.5 (符合 O(1/√n) 理论) |
| 最强风险因子 | Volatility (Vega) |
| 主要局限性 | 波动率时变性 (CV > 0.5) |
| BSM 评分 | ~50/100 (中等适用性) |

---

## 后续周次

- **Week 5–6**: ML 波动率预测 (LSTM/RF/XGBoost) 与端到端定价
- **Week 7–8**: 定价工具开发 (Streamlit/FastAPI) 与最终交付
