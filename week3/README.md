# Week 3 — Chooser Option 定价模型 (BSM)

基于 Rubinstein (1991) 的 Black-Scholes 框架选择者期权（Chooser Option）定价模型，
结合 JPMorgan 真实市场数据（2018–2024）进行实盘验证与敏感性分析。

---

## 目录结构

```
week3/
├── chooser_option_pricer.py      # 核心定价模块
├── test_chooser_pricer.py        # 测试与验证脚本
├── week3_experiment_report.ipynb # 完整实验报告（含可视化）
├── README.md                     # 本文件
└── assets/                       # 生成的可视化图表
    ├── fig_1_8.png               # Chooser 价格时间序列与分解
    ├── fig_2_9.png               # 敏感性分析（6 面板）
    └── fig_3_11.png              # 多 t₁ 价格对比
```

---

## 文件说明

### 1. `chooser_option_pricer.py` — 核心定价模块

独立可用的 Python 模块，包含全部定价函数，无外部依赖（仅需 `numpy`、`scipy`）。

| 函数 | 说明 |
|------|------|
| `bs_call(S, K, T, r, q, sigma)` | Black-Scholes-Merton 看涨期权定价 |
| `bs_put(S, K, T, r, q, sigma)` | Black-Scholes-Merton 看跌期权定价 |
| `simple_chooser(S, K, t1, T2, r, q, sigma)` | Rubinstein (1991) 简单选择者期权解析定价 |
| `simple_chooser_alt(S, K, t1, T2, r, q, sigma)` | 基于 put-call parity 的等价定价形式 |
| `simple_chooser_decomposed(S, K, t1, T2, r, q, sigma)` | 分解为基础看涨部分 + 延期抉择权价值 |
| `simple_chooser_mc(S, K, t1, T2, r, q, sigma, n_sims)` | Monte Carlo 模拟验证 |
| `complex_chooser(S_A, S_B, K_A, K_B, t1, T_A, T_B, r, q_A, q_B, sigma_A, sigma_B, rho)` | 复杂选择者期权（双标的资产，二元正态分布） |

### 2. `test_chooser_pricer.py` — 测试与验证脚本

独立可运行的测试套件，执行五项自动化验证：

```bash
python test_chooser_pricer.py
```

| 测试项 | 验证内容 |
|--------|---------|
| BSM 定价 | BS Call/Put 与已知标准值对比 |
| 边界条件 | t₁=0 → max(C,P)；t₁=T₂ → C+P (straddle)；0<t₁<T₂ → 在两者之间 |
| 解析 vs MC | Rubinstein 解析解与 200K 次 MC 模拟的偏差 < 0.1% |
| 参考论文对比 | 使用论文参数重现结果并对比偏差 |
| 单调性 | Chooser 价格关于 t₁ 严格单调递增 |

额外包含敏感性扫描功能，输出参数变化对价格的影响。

### 3. `week3_experiment_report.ipynb` — 实验报告

Jupyter Notebook 格式的完整报告，包含 14 个 Cell：

| Cell | 内容 |
|------|------|
| 1–2 | 依赖导入与 BSM 核心函数 |
| 3–4 | Simple Chooser 定价与 Monte Carlo 验证 |
| 5 | Complex Chooser 定价实现 |
| 6 | 特征数据集加载与股息率推算 |
| 7 | 实盘数据 Chooser 定价（1722 个交易日） |
| 8 | 价格时间序列可视化（3 面板） |
| 9 | 六维敏感性分析（含 Call/Put/Chooser 对比） |
| 10–11 | 敏感性总结表与多 t₁ 对比 |
| 12 | 复杂选择者实盘演示（JPM + VIX） |
| 13 | Pipeline 运行概览 |

---

## 核心公式

### Simple Chooser Option (Rubinstein, 1991)

$$V = Se^{-qT_2}\bigl[N(d_1) - N(-d_3)\bigr] - Ke^{-rT_2}\bigl[N(d_2) - N(-d_4)\bigr]$$

其中：
- $d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T_2}{\sigma\sqrt{T_2}},\quad d_2 = d_1 - \sigma\sqrt{T_2}$
- $d_3 = \frac{\ln(S/K) + (r - q)T_2 + \sigma^2 t_1/2}{\sigma\sqrt{t_1}},\quad d_4 = d_3 - \sigma\sqrt{t_1}$

### 边界条件

| 条件 | 值 | 含义 |
|------|-----|------|
| $t_1 = 0$ | $\max(C_{\text{BSM}}, P_{\text{BSM}})$ | 立即抉择 |
| $t_1 = T_2$ | $C_{\text{BSM}} + P_{\text{BSM}}$ | 到期抉择（跨式组合） |
| $0 < t_1 < T_2$ | $\max(C,P) < V < C+P$ | 正常区间 |

---

## 参数来源

| 参数 | 来源 | 说明 |
|------|------|------|
| $S$ (股价) | `close_jpm` (Yahoo Finance) | 2018–2024 日数据 |
| $r$ (无风险利率) | `value_treasury_3mo` (FRED) | 3 个月国债收益率 |
| $q$ (股息率) | yfinance JPM 股息记录推算 | 年化股息 / 股价 |
| $\sigma$ (波动率) | `vol_21d` (21 天滚动年化波动率) | 基于日收益率计算 |
| $K$ (行权价) | 固定 $150 (与参考论文一致) | |
| $T_2$ (到期) | 固定 1 年 | |
| $t_1$ (选择日) | 基准 0.5 年，敏感性分析可变 | |

---

## 模型验证结果

| 项目 | 结果 |
|------|------|
| Analytic vs MC (200K) | 偏差 0.011% |
| t₁=0 边界 | max(Call, Put) = \$12.11 → Chooser = \$12.11 ✓ |
| t₁=T₂ 边界 | C+P = \$16.52 → Chooser = \$16.52 ✓ |
| 单调性 | t₁ 从 0 到 T₂ 严格递增 ✓ |

### 参考论文参数对比 (S=156.7, K=150, r=0.15%, q=2.33%, σ=28.2%)

| 方法 | 价格 |
|------|------|
| Rubinstein 解析解 | \$29.13 |
| MC (200K) | \$29.13 ± \$0.04 |
| 论文原始 10 路径 | ~\$22.68 (未折现，样本量过小) |
| 扩展 10K 路径 MC | \$29.06 (折现后) |

---

## 依赖

- Python ≥ 3.8
- numpy
- scipy
- matplotlib
- seaborn
- pandas
- yfinance（可选，用于动态股息率查询）
- jupyter（用于运行 notebook）

---

## 参考论文

1. Huang, Z., Wang, X. & Wan, W. (2021). *Exploration of JPMorgan Chooser Option Pricing*. EMSD 2021, Volume 15.
2. Rubinstein, M. (1991). *Exotic Options*. Research Program in Finance Working Papers RPF-220, UC Berkeley.

---

## 后续周次

- **Week 4**：BSM 基线模型性能评估（MAE/RMSE 误差分析、局限性识别）
- **Week 5–6**：ML 波动率预测（LSTM/RF/XGBoost）与端到端定价
- **Week 7–8**：定价工具开发（Streamlit/FastAPI）与最终交付
