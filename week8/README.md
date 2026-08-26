# Chooser Option Pricing Tool — 基于真实数据与机器学习的双轨定价工具

高级选择权（Chooser Option）定价工具：以 **BSM 基线 + 最优机器学习波动率模型** 双轨定价，内置完整可视化仪表板（价格趋势、敏感性分析、极端场景、性能指标、误差范围、实时数据）。

本仓库为 8 周量化实习项目的**最终交付物（Week 8）**。全项目按周划分于 `E:/实习交付/week2/week1..week8`；本目录为可部署工具与最终报告/演示。

## 功能

- **双轨定价**：BSM(vol_21d) 基线 vs 最优 ML 波动率模型（XGBoost = 实盘首选 / VIX-proxy = 隐含波动率对齐 / GBDT-anchored = 波动率比率），基于 Week-3 验证的 Rubinstein 简单选择权公式。
- **误差范围**：每项定价附带 Week-6 测试集 MAE/RMSE（`价格 ± MAE`）。
- **可视化仪表板**（6 个标签页）：
  - 📈 价格趋势：2018–2024 历史双轨重定价 + 实时市场点
  - 📊 敏感性分析：价格 vs S / σ / t1 / r 交互图表
  - 🌪 极端场景：波动率 +50% / 利率 +2% / 组合 / VIX 冲击
  - 🏆 性能指标：Week-6 测试集与实盘快照指标图表
  - 🔬 误差范围：各模型测试集 MAE/RMSE/R²
  - 📡 实时数据：JPM/VIX/3M 最新状态 + 自动更新
- **实时数据集成**：`data_updater.py` + GitHub Actions 每工作日自动刷新市场数据，增量合并进特征数据集。

## 项目结构

```
week8/
├── streamlit_app.py        # 定价工具（Streamlit 完整仪表板）
├── dashboard.py            # 价格趋势序列 + 性能指标图表（Week-8 新增）
├── w8config.py             # 路径/常量配置（复用 week5–week7）
├── run_week8.py            # 全流程编排（仪表板 + 自检 + 报告 + 演示）
├── test_week8.py           # 单元测试 + Streamlit AppTest
├── make_presentation.py    # 生成最终演示文稿 PPTX
├── capture_ui.py           # 抓取工具 UI 截图（Playwright）
├── demo_script.md          # 5–10 分钟工具演示视频脚本
├── Week_8_实验报告.tex     # 最终报告源（11 页）
├── requirements.txt
├── data_updater.py / sensitivity.py / tool_engine.py   # 复用 week7（见下）
├── assets/                 # 报告图表 + UI 截图
├── output/                 # 价格趋势 CSV、指标
└── presentation/           # Week_8_Final_Presentation.pptx
```

> **依赖的兄弟目录**：工具复用第 5–7 周已训练/验证的资产，需与以下目录并列存在：
> `week3`（`chooser_option_pricer.py`）、`week5`（特征/目标构建）、`week6`（`models/*.pkl` 训练模型与测试集指标）、`week7`（`tool_engine.py` / `sensitivity.py` / `data_updater.py`）。

## 安装与运行

```bash
cd E:/实习交付/week2/week8
pip install -r requirements.txt          # + week2/week5 依赖（streamlit、plotly、shap、xgboost、joblib、yfinance）

python test_week8.py                     # 单元测试（含 Streamlit AppTest）
python run_week8.py --no-tex             # 全流程（仪表板 + 工具自检 + 数据新鲜度）
python run_week8.py                      # 全流程 + 编译最终报告 + 生成演示文稿

streamlit run streamlit_app.py           # 启动定价工具（http://localhost:8511）

python data_updater.py --check           # 实时数据新鲜度（只读）
python data_updater.py --dry-run         # 拉取 + 重建特征，报告增量（不写入）
python data_updater.py --commit          # 增量合并到 week2/data/feature_dataset.csv
```

## 实时数据自动更新（GitHub Actions）

`.github/workflows/week8_data_update.yml` 每工作日 06:30 UTC 运行 `--commit`，自动拉取 JPM / VIX / IRX、重建特征并增量合并。离线缓存回退默认被拒绝（避免模拟数据污染训练集，需 `--force`）。

## 关键结论

1. **以市场隐含波动率为目标是定价改善最大来源**：VIX-proxy 在合成测试集的 Chooser 定价 MAE 上以 \$1.17 领先 BSM 基线（\$1.52）**23%**。
2. **超参调优越过 persistence 门槛**：GBDT-anchored 波动率 MAE 7.16% 首次优于基线 7.32%。
3. **实盘三项目标全部达成**：调优后 XGBoost 实盘 MAE \$0.79、ATM IV 溢价 0.76%、OTM 看跌偏差 -29.5%。
4. **情感/VIX 新特征是价格第一驱动**：sentiment_score 的 SHAP×Vega 价格影响是次高特征的 3 倍。
5. **双轨结论**：将 ML 集中于 BSM 唯一不确定的输入（波动率）优于端到端重建定价函数。

## 交付物

| 交付物 | 文件 |
|---|---|
| 可部署定价工具 | `streamlit_app.py` + 本 README（GitHub 仓库） |
| 最终项目报告（11 页 PDF） | `E:/实习交付/实验报告/Week_8_实验报告.pdf` |
| 结项报告（19 页 PDF，≥8000 字） | `E:/实习交付/实验报告/Week_1-8_结项报告.pdf` |
| 最终演示文稿（14 页） | `presentation/Week_8_Final_Presentation.pptx` |
| 工具演示视频脚本（5–10 分钟） | `demo_script.md` |

## 参考文献

- Rubinstein, M. (1991). *Chooser Options*.
- Hull, J. C., & White, A. (1987). *The Pricing of Options on Assets with Stochastic Volatilities*.
- Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing with Machine Learning*.
- Bakshi, G., Cao, C., & Chen, Z. (1997). *Empirical Performance of Alternative Option Pricing Models*.
