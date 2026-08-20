# Week 7 — 高级分析与工具开发
## Advanced Analysis & Tool Development

扩展敏感性分析、双轨定价工具原型与实时数据集成。

### 交付物 (Deliverables)

| 交付物 | 文件 | 说明 |
|---|---|---|
| 敏感性分析报告 | `Week_7_实验报告.pdf` | SHAP×Vega 价格影响 + 极端场景测试 |
| 定价工具原型 | `streamlit_app.py` | 双轨定价 UI（BSM + 最优 ML 模型） |
| 实时数据集成 | `data_updater.py` + `.github/workflows/week7_data_update.yml` | 市场数据自动更新模块 |
| 工具引擎 | `tool_engine.py` | 模型加载 / 特征构建 / 双轨定价 / 误差条 |
| 敏感性分析 | `sensitivity.py` | SHAP×Vega 分解 + 单变量扰动 + 极端场景 |
| 测试 | `test_week7.py` | 28 项单元断言 |
| 调度 | `run_week7.py` | 全流程编排 + 报告编译 |
| UI 截图 | `capture_ui.py` | Playwright 无头抓取工具 UI 截图（写入报告） |

### 运行方式

```bash
cd E:/实习交付/week2/week7

python test_week7.py            # 单元测试
python run_week7.py --no-tex    # 全流程（敏感性 + 工具自检 + 数据新鲜度）
python run_week7.py             # 全流程 + 编译 Week_7_实验报告.pdf

python data_updater.py --check  # 实时数据新鲜度（只读）
python data_updater.py --dry-run
python data_updater.py --commit # 增量合并到 week2/data/feature_dataset.csv

streamlit run streamlit_app.py  # 启动定价工具原型
```

### 依赖

```bash
pip install -r requirements.txt   # + week2 / week5 依赖（见工作流 week7_data_update.yml）
```

### 关键结论 (Week 7)

1. **情感/VIX 新特征是价格第一驱动**：VIX-proxy 模型下 `sentiment_score` 的平均价格影响（\$0.314）是次高特征的三倍；单变量扰动显示情感 0.4→1.0 使 ATM 价格 -18%。
2. **ATM 基座下波动率风险放大**：波动率 +50% → BSM +49%、XGB +70%、VIX-proxy +81%；ML 通过特征重工程捕获 VIX 冲击（+14%~+21%）。
3. **实盘基座深度实值**：$S=356 \gg K=150$ 时 Chooser 由内在价值主导，波动率敏感性趋零（<0.3%）。
4. **实时数据**：本地回退快照 (2026-07-27)；GitHub Actions 每工作日 06:30 UTC 自动刷新。
