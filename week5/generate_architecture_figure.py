"""
Generate the Week-5 dual-track ML architecture diagram
======================================================
Produces assets/fig_w5_architecture.png (clean box-and-arrow diagram used in
the Week 5 experiment report).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# color palette
C_DATA = '#8ecae6'
C_FEAT = '#a8dadc'
C_A1 = '#219ebc'
C_A2 = '#fb8500'
C_BSM = '#457b9d'
C_EVAL = '#2a9d8f'
C_SIDE = '#6d6875'


def box(ax, x, y, w, h, text, fc, fs=10, tc='black', lw=1.2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.25,rounding_size=0.6',
                       linewidth=lw, edgecolor=tc, facecolor=fc, alpha=0.92)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc, linespacing=1.4)


def arrow(ax, xy1, xy2, color='#333333', lw=1.6, style='->', ls='-'):
    a = FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=16,
                        linewidth=lw, color=color, linestyle=ls)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(12, 8.2), dpi=160)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # ── 1. Data source ──────────────────────────────────────────────
    box(ax, 25, 90, 50, 7,
        '真实市场数据 (2018–2024)\nJPM日线 · VIX · 3Mo国债 · 股息 · 期权链', C_DATA, fs=11)

    # ── 2. Feature engineering ──────────────────────────────────────
    arrow(ax, (50, 89.2), (50, 84.5))
    box(ax, 22, 77, 56, 7,
        '特征工程 (Week2 16维 → 增强 34维 → 精选 15维)\n滚动波动率/收益形态/VIX联动/利率动量/情绪/滞后', C_FEAT, fs=10)

    # split line below feature box
    arrow(ax, (50, 76.2), (30, 70.3))
    arrow(ax, (50, 76.2), (70, 70.3))

    # ── 3. Approach 1 (left) ────────────────────────────────────────
    box(ax, 6, 61, 48, 8,
        '方法一 · Approach 1\nML 波动率预测 + BSM 定价', C_A1, fs=11, tc='white')
    arrow(ax, (30, 60.2), (30, 55.3))
    box(ax, 5, 45, 50, 10,
        '波动率模型\nRF · GBDT · XGBoost · LSTM\n'
        '目标: 前向已实现波动率 / VIX(IV代理)', '#e0fbfc', fs=9.5, tc='black')
    arrow(ax, (30, 44.2), (30, 38.3))
    box(ax, 5, 30, 50, 8,
        'BSM 混合定价引擎\nChooser Price = f(S, K, t1, T2, r, q, σ_ML)\n'
        '(Week3 Rubinstein 解析公式, 向量化)', C_BSM, fs=9.5, tc='white')

    # ── 4. Approach 2 (right) ───────────────────────────────────────
    box(ax, 55, 61, 40, 8,
        '方法二 · Approach 2\n端到端监督定价', C_A2, fs=11, tc='white')
    arrow(ax, (75, 60.2), (75, 55.3))
    box(ax, 54, 45, 42, 10,
        '定价模型\nLinear · GBDT · MLP-NN\n'
        '目标: 期权价格 (合约网格标签)\n'
        '特征: 市场特征 + moneyness/tenor/type', '#fdf0d5', fs=9.5, tc='black')
    arrow(ax, (75, 44.2), (75, 38.3))
    box(ax, 60, 30, 30, 8,
        '直接定价输出\n(不依赖 BSM 结构假设)', '#ffc8dd', fs=9.5, tc='black')

    # ── 5. Merge to evaluation ──────────────────────────────────────
    arrow(ax, (30, 29.2), (30, 22.3))
    arrow(ax, (75, 29.2), (75, 22.3))
    box(ax, 18, 12, 64, 10,
        '评估与对比\nMAE / RMSE / MAPE / R²   vs   BSM(σ21d) 基线\n'
        '波动率精度 · Chooser定价误差 · 期权定价误差', C_EVAL, fs=10, tc='white')

    # ── side note: anti-look-ahead ─────────────────────────────────
    box(ax, 86, 77, 13, 14,
        '时序分割\n70/15/15\n+ purge gap\n无前瞻偏差', C_SIDE, fs=8.5, tc='white')

    ax.set_title('Week 5 双轨 ML 定价架构 (Dual-Track ML Pricing Architecture)',
                 fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig('assets/fig_w5_architecture.png', bbox_inches='tight')
    plt.close(fig)
    print('saved assets/fig_w5_architecture.png')


if __name__ == '__main__':
    main()
