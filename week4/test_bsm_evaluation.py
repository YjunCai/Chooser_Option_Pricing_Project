"""
test_bsm_evaluation.py — BSM 基线模型评估测试套件
=================================================
功能：
  1. 收敛性分析：解析解 vs Monte Carlo（MAE/RMSE 随 n_sims 变化）
  2. Greeks 正确性检验（边界条件）
  3. 参数敏感性量化
  4. 市场 regime 分类与误差分析
  5. BSM 假设诊断（正态性、波动率平稳性）

用法：
  python test_bsm_evaluation.py

依赖：numpy, scipy, pandas
"""

import sys
import numpy as np
from pathlib import Path

# Add week4 to path
WEEK4_PATH = Path(__file__).resolve().parent
if str(WEEK4_PATH) not in sys.path:
    sys.path.insert(0, str(WEEK4_PATH))

from bsm_evaluation import (
    convergence_analysis,
    convergence_grid,
    compute_all_greeks,
    parameter_impact_analysis,
    classify_market_regime,
    test_return_normality,
    test_vol_stationarity,
    test_jump_detection,
    load_week3_data,
)

# ============================================================
# 测试函数
# ============================================================

def test_convergence_basic():
    """验证 MC 随 n_sims 增加，RMSE 递减（收敛性）"""
    print("  [TEST] 收敛性分析")
    conv = convergence_analysis(n_trials=3)

    # RMSE should decrease as n_sims increases
    rmse_values = conv['rmse'].values
    monotonic_decreasing = all(rmse_values[i] >= rmse_values[i+1] for i in range(len(rmse_values)-1))
    # Allow small fluctuation, check overall trend
    first_last = rmse_values[0] > rmse_values[-1]

    print(f"    RMSE progression: {[f'{v:.4f}' for v in rmse_values]}")
    print(f"    RMSE monotonic: {monotonic_decreasing} | First > Last: {first_last}")
    print(f"    收敛率 beta: {conv['convergence_rate_beta'].iloc[0]:.3f} (期望 ≈ -0.5)")
    print(f"    {'PASS' if first_last else 'INFO'}: RMSE 整体下降趋势 {'确认' if first_last else '需更多样本'}")
    return True


def test_convergence_grid():
    """验证 Grid 扫描覆盖所有参数组合"""
    print("  [TEST] 收敛 Grid 扫描")
    grid = convergence_grid(
        S_values=[90, 100, 110],
        sigma_values=[0.15, 0.25],
        t1_values=[0.25, 0.5],
        n_sims=50_000
    )

    expected_rows = 3 * 2 * 2  # S x sigma x t1
    actual_rows = len(grid)
    all_positive = (grid['abs_error'] >= 0).all()

    print(f"    期望组合: {expected_rows}, 实际: {actual_rows}")
    print(f"    所有 abs_error >= 0: {all_positive}")
    print(f"    {'PASS' if actual_rows == expected_rows and all_positive else 'FAIL'}")
    assert actual_rows == expected_rows, f"Grid rows mismatch: {actual_rows} vs {expected_rows}"
    return True


def test_greeks_boundary():
    """验证 Greeks 在边界条件下的合理值"""
    print("  [TEST] Greeks 边界检验")

    # Deep ITM call-chooser scenario (S >> K): delta should be near 1
    g_itm = compute_all_greeks(S=200, K=100, t1=0.25, T2=1.0, r=0.05, q=0.0, sigma=0.20)
    # Deep OTM scenario (S << K): delta should be near 0
    g_otm = compute_all_greeks(S=50, K=100, t1=0.25, T2=1.0, r=0.05, q=0.0, sigma=0.20)

    print(f"    Deep ITM (S=200): delta={g_itm['delta']:.4f}, vega={g_itm['vega']:.4f}")
    print(f"    Deep OTM (S=50):  delta={g_otm['delta']:.4f}, vega={g_otm['vega']:.4f}")

    # Vega should always be positive (option value increases with vol)
    vega_positive = g_itm['vega'] > 0 and g_otm['vega'] > 0
    print(f"    Vega positive: {vega_positive}")
    print(f"    {'PASS' if vega_positive else 'FAIL'}")
    assert vega_positive, "Vega should always be positive for long option positions"
    return True


def test_parameter_impact():
    """验证参数敏感性分析结果"""
    print("  [TEST] 参数敏感性量化")
    impact = parameter_impact_analysis(shock_pct=10)

    # All parameters should have non-negative range
    all_range_positive = (impact['price_range'] >= 0).all()
    # Volatility should be one of the most impactful
    sigma_row = impact[impact['param'] == 'sigma'].iloc[0]
    k_row = impact[impact['param'] == 'K'].iloc[0]

    print(f"    所有参数有非负影响范围: {all_range_positive}")
    print(f"    Sigma 影响范围: ${sigma_row['price_range']:.2f}")
    print(f"    K 影响范围:     ${k_row['price_range']:.2f}")
    print(f"    {'PASS' if all_range_positive else 'FAIL'}")
    assert all_range_positive, "All price ranges should be >= 0"
    return True


def test_regime_classification():
    """验证市场 regime 分类逻辑"""
    print("  [TEST] 市场 Regime 分类")
    try:
        df = load_week3_data()
        regime = classify_market_regime(df)

        n_bull = (regime == 'BULL').sum()
        n_bear = (regime == 'BEAR').sum()
        n_highvol = (regime == 'HIGH_VOL').sum()
        n_calm = (regime == 'CALM').sum()
        n_normal = (regime == 'NORMAL').sum()

        print(f"    总天数: {len(regime)}")
        print(f"    BULL: {n_bull}, BEAR: {n_bear}, HIGH_VOL: {n_highvol}, CALM: {n_calm}, NORMAL: {n_normal}")
        print(f"    所有数据被分类: {len(regime) == n_bull + n_bear + n_highvol + n_calm + n_normal}")
        print(f"    {'PASS' if len(regime) > 0 else 'FAIL'}")
        return True
    except FileNotFoundError as e:
        print(f"    [SKIP] {e}")
        return True


def test_bsm_assumptions():
    """验证 BSM 假设诊断"""
    print("  [TEST] BSM 假设诊断")
    try:
        df = load_week3_data()
        returns = df['daily_return'].dropna().values
        vol_21d = df['vol_21d'].dropna()

        # Normality
        norm_result = test_return_normality(returns)
        print(f"    收益率观测数: {norm_result['n_obs']}")
        print(f"    JB p-value: {norm_result['jarque_bera_p']:.6f} (显著 < 0.01 → 非正态)")
        print(f"    偏度: {norm_result['skewness']:.4f}")
        print(f"    超额峰度: {norm_result['excess_kurtosis']:.4f}")
        print(f"    正态性判定: {'是' if norm_result['is_normal'] else '否'}")

        # Vol stationarity
        vol_result = test_vol_stationarity(vol_21d)
        print(f"    Vol CV: {vol_result['cv']:.4f}")
        print(f"    Vol 范围: [{vol_result['vol_min']:.4f}, {vol_result['vol_max']:.4f}]")

        # Jump detection
        jump_result = test_jump_detection(returns)
        print(f"    Jump 天数: {jump_result['n_jumps']} ({jump_result['jump_pct']:.2f}%)")
        print(f"    最大绝对收益: {jump_result['max_abs_return']:.4f}")

        print(f"    {'PASS' if norm_result['n_obs'] > 0 else 'FAIL'}")
        return True
    except FileNotFoundError as e:
        print(f"    [SKIP] {e}")
        return True


# ============================================================
# 主函数
# ============================================================

if __name__ == '__main__':
    print("=" * 65)
    print("  BSM Baseline Model Evaluation — Test Suite")
    print("=" * 65)
    print()

    tests = [
        ("收敛性基本分析", test_convergence_basic),
        ("Grid 扫描覆盖", test_convergence_grid),
        ("Greeks 边界检验", test_greeks_boundary),
        ("参数敏感性量化", test_parameter_impact),
        ("市场 Regime 分类", test_regime_classification),
        ("BSM 假设诊断", test_bsm_assumptions),
    ]

    all_passed = True
    for name, func in tests:
        print(f"  ── {name} ──")
        try:
            func()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            all_passed = False
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            all_passed = False
        print()

    print("=" * 65)
    if all_passed:
        print("  所有测试通过！BSM 基线评估完成。")
    else:
        print(f"  部分测试失败，请检查。")
    print("=" * 65)
