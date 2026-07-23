"""
test_chooser_pricer.py — Chooser Option 定价模型测试与验证脚本
================================================================
功能：
  1. 验证 BSM 看涨/看跌期权定价正确性
  2. 验证 Simple Chooser Option 解析公式 vs Monte Carlo
  3. 验证边界条件：t1=0 → max(C,P), t1=T2 → Straddle
  4. 与参考论文参数对比验证
  5. 敏感性分析扫描

用法：
  python test_chooser_pricer.py

依赖：numpy, scipy
"""

import numpy as np
from scipy.stats import norm

# ============================================================
# 1. BSM 核心定价函数
# ============================================================

def bs_call(S, K, T, r, q, sigma):
    """Black-Scholes-Merton 看涨期权定价"""
    if T <= 1e-10:
        return np.maximum(S - K, 0.0)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(np.maximum(S / K, 1e-10)) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, q, sigma):
    """Black-Scholes-Merton 看跌期权定价"""
    if T <= 1e-10:
        return np.maximum(K - S, 0.0)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(np.maximum(S / K, 1e-10)) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ============================================================
# 2. Simple Chooser Option 定价（Rubinstein 1991）
# ============================================================

def simple_chooser(S, K, t1, T2, r, q, sigma):
    """
    Rubinstein (1991) 简单选择者期权解析定价公式

    V = Se^{-qT2}[N(d1) - N(-d3)] - Ke^{-rT2}[N(d2) - N(-d4)]

    参数边界：
        t1=0     → max(C, P)     ··· 立即抉择
        t1=T2    → C + P         ··· 到期抉择（= 跨式组合）
        0<t1<T2  → max(C,P) < V < C+P
    """
    sigma = np.maximum(sigma, 1e-10)

    if t1 <= 0:
        return np.maximum(bs_call(S, K, T2, r, q, sigma),
                          bs_put(S, K, T2, r, q, sigma))
    if t1 >= T2:
        return bs_call(S, K, T2, r, q, sigma) + bs_put(S, K, T2, r, q, sigma)

    d1 = (np.log(np.maximum(S / K, 1e-10)) + (r - q + 0.5 * sigma ** 2) * T2) / (sigma * np.sqrt(T2))
    d2 = d1 - sigma * np.sqrt(T2)
    d3 = (np.log(np.maximum(S / K, 1e-10)) + (r - q) * T2 + 0.5 * sigma ** 2 * t1) / (sigma * np.sqrt(t1))
    d4 = d3 - sigma * np.sqrt(t1)

    disc_S = S * np.exp(-q * T2)
    disc_K = K * np.exp(-r * T2)
    return disc_S * (norm.cdf(d1) - norm.cdf(-d3)) - disc_K * (norm.cdf(d2) - norm.cdf(-d4))


# ============================================================
# 3. Monte Carlo 验证
# ============================================================

def simple_chooser_mc(S, K, t1, T2, r, q, sigma, n_sims=200_000, seed=42):
    """
    Monte Carlo 模拟验证。

    逻辑：在 t1 时刻模拟 S_t1，计算此时 BS 看涨/看跌价值，
          取最大值作为 t1 时刻 payoff，再折现到 0 时刻。
    """
    np.random.seed(seed)
    Z = np.random.normal(0, 1, n_sims)

    S_t1 = S * np.exp((r - q - 0.5 * sigma ** 2) * t1 + sigma * np.sqrt(t1) * Z)

    remaining = T2 - t1
    if remaining > 1e-10:
        d1 = (np.log(np.maximum(S_t1 / K, 1e-10)) + (r - q + 0.5 * sigma ** 2) * remaining) / (sigma * np.sqrt(remaining))
        d2 = d1 - sigma * np.sqrt(remaining)
        call_t1 = S_t1 * np.exp(-q * remaining) * norm.cdf(d1) - K * np.exp(-r * remaining) * norm.cdf(d2)
        put_t1 = K * np.exp(-r * remaining) * norm.cdf(-d2) - S_t1 * np.exp(-q * remaining) * norm.cdf(-d1)
    else:
        call_t1 = np.maximum(S_t1 - K, 0)
        put_t1 = np.maximum(K - S_t1, 0)

    payoff = np.maximum(call_t1, put_t1)
    mc_price = np.exp(-r * t1) * np.mean(payoff)
    se = np.std(payoff * np.exp(-r * t1)) / np.sqrt(n_sims)
    return mc_price, se


# ============================================================
# 4. 参考论文风格的两期 MC（用于对比）
# ============================================================

def reference_style_mc(S, K, t1, T2, r, q, sigma, n_sims=10_000, seed=42):
    """
    模拟参考论文《Exploration of JPMorgan Chooser Option Pricing》的两期方法：
      1) 第一期模拟 S_t1，比较与 K 决定选择看涨/看跌
      2) 第二期模拟 S_T2，计算 payoff
      3) 可选折现
    """
    np.random.seed(seed)
    Z1 = np.random.normal(0, 1, n_sims)
    Z2 = np.random.normal(0, 1, n_sims)

    S_t1 = S * np.exp((r - q - 0.5 * sigma ** 2) * t1 + sigma * np.sqrt(t1) * Z1)
    choice_call = S_t1 > K  # True → 选看涨, False → 选看跌

    S_t2 = S_t1 * np.exp((r - q - 0.5 * sigma ** 2) * (T2 - t1) + sigma * np.sqrt(T2 - t1) * Z2)

    payoff = np.where(choice_call, np.maximum(S_t2 - K, 0), np.maximum(K - S_t2, 0))
    avg_payoff = np.mean(payoff)
    discounted = avg_payoff * np.exp(-r * T2)
    return avg_payoff, discounted


# ============================================================
# 5. 验证函数集
# ============================================================

def test_bs_pricing():
    """验证 BSM 定价：与已知结果对比（S=100, K=100, T=1, r=5%, q=0, sigma=20%）"""
    S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.0, 0.20
    call = bs_call(S, K, T, r, q, sigma)
    put = bs_put(S, K, T, r, q, sigma)

    # 已知的 BS 价格（四舍五入到 4 位小数）
    expected_call = 10.4506
    expected_put = 5.5735

    passed_call = abs(call - expected_call) < 0.001
    passed_put = abs(put - expected_put) < 0.001

    print("  [TEST] BSM 定价验证")
    print(f"    BS Call:  {call:.4f}  (期望: {expected_call})  {'PASS' if passed_call else 'FAIL'}")
    print(f"    BS Put:   {put:.4f}  (期望: {expected_put})  {'PASS' if passed_put else 'FAIL'}")

    assert passed_call and passed_put, "BSM 定价验证失败！"
    return True


def test_boundary_conditions():
    """验证 Chooser 期权边界条件"""
    S, K, T2, r, q, sigma = 100, 100, 1.0, 0.08, 0.0, 0.20
    c = bs_call(S, K, T2, r, q, sigma)
    p = bs_put(S, K, T2, r, q, sigma)

    v_t1_0 = simple_chooser(S, K, 0.0, T2, r, q, sigma)     # t1=0
    v_t1_T2 = simple_chooser(S, K, T2, T2, r, q, sigma)     # t1=T2
    v_t1_mid = simple_chooser(S, K, 0.25, T2, r, q, sigma)  # 0<t1<T2

    b1 = abs(v_t1_0 - max(c, p)) < 1e-6
    b2 = abs(v_t1_T2 - (c + p)) < 1e-6
    b3 = (v_t1_mid > max(c, p)) and (v_t1_mid < c + p)

    print("  [TEST] 边界条件验证")
    print(f"    t1=0   -> max(C,P)={max(c, p):.2f}:  V={v_t1_0:.4f}  {'PASS' if b1 else 'FAIL'}")
    print(f"    t1=T2  -> C+P={c+p:.2f}:             V={v_t1_T2:.4f}  {'PASS' if b2 else 'FAIL'}")
    print(f"    0<t1<T2:  max(C,P)={max(c, p):.2f} < V={v_t1_mid:.4f} < {c+p:.2f}  {'PASS' if b3 else 'FAIL'}")

    assert b1 and b2 and b3, "边界条件验证失败！"
    return True


def test_vs_mc():
    """验证解析解 vs Monte Carlo"""
    S, K, t1, T2, r, q, sigma = 100, 100, 0.25, 1.0, 0.08, 0.0, 0.20

    analytic = simple_chooser(S, K, t1, T2, r, q, sigma)
    mc_price, mc_se = simple_chooser_mc(S, K, t1, T2, r, q, sigma, n_sims=200_000)

    bias_pct = abs(analytic - mc_price) / mc_price * 100
    passed = bias_pct < 0.1  # 偏差 < 0.1%

    print("  [TEST] 解析解 vs Monte Carlo")
    print(f"    Analytic:  {analytic:.4f}")
    print(f"    MC (200K): {float(mc_price):.4f} ± {float(mc_se):.4f}")
    print(f"    偏差: {bias_pct:.3f}%  {'PASS' if passed else 'FAIL'}")

    assert passed, f"MC 偏差过大: {bias_pct:.3f}%"
    return True


def test_vs_reference_paper():
    """与参考论文参数对比验证"""
    S, K, t1, T2, r, q, sigma = 156.7, 150.0, 0.5, 1.0, 0.0015, 0.0233, 0.282

    analytic = simple_chooser(S, K, t1, T2, r, q, sigma)
    mc_price, mc_se = simple_chooser_mc(S, K, t1, T2, r, q, sigma, n_sims=200_000)
    ref_raw, ref_disc = reference_style_mc(S, K, t1, T2, r, q, sigma, n_sims=10_000)
    c = bs_call(S, K, T2, r, q, sigma)
    p = bs_put(S, K, T2, r, q, sigma)

    print("  [TEST] 参考论文参数对比 (S=156.7, K=150, r=0.15%, q=2.33%, σ=28.2%)")
    print(f"    BS Call:              ${c:.4f}")
    print(f"    BS Put:               ${p:.4f}")
    print(f"    本项目 Analytic:       ${analytic:.4f}")
    print(f"    本项目 MC (200K):      ${float(mc_price):.4f} ± {float(mc_se):.4f}")
    print(f"    参考论文风格 MC (10K):  ${ref_raw:.4f} (未折现) / ${ref_disc:.4f} (折现)")
    print(f"    Analytic vs MC 偏差:   {abs(analytic-float(mc_price))/float(mc_price)*100:.3f}%")

    return True


def test_monotonicity():
    """验证 Chooser 价格关于 t1 单调递增"""
    S, K, T2, r, q, sigma = 100, 100, 1.0, 0.08, 0.0, 0.20

    t1_values = np.linspace(0, T2, 11)
    prices = [simple_chooser(S, K, t, T2, r, q, sigma) for t in t1_values]

    monotonic = all(prices[i] <= prices[i + 1] for i in range(len(prices) - 1))

    print("  [TEST] 单调性验证 (t1 从 0 到 T2)")
    for t, p in zip(t1_values, prices):
        print(f"    t1={t:.1f}:  {p:.4f}")
    print(f"    单调递增:  {'PASS' if monotonic else 'FAIL'}")

    assert monotonic, "单调性验证失败！"
    return True


# ============================================================
# 6. 敏感性分析
# ============================================================

def sensitivity_scan():
    """对关键参数进行扫描并打印结果"""
    S_base, K_base, t1_base, T2 = 156.7, 150.0, 0.5, 1.0
    r_base, q_base, sigma_base = 0.0015, 0.0233, 0.282

    print("  [SCAN] 敏感性分析扫描")
    print()

    # 波动率
    print("  --- 波动率 σ ---")
    for s in [0.05, 0.10, 0.20, 0.282, 0.40, 0.60]:
        p = simple_chooser(S_base, K_base, t1_base, T2, r_base, q_base, s)
        print(f"    σ={s*100:5.1f}%  →  Chooser Price = ${p:.2f}")

    # 行权价
    print("  --- 行权价 K ---")
    for k in [80, 120, 150, 180, 200]:
        p = simple_chooser(S_base, k, t1_base, T2, r_base, q_base, sigma_base)
        print(f"    K={k:3d}  →  Chooser Price = ${p:.2f}")

    # 无风险利率
    print("  --- 无风险利率 r ---")
    for r_val in [0.0, 0.0015, 0.02, 0.05, 0.08]:
        p = simple_chooser(S_base, K_base, t1_base, T2, r_val, q_base, sigma_base)
        print(f"    r={r_val*100:5.2f}%  →  Chooser Price = ${p:.2f}")

    # 股息率
    print("  --- 股息率 q ---")
    for q_val in [0.0, 0.01, 0.0233, 0.04, 0.06]:
        p = simple_chooser(S_base, K_base, t1_base, T2, r_base, q_val, sigma_base)
        print(f"    q={q_val*100:5.2f}%  →  Chooser Price = ${p:.2f}")

    # 选择日
    print("  --- 选择日 t1 ---")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p = simple_chooser(S_base, K_base, t, T2, r_base, q_base, sigma_base)
        print(f"    t1={t:.2f}  →  Chooser Price = ${p:.2f}")


# ============================================================
# 7. 主函数：运行所有测试
# ============================================================

if __name__ == '__main__':
    import sys

    print("=" * 65)
    print("  Chooser Option Pricing Model — Test Suite")
    print("=" * 65)
    print(f"  NumPy {np.__version__} | SciPy {sys.modules['scipy'].__version__}")
    print()

    tests = [
        ("BSM 定价", test_bs_pricing),
        ("边界条件", test_boundary_conditions),
        ("解析 vs MC", test_vs_mc),
        ("参考论文对比", test_vs_reference_paper),
        ("单调性", test_monotonicity),
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
        print()

    # 敏感性扫描（非测试，仅输出）
    sensitivity_scan()
    print()

    print("=" * 65)
    if all_passed:
        print("  所有测试通过！模型验证成功。")
    else:
        print(f"  部分测试失败，请检查。")
    print("=" * 65)
