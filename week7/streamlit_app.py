"""
Week 7 Pricing-Tool Prototype -- Streamlit
==========================================
Basic UI / functionality for the Week 7 "pricing tool prototype" deliverable:

  * dual-track pricing -- BSM(vol_21d) baseline vs the best ML volatility
    models (XGBoost = live-market pick, VIX-proxy = IV-aligned pick) priced
    with the Week-3 Rubinstein chooser formula;
  * Week-6 test-set error margins attached to every price;
  * interactive sensitivity charts (spot / volatility / choice date / rate);
  * extreme-scenario presets (vol +50%, rate +2%, combined);
  * real-time data panel with an on-demand refresh (data_updater).

Run:
    cd E:/实习交付/week2/week7
    streamlit run streamlit_app.py
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import w7config as cfg
import tool_engine as te

import streamlit as st

st.set_page_config(page_title='Chooser Option Pricing Tool — Week 7',
                   layout='wide', initial_sidebar_state='expanded')


# ═══════════════════════════════════════════════════════════════════════════════
# Cached heavy artifacts
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def cached_models():
    return {fam: te.load_vol_model(fam) for fam in ('vol_xgb', 'vol_vix_proxy', 'vol_gbdt_anchored')}


@st.cache_data(show_spinner=False)
def cached_live_state():
    market = te.fetch_market_history(period='2y')
    feat = te.build_features(market)
    feat = pd.concat([feat, te.regime_dummies(feat)], axis=1).dropna()
    row = feat.iloc[-1]
    return {
        'row': row, 'spot': float(row[cfg.SPOT_COL]),
        'rate': float(row[cfg.RATE_COL]) / 100.0,
        'q': cfg.Q_YIELD, 'vol21': float(row['vol_21d']),
        'vix': float(row[cfg.VIX_COL]), 'last_date': str(feat.index[-1].date()),
    }


@st.cache_data(show_spinner=False)
def cached_error_margins():
    return te.error_margins()


def refresh_live_data():
    cached_live_state.clear()
    st.cache_resource.clear()
    import data_updater as du
    return du.run(mode='dry_run', verbose=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Pricing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def predict_sigma(family: str, row: pd.Series) -> float:
    payload, feats, _ = cached_models()[family]
    X = pd.DataFrame([row[feats].values], columns=feats).astype(float)
    return float(te.predict_sigma_from_pipe(payload, X)[0])


def price_arrays(S0, K, t1, T2, r, q, sigma_ml, sigma_base, grid, which):
    """Vectorized price for a sensitivity sweep. `which` indexes grid."""
    sig_ml = np.full(len(grid), sigma_ml)
    sig_base = np.full(len(grid), sigma_base)
    if which == 'spot':
        x = grid
        p_ml = te.price_chooser_vec(grid, K, t1, T2, r, q, sig_ml)
        p_bsm = te.price_chooser_vec(grid, K, t1, T2, r, q, sig_base)
        return x, p_ml, p_bsm
    if which == 'sigma':
        x = grid
        p_ml = te.price_chooser_vec(np.full(len(grid), S0), K, t1, T2, r, q, grid)
        p_bsm = te.price_chooser_vec(np.full(len(grid), S0), K, t1, T2, r, q, grid)
        return x, p_ml, p_bsm
    if which == 't1':
        x = grid
        p_ml = [te.price_chooser(S0, K, g, T2, r, q, sigma_ml) for g in grid]
        p_bsm = [te.price_chooser(S0, K, g, T2, r, q, sigma_base) for g in grid]
        return x, p_ml, p_bsm
    if which == 'rate':
        x = grid
        p_ml = [te.price_chooser(S0, K, t1, T2, g, q, sigma_ml) for g in grid]
        p_bsm = [te.price_chooser(S0, K, t1, T2, g, q, sigma_base) for g in grid]
        return x, p_ml, p_bsm


def line_fig(x, y_ml, y_bsm, xlabel, title):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_ml, name='ML 波动率', mode='lines',
                             line=dict(color='#d73027', width=2.5)))
    fig.add_trace(go.Scatter(x=x, y=y_bsm, name='BSM(vol_21d)', mode='lines',
                             line=dict(color='#2166ac', width=2.5, dash='dash')))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title='Chooser 价格 ($)',
                      height=380, margin=dict(t=50, b=30, l=40, r=20))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title('⚙️ 参数设置')
    st.caption('Week 7 定价工具原型 · 双轨定价 (BSM + ML)')

    live = cached_live_state()
    with st.expander('📈 实时市场数据', expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric('JPM', f"${live['spot']:.2f}")
        c2.metric('VIX', f"{live['vix']:.2f}")
        c3.metric('3M 利率', f"{live['rate']*100:.2f}%")
        st.caption(f"数据日期: {live['last_date']}")
        if st.button('🔄 刷新数据 (dry-run)', width='stretch'):
            st.session_state['updater_status'] = refresh_live_data()
            st.rerun()
        st.caption('(自动更新由 data_updater.py + GitHub Actions 调度)')

    st.subheader('合约参数')
    K = st.number_input('行权价 K ($)', value=float(cfg.CHOOSER_PARAMS['K']), step=5.0)
    t1 = st.slider('选择日 t1 (年)', 0.05, 1.0, float(cfg.CHOOSER_PARAMS['t1']), 0.05)
    T2 = st.slider('最终到期 T2 (年)', 0.25, 3.0, float(cfg.CHOOSER_PARAMS['T2']), 0.25)
    S = st.number_input('标的现价 S ($)', value=float(live['spot']), step=5.0)
    r_pct = st.slider('无风险利率 r (%)', 0.0, 12.0, float(live['rate'] * 100), 0.1)
    q_pct = st.slider('股息率 q (%)', 0.0, 5.0, float(cfg.Q_YIELD * 100), 0.1)
    r, q = r_pct / 100.0, q_pct / 100.0

    st.subheader('波动率输入')
    vol_mode = st.radio('定价模式', ['ML 自动预测', '手动设定 σ'], horizontal=True)
    family_labels = {'XGBoost (实盘首选)': 'vol_xgb',
                     'VIX-proxy (隐含波动率)': 'vol_vix_proxy',
                     'GBDT-anchored (波动率比率)': 'vol_gbdt_anchored'}
    if vol_mode == 'ML 自动预测':
        fam_label = st.selectbox('ML 模型', list(family_labels.keys()))
        family = family_labels[fam_label]
        sigma_ml = predict_sigma(family, live['row'])
        st.caption(f'预测波动率 σ = {sigma_ml*100:.2f}% (基于最新市场特征)')
    else:
        family = 'vol_xgb'
        sigma_ml = st.slider('σ (%)', 5.0, 80.0, 25.0, 0.5) / 100.0
    sigma_base = st.slider('BSM 基准 σ(vol_21d) (%)', 5.0, 80.0, float(live['vol21'] * 100), 0.5) / 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main area
# ═══════════════════════════════════════════════════════════════════════════════

st.title('🎯 Chooser Option Pricing Tool')
st.caption('Week 7 工具原型 — 双轨定价 (BSM + 最优 ML 模型) · 数据日期 '
           f"{live['last_date']} · 模型来自 Week 6 (models/*.pkl)")

p = te.price_dual(S, K, t1, T2, r, q, sigma_ml, sigma_base)

c1, c2, c3, c4 = st.columns(4)
c1.metric('BSM(vol_21d) 价格', f"${p['price_BSM']:.4f}")
c2.metric('ML 定价', f"${p['price_ML']:.4f}",
          delta=f"{p['spread_$']:+.4f} (ML-BSM)")
c3.metric('ML 波动率', f"{p['sigma_ML']*100:.2f}%")
c4.metric('BSM 波动率', f"{p['sigma_base']*100:.2f}%")

st.caption('模型选择: ' + (family_labels.get(family, family)))

tab1, tab2, tab3, tab4 = st.tabs(['📊 敏感性分析', '🌪 极端场景', '🔬 误差范围', '📡 实时数据'])

with tab1:
    st.subheader('敏感性图表 (可交互)')
    sc1, sc2 = st.columns(2)
    with sc1:
        Sgrid = np.linspace(max(S * 0.6, 20), S * 1.4, 60)
        x, yml, yb = price_arrays(S, K, t1, T2, r, q, sigma_ml, sigma_base, Sgrid, 'spot')
        st.plotly_chart(line_fig(x, yml, yb, '标的现价 S ($)', '价格 vs 标的现价 S'),
                        width='stretch')
        tg = np.linspace(0.1, max(T2, 0.2), 40)
        x, yml, yb = price_arrays(S, K, t1, T2, r, q, sigma_ml, sigma_base, tg, 't1')
        st.plotly_chart(line_fig(x, yml, yb, '选择日 t1 (年)', '价格 vs 选择日 t1'),
                        width='stretch')
    with sc2:
        sig = np.linspace(0.05, 0.60, 60)
        x, yml, yb = price_arrays(S, K, t1, T2, r, q, sigma_ml, sigma_base, sig, 'sigma')
        st.plotly_chart(line_fig(x, yml, yb, '波动率 σ', '价格 vs 波动率 σ'),
                        width='stretch')
        rg = np.linspace(0.0, 0.10, 40)
        x, yml, yb = price_arrays(S, K, t1, T2, r, q, sigma_ml, sigma_base, rg, 'rate')
        st.plotly_chart(line_fig(x, yml, yb, '无风险利率 r', '价格 vs 利率 r'),
                        width='stretch')

with tab2:
    st.subheader('极端场景测试 (波动率激增 50% / 利率上调 2%)')
    import sensitivity as sens
    base_row, spot0, rate0, q0, vol21 = live['row'], live['spot'], live['rate'], cfg.Q_YIELD, live['vol21']
    sc = sens.extreme_scenarios(base_row, S, r, q, sigma_base, verbose=False)
    st.dataframe(sc.round(3), width='stretch', hide_index=True)
    st.caption('表中价格为当前合约参数下的取值；场景机制见 sensitivity.py。'
               '波动率激增同时放大 ML 预测 σ，故 ML 价格变动通常大于 BSM。')

with tab3:
    st.subheader('Week 6 测试集误差范围 (用于展示定价误差条)')
    em = cached_error_margins()
    em = em.rename(columns={'label': '模型', 'price_MAE_$': '定价 MAE ($)',
                            'price_RMSE_$': '定价 RMSE ($)', 'price_R2': 'R²'})
    st.dataframe(em[['模型', '定价 MAE ($)', '定价 RMSE ($)', 'R²']].round(3),
                 width='stretch', hide_index=True)
    vix_row = em.loc[em['模型'] == 'vix_proxy']
    ma = float(vix_row['定价 MAE ($)'].iloc[0]) if len(vix_row) else float('nan')
    st.metric('示例: VIX-proxy 定价 ± MAE', f"${p['price_ML']:.4f} ± ${ma:.3f}")
    st.caption('误差条 = 各模型在 2024 测试集上的定价 MAE（上表），当前以 VIX-proxy 为例。')

with tab4:
    st.subheader('实时数据集成状态')
    st.markdown(f"""
    | 项目 | 值 |
    |---|---|
    | 最新数据日期 | **{live['last_date']}** |
    | JPM 现价 | **${live['spot']:.2f}** |
    | VIX | **{live['vix']:.2f}** |
    | 3M 利率 | **{live['rate']*100:.2f}%** |
    | 特征数据集末行 | **2024-12-30** (训练区间 2018-2024) |
    """)
    st.info('自动更新: `data_updater.py --commit` 在 GitHub Actions 中每工作日执行, '
            '将新市场数据增量合并进特征数据集。本地演示使用快照缓存 (2026-07-27)。')
    if 'updater_status' in st.session_state:
        st.json(st.session_state['updater_status'])

st.caption('---  Week 7 交付 · 定价工具原型 (基础版)  |  双轨: BSM(vol_21d) vs ML 波动率 + BSM 引擎')
