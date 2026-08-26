"""
Week 8 Final Pricing Tool -- Streamlit Dashboard
================================================
Complete visualization dashboard for the deployable chooser-option pricing
tool. Delivers the Week-8 "tool feature completion" block:

  * dual pricing -- BSM(vol_21d) baseline vs the best ML volatility models
    (XGBoost = live pick, VIX-proxy = IV-aligned pick), priced with the
    Week-3 validated Rubinstein chooser formula;
  * error-margin displays -- Week-6 test-set MAE/RMSE attached to every price;
  * full visualization dashboard --
      1. price trends over time (2018-2024 history + live point),
      2. interactive sensitivity charts (spot / vol / choice date / rate),
      3. performance metrics (approach-1 chooser, vol, live targets, end-to-end),
      4. extreme-scenario presets (vol +50%, rate +2%, combined, VIX shock);
  * real-time data panel with an on-demand refresh (data_updater).

Run:
    cd E:/实习交付/week2/week8
    streamlit run streamlit_app.py
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import w8config as cfg
import tool_engine as te
import dashboard as db

import streamlit as st

st.set_page_config(page_title='Chooser Option Pricing Tool — Final',
                   layout='wide', initial_sidebar_state='expanded')


# ═══════════════════════════════════════════════════════════════════════════════
# Cached heavy artifacts
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def cached_models():
    return {fam: te.load_vol_model(fam)
            for fam in ('vol_xgb', 'vol_vix_proxy', 'vol_gbdt_anchored')}


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


@st.cache_data(show_spinner=False)
def cached_trend_series():
    return db.historical_price_series(max_rows=cfg.PRICE_SERIES_SAMPLE)


@st.cache_data(show_spinner=False)
def cached_metrics():
    return db.performance_metrics()


def refresh_live_data():
    cached_live_state.clear()
    cached_trend_series.clear()
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
    sig_ml = np.full(len(grid), sigma_ml)
    sig_base = np.full(len(grid), sigma_base)
    if which == 'spot':
        p_ml = te.price_chooser_vec(grid, K, t1, T2, r, q, sig_ml)
        p_bsm = te.price_chooser_vec(grid, K, t1, T2, r, q, sig_base)
        return grid, p_ml, p_bsm
    if which == 'sigma':
        p_ml = te.price_chooser_vec(np.full(len(grid), S0), K, t1, T2, r, q, grid)
        p_bsm = te.price_chooser_vec(np.full(len(grid), S0), K, t1, T2, r, q, grid)
        return grid, p_ml, p_bsm
    if which == 't1':
        p_ml = [te.price_chooser(S0, K, g, T2, r, q, sigma_ml) for g in grid]
        p_bsm = [te.price_chooser(S0, K, g, T2, r, q, sigma_base) for g in grid]
        return grid, p_ml, p_bsm
    if which == 'rate':
        p_ml = [te.price_chooser(S0, K, t1, T2, g, q, sigma_ml) for g in grid]
        p_bsm = [te.price_chooser(S0, K, t1, T2, g, q, sigma_base) for g in grid]
        return grid, p_ml, p_bsm


def line_fig(x, y_ml, y_bsm, xlabel, title, y3=None, y3_label=None):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_ml, name='ML 波动率', mode='lines',
                             line=dict(color='#d73027', width=2.5)))
    fig.add_trace(go.Scatter(x=x, y=y_bsm, name='BSM(vol_21d)', mode='lines',
                             line=dict(color='#2166ac', width=2.5, dash='dash')))
    if y3 is not None:
        fig.add_trace(go.Scatter(x=x, y=y3, name=y3_label, mode='lines',
                                 line=dict(color='#fdae61', width=2.0)))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title='Chooser 价格 ($)',
                      height=380, margin=dict(t=50, b=30, l=40, r=20))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title('⚙️ 参数设置')
    st.caption('Week 8 最终定价工具 · 双轨定价 (BSM + 最优 ML 模型)')

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
    sigma_base = st.slider('BSM 基准 σ(vol_21d) (%)', 5.0, 80.0,
                           float(live['vol21'] * 100), 0.5) / 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main area -- metric cards
# ═══════════════════════════════════════════════════════════════════════════════

st.title('🎯 Chooser Option Pricing Tool — 最终版')
st.caption('Week 8 交付 · 双轨定价 (BSM(vol_21d) + 最优 ML 波动率模型) · 数据日期 '
           f"{live['last_date']} · 模型来自 Week 6 (models/*.pkl)")

p = te.price_dual(S, K, t1, T2, r, q, sigma_ml, sigma_base)

# best-model error margin (from the model family actually used)
em = cached_error_margins()
em_row = em[em['family'] == family]
best_mae = float(em_row['price_MAE_$'].iloc[0]) if len(em_row) else float('nan')

c1, c2, c3, c4 = st.columns(4)
c1.metric('BSM(vol_21d) 价格', f"${p['price_BSM']:.4f}")
c2.metric('ML 定价', f"${p['price_ML']:.4f}",
          delta=f"{p['spread_$']:+.4f} (ML-BSM)")
c3.metric('ML 波动率', f"{p['sigma_ML']*100:.2f}%")
c4.metric('BSM 波动率', f"{p['sigma_base']*100:.2f}%")

st.caption(f"模型: {cfg.MODEL_LABELS.get(family, family)}  |  "
           f"当前定价误差条: ± ${best_mae:.3f} (Week-6 测试集 MAE, 2024 共 208 日)")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ['📈 价格趋势', '📊 敏感性分析', '🌪 极端场景', '🏆 性能指标', '🔬 误差范围', '📡 实时数据'])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 -- price trends over time
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader('双轨定价价格趋势 (2018–2024 + 实时点)')
    st.caption('历史区间以 BSM(vol_21d) 与最优 ML 波动率模型重新定价；末尾标记点为最新实时市场状态。')
    trend = cached_trend_series()
    trend = trend[(trend['date'] <= pd.Timestamp(live['last_date']))].copy()
    # extend / re-price the last live point with the current sidebar contract params
    import plotly.graph_objects as go
    fig = go.Figure()
    for col, name, color in [('price_BSM', 'BSM(vol_21d) 基线', '#2166ac'),
                             ('price_xgb', 'XGBoost (实盘首选)', '#d73027'),
                             ('price_vix_proxy', 'VIX-proxy (IV 对齐)', '#fdae61')]:
        if col in trend.columns:
            fig.add_trace(go.Scatter(x=trend['date'], y=trend[col], name=name,
                                     mode='lines', line=dict(color=color, width=1.6)))
    fig.add_trace(go.Scatter(x=[trend['date'].iloc[-1]], y=[p['price_ML']], name='实时 ML 点',
                             mode='markers', marker=dict(color='#d73027', size=10,
                                                         symbol='star', line=dict(width=1, color='black'))))
    fig.add_trace(go.Scatter(x=[trend['date'].iloc[-1]], y=[p['price_BSM']], name='实时 BSM 点',
                             mode='markers', marker=dict(color='#2166ac', size=10,
                                                         symbol='star', line=dict(width=1, color='black'))))
    fig.update_layout(title='价格趋势：BSM 与 ML 的 Chooser 价格 (随最新市场状态延伸)',
                      xaxis_title='日期', yaxis_title='Chooser 价格 ($)',
                      height=460, margin=dict(t=50, b=30, l=40, r=20),
                      hovermode='x unified')
    st.plotly_chart(fig, width='stretch')

    st.subheader('波动率输入趋势')
    fig2 = go.Figure()
    for col, name, color in [('vol_21d', 'vol_21d (BSM 基线)', '#2166ac'),
                             ('sigma_xgb', 'XGBoost 预测 σ', '#d73027'),
                             ('sigma_vix_proxy', 'VIX-proxy 预测 σ', '#fdae61')]:
        if col in trend.columns:
            fig2.add_trace(go.Scatter(x=trend['date'], y=trend[col] * 100, name=name,
                                      mode='lines', line=dict(color=color, width=1.5)))
    fig2.update_layout(title='波动率输入：vol_21d vs ML 预测 (年化 %)',
                       xaxis_title='日期', yaxis_title='年化波动率 (%)',
                       height=340, margin=dict(t=50, b=30, l=40, r=20),
                       hovermode='x unified')
    st.plotly_chart(fig2, width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 -- interactive sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
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

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 -- extreme scenarios
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader('极端场景测试 (波动率激增 50% / 利率上调 2% / VIX 冲击)')
    import sensitivity as sens
    base_row, spot0, rate0, q0, vol21 = live['row'], live['spot'], live['rate'], cfg.Q_YIELD, live['vol21']
    sc = sens.extreme_scenarios(base_row, S, r, q, sigma_base, verbose=False)
    st.dataframe(sc.round(3), width='stretch', hide_index=True)
    st.caption('表中价格为当前合约参数下的取值；场景机制见 sensitivity.py。'
               '波动率激增同时放大 ML 预测 σ，故 ML 价格变动通常大于 BSM。')

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 -- performance metrics dashboard
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader('性能指标仪表板 (Week-6 测试集 + 实盘快照)')
    mtr = cached_metrics()
    c_left, c_right = st.columns(2)
    with c_left:
        st.image(str(cfg.ASSETS_DIR / 'fig_w8_chooser_metrics.png'), width='stretch')
        st.image(str(cfg.ASSETS_DIR / 'fig_w8_vol_metrics.png'), width='stretch')
    with c_right:
        st.image(str(cfg.ASSETS_DIR / 'fig_w8_live_metrics.png'), width='stretch')
        st.image(str(cfg.ASSETS_DIR / 'fig_w8_end2end.png'), width='stretch')
    st.caption('图(左上) 方法一 Chooser 定价测试集误差; (右上) 方法一波动率预测 MAE; '
               '(左下) 实盘快照目标对照; (右下) 方法二端到端定价。数字源: week6/output。')

    with st.expander('查看明细表'):
        st.markdown('**方法一 · Chooser 定价测试集 (2024)**')
        st.dataframe(mtr['chooser'].round(3), width='stretch', hide_index=True)
        st.markdown('**方法一 · 波动率预测测试集 (2024)**')
        st.dataframe(mtr['vol'].round(3), width='stretch', hide_index=True)
        st.markdown('**实盘快照指标 (2026-07-27)**')
        st.dataframe(mtr['live'].round(3), width='stretch', hide_index=True)
        st.markdown('**方法二 · 端到端定价测试集**')
        st.dataframe(mtr['end2end'].round(3), width='stretch', hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 -- error margins
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader('Week-6 测试集误差范围 (用于展示定价误差条)')
    emd = em.rename(columns={'label': '模型', 'price_MAE_$': '定价 MAE ($)',
                             'price_RMSE_$': '定价 RMSE ($)', 'price_R2': 'R²'})
    st.dataframe(emd[['模型', '定价 MAE ($)', '定价 RMSE ($)', 'R²']].round(3),
                 width='stretch', hide_index=True)
    vix_row = em.loc[em['family'] == 'vol_vix_proxy']
    ma = float(vix_row['price_MAE_$'].iloc[0]) if len(vix_row) else float('nan')
    st.metric('示例: VIX-proxy 定价 ± MAE', f"${p['price_ML']:.4f} ± ${ma:.3f}")
    st.metric(f"当前模型 ({cfg.MODEL_LABELS.get(family, family)}) 定价 ± MAE",
              f"${p['price_ML']:.4f} ± ${best_mae:.3f}")
    st.caption('误差条 = 各模型在 2024 测试集上的定价 MAE（上表）。')

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 6 -- real-time data
# ═══════════════════════════════════════════════════════════════════════════════

with tab6:
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

st.caption('---  Week 8 最终交付 · 可部署定价工具  |  双轨: BSM(vol_21d) vs 最优 ML 波动率 + BSM 引擎  |  '
           '配套: 最终报告 + 演示文稿 + 演示视频脚本')
