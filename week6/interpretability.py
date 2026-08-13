"""
Interpretability Analysis (SHAP / LIME) -- Week 6
=================================================
Explains the best volatility model and the best end-to-end pricing model:

  * SHAP (TreeExplainer) on the tree stage inside each final (scale, model)
    pipeline:
      - global summary (beeswarm) plot
      - mean |SHAP| bar chart  (global feature importance)
      - top-feature dependence plots
  * LIME (LimeTabularExplainer) local explanations on a handful of test rows,
    demonstrating per-prediction feature attribution.
  * A comparison table of Week 5 RF importances vs Week 6 SHAP importances.

Figures land in assets/ and the table in output/feature_importance_w6.csv.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import w6config

WEEK5_IMP_PATH = w6config.WEEK5_DIR / 'output' / 'feature_importance.csv'


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SHAP helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _extract(pipeline):
    """(inner_model, scaler) from a (scale, model) sklearn Pipeline."""
    if hasattr(pipeline, 'named_steps') and 'model' in pipeline.named_steps:
        return pipeline.named_steps['model'], pipeline.named_steps.get('scale')
    if isinstance(pipeline, dict):                     # vix_proxy artifact
        p = pipeline['pipeline']
        return p.named_steps['model'], p.named_steps.get('scale')
    return pipeline, None


def compute_shap(pipeline, X_raw: pd.DataFrame, sample: int = 500) -> Tuple:
    """Return (explainer, shap_values, X_scaled_sub, X_raw_sub)."""
    import shap
    model, scaler = _extract(pipeline)
    X = X_raw.values if hasattr(X_raw, 'values') else np.asarray(X_raw, dtype=float)
    if len(X) > sample:
        idx = np.random.RandomState(0).choice(len(X), sample, replace=False)
        X = X[idx]
    X_scaled = scaler.transform(X) if scaler is not None else X
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_scaled)
    return explainer, np.asarray(sv), X_scaled, X


def save_shap_summary(sv, X_scaled, feature_names: List[str], out_path,
                      display_X=None):
    """Beeswarm + mean-|SHAP| bar, using original-scale values for the colorbar."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import shap

    out_path = str(out_path)
    disp = X_scaled if display_X is None else display_X
    if len(disp.shape) == 2 and disp.shape[1] != len(feature_names):
        disp = X_scaled

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(sv, disp, feature_names=feature_names, show=False,
                      max_display=15)
    plt.title('SHAP feature importance (beeswarm)')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    bar_path = out_path.replace('.png', '_bar.png')
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(sv, disp, feature_names=feature_names, show=False,
                      plot_type='bar', max_display=15)
    plt.title('Mean |SHAP| (global feature importance)')
    fig.tight_layout()
    fig.savefig(bar_path, bbox_inches='tight')
    plt.close(fig)
    return bar_path


def save_shap_dependence(sv, X_scaled, feature_names: List[str], out_path,
                         top_k: int = 2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import shap

    out_path = str(out_path)
    mean_abs = np.abs(sv).mean(axis=0)
    top = np.argsort(mean_abs)[::-1][:top_k]
    for rank, idx in enumerate(top):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        shap.dependence_plot(idx, sv, X_scaled, feature_names=feature_names,
                             show=False)
        plt.title(f'SHAP dependence: {feature_names[idx]} (rank {rank+1})')
        fig.tight_layout()
        fig.savefig(out_path.replace('.png', f'_dep{rank+1}.png'), bbox_inches='tight')
        plt.close(fig)


def global_shap_importance(sv, feature_names: List[str]) -> pd.DataFrame:
    mean_abs = np.abs(sv).mean(axis=0)
    return pd.DataFrame({'feature': feature_names,
                         'mean_abs_SHAP': mean_abs}).sort_values(
        'mean_abs_SHAP', ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LIME local explanation
# ═══════════════════════════════════════════════════════════════════════════════

def lime_explain(pipeline, X_raw, feature_names: List[str], target_names=None,
                 n_samples: int = 3, seed: int = 0):
    """Local LIME explanations for a few test rows (works on raw-scale features)."""
    from lime.lime_tabular import LimeTabularExplainer

    model, scaler = _extract(pipeline)
    X = X_raw.values if hasattr(X_raw, 'values') else np.asarray(X_raw, dtype=float)

    def predict_fn(x):
        x = scaler.transform(x) if scaler is not None else x
        return model.predict(x)

    explainer = LimeTabularExplainer(X, feature_names=feature_names,
                                     mode='regression', random_state=seed)
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(X), min(n_samples, len(X)), replace=False)
    out = []
    for i in idxs:
        exp = explainer.explain_instance(X[i], predict_fn, num_features=8)
        out.append({'row': int(i), 'pred': float(predict_fn(X[i:i+1])[0]),
                    'explanation': exp})
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

def run_interpretability(a1: Dict, a2: Dict, verbose: bool = True) -> Dict:
    """SHAP + LIME for the best vol model and the best pricing model."""
    import shap  # noqa: F401

    feats = a1['feats']
    X_test = a1['X_test']

    # pick the best volatility model by chooser-price MAE, else vol MAE
    best_vol = _best_vol_key(a1)
    best_price = _best_price_key(a2)
    if verbose:
        print(f'  interpretability targets: vol_model={best_vol}, '
              f'price_model={best_price}')

    results = {}

    # --- SHAP: best volatility model ---
    vol_pipeline = a1['fitted'][best_vol]
    explainer, sv, Xs, Xraw = compute_shap(vol_pipeline, X_test)
    save_shap_summary(sv, Xs, feats, w6config.ASSETS_DIR / f'fig_w6_shap_{best_vol}.png',
                      display_X=Xraw)
    save_shap_dependence(sv, Xs, feats, w6config.ASSETS_DIR / f'fig_w6_shap_{best_vol}.png')
    imp = global_shap_importance(sv, feats)
    imp.to_csv(w6config.OUTPUT_DIR / f'shap_importance_{best_vol}.csv', index=False)
    results['vol'] = {'key': best_vol, 'importance': imp,
                      'lime': lime_explain(vol_pipeline, X_test, feats)}
    if verbose:
        print('  [vol] top-8 SHAP:')
        print(imp.head(8).to_string(index=False))

    # --- SHAP: best pricing model ---
    price_pipeline = a2['fitted'][best_price]
    X_test2 = a2['X_test']
    feats2 = a2['feats']
    pfeats = _to_dataframe(X_test2, feats2)
    explainer2, sv2, Xs2, Xraw2 = compute_shap(price_pipeline, pfeats)
    save_shap_summary(sv2, Xs2, feats2, w6config.ASSETS_DIR / f'fig_w6_shap_{best_price}.png',
                      display_X=Xraw2)
    save_shap_dependence(sv2, Xs2, feats2, w6config.ASSETS_DIR / f'fig_w6_shap_{best_price}.png',
                         top_k=2)
    imp2 = global_shap_importance(sv2, feats2)
    imp2.to_csv(w6config.OUTPUT_DIR / f'shap_importance_{best_price}.csv', index=False)
    results['price'] = {'key': best_price, 'importance': imp2,
                        'lime': lime_explain(price_pipeline, pfeats, feats2)}
    if verbose:
        print('  [price] top-8 SHAP:')
        print(imp2.head(8).to_string(index=False))

    # --- Week5 RF importance vs Week6 SHAP comparison ---
    comp = compare_importance(imp, week5_path=WEEK5_IMP_PATH)
    comp.to_csv(w6config.OUTPUT_DIR / 'feature_importance_w6.csv', index=False)
    results['comparison'] = comp
    return results


def _to_dataframe(X, feats: List[str]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=feats)


def _best_vol_key(a1: Dict) -> str:
    # the VIX-proxy model won the chooser-price comparison -> explain it
    if 'vix_proxy' in a1['fitted']:
        return 'vix_proxy'
    # fallback: lowest vol MAE among fitted keys
    m = a1['metrics'].set_index('family')['MAE']
    return m.dropna().idxmin()


def _best_price_key(a2: Dict) -> str:
    cmp = a2['comparison']
    for _, r in cmp.iterrows():
        if r['family'] in ('pricing_gbdt', 'pricing_nn'):
            return r['family']
    return 'pricing_gbdt'


def compare_importance(shap_imp: pd.DataFrame,
                       week5_path=WEEK5_IMP_PATH) -> pd.DataFrame:
    """Merge Week-6 SHAP importance with Week-5 RF importance (if available)."""
    try:
        w5 = pd.read_csv(week5_path)
        w5 = w5[['feature', 'importance']].rename(columns={'importance': 'rf_importance_w5'})
    except Exception:
        w5 = pd.DataFrame(columns=['feature', 'rf_importance_w5'])
    out = shap_imp.merge(w5, on='feature', how='left')
    out['rf_rank_w5'] = w5.set_index('feature')['rf_importance_w5'].rank(
        ascending=False).reindex(out['feature']).values
    return out


if __name__ == '__main__':
    pass
