"""
Approach 2 -- End-to-End Supervised Pricing Models -- Week 5
============================================================
Directly map (market features + contract features) -> option price, bypassing
the explicit BSM step:

  - Linear Regression  : interpretable affine pricing baseline
  - GBDT               : gradient-boosted trees over the pricing surface
  - MLP / Neural Net   : multi-layer perceptron (scikit-learn backend)

The supervised target is the forward-vol-priced BSM label (see targets.py);
a static-vol BSM price (vol_21d) is kept as the benchmark in evaluation.py.
"""

from typing import Dict, Optional

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from .base import SklearnModel


def build_pricing_models(params: Optional[Dict] = None) -> Dict[str, SklearnModel]:
    """Instantiate the Approach-2 model registry."""
    import config

    p = params or config.PRICING_MODELS
    models = {
        'linear': SklearnModel(p['linear']['name'], LinearRegression,
                               **{k: v for k, v in p['linear'].items() if k != 'name'}),
        'gbdt': SklearnModel(p['gbdt']['name'], GradientBoostingRegressor,
                             **{k: v for k, v in p['gbdt'].items() if k != 'name'}),
        'nn': SklearnModel(p['nn']['name'], MLPRegressor,
                           **{k: v for k, v in p['nn'].items() if k != 'name'}),
    }
    return models
