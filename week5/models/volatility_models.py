"""
Approach 1 -- ML Volatility Prediction Models -- Week 5
=======================================================
Models that predict a volatility input for the BSM engine:

  - Random Forest   (scikit-learn)     : nonlinear feature interactions
  - GBDT            (scikit-learn)     : gradient-boosted trees
  - XGBoost         (if installed)     : gradient boosting, stronger defaults
  - LSTM            (PyTorch if usable, else windowed-MLP fallback)
                                          : sequence memory over features

Every model follows the BaseModel interface. The LSTM consumes sliding-window
sequences built by data_preparation.make_sequences(); the tree/linear models
consume the flat feature matrix.
"""

from typing import Dict, List, Optional

import numpy as np

from .base import BaseModel, SklearnModel


def _torch_available():
    """Import torch defensively (a broken install must not break the framework)."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _xgb_available():
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Tree models
# ═══════════════════════════════════════════════════════════════════════════════

def build_volatility_models(params: Optional[Dict] = None) -> Dict[str, BaseModel]:
    """
    Instantiate the Approach-1 model registry. Models whose optional backend
    (xgboost, torch) is unavailable are skipped and reported.
    """
    import config
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    p = params or config.VOL_MODELS
    models: Dict[str, BaseModel] = {}

    models['rf'] = SklearnModel(p['rf']['name'], RandomForestRegressor,
                                **{k: v for k, v in p['rf'].items() if k != 'name'})
    models['gbdt'] = SklearnModel(p['gbdt']['name'], GradientBoostingRegressor,
                                  **{k: v for k, v in p['gbdt'].items() if k != 'name'})

    if _xgb_available():
        from xgboost import XGBRegressor
        models['xgb'] = SklearnModel(p['xgb']['name'], XGBRegressor,
                                     **{k: v for k, v in p['xgb'].items() if k != 'name'},
                                     verbosity=0)
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        models['xgb'] = SklearnModel('XGBoost (HistGB fallback)', HistGradientBoostingRegressor,
                                     max_iter=200, learning_rate=0.05,
                                     max_depth=4, random_state=config.RANDOM_SEED)

    return models


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LSTM (sequence model)
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMVolatilityPredictor(BaseModel):
    """Sliding-window sequence model for volatility forecasting.

    Backend resolution (config-driven, pluggable):
      * PyTorch available  -> a real nn.LSTM regressor.
      * otherwise          -> scikit-learn MLPRegressor on flattened windows
                              (documented fallback; sklearn has no LSTM).
    """

    name = 'LSTM'

    def __init__(self, seq_len=20, hidden_size=32, num_layers=1, dropout=0.1,
                 epochs=30, batch_size=32, lr=1e-3, seed=42, **kwargs):
        self.seq_len = int(seq_len)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.seed = int(seed)
        self.backend = 'torch' if _torch_available() else 'mlp-fallback'
        self.model = None

    # -- fit / predict ------------------------------------------------------

    def fit(self, X, y) -> 'LSTMVolatilityPredictor':
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 3:
            raise ValueError('LSTM expects 3D sequences (n, seq_len, n_feat)')
        if self.backend == 'torch':
            self.model = self._fit_torch(X, y)
        else:
            self.model = self._fit_mlp(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError('LSTM expects 3D sequences (n, seq_len, n_feat)')
        if self.model is None:
            raise RuntimeError('model not fitted')
        if self.backend == 'torch':
            return self._predict_torch(X)
        return np.asarray(self.model.predict(X.reshape(X.shape[0], -1)), dtype=float)

    # -- PyTorch backend ----------------------------------------------------

    def _build_torch_model(self, input_size):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)

        hidden_size, num_layers, dropout = self.hidden_size, self.num_layers, self.dropout

        class _LSTMReg(nn.Module):
            def __init__(self, in_sz):
                super().__init__()
                self.lstm = nn.LSTM(in_sz, hidden_size, num_layers, batch_first=True,
                                    dropout=dropout if num_layers > 1 else 0.0)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return _LSTMReg(input_size)

    def _fit_torch(self, X, y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        n, seq_len, n_feat = X.shape
        model = self._build_torch_model(n_feat)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        model.train()
        for epoch in range(self.epochs):
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            if self._verbose_epoch(epoch):
                print(f'      LSTM epoch {epoch+1:2d}/{self.epochs}  loss={total/len(Xt):.6f}')
        return model

    def _predict_torch(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(X, dtype=torch.float32))
        return out.numpy()

    # -- sklearn fallback ---------------------------------------------------

    def _fit_mlp(self, X, y):
        from sklearn.neural_network import MLPRegressor
        X2 = X.reshape(X.shape[0], -1)
        mlp = MLPRegressor(hidden_layer_sizes=(max(32, X2.shape[1] // 2), 16),
                           activation='relu', alpha=1e-3,
                           learning_rate_init=self.lr, max_iter=300,
                           early_stopping=True, n_iter_no_change=15,
                           validation_fraction=0.1, random_state=self.seed)
        mlp.fit(X2, y)
        return mlp

    def _verbose_epoch(self, epoch) -> bool:
        """Print sparsely (first, last, and every 10th epoch)."""
        return epoch == 0 or epoch == self.epochs - 1 or (epoch + 1) % 10 == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Sequence-data helper
# ═══════════════════════════════════════════════════════════════════════════════

def build_lstm_datasets(
    split: Dict[str, dict],
    seq_len: int = 20,
) -> Dict[str, dict]:
    """Turn the flat-split dict into LSTM-ready sequence dicts (same keys).

    Windows are cut from the FULL scaled frame (split['X_full']) using each
    set's original row positions, so a sequence at anchor j always covers
    rows [j-seq_len+1 .. j] -- strictly backward-looking.
    """
    from data_preparation import make_sequences
    X_full = split['X_full']
    y_full = split['y_full']
    out = {}
    for key in ('train', 'val', 'test'):
        Xs, ys, idx = make_sequences(X_full, y_full, seq_len, split[key]['idx'])
        out[key] = {'X': Xs, 'y': ys, 'idx': idx}
    return out
