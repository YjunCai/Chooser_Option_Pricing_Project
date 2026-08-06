"""
Base Model Interface -- Week 5
==============================
Common contract for every model in the dual-track framework. Both the
volatility-prediction track (Approach 1) and the end-to-end pricing track
(Approach 2) register models through this interface, which keeps the
pipeline, evaluation and future hyper-parameter search (Week 6) model-agnostic.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Uniform sklearn-compatible interface.

    Subclasses must expose:
      - name  : human-readable model id
      - fit(X, y) -> self
      - predict(X) -> np.ndarray
    """

    name: str = 'base'

    @abstractmethod
    def fit(self, X, y) -> 'BaseModel':
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        ...

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.name}>'


class SklearnModel(BaseModel):
    """Thin adapter around any scikit-learn (or sklearn-API) regressor."""

    def __init__(self, name: str, estimator, **params):
        self.name = name
        self.estimator_cls = estimator
        self.params = params
        self.model = None

    def fit(self, X, y) -> 'SklearnModel':
        self.model = self.estimator_cls(**self.params)
        self.model.fit(X, np.asarray(y, dtype=float).ravel())
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('model not fitted')
        return np.asarray(self.model.predict(X), dtype=float)
