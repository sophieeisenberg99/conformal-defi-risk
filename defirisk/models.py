"""Base predictors. Conformal coverage does not depend on their quality, only the width does."""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class Persistence:
    """Predict tomorrow's log-value as today's. The baseline any model must beat."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X["log_value"], dtype=float)


def gradient_boosting(quantile: float = None) -> GradientBoostingRegressor:
    """Small, heavily regularised GBM (the training window is only tens of days)."""
    kw = dict(
        n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=0
    )
    if quantile is not None:
        return GradientBoostingRegressor(loss="quantile", alpha=quantile, **kw)
    return GradientBoostingRegressor(**kw)
