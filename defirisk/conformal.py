"""Conformal prediction intervals: split, CQR, and adaptive (ACI).

All methods work in the model's output space (here log1p of liquidated value).
Back-transforming interval endpoints with a monotone map preserves coverage.

References
- Lei et al. (2018), Distribution-free predictive inference for regression.
- Romano, Patterson & Candes (2019), Conformalized quantile regression.
- Gibbs & Candes (2021), Adaptive conformal inference under distribution shift.
"""
import numpy as np


def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) quantile of calibration scores.

    Takes the ceil((1 - alpha)(n + 1))-th smallest score. If that rank exceeds n
    the honest answer is an infinite interval, not the largest score.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil(round((1 - alpha) * (n + 1), 9)))
    if k <= 0:
        return -np.inf
    if k > n:
        return np.inf
    return float(s[k - 1])


def to_original(point, lo, hi):
    """Back-transform log1p-space predictions; clip at 0 (values can't be negative)."""
    f = lambda a: np.clip(np.expm1(np.asarray(a, dtype=float)), 0.0, None)
    return f(point), f(lo), f(hi)


class SplitConformal:
    """Split conformal with absolute-residual scores: interval = prediction +/- q."""

    def __init__(self, model, alpha: float = 0.1):
        self.model, self.alpha = model, alpha

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def calibrate(self, X, y):
        self.scores_ = np.abs(np.asarray(y, float) - self.model.predict(X))
        self.q_ = conformal_quantile(self.scores_, self.alpha)
        return self

    def predict_interval(self, X):
        p = np.asarray(self.model.predict(X), dtype=float)
        return p, p - self.q_, p + self.q_


class CQR:
    """Conformalized quantile regression: interval width adapts to the inputs."""

    def __init__(self, lo_model, hi_model, alpha: float = 0.1):
        self.lo_model, self.hi_model, self.alpha = lo_model, hi_model, alpha

    def fit(self, X, y):
        self.lo_model.fit(X, y)
        self.hi_model.fit(X, y)
        return self

    def _bounds(self, X):
        lo, hi = self.lo_model.predict(X), self.hi_model.predict(X)
        return np.minimum(lo, hi), np.maximum(lo, hi)  # fix quantile crossing

    def calibrate(self, X, y):
        lo, hi = self._bounds(X)
        y = np.asarray(y, float)
        self.scores_ = np.maximum(lo - y, y - hi)
        self.q_ = conformal_quantile(self.scores_, self.alpha)
        return self

    def predict_interval(self, X):
        lo, hi = self._bounds(X)
        return (lo + hi) / 2, lo - self.q_, hi + self.q_


class AdaptiveConformal:
    """Adaptive conformal inference (Gibbs & Candes 2021) around a fitted point model.

    Predicts one day at a time. After each outcome is revealed it (a) adds the new
    score to the calibration pool and (b) nudges the working miscoverage level:
        alpha_{t+1} = alpha_t + gamma * (alpha - 1[miss_t])
    so the interval widens after misses and tightens after hits. Guarantee, for any
    data sequence: |mean miss rate - alpha| <= (max(alpha_1, 1-alpha_1) + gamma) / (gamma * T).
    It is a long-run guarantee; at T ~ 100 it is loose.
    """

    def __init__(self, model, alpha: float = 0.1, gamma: float = 0.05, window: int = None):
        self.model, self.alpha, self.gamma, self.window = model, alpha, gamma, window

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def run(self, X_calib, y_calib, X_test, y_test):
        """Return (point, lower, upper, alpha_path) over the test period."""
        pool = list(np.abs(np.asarray(y_calib, float) - self.model.predict(X_calib)))
        pts = np.asarray(self.model.predict(X_test), dtype=float)
        y_test = np.asarray(y_test, float)
        a_t = self.alpha
        lo, hi, path = np.empty(len(pts)), np.empty(len(pts)), np.empty(len(pts))
        for i, (p, y) in enumerate(zip(pts, y_test)):
            recent = pool if self.window is None else pool[-self.window:]
            q = conformal_quantile(recent, a_t)
            lo[i], hi[i], path[i] = p - q, p + q, a_t
            miss = float(not (lo[i] <= y <= hi[i]))
            a_t += self.gamma * (self.alpha - miss)
            pool.append(abs(y - p))
        return pts, lo, hi, path
