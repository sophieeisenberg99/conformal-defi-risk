"""Coverage metrics that don't flatter a zero-inflated series."""
import numpy as np


def coverage_report(y_log, lo_log, hi_log, alpha: float) -> dict:
    """Coverage overall and split by zero / nonzero target days, plus width and interval score.

    Inputs are in log1p space (monotone, so coverage is identical in original units).
    """
    y, lo, hi = (np.asarray(a, float) for a in (y_log, lo_log, hi_log))
    covered = (y >= lo) & (y <= hi)
    nz = y > 0
    width = hi - lo
    # Winkler interval score: width + (2/alpha) * distance outside the interval (lower is better).
    # Reported as a median because adaptive methods can emit infinite intervals.
    with np.errstate(invalid="ignore"):
        score = width + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    mean_or_nan = lambda a: float(a.mean()) if a.size else float("nan")
    return {
        "n_days": int(len(y)),
        "n_nonzero": int(nz.sum()),
        "coverage": float(covered.mean()),
        "coverage_nonzero": mean_or_nan(covered[nz]),
        "coverage_zero": mean_or_nan(covered[~nz]),
        "median_upper": float(np.median(np.expm1(np.clip(hi, 0, 50)))),  # typical upper bound, in original units
        "median_width_log": float(np.median(width)),
        "frac_infinite": float(np.isinf(width).mean()),
        "median_interval_score": float(np.median(score)),
    }


def block_bootstrap_ci(x, block: int = 7, n_boot: int = 2000, level: float = 0.95, seed: int = 0):
    """Moving-block bootstrap CI for a mean. Blocks keep day-to-day dependence, which
    a plain binomial interval would ignore."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return tuple(np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2]))


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion (assumes independent days)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)
