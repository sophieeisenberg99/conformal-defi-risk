"""Run the coverage-under-shift experiment.

    python -m defirisk.experiment --data data/aave_v3_liquidations.csv
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .conformal import CQR, AdaptiveConformal, SplitConformal
from .data import FEATURES, build_features, load_daily, split_by_date
from .evaluate import block_bootstrap_ci, coverage_report
from .models import Persistence, gradient_boosting


def run(data, value_col, train_end, calib_end, alpha=0.1, gamma=0.05):
    """Fit every method on the same split. Returns (test frame, {name: (lo, hi)}, results, alpha paths)."""
    feats = build_features(load_daily(data, value_col))
    train, calib, test = split_by_date(feats, train_end, calib_end)
    X = lambda d: d[FEATURES]
    y_tr, y_ca, y_te = train["target"], calib["target"], test["target"]

    intervals, aci_paths = {}, {}

    for name, base in [("Persistence", Persistence()), ("GBM", gradient_boosting())]:
        base.fit(X(train), y_tr)
        sc = SplitConformal(base, alpha).calibrate(X(calib), y_ca)
        _, lo, hi = sc.predict_interval(X(test))
        intervals[f"{name} + split conformal"] = (lo, hi)
        _, lo, hi, path = AdaptiveConformal(base, alpha, gamma).run(X(calib), y_ca, X(test), y_te)
        intervals[f"{name} + adaptive (ACI)"] = (lo, hi)
        aci_paths[name] = path

    cqr = CQR(gradient_boosting(alpha / 2), gradient_boosting(1 - alpha / 2), alpha)
    cqr.fit(X(train), y_tr).calibrate(X(calib), y_ca)
    _, lo, hi = cqr.predict_interval(X(test))
    intervals["GBM + CQR"] = (lo, hi)

    y = y_te.to_numpy()
    rows = {}
    for name, (lo, hi) in intervals.items():
        rep = coverage_report(y, lo, hi, alpha)
        covered = (y >= lo) & (y <= hi)
        rep["coverage_ci95"] = block_bootstrap_ci(covered.astype(float))
        rows[name] = rep
    sizes = {"train": len(train), "calib": len(calib), "test": len(test)}
    return test, intervals, pd.DataFrame(rows).T, aci_paths, sizes


def plot(test, intervals, alpha, shift, out_path, window=14):
    y = test["target"].to_numpy()
    x = test.index + pd.Timedelta(days=1)  # plot on the label date, the day being predicted
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for name, (lo, hi) in intervals.items():
        cov = pd.Series(((y >= lo) & (y <= hi)).astype(float), index=x)
        a1.plot(cov.rolling(window, min_periods=window // 2).mean(), label=name, lw=1.6)
    a1.axhline(1 - alpha, color="k", ls="--", lw=1, label=f"target {1 - alpha:.0%}")
    a1.set_ylim(0, 1.05)
    a1.set_ylabel(f"{window}-day rolling coverage")
    a1.legend(fontsize=8, loc="lower left", ncol=2)
    a1.set_title("Does the coverage guarantee survive the shift?")

    for name, color in [("GBM + split conformal", "tab:blue"), ("GBM + adaptive (ACI)", "tab:orange")]:
        lo, hi = intervals[name]
        a2.fill_between(x, np.clip(lo, 0, None), np.clip(hi, 0, y.max() * 1.3),
                        alpha=0.25, color=color, label=name)
    a2.scatter(x, y, s=10, color="k", zorder=3, label="actual next-day value")
    a2.set_ylabel("log(1 + liquidated value)")
    a2.legend(fontsize=8, loc="upper right")

    if shift:
        for ax in (a1, a2):
            ax.axvspan(pd.Timestamp(shift[0], tz="UTC"), pd.Timestamp(shift[1], tz="UTC"),
                       color="red", alpha=0.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/aave_v3_liquidations.csv")
    p.add_argument("--value-col", default="debt_repaid_usd")
    p.add_argument("--train-end", default="2023-02-25")
    p.add_argument("--calib-end", default="2023-03-08")
    p.add_argument("--shift", nargs=2, default=["2023-03-09", "2023-03-14"], metavar=("START", "END"))
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--out", default="outputs")
    a = p.parse_args()

    test, intervals, results, _, sizes = run(
        a.data, a.value_col, a.train_end, a.calib_end, a.alpha, a.gamma
    )
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    results.to_csv(out / "results.csv")
    plot(test, intervals, a.alpha, a.shift, out / "coverage.png")

    print(f"days: {sizes}  target coverage: {1 - a.alpha:.0%}")
    show = results.drop(columns=["n_days", "n_nonzero", "coverage_ci95"]).astype(float).round(3)
    show.insert(1, "coverage_ci95", results["coverage_ci95"].map(lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]"))
    print(show.to_string())
    print(f"\nwrote {out / 'results.csv'} and {out / 'coverage.png'}")


if __name__ == "__main__":
    main()
