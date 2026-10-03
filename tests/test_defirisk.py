import numpy as np
import pandas as pd
import pytest

from defirisk.conformal import CQR, AdaptiveConformal, SplitConformal, conformal_quantile, to_original
from defirisk.data import add_debt_repaid_usd, build_features, load_daily, split_by_date
from defirisk.evaluate import block_bootstrap_ci, coverage_report


class Identity:
    """Model that predicts its single feature, so residuals are pure noise."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float).ravel()


def test_quantile_is_infinite_when_calibration_set_too_small():
    assert conformal_quantile(np.arange(5.0), alpha=0.1) == np.inf


def test_quantile_picks_corrected_rank():
    # n=11, alpha=0.1 -> rank ceil(0.9 * 12) = 11 -> the maximum score
    assert conformal_quantile(np.arange(11.0), alpha=0.1) == 10.0


def test_split_conformal_covers_on_exchangeable_data():
    rng = np.random.default_rng(0)
    hits = []
    for _ in range(300):
        x = rng.normal(size=250)
        y = x + rng.normal(size=250)
        sc = SplitConformal(Identity(), alpha=0.1).calibrate(x[:50], y[:50])
        _, lo, hi = sc.predict_interval(x[50:])
        hits.append(((y[50:] >= lo) & (y[50:] <= hi)).mean())
    assert 0.88 < np.mean(hits) < 0.95  # theory: >= 1 - alpha, about 0.90 for n=50


def shifted_stream(rng, n_calib=50, n_test=2000):
    """Noise sd jumps from 1 to 3 once the test period starts."""
    x = rng.normal(size=n_calib + n_test)
    sd = np.r_[np.ones(n_calib), np.ones(n_test // 4), 3 * np.ones(n_test - n_test // 4)]
    return x, x + rng.normal(size=x.size) * sd


def test_split_conformal_undercovers_but_aci_recovers_under_shift():
    rng = np.random.default_rng(1)
    x, y = shifted_stream(rng)
    sc = SplitConformal(Identity(), alpha=0.1).calibrate(x[:50], y[:50])
    _, lo, hi = sc.predict_interval(x[50:])
    split_cov = ((y[50:] >= lo) & (y[50:] <= hi)).mean()

    _, lo, hi, _ = AdaptiveConformal(Identity(), alpha=0.1, gamma=0.02).run(x[:50], y[:50], x[50:], y[50:])
    aci_cov = ((y[50:] >= lo) & (y[50:] <= hi)).mean()

    assert split_cov < 0.80
    assert abs(aci_cov - 0.90) < 0.04


def test_cqr_interval_is_ordered_and_covers():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(600, 1))
    y = x.ravel() + rng.normal(size=600)

    class Q:
        def __init__(self, z):
            self.z = z

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.asarray(X).ravel() + self.z

    cqr = CQR(Q(-1.0), Q(1.0), alpha=0.1).calibrate(x[:300], y[:300])
    _, lo, hi = cqr.predict_interval(x[300:])
    assert (lo <= hi).all()
    assert 0.85 < ((y[300:] >= lo) & (y[300:] <= hi)).mean() < 0.96


def test_back_transform_never_goes_negative():
    point, lo, hi = to_original([0.2], [-3.0], [1.0])
    assert lo[0] == 0.0 and hi[0] > point[0] > 0


def test_daily_aggregation_fills_gaps_with_zero(tmp_path):
    csv = tmp_path / "ev.csv"
    pd.DataFrame(
        {
            "block_time": ["2023-01-01 01:00:00 UTC", "2023-01-01 05:00:00 UTC", "2023-01-04 12:00:00 UTC"],
            "collateral_amount": [1.0, 2.0, 5.0],
        }
    ).to_csv(csv, index=False)
    daily = load_daily(csv, value_col="collateral_amount")
    assert list(daily["value"]) == [3.0, 0.0, 0.0, 5.0]
    assert list(daily["n_events"]) == [2, 0, 0, 1]


def test_features_use_only_past_information():
    daily = pd.DataFrame(
        {"value": [0, 0, 7.0, 0, 0], "n_events": [0, 0, 1, 0, 0]},
        index=pd.date_range("2023-01-01", periods=5, tz="UTC"),
    )
    f = build_features(daily)
    assert f.loc["2023-01-02", "target_raw"] == 7.0  # label is the NEXT day
    assert f.loc["2023-01-02", "log_value"] == 0.0  # feature does not see the spike
    assert len(f) == 4  # last day has no label


def test_split_by_date_cuts_on_label_date():
    idx = pd.date_range("2023-01-01", periods=30, tz="UTC")
    f = pd.DataFrame({"target": np.arange(30.0)}, index=idx)
    tr, ca, te = split_by_date(f, "2023-01-10", "2023-01-20")
    one = pd.Timedelta(days=1)
    assert (tr.index + one).max() <= pd.Timestamp("2023-01-10", tz="UTC")
    assert (ca.index + one).min() > pd.Timestamp("2023-01-10", tz="UTC")
    assert (ca.index + one).max() <= pd.Timestamp("2023-01-20", tz="UTC")
    assert (te.index + one).min() > pd.Timestamp("2023-01-20", tz="UTC")  # no calibration label touches test
    assert len(tr) + len(ca) + len(te) == 30


def test_coverage_report_separates_zero_days():
    y = np.array([0, 0, 0, 2.0])
    rep = coverage_report(y, lo_log=np.zeros(4), hi_log=np.ones(4), alpha=0.1)
    assert rep["coverage"] == 0.75 and rep["coverage_zero"] == 1.0 and rep["coverage_nonzero"] == 0.0


def test_block_bootstrap_ci_brackets_the_mean():
    x = np.r_[np.ones(90), np.zeros(10)]
    lo, hi = block_bootstrap_ci(x, block=5, n_boot=500)
    assert lo < 0.9 < hi


def test_stablecoin_debt_is_rescaled_by_token_decimals():
    ev = pd.DataFrame(
        {
            "debt_asset": [
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC, 6 decimals (mixed case)
                "0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI, 18 decimals
                "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH: not a stablecoin
            ],
            "debt_repaid": [2.5e-9, 1000.0, 3.0],  # as exported: raw / 1e18
        }
    )
    usd = add_debt_repaid_usd(ev)
    assert usd.iloc[0] == pytest.approx(2500.0)
    assert usd.iloc[1] == pytest.approx(1000.0)
    assert np.isnan(usd.iloc[2])
