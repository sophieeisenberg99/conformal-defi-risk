"""Load Aave liquidation data and build the one-step-ahead daily forecasting table."""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Stablecoin debt tokens on Ethereum -> decimals. Debt repaid in these is a USD amount
# (assuming a $1 peg; USDC briefly traded near $0.88 on 2023-03-11, ignored here).
STABLECOIN_DECIMALS = {
    "0xdac17f958d2ee523a2206206994597c13d831ec7": 6,  # USDT
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 6,  # USDC
    "0x6b175474e89094c44da98b954eedeac495271d0f": 18,  # DAI
    "0x5f98805a4e8be255a32880fdec7f6728c6568ba0": 18,  # LUSD
}

FEATURES = [
    "log_value",
    "log_value_mean3",
    "log_value_mean7",
    "n_events",
    "n_events_sum7",
    "days_since_event",
]


def add_debt_repaid_usd(events: pd.DataFrame, exported_decimals: int = 18) -> pd.Series:
    """USD debt repaid per event, for stablecoin-denominated debt only (NaN otherwise).

    The Dune export divided every raw amount by 1e18, which is wrong for 6-decimal
    tokens (USDT/USDC). Multiplying by 10**(18 - decimals) undoes that.
    """
    asset = events["debt_asset"].str.lower()
    scale = asset.map(
        lambda a: 10.0 ** (exported_decimals - STABLECOIN_DECIMALS[a]) if a in STABLECOIN_DECIMALS else np.nan
    )
    return events["debt_repaid"] * scale


def load_daily(path, value_col: str = "debt_repaid_usd") -> pd.DataFrame:
    """Return a gap-free daily frame with columns `value` and `n_events` (UTC index).

    Accepts either a per-event CSV (has `block_time`) or an already-daily CSV
    (has `date` and `n_events`). `value_col` is the column summed per day;
    `debt_repaid_usd` is derived from the event columns (stablecoin debt only, other
    events count toward `n_events` but add 0 to the value).
    """
    df = pd.read_csv(Path(path))
    if value_col == "debt_repaid_usd" and value_col not in df.columns and "debt_asset" in df.columns:
        df[value_col] = add_debt_repaid_usd(df).fillna(0.0)
    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not in {list(df.columns)}")

    if "block_time" in df.columns:
        df["block_time"] = pd.to_datetime(df["block_time"], utc=True)
        day = df["block_time"].dt.floor("D")
        g = df.groupby(day)
        daily = pd.DataFrame({"value": g[value_col].sum(), "n_events": g.size()})
    elif "date" in df.columns and "n_events" in df.columns:
        idx = pd.to_datetime(df["date"], utc=True).dt.floor("D")
        daily = pd.DataFrame(
            {"value": df[value_col].to_numpy(), "n_events": df["n_events"].to_numpy()},
            index=idx,
        )
    else:
        raise ValueError("need a `block_time` column (events) or `date` + `n_events` (daily)")

    if (daily["value"] < 0).any():
        raise ValueError("negative liquidation value; check units")
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
    return daily.reindex(full, fill_value=0.0).rename_axis("date")


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Features known at the end of day t; `target` is log1p(value on day t+1).

    The target is log-transformed so the (zero-inflated, heavy-tailed) series
    is better behaved and back-transformed intervals can never go below zero.
    """
    lv = np.log1p(daily["value"])
    pos = np.arange(len(daily))
    last_event = pd.Series(np.where(daily["n_events"] > 0, pos, np.nan), index=daily.index).ffill()

    f = pd.DataFrame(index=daily.index)
    f["log_value"] = lv
    f["log_value_mean3"] = lv.rolling(3, min_periods=1).mean()
    f["log_value_mean7"] = lv.rolling(7, min_periods=1).mean()
    f["n_events"] = daily["n_events"]
    f["n_events_sum7"] = daily["n_events"].rolling(7, min_periods=1).sum()
    f["days_since_event"] = (pos - last_event).fillna(30).clip(upper=30)
    f["target"] = lv.shift(-1)
    f["target_raw"] = daily["value"].shift(-1)
    return f.dropna(subset=["target"])


def split_by_date(
    feats: pd.DataFrame, train_end: str, calib_end: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered train / calibration / test split on the LABEL date (feature date + 1 day).

    Cutting on label dates keeps every calibration label strictly before the shock;
    cutting on feature dates would let the first shock day's label into calibration.
    """
    t1 = pd.Timestamp(train_end, tz="UTC")
    t2 = pd.Timestamp(calib_end, tz="UTC")
    if not t1 < t2:
        raise ValueError("train_end must be before calib_end")
    label_date = feats.index + pd.Timedelta(days=1)
    train = feats[label_date <= t1]
    calib = feats[(label_date > t1) & (label_date <= t2)]
    test = feats[label_date > t2]
    if min(len(train), len(calib), len(test)) == 0:
        raise ValueError("a split is empty; check the dates against the data range")
    return train, calib, test
