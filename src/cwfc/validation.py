from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp | int | float
    validation_start: pd.Timestamp | int | float
    validation_end: pd.Timestamp | int | float


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 288,
    min_train_size: int = 7 * 24 * 12,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    `horizon=288` is one day at 5-minute cadence.
    """
    step = horizon if step is None else step
    if horizon <= 0 or min_train_size <= 0 or step <= 0:
        raise ValueError("horizon, min_train_size, and step must be positive")
    if frame.empty:
        raise ValueError("frame must not be empty")
    raw = frame[timestamp_col]
    # Preserve source-relative seconds instead of interpreting numbers as
    # Unix nanoseconds. Calendar timestamps retain the starter's UTC behavior.
    ts = raw if pd.api.types.is_numeric_dtype(raw) else pd.to_datetime(raw, utc=True)
    if ts.isna().any() or (pd.api.types.is_numeric_dtype(ts) and not np.isfinite(ts).all()):
        raise ValueError("timestamps must be finite and nonmissing")
    if not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError("timestamps must be sorted and unique")
    deltas = ts.diff().dropna()
    if not deltas.empty and not deltas.eq(deltas.iloc[0]).all():
        raise ValueError("timestamps must be evenly sampled")
    n = len(frame)
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        train_end = ts.iloc[train_end_idx - 1]
        val_start = ts.iloc[train_end_idx]
        val_end = ts.iloc[train_end_idx + horizon - 1]
        out.append(TimeSplit(train_end, val_start, val_end))
        train_end_idx += step
    return out
