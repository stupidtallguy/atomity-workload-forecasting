from __future__ import annotations

import pandas as pd


def persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    if train.empty:
        raise ValueError("train must not be empty")
    last_value = float(train.iloc[-1][target])
    return pd.Series(last_value, index=template.index, dtype=float)


def moving_average_forecast(
    train: pd.DataFrame, template: pd.DataFrame, target: str, window: int,
) -> pd.Series:
    """Forecast every lead with the mean of the last observed window.

    This is a fixed-origin forecast: it never updates the window using actual
    validation readings, or values provided in the forecast template.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if len(train) < window:
        raise ValueError("not enough history for moving-average forecast")
    value = float(train[target].iloc[-window:].mean())
    return pd.Series(value, index=template.index, dtype=float)


def seasonal_naive_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, season_length: int) -> pd.Series:
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = train[target].astype(float).to_numpy()
    if len(values) < season_length:
        raise ValueError("not enough history for seasonal-naive forecast")
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)
