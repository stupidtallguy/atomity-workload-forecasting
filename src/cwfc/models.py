"""Small, fixed CPU model set; all fitting uses forecast-origin history only."""
from __future__ import annotations

import numpy as np

CPU_MODEL_CONFIG = {
    "ses_alpha_0.2": {"family": "exponential_smoothing", "alpha": 0.2},
    "damped_trend_12_phi_0.8": {"family": "damped_trend", "window": 12, "damping": 0.8},
    "ar1_stable": {"family": "autoregressive_one", "phi_bound": 0.95},
}


def _history(values, horizon: int) -> np.ndarray:
    history = np.asarray(values, dtype=float)
    if history.ndim != 1 or len(history) == 0:
        raise ValueError("CPU history must be a nonempty one-dimensional array")
    if not np.isfinite(history).all() or ((history < 0) | (history > 100)).any():
        raise ValueError("CPU history must contain finite percentages in [0, 100]")
    if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    return history


def exponential_smoothing(values, horizon: int, alpha: float = 0.2) -> np.ndarray:
    """Initialize at the first reading and update level = alpha*y + (1-alpha)*level."""
    history = _history(values, horizon)
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    level = history[0]
    for value in history[1:]:
        level = alpha * value + (1 - alpha) * level
    return np.clip(np.repeat(level, horizon), 0, 100)


def damped_trend(
    values, horizon: int, window: int = 12, damping: float = 0.8,
) -> np.ndarray:
    """Fit a local OLS line, then damp its future slope geometrically.

    This is a local regression extrapolator, not a fitted Holt state-space
    model. For lead h: fitted_last_level + slope * sum(damping**j, j=1..h).
    """
    history = _history(values, horizon)
    if window < 2 or len(history) < window:
        raise ValueError("window must be at least two and fit the observed history")
    if not 0 < damping < 1:
        raise ValueError("damping must be in (0, 1)")
    t = np.arange(window, dtype=float)
    centered_t = t - t.mean()
    recent = history[-window:]
    slope = float(centered_t @ (recent - recent.mean()) / (centered_t @ centered_t))
    level = float(recent.mean() + slope * (t[-1] - t.mean()))
    drift = np.cumsum(damping ** np.arange(1, horizon + 1))
    return np.clip(level + slope * drift, 0, 100)


def autoregressive_one(values, horizon: int, phi_bound: float = 0.95) -> np.ndarray:
    """OLS AR(1), constrained to a stable coefficient, with recursive forecasts.

    After constraining phi, refit the intercept as mean(y) - phi*mean(x).
    A constant lag history falls back to its latest observation. Predictions
    are bounded to [0, 100] at each recursive step.
    """
    history = _history(values, horizon)
    if len(history) < 3:
        raise ValueError("AR(1) requires at least three observations")
    if not 0 < phi_bound < 1:
        raise ValueError("phi_bound must be in (0, 1)")
    x, y = history[:-1], history[1:]
    centered_x = x - x.mean()
    denominator = float(centered_x @ centered_x)
    if denominator <= np.finfo(float).eps * max(1., float(x @ x)):
        return np.repeat(history[-1], horizon)
    phi = float(np.clip(centered_x @ (y - y.mean()) / denominator, -phi_bound, phi_bound))
    intercept = float(y.mean() - phi * x.mean())
    value = history[-1]
    predictions = []
    for _ in range(horizon):
        value = float(np.clip(intercept + phi * value, 0, 100))
        predictions.append(value)
    return np.asarray(predictions)


def forecast_cpu(values, horizon: int, model: str) -> np.ndarray:
    parameters = CPU_MODEL_CONFIG[model].copy()
    family = parameters.pop("family")
    forecaster = {
        "exponential_smoothing": exponential_smoothing,
        "damped_trend": damped_trend,
        "autoregressive_one": autoregressive_one,
    }[family]
    return forecaster(values, horizon, **parameters)
