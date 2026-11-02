"""Empirical, one-sided CPU planning allowance and chronological diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate_margin(underprediction, quantile: float = 0.95) -> float:
    """Use signed actual-minus-forecast errors; no coverage guarantee is implied."""
    errors = np.asarray(underprediction, dtype=float)
    if errors.ndim != 1 or not len(errors) or not np.isfinite(errors).all():
        raise ValueError("Calibration requires nonempty finite one-dimensional errors")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    return max(0., float(np.quantile(errors, quantile, method="higher")))


def planning_ceiling(point_forecast, margin_pp: float) -> np.ndarray:
    forecast = np.asarray(point_forecast, dtype=float)
    if (forecast.ndim != 1 or not len(forecast) or not np.isfinite(forecast).all()
            or ((forecast < 0) | (forecast > 100)).any()):
        raise ValueError("CPU forecasts must be finite percentages in [0, 100]")
    if not np.isfinite(margin_pp) or margin_pp < 0:
        raise ValueError("margin_pp must be finite and nonnegative")
    return np.minimum(100., forecast + margin_pp)


def breach_signals(actual, ceiling, consecutive: int = 2) -> pd.DataFrame:
    """Alert once when a breach run reaches the required length; reset on recovery.

    Readings correspond to five-minute intervals. Two consecutive readings
    are two elevated interval averages, not proof of continuous overload.
    """
    actual, ceiling = np.asarray(actual, dtype=float), np.asarray(ceiling, dtype=float)
    if (actual.ndim != 1 or actual.shape != ceiling.shape or not len(actual)
            or not np.isfinite(actual).all() or not np.isfinite(ceiling).all()
            or ((actual < 0) | (actual > 100) | (ceiling < 0) | (ceiling > 100)).any()):
        raise ValueError("Actuals and ceilings must be aligned finite CPU percentages")
    if not isinstance(consecutive, (int, np.integer)) or consecutive < 1:
        raise ValueError("consecutive must be a positive integer")
    streak = 0
    records = []
    for observed, limit in zip(actual, ceiling):
        breached = bool(observed > limit)
        streak = streak + 1 if breached else 0
        records.append({"breach": breached, "breach_streak": streak,
                        "review_active": streak >= consecutive,
                        "alert_event": streak == consecutive})
    return pd.DataFrame(records)


def _assess(group: pd.DataFrame, margin: float) -> tuple[pd.DataFrame, dict]:
    frame = group.sort_values("timestamp").reset_index(drop=True).copy()
    frame["planning_ceiling_pct"] = planning_ceiling(frame.prediction, margin)
    frame["margin_pp"] = margin
    signals = breach_signals(frame.actual, frame.planning_ceiling_pct)
    frame = pd.concat([frame, signals], axis=1)
    peak = frame[frame.is_peak]
    return frame, {
        "n": len(frame), "margin_pp": margin,
        "covered": int((~frame.breach).sum()), "breaches": int(frame.breach.sum()),
        "point_coverage": float((~frame.breach).mean()),
        "mean_unused_allowance_pp": float(np.maximum(frame.planning_ceiling_pct - frame.actual, 0).mean()),
        "max_exceedance_pp": float(np.maximum(frame.actual - frame.planning_ceiling_pct, 0).max()),
        "peak_n": len(peak), "peak_breaches": int(peak.breach.sum()),
        "alert_events": int(frame.alert_event.sum()),
    }


def evaluate_headroom(validation: pd.DataFrame, quantile: float = .95) -> dict:
    """Separate reused calibration diagnostics from prior-fold-only checks.

    Input contains one selected CPU model, with development validation labels
    only. Every later-fold margin uses observations available at its origin.
    """
    required = {"fold", "timestamp", "origin_timestamp", "actual", "prediction", "is_peak"}
    if not required.issubset(validation.columns) or validation.empty:
        raise ValueError("Missing development validation predictions")
    if validation.timestamp.duplicated().any():
        raise ValueError("Expected one selected model on nonoverlapping validation timestamps")
    if not (validation.timestamp > validation.origin_timestamp).all():
        raise ValueError("Validation observations must follow their forecast origin")
    margin = calibrate_margin(validation.actual - validation.prediction, quantile)
    calibration, calibration_folds = [], []
    sequential, sequential_folds = [], []
    warmup = []
    for fold, group in validation.groupby("fold", sort=True):
        if group.origin_timestamp.nunique() != 1:
            raise ValueError("A fold must have exactly one forecast origin")
        frame, metrics = _assess(group, margin)
        calibration.append(frame)
        calibration_folds.append({"fold": fold, **metrics})
        prior = validation[validation.timestamp <= group.origin_timestamp.iloc[0]]
        if prior.empty:
            warmup.append(int(fold))
            continue
        past_margin = calibrate_margin(prior.actual - prior.prediction, quantile)
        frame, metrics = _assess(group, past_margin)
        sequential.append(frame)
        sequential_folds.append({"fold": fold, "prior_residuals": len(prior), **metrics})
    return {
        "margin_pp": margin, "warmup_folds": warmup,
        "calibration_predictions": pd.concat(calibration, ignore_index=True),
        "calibration_folds": pd.DataFrame(calibration_folds),
        "sequential_predictions": pd.concat(sequential, ignore_index=True) if sequential else pd.DataFrame(),
        "sequential_folds": pd.DataFrame(sequential_folds),
    }
