"""Training-only two-hour baseline evaluation; never scores the final holdout."""
from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.audit import markdown_table
from cwfc.baselines import moving_average_forecast, persistence_forecast
from cwfc.data import build_targets, split_targets
from cwfc.metrics import mae, rmse
from cwfc.validation import rolling_origin_splits

TARGETS = ("active_vm_count", "aggregate_cpu_utilization")
DEFAULT_WINDOWS = (3, 6, 12, 24)


def score_predictions(predictions: pd.DataFrame) -> dict:
    """Pool individual forecast errors, rather than averaging fold RMSEs."""
    actual, forecast = predictions.actual, predictions.prediction
    error = forecast - actual
    reference_mae = mae(actual, predictions.persistence_prediction)
    peak = predictions[predictions.is_peak]
    return {
        "n": len(predictions), "mae": mae(actual, forecast),
        "rmse": rmse(actual, forecast), "bias": float(error.mean()),
        "nmae_pct_training_mean": float(predictions.scaled_absolute_error.mean()),
        "mae_ratio_to_persistence": mae(actual, forecast) / reference_mae if reference_mae else np.nan,
        "mae_improvement_pct": 100 * (1 - mae(actual, forecast) / reference_mae) if reference_mae else np.nan,
        "underforecast_fraction": float((error < 0).mean()),
        "peak_n": len(peak),
        "peak_mae": mae(peak.actual, peak.prediction) if len(peak) else np.nan,
        "peak_bias": float((peak.prediction - peak.actual).mean()) if len(peak) else np.nan,
    }


def evaluate_forecasters(
    train: pd.DataFrame, horizon: int = 24, min_train_size: int = 38,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    cpu_forecasters: dict[str, Callable] | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate baselines and optional CPU models on identical origin histories.

    This function accepts training data only. All statistics used to make
    predictions, normalize errors, or define peak thresholds are fitted on
    the history available at the corresponding forecast origin.
    """
    if any(window <= 0 or window > min_train_size for window in windows):
        raise ValueError("Moving-average windows must be positive and fit the initial history")
    if len(set(windows)) != len(windows):
        raise ValueError("Moving-average windows must be unique")
    values = train[list(TARGETS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Training targets must be finite")
    if (values[:, 0] < 0).any() or ((values[:, 1] < 0) | (values[:, 1] > 100)).any():
        raise ValueError("Training targets are outside their valid ranges")
    splits = rolling_origin_splits(train, horizon=horizon, min_train_size=min_train_size,
                                   step=horizon)
    if not splits:
        raise ValueError("Insufficient training timestamps for a complete validation fold")
    specifications = [("persistence", None)] + [(f"moving_average_{w}", w) for w in windows]
    if set(cpu_forecasters or {}) & {name for name, _ in specifications}:
        raise ValueError("CPU model names must not collide with baseline names")
    fold_rows, prediction_frames = [], []
    for fold, split in enumerate(splits, start=1):
        history = train[train.timestamp <= split.train_end]
        validation = train[train.timestamp.between(split.validation_start, split.validation_end)]
        template = validation[["timestamp"]]
        cutoff = float(history.aggregate_cpu_utilization.quantile(.9))
        fold_rows.append({
            "fold": fold, "train_n": len(history), "train_start": history.timestamp.iloc[0],
            "train_end": split.train_end, "validation_start": split.validation_start,
            "validation_end": split.validation_end, "validation_n": len(validation),
            "cpu_peak_threshold_pct": cutoff,
        })
        for target in TARGETS:
            reference = persistence_forecast(history, template, target).to_numpy()
            actual = validation[target].to_numpy(dtype=float)
            scale = float(history[target].abs().mean())
            forecasts = {
                model: (persistence_forecast(history, template, target).to_numpy() if window is None
                        else moving_average_forecast(history, template, target, window).to_numpy())
                for model, window in specifications
            }
            if target == "aggregate_cpu_utilization":
                for model, forecaster in (cpu_forecasters or {}).items():
                    # Callbacks receive observed values and horizon length only.
                    forecasts[model] = forecaster(history[target].to_numpy(dtype=float).copy(),
                                                   len(validation))
            for model, forecast in forecasts.items():
                predicted = np.asarray(forecast, dtype=float)
                if predicted.shape != actual.shape or not np.isfinite(predicted).all():
                    raise ValueError(f"Invalid forecast shape or nonfinite predictions from {model}")
                if (predicted < 0).any() or (target == "aggregate_cpu_utilization" and (predicted > 100).any()):
                    raise ValueError(f"Predictions outside target bounds from {model}")
                prediction_frames.append(pd.DataFrame({
                    "fold": fold, "model": model, "target": target,
                    "origin_timestamp": split.train_end,
                    "timestamp": validation.timestamp.to_numpy(),
                    "lead": np.arange(1, len(validation) + 1), "actual": actual,
                    "prediction": predicted, "persistence_prediction": reference,
                    "training_mean_scale": scale,
                    "scaled_absolute_error": 100 * np.abs(predicted - actual) / scale if scale else np.nan,
                    "peak_threshold": cutoff if target == "aggregate_cpu_utilization" else np.nan,
                    "is_peak": actual >= cutoff if target == "aggregate_cpu_utilization" else False,
                }))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    aggregates = []
    for keys in [("target", "model"), ("target", "model", "fold"), ("target", "model", "lead")]:
        records = []
        for key, group in predictions.groupby(list(keys), sort=False):
            records.append(dict(zip(keys, key)) | score_predictions(group))
        aggregates.append(pd.DataFrame(records))
    return dict(zip(("metrics", "fold_metrics", "horizon_metrics"), aggregates)) | {
        "predictions": predictions, "folds": pd.DataFrame(fold_rows),
    }


def evaluate_baselines(
    train: pd.DataFrame, horizon: int = 24, min_train_size: int = 38,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict[str, pd.DataFrame]:
    """Keep the baseline-only entry point and its original output contract."""
    return evaluate_forecasters(train, horizon, min_train_size, windows)


def plot_backtests(train: pd.DataFrame, results: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cpu_metrics = results["metrics"].query("target == 'aggregate_cpu_utilization'")
    best_ma = cpu_metrics[cpu_metrics.model != "persistence"].sort_values("mae", kind="stable").iloc[0].model
    choices = ["persistence", best_ma]
    colors = {"persistence": "#cc7526", best_ma: "#347a53"}
    labels = {"persistence": "Persistence", best_ma: best_ma.replace("_", " ").title()}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained", dpi=150)
    first = train.timestamp.iloc[0]
    axes[0].plot((train.timestamp - first) / 3600, train.aggregate_cpu_utilization,
                 color="#176b87", linewidth=1.7, label="Observed CPU (training window)")
    cpu_predictions = results["predictions"].query("target == 'aggregate_cpu_utilization'")
    for model in choices:
        for i, (_, group) in enumerate(cpu_predictions[cpu_predictions.model == model].groupby("fold")):
            axes[0].plot((group.timestamp - first) / 3600, group.prediction,
                         color=colors[model], linestyle="--", linewidth=1.6,
                         label=labels[model] if i == 0 else None)
    for fold in results["folds"].itertuples(index=False):
        axes[0].axvline((fold.validation_start - first) / 3600, color="#777777", alpha=.4, linewidth=.8)
    axes[0].set_title("Two-hour baseline forecasts | validation inside the training window", loc="left")
    axes[0].set_xlabel("Hours since first source timestamp | vertical lines mark validation starts")
    axes[0].set_ylabel("Mean CPU (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    for i, model in enumerate(choices):
        frame = results["fold_metrics"].query("target == 'aggregate_cpu_utilization' and model == @model")
        axes[1].bar(np.arange(len(frame)) + (i - .5) * .32, frame.mae, width=.32,
                    color=colors[model], label=labels[model])
    axes[1].set_xticks(np.arange(len(results["folds"])),
                       [f"Fold {f.fold}: {f.train_n} training points" for f in results["folds"].itertuples()])
    axes[1].set_ylabel("CPU MAE (percentage points)")
    axes[1].set_title("Each fold exposes a different workload regime", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(output / "baseline_backtests.png")
    plt.close(fig)


def run_backtest(
    data: Path, output: Path, train_size: int = 110, horizon: int = 24,
    min_train_size: int = 38, windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict:
    if not windows:
        raise ValueError("Include at least one moving-average window")
    readings = pd.read_csv(data / "vm_cpu_readings.csv.gz")
    train, holdout_timestamps = split_targets(build_targets(readings), train_size, horizon)
    results = evaluate_baselines(train, horizon, min_train_size, windows)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(output / f"baseline_{name}.csv", index=False)
    metrics, folds = results["metrics"], results["folds"]
    provisional = {
        target: group.sort_values("mae", kind="stable").iloc[0].model
        for target, group in metrics.groupby("target", sort=False)
    }
    summary = {
        "training_timestamps": len(train), "horizon_steps": horizon,
        "min_train_size": min_train_size, "step": horizon,
        "moving_average_windows": list(windows),
        "fold_count": len(folds), "validation_points_per_model_target": int(folds.validation_n.sum()),
        "provisional_best_baseline_by_mae": provisional,
        "holdout_labels_used": False, "holdout_timestamps": holdout_timestamps.timestamp.tolist(),
        "folds": json.loads(folds.to_json(orient="records")),
        "metrics": json.loads(metrics.to_json(orient="records", double_precision=12)),
    }
    (output / "baseline_evaluation.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_backtests(train, results, output)
    cpu = metrics[metrics.target == "aggregate_cpu_utilization"].sort_values("mae", kind="stable")
    counts = metrics[metrics.target == "active_vm_count"]
    best = cpu.iloc[0]
    persistence = cpu[cpu.model == "persistence"].iloc[0]
    best_folds = results["fold_metrics"].query(
        "target == 'aggregate_cpu_utilization' and model == @best.model")
    fold_changes = ", ".join(
        f"fold {row.fold}: {row.mae_improvement_pct:+.2f}%"
        for row in best_folds.itertuples(index=False)
    )
    if best.peak_n:
        peak_comparison = (
            f"For the {int(best.peak_n)} CPU peak observations, `{best.model}` has "
            f"MAE {best.peak_mae:.4f} percentage points versus "
            f"{persistence.peak_mae:.4f} for persistence. Its peak bias is "
            f"{best.peak_bias:+.4f}, versus {persistence.peak_bias:+.4f} for "
            "persistence. Negative peak bias means systematic underforecasting "
            "on this subset. Overall MAE and peak risk therefore require "
            "separate assessment when choosing a capacity policy."
        )
    else:
        peak_comparison = "No validation CPU observations exceed their training-derived peak thresholds."
    columns = ["model", "mae", "rmse", "bias", "nmae_pct_training_mean",
               "mae_improvement_pct", "peak_n", "peak_mae", "peak_bias"]
    # Blank cells mean undefined metrics, never missing predictions.
    cpu_table = cpu[columns].astype(object).where(cpu[columns].notna(), "N/A")
    fold_table = results["fold_metrics"].query("target == 'aggregate_cpu_utilization'")
    fold_table = fold_table[["fold", "model", "mae", "rmse", "bias", "peak_n", "peak_mae"]]
    fold_table = fold_table.astype(object).where(fold_table.notna(), "N/A")
    report = f"""# Baseline forecasting and training-only validation

Reproduce with `python -m cwfc.backtest --data challenge_data --output reports`.
This is baseline development, not final test evaluation. No final holdout
target is scored, summarized, plotted, or used to select a baseline.

## Validation design

Construct the full aggregate targets, then protect the last {horizon} labels.
Only the first {len(train)} timestamps enter this evaluation. Expanding
histories start with {min_train_size} observations and advance by {horizon}.
Every forecast is made once at its origin for all {horizon} future steps.
There is no updating with validation actuals during that forecast horizon.

{markdown_table(folds)}

There are {len(folds)} folds and {int(folds.validation_n.sum())} distinct scored
timestamps per model and target. The folds do not overlap. Histories expand,
so labels from an earlier fold are legitimately available at later forecast
origins. This matches repeated operational forecasting without using future
readings at any individual origin. Numeric timestamps retain their original
source-relative second units.

The three-fold default uses origins after 38, 62, and 86 observations. It
maximizes full nonoverlapping two-hour validation blocks after a history long
enough for the largest baseline window. Only three workload regimes are
available; these are limited development estimates, not independent evidence
of deployment performance or a test of daily/weekly seasonality.

## Baselines fixed before evaluation

- Persistence: repeat the last observed target value at every lead.
- Moving averages: repeat the mean of the last {', '.join(map(str, windows))}
  readings, respectively. At five-minute cadence the default windows cover
  15, 30, 60, and 120 minutes. The forecast stays flat throughout its horizon.
- Both targets use the same folds and all models see the same histories.
- No seasonal baseline is justified by fewer than 12 hours of data.
- No metadata or source-wide CPU summaries are used as predictors.

## Metrics and units

CPU MAE, RMSE, and bias are in **percentage points**. VM-count errors are in
VMs. Bias is prediction minus actual: negative values mean underforecasting.
Pooled MAE/RMSE are calculated over individual errors, not by averaging fold
RMSE values.

Normalized MAE is the mean absolute error divided by that fold's mean absolute
training target, multiplied by 100 (`nmae_pct_training_mean`). It is not
pointwise MAPE. A zero training scale yields N/A. Relative improvement is
`100 * (1 - model_MAE / persistence_MAE)`, using pooled errors. When
persistence has zero error, improvement and the relative MAE ratio are N/A;
we do not claim a percentage improvement over a perfect baseline.

A CPU peak is an observed validation reading at or above the 90th percentile
of that fold's **training** CPU target. Thresholds are fixed at the forecast
origin; validation actuals only determine which errors are subsequently
scored as peaks. No peak points gives N/A peak metrics. Active-count peak
metrics are not used because the target is constant in this release.

## Aggregate CPU results

{markdown_table(cpu_table)}

The lowest pooled CPU MAE belongs to `{best.model}`: {best.mae:.4f} percentage
points, with RMSE {best.rmse:.4f}. This identifies a provisional baseline
using development folds; its reported score is not an unbiased estimate after
choosing it among several windows. No final forecasting model is selected here.

Its fold-specific MAE improvements over persistence are {fold_changes}.
A positive number means lower error; a negative number means higher error.
The pooled ranking does not imply a win in every workload regime.

{peak_comparison}

## Active VM count results

{markdown_table(counts[['model', 'mae', 'rmse', 'bias']])}

All histories and validation blocks contain the same observed VM population.
Exact constant-count predictions reflect complete-trace cohort selection,
not demonstrated ability to forecast arrivals or departures. Persistence is
preferred on ties because it requires no window choice. No VM count is
hard-coded in the implementation.

## Fold-level CPU errors and failure analysis

{markdown_table(fold_table)}

![Baseline validation](baseline_backtests.png)

Flat forecasts cannot anticipate a workload change that begins after their
origin. Persistence can amplify an unusually high or low final reading;
averaging can stabilize such noise but can also lag a recent demand change.
Use the fold rows and signed bias to distinguish underprediction during a
rise from overprediction after a decline. Peak errors expose whether average
accuracy conceals under-provisioning risk.

`baseline_predictions.csv` preserves every scored prediction, origin, lead,
actual value, training scale, and peak flag for inspection. All actuals in
that file belong to development validation, not the protected final holdout.
`baseline_horizon_metrics.csv` gives lead-specific results, with only
{len(folds)} errors per model/target/lead. That sample is too small to establish
a reliable horizon-dependent uncertainty interval.

## What this stage supports

These results establish a reproducible reference for later CPU modeling.
Compare any additional model with persistence and the strongest moving
average using these same chronological folds. Freeze the final method and
parameters before generating and scoring the two-hour public holdout.
Capacity margins should be assessed using signed/peak forecast errors and
the consequences of under-provisioning, rather than low mean utilization
alone. This baseline experiment does not yet justify a calibrated uncertainty
band, a concrete scaling threshold, or generalization to a changing VM fleet.
"""
    (output / "baseline_evaluation.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("challenge_data"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--train-size", type=int, default=110)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--min-train-size", type=int, default=38)
    parser.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    args = parser.parse_args()
    summary = run_backtest(args.data, args.output, args.train_size, args.horizon,
                           args.min_train_size, tuple(args.windows))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
