"""Generate point predictions and a separate operational plan from the frozen method."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from cwfc.baselines import moving_average_forecast, persistence_forecast
from cwfc.data import build_targets, split_targets
from cwfc.freeze import load_frozen_method, training_fingerprint
from cwfc.policy import planning_ceiling

PREDICTION_COLUMNS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
DEFAULT_METHOD = Path(__file__).resolve().parents[2] / "config" / "frozen_method.json"


def validate_predictions(predictions: pd.DataFrame, timestamps) -> None:
    """Validate the submission schema against the supplied forecast template."""
    expected = np.asarray(timestamps)
    if predictions.columns.tolist() != PREDICTION_COLUMNS:
        raise ValueError("Prediction columns must match the submission schema exactly")
    if not len(expected) or len(predictions) != len(expected):
        raise ValueError("Expected one prediction for each forecast timestamp")
    if not all(pd.api.types.is_numeric_dtype(predictions[c]) for c in PREDICTION_COLUMNS):
        raise ValueError("All prediction columns must be numeric")
    values = predictions.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Predictions must be finite without missing values")
    timestamps = predictions.timestamp
    if (timestamps.duplicated().any() or not timestamps.is_monotonic_increasing
            or not np.array_equal(timestamps.to_numpy(), expected)):
        raise ValueError("Forecast timestamps must match the sorted, unique template exactly")
    if not (timestamps.to_numpy() == np.floor(timestamps.to_numpy())).all():
        raise ValueError("Forecast timestamps must be integer source-relative seconds")
    if (predictions.active_vm_count < 0).any():
        raise ValueError("VM-count predictions must be nonnegative")
    if not predictions.aggregate_cpu_utilization.between(0, 100).all():
        raise ValueError("CPU predictions must be percentages in [0, 100]")


def forecast_from_training(train: pd.DataFrame, template: pd.DataFrame, method: dict) -> pd.DataFrame:
    """Receive historical targets and future timestamps only, never future labels."""
    if template.columns.tolist() != ["timestamp"]:
        raise ValueError("The future template must contain timestamps only")
    if train.empty:
        raise ValueError("Training history must not be empty")
    validate_predictions(train[PREDICTION_COLUMNS], train.timestamp)
    cadence = method["cadence_seconds"]
    if cadence <= 0 or not np.all(np.diff(train.timestamp.to_numpy()) == cadence):
        raise ValueError("Training timestamps must follow the frozen cadence")
    expected = train.timestamp.iloc[-1] + cadence * np.arange(1, len(template) + 1)
    if not np.array_equal(template.timestamp.to_numpy(), expected):
        raise ValueError("The forecast template must immediately follow training at the frozen cadence")
    cpu = method["targets"]["aggregate_cpu_utilization"]
    count = method["targets"]["active_vm_count"]
    if cpu["method"] != "moving_average" or count["method"] != "persistence":
        raise ValueError("Unsupported frozen forecasting methods")
    window = cpu["window"]
    if not isinstance(window, int) or window < 1:
        raise ValueError("Frozen moving-average window must be a positive integer")
    predictions = template.reset_index(drop=True).copy()
    # Template indices need not start at zero; use arrays to avoid label alignment.
    predictions["active_vm_count"] = persistence_forecast(
        train, template, "active_vm_count").to_numpy()
    predictions["aggregate_cpu_utilization"] = moving_average_forecast(
        train, template, "aggregate_cpu_utilization", window).to_numpy()
    validate_predictions(predictions, template.timestamp)
    return predictions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_prediction(
    data: Path, output: Path, reports: Path,
    method_path: Path = DEFAULT_METHOD, horizon: int | None = None,
    train_size: int | None = None,
) -> dict:
    method = load_frozen_method(method_path)
    horizon = method["forecast_horizon"]["observations"] if horizon is None else horizon
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    source = data / "vm_cpu_readings.csv.gz"
    targets = build_targets(pd.read_csv(source))
    train_size = len(targets) - horizon if train_size is None else train_size
    if train_size < method["targets"]["aggregate_cpu_utilization"]["window"]:
        raise ValueError("Insufficient history for the frozen 12-reading window; supply more data or a shorter --horizon")
    train, template = split_targets(targets, train_size, horizon)
    predictions = forecast_from_training(train, template, method)
    margin = method["headroom_policy"]["margin_percentage_points"]
    ceilings = planning_ceiling(predictions.aggregate_cpu_utilization, margin)
    plan = pd.DataFrame({
        "timestamp": predictions.timestamp,
        "active_vm_count_forecast": predictions.active_vm_count,
        "cpu_point_forecast_pct": predictions.aggregate_cpu_utilization,
        "cpu_planning_ceiling_pct": ceilings,
        "headroom_percentage_points": margin,
        "consecutive_breaches_for_review": method["headroom_policy"]["consecutive_breaches_for_review"],
    })
    fingerprint = training_fingerprint(train)
    same_training = fingerprint == method["training"]["aggregate_targets_sha256"]
    same_split = (
        len(train) == method["training"]["observations"]
        and len(template) == method["forecast_horizon"]["observations"]
        and int(train.timestamp.iloc[0]) == method["training"]["first_timestamp"]
        and int(train.timestamp.iloc[-1]) == method["training"]["last_timestamp"]
        and int(template.timestamp.iloc[0]) == method["forecast_horizon"]["first_timestamp"]
        and int(template.timestamp.iloc[-1]) == method["forecast_horizon"]["last_timestamp"]
    )
    plan_path = reports / "operational_plan.csv"
    run_path = reports / "prediction_run.json"
    report_path = reports / "prediction_generation.md"
    output_paths = [output, plan_path, run_path, report_path]
    protected = [method_path, method_path.with_suffix(method_path.suffix + ".sha256"),
                 *data.iterdir(), *Path(__file__).resolve().parent.glob("*.py")]
    if len({p.resolve() for p in output_paths}) != len(output_paths):
        raise ValueError("Prediction and report outputs must be separate files")
    if {p.resolve() for p in output_paths} & {p.resolve() for p in protected}:
        raise ValueError("Outputs must not overwrite source data, frozen decisions, or implementation")
    output.parent.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False, float_format="%.12f", lineterminator="\n")
    plan.to_csv(plan_path, index=False, float_format="%.12f", lineterminator="\n")
    # Validate the serialized artifact as a reviewer will read it.
    written = pd.read_csv(output)
    validate_predictions(written, template.timestamp)
    # The only expected difference is decimal serialization precision.
    if not np.allclose(written.iloc[:, 1:], predictions.iloc[:, 1:], rtol=0, atol=1e-12):
        raise ValueError("Serialized predictions differ from the frozen forecast")
    summary = {
        "status": "predictions_generated_without_holdout_scoring",
        "rows": len(written), "columns": written.columns.tolist(),
        "first_timestamp": int(written.timestamp.iloc[0]),
        "last_timestamp": int(written.timestamp.iloc[-1]),
        "training_observations": len(train), "training_last_timestamp": int(train.timestamp.iloc[-1]),
        "training_targets_sha256": fingerprint,
        "matches_frozen_training": same_training, "matches_frozen_split": same_split,
        "application": "frozen_data_reproduction" if same_training and same_split else "fixed_method_on_alternative_data_or_horizon",
        "cpu_method": method["targets"]["aggregate_cpu_utilization"]["method"],
        "cpu_window": method["targets"]["aggregate_cpu_utilization"]["window"],
        "active_vm_method": method["targets"]["active_vm_count"]["method"],
        "margin_percentage_points": margin,
        "headroom_calibration_matches_input_training": same_training and same_split,
        "headroom_coverage_guaranteed": False,
        "source_cpu_sha256": _sha256(source),
        "frozen_method_sha256": method_path.with_suffix(method_path.suffix + ".sha256").read_text().strip(),
        "prediction_implementation_sha256": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
        "predictions_sha256": _sha256(output), "operational_plan_sha256": _sha256(plan_path),
        "holdout_labels_used_for_forecasting": False, "holdout_scored": False,
        "environment": {"python": platform.python_version(), "numpy": version("numpy"), "pandas": version("pandas")},
    }
    run_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    cpu_forecast = float(predictions.aggregate_cpu_utilization.iloc[0])
    count_forecast = float(predictions.active_vm_count.iloc[0])
    budget = method["headroom_policy"]["capacity_budget_pct"]
    budget_note = ("No external capacity budget B was supplied, so no budget-based scaling decision is asserted."
                   if budget is None else f"The supplied sustainable utilization budget is {budget}%.")
    report = f"""# Final prediction generation

The frozen configuration and source fingerprints were verified before
forecasting. This stage generates predictions only: no final holdout target
is scored, summarized, plotted, or used to alter a model or headroom margin.

## Submission artifact

- Rows: **{len(written)}**; columns: `timestamp,active_vm_count,aggregate_cpu_utilization`.
- Forecast timestamps: **{summary['first_timestamp']} through {summary['last_timestamp']}**
  in source-relative seconds, at the frozen five-minute cadence.
- Training: {len(train)} observations, ending at {summary['training_last_timestamp']} seconds.
- CPU point forecast: **{cpu_forecast:.8f}%** at every lead, calculated once
  from the last {summary['cpu_window']} observed training aggregate values.
- Active VM count: **{count_forecast:g}** at every lead, taken from the last
  observed training count. It is not a hard-coded VM population.
- The flat forecasts are intentional consequences of the frozen methods;
  no future actuals are substituted or used for updates.
- The serialized CSV has exactly the required columns, ordered unique
  timestamps, no index, no missing/nonfinite predictions, nonnegative VM
  counts, and CPU predictions in [0, 100]. Values are written to 12 decimal
  places and revalidated after loading the file.

## Separate operational plan

The planning ceiling is **{ceilings[0]:.8f}%**, including the fixed allowance
of **{margin:.8f} percentage points**, capped at 100%. It appears only in
`operational_plan.csv`; the submission contains the CPU point forecast.
Two consecutive actual readings above their issued ceilings trigger the
frozen review policy when observations arrive. No future breach or alert is
claimed before observing those readings. {budget_note}

## Reproduction and provenance

The training-target fingerprint matches the freeze: **{same_training}**.
The timestamp split matches the freeze: **{same_split}**.
The machine-readable run record preserves source, method, implementation,
prediction, and plan hashes, plus runtime versions. The frozen record and
its source hashes are unchanged by prediction generation.

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

Input-derived VM counts and timestamps allow the same command to operate
on a smaller VM population. The default horizon comes from the frozen record;
training length is inferred as total timestamps minus that horizon. For a
shorter smoke trace, `--horizon` may shorten the forecast while retaining
the fixed 12-reading model window. At least 12 training timestamps are
required. Alternative inputs are identified in the run record; the supplied
headroom margin has not been validated for their population or horizon.

The original release's final 24 labels remain excluded from this generation
stage. Holdout accuracy and realized policy breaches require a separate,
subsequent evaluation of these saved forecasts, without retuning.
"""
    report_path.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("challenge_data"))
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--method", type=Path, default=DEFAULT_METHOD)
    parser.add_argument("--horizon", type=int, help="Override output length for a shorter smoke trace")
    parser.add_argument("--train-size", type=int, help="Optional explicit training length; otherwise inferred")
    args = parser.parse_args()
    summary = run_prediction(args.data, args.output, args.reports, args.method, args.horizon, args.train_size)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
