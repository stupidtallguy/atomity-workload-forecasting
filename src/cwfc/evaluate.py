"""Score saved, frozen forecasts once decisions are fixed; never retune them."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.audit import markdown_table
from cwfc.backtest import DEFAULT_WINDOWS, TARGETS, score_predictions
from cwfc.baselines import moving_average_forecast, persistence_forecast
from cwfc.data import build_targets, split_targets
from cwfc.freeze import load_frozen_method, training_fingerprint
from cwfc.policy import breach_signals, planning_ceiling
from cwfc.predict import DEFAULT_METHOD, _sha256, forecast_from_training, validate_predictions


def display_table(frame: pd.DataFrame) -> str:
    return markdown_table(frame.astype(object).where(frame.notna(), "N/A"))


def evaluate_saved_forecasts(
    train: pd.DataFrame, actual: pd.DataFrame, predictions: pd.DataFrame,
    plan: pd.DataFrame, method: dict,
) -> dict:
    """Score supplied labels without fitting anything on them or selecting a winner."""
    validate_predictions(actual, actual.timestamp)
    validate_predictions(predictions, actual.timestamp)
    expected = forecast_from_training(train, actual[["timestamp"]], method)
    if not np.allclose(predictions.iloc[:, 1:], expected.iloc[:, 1:], rtol=0, atol=1e-12):
        raise ValueError("Saved predictions differ from the frozen training-only forecast")
    plan_columns = ["timestamp", "active_vm_count_forecast", "cpu_point_forecast_pct",
                    "cpu_planning_ceiling_pct", "headroom_percentage_points",
                    "consecutive_breaches_for_review"]
    if plan.columns.tolist() != plan_columns or len(plan) != len(actual):
        raise ValueError("Operational plan has an invalid schema or row count")
    if not np.isfinite(plan.to_numpy(dtype=float)).all():
        raise ValueError("Operational plan must contain finite numeric values")
    if not np.array_equal(plan.timestamp.to_numpy(), actual.timestamp.to_numpy()):
        raise ValueError("Operational plan timestamps differ from the forecast template")
    policy = method["headroom_policy"]
    margin, consecutive = policy["margin_percentage_points"], policy["consecutive_breaches_for_review"]
    if (not np.allclose(plan.cpu_point_forecast_pct, predictions.aggregate_cpu_utilization, rtol=0, atol=1e-12)
            or not np.allclose(plan.active_vm_count_forecast, predictions.active_vm_count, rtol=0, atol=1e-12)
            or not np.allclose(plan.headroom_percentage_points, margin, rtol=0, atol=1e-12)
            or not np.allclose(plan.cpu_planning_ceiling_pct,
                               planning_ceiling(expected.aggregate_cpu_utilization, margin), rtol=0, atol=1e-12)
            or not plan.consecutive_breaches_for_review.eq(consecutive).all()):
        raise ValueError("Operational plan differs from the frozen policy")
    cutoff = float(train.aggregate_cpu_utilization.quantile(.9))
    frames = []
    for target in TARGETS:
        reference = persistence_forecast(train, actual[["timestamp"]], target).to_numpy()
        forecasts = {"frozen_submission": predictions[target].to_numpy()}
        if target == "aggregate_cpu_utilization":
            forecasts["persistence"] = reference
            # The selected 12-reading baseline is represented by the saved file,
            # not recomputed as a second entry with insignificant rounding differences.
            for window in DEFAULT_WINDOWS:
                if window != method["targets"][target]["window"]:
                    forecasts[f"moving_average_{window}"] = moving_average_forecast(
                        train, actual[["timestamp"]], target, window).to_numpy()
        scale = float(train[target].abs().mean())
        observed = actual[target].to_numpy(dtype=float)
        for model, forecast in forecasts.items():
            error = forecast - observed
            frames.append(pd.DataFrame({
                "target": target, "model": model, "timestamp": actual.timestamp.to_numpy(),
                "origin_timestamp": train.timestamp.iloc[-1], "lead": np.arange(1, len(actual) + 1),
                "actual": observed, "prediction": forecast, "error": error,
                "absolute_error": np.abs(error), "persistence_prediction": reference,
                "training_mean_scale": scale,
                "scaled_absolute_error": 100 * np.abs(error) / scale if scale else np.nan,
                "is_peak": observed >= cutoff if target == "aggregate_cpu_utilization" else False,
                "peak_threshold": cutoff if target == "aggregate_cpu_utilization" else np.nan,
            }))
    errors = pd.concat(frames, ignore_index=True)
    metrics = pd.DataFrame([
        {"target": target, "model": model, **score_predictions(group)}
        for (target, model), group in errors.groupby(["target", "model"], sort=False)
    ])
    chosen = errors.query("target == 'aggregate_cpu_utilization' and model == 'frozen_submission'").copy()
    blocks = []
    for start in range(1, len(actual) + 1, 6):
        end = min(start + 5, len(actual))
        group = chosen[chosen.lead.between(start, end)]
        blocks.append({"lead_start": start, "lead_end": end, **score_predictions(group)})
    policy_rows = chosen[["timestamp", "lead", "actual", "prediction", "error", "is_peak"]].reset_index(drop=True)
    policy_rows["planning_ceiling_pct"] = plan.cpu_planning_ceiling_pct.to_numpy()
    policy_rows = pd.concat([
        policy_rows, breach_signals(policy_rows.actual, policy_rows.planning_ceiling_pct, consecutive),
    ], axis=1)
    peak = policy_rows[policy_rows.is_peak]
    policy_metrics = {
        "margin_percentage_points": margin, "peak_threshold_pct": cutoff,
        "n": len(actual), "covered": int((~policy_rows.breach).sum()),
        "point_coverage": float((~policy_rows.breach).mean()),
        "entire_horizon_covered": bool((~policy_rows.breach).all()),
        "breaches": int(policy_rows.breach.sum()),
        "breach_timestamps": policy_rows.loc[policy_rows.breach, "timestamp"].tolist(),
        "peak_n": len(peak), "peak_breaches": int(peak.breach.sum()),
        "alert_events": int(policy_rows.alert_event.sum()),
        "alert_timestamps": policy_rows.loc[policy_rows.alert_event, "timestamp"].tolist(),
        "mean_unused_allowance_pp": float(np.maximum(policy_rows.planning_ceiling_pct - policy_rows.actual, 0).mean()),
        "max_exceedance_pp": float(np.maximum(policy_rows.actual - policy_rows.planning_ceiling_pct, 0).max()),
        "largest_point_underprediction_pp": float(np.maximum(policy_rows.actual - policy_rows.prediction, 0).max()),
        "minimum_ceiling_slack_pp": float((policy_rows.planning_ceiling_pct - policy_rows.actual).min()),
        "coverage_guaranteed": False,
    }
    return {"metrics": metrics, "errors": errors, "horizon_blocks": pd.DataFrame(blocks),
            "policy_rows": policy_rows, "policy_metrics": policy_metrics}


def plot_evaluation(train: pd.DataFrame, actual: pd.DataFrame, results: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = results["policy_rows"]
    origin = train.timestamp.iloc[-1]
    minutes = (rows.timestamp - origin) / 60
    history = train.tail(24)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=150, layout="constrained")
    axes[0].plot((history.timestamp - origin) / 60, history.aggregate_cpu_utilization,
                 color="#7b9eae", label="Recent training CPU", linewidth=1.5)
    axes[0].plot(minutes, rows.actual, color="#176b87", label="Holdout actual", linewidth=1.8)
    axes[0].plot(minutes, rows.prediction, color="#347a53", linestyle="--", label="Frozen point forecast")
    axes[0].plot(minutes, rows.planning_ceiling_pct, color="#ac752d", linestyle=":", label="Planning ceiling")
    axes[0].fill_between(minutes, rows.prediction, rows.planning_ceiling_pct, color="#e2c995", alpha=.25)
    reference = results["errors"].query("target == 'aggregate_cpu_utilization' and model == 'persistence'")
    axes[0].plot(minutes, reference.prediction, color="#8b56a5", linestyle="--", linewidth=1,
                 label="Persistence comparator")
    breached = rows[rows.breach]
    if len(breached):
        axes[0].scatter((breached.timestamp - origin) / 60, breached.actual,
                        color="#b13434", marker="^", s=35, label="Ceiling breach", zorder=4)
    axes[0].axvline(0, color="#777777", linewidth=1)
    axes[0].set_title("Final two-hour holdout | methods and margin unchanged", loc="left")
    axes[0].set_xlabel("Minutes from forecast origin | shaded area is a planning allowance")
    axes[0].set_ylabel("Aggregate CPU (%)")
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    axes[1].bar(minutes, rows.error, width=3.5,
                color=np.where(rows.error < 0, "#b45c49", "#4c8664"))
    axes[1].axhline(0, color="#777777", linewidth=1)
    axes[1].set_title("Signed error | negative values mean underforecasting", loc="left")
    axes[1].set_xlabel("Forecast lead (minutes)")
    axes[1].set_ylabel("Prediction minus actual (pp)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(output / "holdout_evaluation.png")
    plt.close(fig)


def run_evaluation(
    data: Path, predictions_path: Path, output: Path,
    method_path: Path = DEFAULT_METHOD, generation_reports: Path = Path("reports"),
) -> dict:
    method = load_frozen_method(method_path)
    run_path, plan_path = generation_reports / "prediction_run.json", generation_reports / "operational_plan.csv"
    source = data / "vm_cpu_readings.csv.gz"
    protected = [predictions_path, method_path, method_path.with_suffix(method_path.suffix + ".sha256"),
                 run_path, plan_path, source]
    before = {str(p.resolve()): _sha256(p) for p in protected}
    generation = json.loads(run_path.read_text(encoding="utf-8"))
    if generation["predictions_sha256"] != _sha256(predictions_path):
        raise ValueError("Saved predictions do not match the pre-evaluation run hash")
    if generation["operational_plan_sha256"] != _sha256(plan_path):
        raise ValueError("Saved operational plan does not match the pre-evaluation run hash")
    if generation["source_cpu_sha256"] != _sha256(source):
        raise ValueError("Source data differs from the prediction-generation input")
    digest = method_path.with_suffix(method_path.suffix + ".sha256").read_text().strip()
    if generation["frozen_method_sha256"] != digest:
        raise ValueError("Prediction-generation record references a different frozen method")
    implementation = Path(__file__).resolve().with_name("predict.py")
    if generation["prediction_implementation_sha256"] != hashlib.sha256(
            implementation.read_text(encoding="utf-8").encode()).hexdigest():
        raise ValueError("Prediction implementation differs from its generation record")
    targets = build_targets(pd.read_csv(source))
    horizon, train_size = generation["rows"], generation["training_observations"]
    train, template = split_targets(targets, train_size, horizon)
    if training_fingerprint(train) != generation["training_targets_sha256"]:
        raise ValueError("Training targets differ from the prediction-generation record")
    predictions = pd.read_csv(predictions_path)
    validate_predictions(predictions, template.timestamp)
    actual = targets.iloc[train_size:].reset_index(drop=True)
    results = evaluate_saved_forecasts(train, actual, predictions, pd.read_csv(plan_path), method)
    output_names = [f"holdout_{name}.csv" for name in ("metrics", "errors", "horizon_blocks", "policy_rows")]
    output_names += ["holdout_evaluation.json", "holdout_evaluation.md", "holdout_evaluation.png"]
    outputs = {(output / name).resolve() for name in output_names}
    protected_paths = {p.resolve() for p in protected}
    protected_paths.update(p.resolve() for p in data.iterdir())
    protected_paths.update(p.resolve() for p in Path(__file__).resolve().parent.glob("*.py"))
    if outputs & protected_paths:
        raise ValueError("Evaluation outputs must not overwrite input or frozen artifacts")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("metrics", "errors", "horizon_blocks", "policy_rows"):
        results[name].to_csv(output / f"holdout_{name}.csv", index=False)
    unchanged = before == {str(p.resolve()): _sha256(p) for p in protected}
    if not unchanged:
        raise ValueError("An input artifact changed during evaluation")
    summary = {
        "status": "frozen_holdout_evaluated_without_retuning", "holdout_observations": len(actual),
        "first_timestamp": int(actual.timestamp.iloc[0]), "last_timestamp": int(actual.timestamp.iloc[-1]),
        "training_observations": len(train),
        "training_matches_freeze": training_fingerprint(train) == method["training"]["aggregate_targets_sha256"],
        "metrics": json.loads(results["metrics"].to_json(orient="records", double_precision=12)),
        "policy": results["policy_metrics"],
        "holdout_labels_used_for_scoring": True, "holdout_labels_used_for_fitting_or_selection": False,
        "parameters_or_margin_changed": False, "protected_artifacts_unchanged": unchanged,
        "predictions_sha256": _sha256(predictions_path), "frozen_method_sha256": digest,
        "generation_record_sha256": _sha256(run_path), "source_cpu_sha256": _sha256(source),
        "evaluation_implementation_sha256": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
    }
    (output / "holdout_evaluation.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_evaluation(train, actual, results, output)
    cpu_metrics = results["metrics"].query("target == 'aggregate_cpu_utilization'")
    chosen = cpu_metrics.query("model == 'frozen_submission'").iloc[0]
    count = results["metrics"].query("target == 'active_vm_count'")
    rows, policy = results["policy_rows"], results["policy_metrics"]
    better_comparators = cpu_metrics[(cpu_metrics.model != "frozen_submission")
                                    & (cpu_metrics.mae < chosen.mae - 1e-12)]
    if len(better_comparators):
        comparator_note = "On this test window, the following pre-existing comparators have lower MAE: "
        comparator_note += ", ".join(f"`{row.model}` ({row.mae:.4f})"
                                     for row in better_comparators.itertuples(index=False))
        comparator_note += (". This test outcome is retained transparently. The submission stays with "
                            "the method selected on development folds; no switch is made after seeing labels.")
    else:
        comparator_note = ("None of the original comparators has lower MAE in this window. "
                           "One test horizon does not establish superiority across workload regimes.")
    if chosen.peak_n:
        peak_note = (f"There are {int(chosen.peak_n)} holdout CPU observations at or above the "
                     f"training-derived threshold of {policy['peak_threshold_pct']:.4f}%. "
                     f"Their MAE is {chosen.peak_mae:.4f} and signed bias "
                     f"{chosen.peak_bias:+.4f} percentage points.")
    else:
        peak_note = (f"No holdout CPU observation reaches the training-derived peak threshold "
                     f"of {policy['peak_threshold_pct']:.4f}%. Peak metrics are N/A. "
                     "This holdout does not test the model's ability to handle a comparable "
                     "high-load regime; the poor peak behavior found in development remains relevant.")
    if np.isfinite(chosen.mae_improvement_pct):
        direction = "lower" if chosen.mae_improvement_pct >= 0 else "higher"
        relative_note = (f"Its MAE is **{abs(chosen.mae_improvement_pct):.2f}% {direction} than "
                         "persistence** on this holdout.")
    else:
        relative_note = "Relative improvement over persistence is N/A because persistence has zero error."
    highest = rows.nlargest(3, "actual")[["timestamp", "lead", "actual", "prediction", "error", "is_peak"]]
    worst = rows.loc[rows.error.abs().idxmax()]
    worst_direction = "underprediction" if worst.error < 0 else "overprediction"
    first_block, last_block = results["horizon_blocks"].iloc[0], results["horizon_blocks"].iloc[-1]
    peak_policy_note = (f"{policy['peak_breaches']}/{policy['peak_n']}" if policy["peak_n"]
                        else "N/A (no peak observations)")
    report = f"""# Final holdout evaluation

The saved predictions were verified against their pre-evaluation hash and
the frozen training-only forecasting method before scoring. The source data,
operational plan, prediction implementation, training fingerprint, and frozen
configuration also match their generation records. **Predictions, model
parameters, and the headroom margin are unchanged.** This is final scoring,
not an opportunity to choose a new model or recalibrate on test errors.

## Protocol and comparison

- Training: {len(train)} observations; origin: {int(train.timestamp.iloc[-1])} seconds.
- Final holdout: {len(actual)} observations, {int(actual.timestamp.iloc[0])} through
  {int(actual.timestamp.iloc[-1])} source-relative seconds.
- The selected CPU forecast is scored directly from `predictions.csv`.
  Other original baseline comparators are calculated using training data only
  and forecast the identical timestamps. The 12-reading moving average is
  represented by `frozen_submission`, without a duplicate recomputed row.
- The peak threshold is the 90th percentile of training CPU, fixed at the
  forecast origin. Holdout values are used only as labels for scoring.
- No new candidate models, windows, thresholds, or margins are selected here.

## CPU accuracy

{display_table(cpu_metrics[['model', 'mae', 'rmse', 'bias', 'nmae_pct_training_mean', 'mae_improvement_pct', 'peak_n', 'peak_mae', 'peak_bias']])}

CPU errors and bias are in percentage points. Bias is prediction minus
actual. Normalized MAE divides absolute errors by the mean absolute training
target and multiplies by 100, avoiding pointwise near-zero denominators. A
zero training scale or perfect persistence reference yields N/A for the
corresponding normalized/relative metric.

The frozen CPU model has MAE **{chosen.mae:.4f}**, RMSE **{chosen.rmse:.4f}**,
and signed bias **{chosen.bias:+.4f}** percentage points. {relative_note}
The comparison is reported without changing the selected method.

{comparator_note}

Observed holdout CPU mean is {actual.aggregate_cpu_utilization.mean():.4f}%,
with range {actual.aggregate_cpu_utilization.min():.4f}% to
{actual.aggregate_cpu_utilization.max():.4f}%.

{peak_note}

## Error progression and highest observed loads

{display_table(results['horizon_blocks'][['lead_start', 'lead_end', 'n', 'mae', 'rmse', 'bias']])}

Highest three holdout CPU values, reported descriptively without redefining peaks:

{display_table(highest)}

A fixed level cannot follow newly developing changes after the origin.
The first lead block has signed bias {first_block.bias:+.4f}, versus
{last_block.bias:+.4f} percentage points in the final block. The largest
absolute error is an {worst_direction} of {abs(worst.error):.4f} percentage
points at lead {int(worst.lead)} (timestamp {int(worst.timestamp)}).
One two-hour window does not establish a general
horizon/error relationship. Development validation remains the broader
available evidence across workload regimes.

![Frozen forecast and realized holdout](holdout_evaluation.png)

## Active VM count

{display_table(count[['model', 'n', 'mae', 'rmse', 'bias']])}

The selected complete-trace VM cohort stays constant. Exact count forecasts
therefore demonstrate consistency with the curated target, not general
churn prediction. Relative improvement over a zero-error persistence
reference is undefined and remains N/A.

## Frozen headroom policy

- Additive margin: **{policy['margin_percentage_points']:.8f} percentage points**, unchanged.
- Covered readings: **{policy['covered']}/{policy['n']} ({policy['point_coverage']:.2%})**.
- Entire two-hour horizon covered: **{policy['entire_horizon_covered']}**.
- Ceiling breaches: **{policy['breaches']}**; peak breaches: **{peak_policy_note}**.
- Two-consecutive-breach review events: **{policy['alert_events']}**;
  timestamps: `{policy['alert_timestamps']}`.
- Maximum exceedance above the ceiling: **{policy['max_exceedance_pp']:.4f} percentage points**.
- Mean positive unused allowance: **{policy['mean_unused_allowance_pp']:.4f} percentage points**.
- Largest point-forecast underprediction: **{policy['largest_point_underprediction_pp']:.4f} percentage points**.
- Minimum ceiling-minus-actual gap: **{policy['minimum_ceiling_slack_pp']:.4f} percentage points**
  (negative would indicate a breach).

These are realized diagnostics for one holdout. Empirical point coverage is
not a calibrated probability or a guarantee for future complete horizons.
The positive ceiling-minus-actual gap is a utilization-budget proxy, not
measured infrastructure cost. No hardware allocation or automatic scaling
action is inferred, and no external alert is sent by this evaluator.

## Reproduction and artifact integrity

Install the analysis extra for the chart, then score the saved artifacts:

```bash
python -m pip install -e ".[analysis]"
python -m cwfc.evaluate --data challenge_data --predictions predictions.csv --output reports
```

`holdout_errors.csv` preserves each actual, prediction, signed error, origin,
lead, and peak flag. `holdout_policy_rows.csv` records each ceiling breach
and review event. The JSON summary links the scored prediction hash to the
earlier generation record and freeze. Historical prediction-generation
records keep their original pre-scoring status; this separate evaluation
record documents that labels have now been examined for scoring only.

All protected inputs are unchanged after evaluation. Final-report conclusions
must include these outcomes and limitations; the model and policy stay frozen.
"""
    (output / "holdout_evaluation.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("challenge_data"))
    parser.add_argument("--predictions", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--generation-reports", type=Path, default=Path("reports"))
    parser.add_argument("--method", type=Path, default=DEFAULT_METHOD)
    args = parser.parse_args()
    print(json.dumps(run_evaluation(args.data, args.predictions, args.output,
                                    args.method, args.generation_reports), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
