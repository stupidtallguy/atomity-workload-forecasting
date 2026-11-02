"""Freeze forecasting decisions and headroom before any final holdout scoring."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import pandas as pd

from cwfc.audit import markdown_table
from cwfc.backtest import evaluate_baselines
from cwfc.data import build_targets, split_targets
from cwfc.policy import evaluate_headroom

FROZEN_SOURCES = (
    "baselines.py", "data.py", "validation.py", "metrics.py", "backtest.py",
    "models.py", "compare.py", "policy.py", "freeze.py",
)


def source_fingerprints() -> dict[str, str]:
    # Universal-newline reads keep the hashes stable across LF/CRLF checkouts.
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in FROZEN_SOURCES}


def training_fingerprint(train: pd.DataFrame) -> str:
    canonical = train.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_frozen_method(path: Path) -> dict:
    """Verify saved decisions and relevant source code; never silently regenerate."""
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    serialized = path.read_text(encoding="utf-8")
    expected = checksum_path.read_text(encoding="utf-8").strip()
    if hashlib.sha256(serialized.encode()).hexdigest() != expected:
        raise ValueError("Frozen method checksum mismatch; do not change it after holdout evaluation")
    method = json.loads(serialized)
    if method.get("schema_version") != 1 or method.get("status") != "frozen_before_holdout":
        raise ValueError("Unsupported frozen method")
    if method.get("source_sha256") != source_fingerprints():
        raise ValueError("Source code differs from the frozen method; review before predicting")
    return method


def freeze_method(data: Path, output: Path, method_path: Path, frozen_on: str) -> dict:
    date.fromisoformat(frozen_on)
    # Build the full series, then expose only training labels to all decisions.
    train, template = split_targets(build_targets(pd.read_csv(data / "vm_cpu_readings.csv.gz")))
    development = evaluate_baselines(train)
    selected = development["predictions"].query(
        "target == 'aggregate_cpu_utilization' and model == 'moving_average_12'").copy()
    policy = evaluate_headroom(selected, quantile=.95)
    folds = policy["calibration_folds"]
    sequential = policy["sequential_folds"]
    method = {
        "schema_version": 1, "status": "frozen_before_holdout", "frozen_on": frozen_on,
        "data_release": "data-v1", "cadence_seconds": 300,
        "training": {"observations": len(train), "first_timestamp": int(train.timestamp.iloc[0]),
                     "last_timestamp": int(train.timestamp.iloc[-1]),
                     "aggregate_targets_sha256": training_fingerprint(train)},
        "forecast_horizon": {"observations": len(template), "minutes": 120,
                             "first_timestamp": int(template.timestamp.iloc[0]),
                             "last_timestamp": int(template.timestamp.iloc[-1])},
        "targets": {
            "active_vm_count": {"aggregation": "distinct observed VM IDs per timestamp",
                                "method": "persistence", "minimum": 0,
                                "churn_uncertainty_calibrated": False},
            "aggregate_cpu_utilization": {
                "aggregation": "unweighted mean of VM avg_cpu per timestamp, percent",
                "method": "moving_average", "window": 12,
                "forecast_mode": "fixed origin; last 12 observed values averaged once for all 24 leads",
                "bounds": [0, 100],
            },
        },
        "selection": {"primary_metric": "pooled development CPU MAE",
                      "fold_training_sizes": development["folds"].train_n.tolist(),
                      "validation_horizon": 24, "validation_points": len(selected),
                      "refit": "use first 110 observations only; no fitting on holdout labels",
                      "missing_data": "fail on missing, invalid, duplicate, or incomplete observations",
                      "post_holdout_retuning": False},
        "headroom_policy": {
            "kind": "empirical one-sided aggregate CPU planning allowance",
            "quantile": .95, "quantile_method": "higher",
            "residual_definition": "actual minus point forecast, development validation only",
            "calibration_residuals": len(selected), "margin_percentage_points": policy["margin_pp"],
            "planning_ceiling": "min(100, CPU point forecast + margin_percentage_points)",
            "coverage_guaranteed": False, "simultaneous_horizon_coverage_guaranteed": False,
            "consecutive_breaches_for_review": 2,
            "breach_rule": "observed aggregate CPU > its fixed-origin planning ceiling",
            "review_rule": "alert once at the second consecutive breach; clear the run on recovery",
            "capacity_budget_pct": None,
            "budget_rule": "if an independently established budget B exists, review capacity before max(ceiling) >= B",
            "automatic_downscaling": False,
            "hardware_capacity_conversion": "unavailable; mean utilization is not total consumed cores",
            "production_refresh_minutes": 5,
            "challenge_refresh_within_horizon": False,
        },
        "holdout_labels_used": False,
        "source_sha256": source_fingerprints(),
        "environment": {"python": platform.python_version(), "numpy": version("numpy"),
                        "pandas": version("pandas")},
    }
    checksum_path = method_path.with_suffix(method_path.suffix + ".sha256")
    if method_path.exists() or checksum_path.exists():
        if load_frozen_method(method_path) != method:
            raise ValueError("An existing freeze differs; refusing to overwrite decisions or calibration")
    else:
        method_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(method, indent=2, allow_nan=False) + "\n"
        with method_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
        with checksum_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(hashlib.sha256(serialized.encode()).hexdigest() + "\n")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("calibration_predictions", "calibration_folds", "sequential_predictions", "sequential_folds"):
        policy[name].to_csv(output / f"headroom_{name}.csv", index=False)
    coverage = int(folds.covered.sum()) / int(folds.n.sum())
    protected_windows = int(folds.breaches.eq(0).sum())
    diagnostics = {
        "margin_percentage_points": policy["margin_pp"],
        "calibration_point_coverage": coverage,
        "calibration_peak_n": int(folds.peak_n.sum()),
        "calibration_peak_breaches": int(folds.peak_breaches.sum()),
        "calibration_horizons_without_breach": protected_windows,
        "calibration_horizons": len(folds), "warmup_folds": policy["warmup_folds"],
        "sequential_prior_fold_checks": json.loads(sequential.to_json(orient="records", double_precision=12)),
        "final_holdout_predictions_generated": False, "holdout_labels_used": False,
    }
    (output / "headroom_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = f"""# Frozen forecasting method and operational headroom policy

Decision date: **{frozen_on}**. Method state: **frozen before final holdout
prediction or evaluation**. The machine-readable record is
`config/frozen_method.json`, accompanied by its SHA-256 digest.

## Forecasting decision

- CPU: an unweighted mean of each VM's interval `avg_cpu`, followed by a
  **12-observation (60-minute) moving-average forecast**. Average the last
  12 training observations once and repeat it for all 24 forecast leads.
- Active VMs: **persistence**, using the last training count, with no hard-coded
  population size. A constant selected cohort does not calibrate churn risk.
- Training: the first 110 timestamps, {int(train.timestamp.iloc[0])} through
  {int(train.timestamp.iloc[-1])} source-relative seconds. The final 24
  timestamps are the two-hour forecast template; their labels remain excluded.
- Keep the five-minute cadence and existing fail-on-defect preprocessing.
  No holdout updates, metadata CPU summaries, lifetime-based features,
  resampling, further model search, or tuning after final test evaluation.
- Preserve point predictions in the required submission columns. Headroom is
  an operational allowance; it must not inflate `aggregate_cpu_utilization`
  in the eventual `predictions.csv`.

The 60-minute average has the lowest pooled development MAE (0.1784 percentage
points), versus 0.1817 for AR(1). AR(1) has lower RMSE but worse peak MAE.
This decision uses the previously stated MAE objective and acknowledges those
trade-offs, rather than claiming dominance on every metric. Only three
development origins exist; small differences are not strong generalization
evidence. See `model_comparison.md` for the full comparison.

## The headroom rule

Let `f[t]` be the CPU point forecast. Define underprediction error as
`actual[t] - f[t]`. From the 72 nonoverlapping development-validation errors
for the selected method, compute:

```text
margin = max(0, empirical_quantile(errors, 0.95, method="higher"))
planning_ceiling[t] = min(100, f[t] + margin)
```

The frozen margin is **{policy['margin_pp']:.8f} percentage points**. Use the
full-precision configuration value when computing ceilings. This is an
additive percentage-point allowance, not a relative percent increase or a
claim about required cores. It is an empirical one-sided planning guardrail,
not a calibrated 95% prediction interval or a 95% guarantee for an entire
two-hour horizon. No future holdout error determines this margin.

The 95th-percentile choice expresses a preference to cover most historical
underprediction while avoiding reserving the largest observed error at every
timestamp. It was fixed before examining policy results; no margin sweep was
used to optimize reported coverage. Under-provisioning risk remains in the
tail. Reserving more allowance would increase unused capacity budget; actual
financial/SLO costs are unavailable, so this is not an optimized cost policy.

## Operational action

1. Plan for the ceiling over the next two hours, retaining current capacity.
   If the team independently supplies a sustainable utilization budget `B`
   for the same cohort and aggregation, initiate a capacity review when
   `max(planning_ceiling) >= B`. No budget value is inferred from this trace.
2. Compare arriving five-minute aggregate readings to the ceilings issued at
   their forecast origin. **Two consecutive readings strictly above their
   ceilings trigger a workload/host-level capacity review.** Emit one alert
   on the second breach, retain the review state during that run, and reset
   when a reading is at or below its ceiling. These are interval averages,
   not proof of continuous saturation. A single breach remains a diagnostic.
3. Inspect loaded VMs and workload segments before allocating capacity. The
   low fleet average masks heavily loaded VMs and does not establish spare
   capacity at a particular host. Do not automatically downscale from this
   aggregate signal. Actual resource changes need core-weighted usage,
   placement constraints, service SLOs, and provisioning lead times.
4. In a deployed monitoring workflow, refresh forecasts every five minutes
   using newly available readings. The frozen challenge evaluation remains
   a single fixed-origin two-hour forecast, with no within-horizon updates.
   Keep the margin fixed for the final evaluation; recalibration belongs to
   a separately versioned future policy, not post-test adjustment.

The policy produces an advisory ceiling and review triggers, not a hardware
capacity estimate or automatic scaling controller. The dataset contains no
host topology, exact bucketed hardware capacities, SLA, or scaling latency
that could justify a numeric cores/VM scaling recommendation. A zero VM-count
validation error likewise does not justify a zero production churn reserve.

## Reused calibration diagnostics

{markdown_table(folds)}

Using the same residuals to set and assess the final margin covers
**{int(folds.covered.sum())}/{int(folds.n.sum())} observations ({coverage:.2%})**.
It covers all timestamps in **{protected_windows}/{len(folds)} two-hour
windows**. There are {int(folds.peak_breaches.sum())} breaches among
{int(folds.peak_n.sum())} peak observations. This is an in-sample diagnostic,
not an independent accuracy or coverage estimate. The all-window count makes
clear why pointwise coverage cannot be read as horizon-wide coverage.

`mean_unused_allowance_pp` is the mean positive ceiling-minus-actual gap in
percentage points. It is a descriptive utilization-budget proxy, not measured
over-provisioning cost. `max_exceedance_pp` reports remaining demand above
the ceiling. No costs or capacity conversions are manufactured.

## Prior-fold-only chronological check

The first validation fold is calibration warmup. For each later fold, use
only residuals whose observed timestamps are at or before its forecast
origin. Freeze that earlier-data margin throughout the next two-hour window.

{markdown_table(sequential)}

These checks avoid looking forward when calibrating each margin. They still
reuse development data for a model chosen on those folds, and cover only two
later origins. Later low-load periods are not a representative sample of
all peak risks. Their observed coverage establishes no calibrated probability
or deployment guarantee. The final margin uses all 72 development residuals
because all are available before the final forecast origin.

## Freeze record and next stage

The JSON records methods, parameters, split, margin, policy actions, dependency
versions, a training-target fingerprint, and relevant source-code hashes.
The digest detects accidental configuration edits. Source hashes normalize
line endings so clean Windows/Linux checkouts agree. Verification checks
both the configuration digest and the frozen sources; the freeze command
refuses to replace an existing differing record. This is a local reproducible
record, not an externally timestamped or signed attestation.

Reproduce or verify this freeze with:

```bash
python -m cwfc.freeze --data challenge_data --output reports --method config/frozen_method.json --frozen-on {frozen_on}
```

Final holdout predictions have **not** been generated by this stage. The next
stage must load and verify this record, apply the fixed methods to the public
training window, write the exact 24-row submission, and save a separate
operational plan if needed. Final holdout scoring must not change the frozen
model, headroom margin, or selection decisions.
"""
    (output / "method_freeze.md").write_text(report, encoding="utf-8")
    return method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("challenge_data"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--method", type=Path, default=Path("config/frozen_method.json"))
    parser.add_argument("--frozen-on", required=True, help="Decision date, YYYY-MM-DD")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    method = (load_frozen_method(args.method) if args.verify_only else
              freeze_method(args.data, args.output, args.method, args.frozen_on))
    print(json.dumps({"status": method["status"], "frozen_on": method["frozen_on"],
                      "cpu_method": method["targets"]["aggregate_cpu_utilization"],
                      "margin_percentage_points": method["headroom_policy"]["margin_percentage_points"],
                      "holdout_predictions_generated": False}, indent=2))


if __name__ == "__main__":
    main()
