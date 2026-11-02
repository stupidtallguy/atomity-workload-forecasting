"""Reproduce the structural audit and exploratory analysis of training data.

Run: python -m cwfc.audit --data challenge_data --output reports
Full-release values are checked for integrity only. CPU summaries, segment
comparisons, and plots use the training window exclusively.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.data import build_targets, split_targets, validate_dataset


def markdown_table(frame: pd.DataFrame) -> str:
    """Small Markdown renderer without an optional tabulate dependency."""
    def cell(value):
        return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def save_plots(train: pd.DataFrame, rows: pd.DataFrame, segment_series: pd.DataFrame,
               output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    hours = (train.timestamp - train.timestamp.iloc[0]) / 3600
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    axes[0].plot(hours, train.aggregate_cpu_utilization, color="#176b87", linewidth=1.8)
    axes[0].set_ylabel("Mean CPU (%)")
    axes[0].set_title("Training window only | holdout CPU values excluded", loc="left")
    axes[1].plot(hours, train.active_vm_count, color="#52734d")
    axes[1].set_ylabel("Observed VMs")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].set_xlabel("Hours since first source timestamp")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output / "training_targets.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].hist(rows.avg_cpu, bins=np.linspace(0, 100, 51), color="#176b87")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("VM interval average CPU (%)")
    axes[0].set_ylabel("Training observations (log scale)")
    for category, group in segment_series.groupby("vm_category", sort=True):
        axes[1].plot((group.timestamp - train.timestamp.iloc[0]) / 3600,
                     group.mean_cpu_pct, label=category, linewidth=1.5)
    axes[1].set_xlabel("Hours since first source timestamp")
    axes[1].set_ylabel("Segment mean CPU (%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("CPU distribution and workload categories | training window only")
    fig.savefig(output / "training_workloads.png")
    plt.close(fig)


def run_audit(data: Path, output: Path, train_size: int = 110, horizon: int = 24) -> dict:
    if train_size < 2:
        raise ValueError("At least two training timestamps are needed for this analysis")
    readings = pd.read_csv(data / "vm_cpu_readings.csv.gz")
    metadata = pd.read_csv(data / "vm_metadata.csv.gz")
    quality = validate_dataset(readings, metadata)
    targets = build_targets(readings)
    train, template = split_targets(targets, train_size, horizon)
    # All value-based EDA starts after this explicit chronological boundary.
    rows = readings[readings.timestamp.isin(train.timestamp)].copy()
    static_columns = ["vm_id", "vm_category", "virtual_core_bucket", "memory_gb_bucket"]
    rows = rows.merge(metadata[static_columns], on="vm_id", validate="many_to_one")
    cpu = train.aggregate_cpu_utilization
    vm_cpu = rows.groupby("vm_id").avg_cpu.mean()
    category_series = rows.groupby(["timestamp", "vm_category"], observed=True).agg(
        mean_cpu_pct=("avg_cpu", "mean"), vm_count=("vm_id", "nunique"),
    ).reset_index()
    segments = []
    for column in static_columns[1:]:
        summary = rows.groupby(column, observed=True).agg(
            vm_count=("vm_id", "nunique"), mean_cpu_pct=("avg_cpu", "mean"),
        ).reset_index().rename(columns={column: "segment"})
        summary.insert(0, "dimension", column)
        segments.append(summary)
    segments = pd.concat(segments, ignore_index=True)
    categories = segments[segments.dimension == "vm_category"].sort_values("mean_cpu_pct")
    lowest_category, highest_category = categories.iloc[0], categories.iloc[-1]
    quantiles = rows.avg_cpu.quantile([0, .25, .5, .75, .9, .95, .99, 1])
    differences = cpu.diff().dropna()
    first_ts, last_ts = int(train.timestamp.iloc[0]), int(train.timestamp.iloc[-1])
    lifetimes = {
        "created_before_window": int((metadata.timestamp_vm_created < first_ts).sum()),
        "created_at_window_start": int((metadata.timestamp_vm_created == first_ts).sum()),
        "created_during_training": int(metadata.timestamp_vm_created.between(
            first_ts + 1, last_ts).sum()),
        "deleted_during_training": int(metadata.timestamp_vm_deleted.between(
            first_ts, last_ts).sum()),
    }
    summary = {
        "quality": quality,
        "split": {"training_observations": len(train), "holdout_observations": len(template),
                  "training_first_timestamp": first_ts, "training_last_timestamp": last_ts,
                  "holdout_first_timestamp": int(template.timestamp.iloc[0]),
                  "holdout_last_timestamp": int(template.timestamp.iloc[-1])},
        "training": {
            "cpu_mean_pct": float(cpu.mean()), "cpu_min_pct": float(cpu.min()),
            "cpu_max_pct": float(cpu.max()), "cpu_std_percentage_points": float(cpu.std()),
            "cpu_first_pct": float(cpu.iloc[0]), "cpu_last_pct": float(cpu.iloc[-1]),
            "lag1_autocorrelation": float(cpu.autocorr(1)) if cpu.nunique() > 1 else None,
            "largest_absolute_5min_change_pp": float(differences.abs().max()),
            "constant_active_vm_count": bool(train.active_vm_count.nunique() == 1),
            "zero_cpu_observation_fraction": float(rows.avg_cpu.eq(0).mean()),
            "observations_at_least_80pct_fraction": float(rows.avg_cpu.ge(80).mean()),
            "vm_mean_cpu_below_5pct_fraction": float(vm_cpu.lt(5).mean()),
            "vm_mean_cpu_at_least_80pct_fraction": float(vm_cpu.ge(80).mean()),
            "cpu_observation_quantiles_pct": {str(q): float(v) for q, v in quantiles.items()},
        },
        "metadata_lifetime_checks": lifetimes,
        "holdout_target_values_exported": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "training_targets.csv", index=False)
    template.to_csv(output / "holdout_timestamps.csv", index=False)
    segments.to_csv(output / "training_segments.csv", index=False)
    (output / "data_audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    save_plots(train, rows, category_series, output)
    peaks = train.nlargest(5, "aggregate_cpu_utilization")
    lows = train.nsmallest(5, "aggregate_cpu_utilization")
    quantile_table = quantiles.rename_axis("quantile").reset_index(name="CPU (%)")
    training = summary["training"]
    report = f"""# Data audit and initial workload analysis

Reproduce with `python -m cwfc.audit --data challenge_data --output reports`.
This is an initial analysis, not the final submission report. No forecasting
model has been fitted or selected, and holdout CPU targets are not summarized,
plotted, or exported. Full-release checks concern structural integrity only.

## Source integrity and preprocessing

- CPU observations: {quality['cpu_rows']:,}; VMs: {quality['vm_count']:,}.
- Metadata records: {quality['metadata_rows']:,}; VM populations match exactly.
- Shared timestamp grid: {quality['timestamps']} observations, from
  {quality['first_timestamp']} to {quality['last_timestamp']} source-relative seconds,
  at exactly {quality['cadence_seconds']}-second cadence.
- Each VM has {quality['observations_per_vm']} observations on that shared grid.
- No missing values, duplicate VM/timestamp keys, duplicate metadata VM IDs,
  missing grid positions, nonfinite CPU readings, out-of-range CPU percentages,
  or violations of `min_cpu <= avg_cpu <= max_cpu`.
- No rows were filled, removed, deduplicated, or resampled. The implementation
  fails on such defects so any future repair must be deliberate and documented.
- Timestamps remain numeric seconds. There is no asserted real calendar date,
  weekday, timezone, or hour of day.

## Targets and the protected holdout

The full target series is constructed first by counting distinct VM IDs and
taking an unweighted mean of their `avg_cpu` values at each timestamp. The CPU
target is already expressed in percent; it is not core-weighted or total CPU
capacity consumed. No core-count approximation is used.

Training uses {len(train)} observations from {first_ts} to {last_ts} seconds.
The {len(template)} holdout timestamps run from {int(template.timestamp.iloc[0])}
to {int(template.timestamp.iloc[-1])} seconds. Only their timestamps are exported.
The forecast horizon is {horizon} steps, or
{horizon * quality['cadence_seconds'] / 60:g} minutes from the forecast origin.
The difference between its first and last timestamps is
{(horizon - 1) * quality['cadence_seconds'] / 60:g} minutes.

## Active VMs, lifetimes, and sampling limitations

Every timestamp has {quality['vm_count']:,} observed VMs. The active-count target
is therefore constant in this release. This follows the documented selection
of complete VM traces and does not demonstrate that the upstream cloud fleet
has no churn. We observe no VM arrivals or departures within the selected grid.
A persistence baseline is an appropriate candidate for this target; no
holdout performance claim is made here.

Metadata creation/deletion checks relative to the training window:

{markdown_table(pd.DataFrame(list(lifetimes.items()), columns=['check', 'VMs']))}

Creation times at zero may be boundary-censored; the data contract does not
establish their censoring semantics. Likewise, a recorded deletion time beyond
the observed window does not establish a complete observed lifetime. We do not
infer population-wide lifetime distributions from this selected cohort.

## Training-window CPU behavior

- Aggregate mean CPU: {cpu.mean():.4f}%; minimum: {cpu.min():.4f}%; maximum: {cpu.max():.4f}%.
- Aggregate standard deviation: {cpu.std():.4f} percentage points.
- First versus last training observation: {cpu.iloc[0]:.4f}% and {cpu.iloc[-1]:.4f}%.
- Largest absolute five-minute aggregate change: {differences.abs().max():.4f} percentage points.
- Lag-one autocorrelation: {training['lag1_autocorrelation']}.
- Zero CPU readings: {training['zero_cpu_observation_fraction']:.2%} of training observations.
- Readings at or above 80%: {training['observations_at_least_80pct_fraction']:.2%}.
- VMs with training mean below 5%: {training['vm_mean_cpu_below_5pct_fraction']:.2%}.
- VMs with training mean at least 80%: {training['vm_mean_cpu_at_least_80pct_fraction']:.2%}.

Distribution across VM-level training observations:

{markdown_table(quantile_table)}

![Training targets](training_targets.png)

### Highest training demand timestamps

{markdown_table(peaks)}

### Lowest training demand timestamps

{markdown_table(lows)}

These ranks identify peaks in the training window only. A narrow aggregate
CPU range can coexist with heavily loaded individual VMs; a fleet average
alone cannot determine which VMs can safely be downsized.

## Workload segments

All CPU summaries below are recomputed from training readings. The metadata
CPU fields `avg_cpu`, `max_cpu`, and `p95_max_cpu` are excluded because their
time coverage is not established. Core and memory buckets are used as labels,
not exact hardware capacities. Category and hardware labels are descriptive;
their availability at forecast origin would need justification before modeling.

{markdown_table(segments)}

The highest category mean is {highest_category['segment']} at
{highest_category['mean_cpu_pct']:.4f}%, versus {lowest_category['segment']} at
{lowest_category['mean_cpu_pct']:.4f}%. Their difference is
{highest_category['mean_cpu_pct'] - lowest_category['mean_cpu_pct']:.4f}
percentage points. This supports monitoring workload classes separately rather
than treating the fleet average as a typical individual workload.

![Training CPU distribution and categories](training_workloads.png)

Differences between segment means can be hidden by a fleet average. Segment
monitoring can help identify a loaded workload class even when overall demand
is stable. Category/core stratification and complete-trace selection mean this
cohort should not be assumed representative of Azure's full VM population.

## Consequences for modeling and operations

1. Start with persistence and recent moving-average baselines. The constant
   active-count target must be reported honestly; forecasting it does not
   establish general capability to forecast VM churn.
2. There are only {len(train)} training timestamps. Daily and weekly seasonality
   cannot be established from this window. Local changes and short-lag
   autocorrelation are descriptive, not proof of a recurring cycle.
3. Compare candidate CPU models using expanding training windows and a
   {horizon}-step validation horizon entirely inside the first {len(train)}
   observations. Choose the model before evaluating final holdout labels.
4. Keep deletion timestamps and source-wide CPU summaries out of forecasting
   features unless their information availability can be proved.
5. A capacity policy needs forecast-error evidence and explicit headroom.
   Unweighted average utilization is insufficient to calculate total required
   cores or justify a specific scaling threshold on its own.

No model accuracy, capacity threshold, or final operational policy is selected
by this audit. Those require the next validation and modeling stage.
"""
    (output / "data_audit.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("challenge_data"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--train-size", type=int, default=110)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()
    summary = run_audit(args.data, args.output, args.train_size, args.horizon)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
