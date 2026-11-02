"""Dataset checks and target construction without calendar-date assumptions."""
from __future__ import annotations

import numpy as np
import pandas as pd

CPU_COLUMNS = ["min_cpu", "avg_cpu", "max_cpu"]
READING_COLUMNS = ["timestamp", "vm_id", *CPU_COLUMNS]
METADATA_COLUMNS = [
    "vm_id", "subscription_id", "deployment_id", "timestamp_vm_created",
    "timestamp_vm_deleted", "max_cpu", "avg_cpu", "p95_max_cpu",
    "vm_category", "virtual_core_bucket", "memory_gb_bucket",
]


def validate_readings(readings: pd.DataFrame, cadence: int = 300) -> dict:
    """Reject ambiguous source data rather than silently filling or deduplicating.

    Missing/invalid CPU observations are not treated as zero utilization or an
    inactive VM. They require an explicit preprocessing decision before use.
    """
    missing = set(READING_COLUMNS) - set(readings.columns)
    if missing:
        raise ValueError(f"Missing reading columns: {sorted(missing)}")
    if readings.empty:
        raise ValueError("CPU readings must not be empty")
    if readings[READING_COLUMNS].isna().any().any():
        raise ValueError("Missing reading values: choose an explicit missing-data policy")
    timestamps = readings["timestamp"].to_numpy(dtype=float)
    if not np.isfinite(timestamps).all() or not (timestamps == np.floor(timestamps)).all():
        raise ValueError("Timestamps must be finite integer source-relative seconds")
    if readings["vm_id"].astype(str).str.strip().eq("").any():
        raise ValueError("VM IDs must not be empty")
    if readings.duplicated(["vm_id", "timestamp"]).any():
        raise ValueError("Duplicate (vm_id, timestamp) observations")
    cpu = readings[CPU_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(cpu).all() or ((cpu < 0) | (cpu > 100)).any():
        raise ValueError("CPU values must be finite percentages in [0, 100]")
    if ((cpu[:, 0] > cpu[:, 1]) | (cpu[:, 1] > cpu[:, 2])).any():
        raise ValueError("CPU observations must satisfy min_cpu <= avg_cpu <= max_cpu")
    grid = np.sort(readings["timestamp"].unique())
    if cadence <= 0 or not np.all(np.diff(grid) == cadence):
        raise ValueError("Timestamp gaps or irregular cadence; no automatic imputation")
    per_vm = readings.groupby("vm_id", sort=False).size()
    if not per_vm.eq(len(grid)).all():
        raise ValueError("Incomplete per-VM coverage; no automatic imputation")
    return {
        "cpu_rows": len(readings), "vm_count": len(per_vm),
        "timestamps": len(grid), "first_timestamp": int(grid[0]),
        "last_timestamp": int(grid[-1]), "cadence_seconds": cadence,
        "observations_per_vm": int(per_vm.iloc[0]),
        "missing_values": 0, "duplicate_vm_timestamp_pairs": 0,
        "invalid_cpu_rows": 0, "missing_vm_timestamp_pairs": 0,
        "input_sorted_by_timestamp": bool(readings.timestamp.is_monotonic_increasing),
    }


def validate_dataset(readings: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    quality = validate_readings(readings)
    missing = set(METADATA_COLUMNS) - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if metadata[METADATA_COLUMNS].isna().any().any():
        raise ValueError("Missing metadata values")
    if metadata.vm_id.duplicated().any():
        raise ValueError("Duplicate metadata VM IDs")
    if set(readings.vm_id) != set(metadata.vm_id):
        raise ValueError("CPU and metadata VM populations differ")
    quality.update(metadata_rows=len(metadata), metadata_missing_values=0,
                   duplicate_metadata_vm_ids=0, matching_vm_populations=True)
    return quality


def build_targets(readings: pd.DataFrame) -> pd.DataFrame:
    """Use distinct observed VMs and an unweighted mean of valid avg_cpu values.

    Validate first: any missing/invalid observations fail rather than changing
    the population silently. Source-relative timestamps remain numeric seconds.
    """
    validate_readings(readings)
    return readings.groupby("timestamp", sort=True).agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
    ).reset_index()


def split_targets(
    targets: pd.DataFrame, train_size: int = 110, horizon: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return training targets and holdout timestamps only, never test labels."""
    if train_size <= 0 or horizon <= 0 or len(targets) != train_size + horizon:
        raise ValueError("Target length must equal positive train_size + horizon")
    if (not targets.timestamp.is_monotonic_increasing
            or targets.timestamp.duplicated().any()):
        raise ValueError("Target timestamps must be sorted and unique")
    return (targets.iloc[:train_size].copy(),
            targets.iloc[train_size:][["timestamp"]].copy())
