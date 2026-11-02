import numpy as np
import pandas as pd
import pytest

from cwfc.data import build_targets, split_targets, validate_readings


@pytest.fixture
def readings():
    return pd.DataFrame([
        {"timestamp": step * 300, "vm_id": vm, "min_cpu": 0.,
         "avg_cpu": float(step + offset), "max_cpu": 100.}
        for step in range(6)
        for vm, offset in [("small", 10), ("large", 30)]
    ])


def test_targets_are_sorted_unweighted_vm_means(readings):
    targets = build_targets(readings.sample(frac=1, random_state=2))
    assert targets.timestamp.tolist() == [0, 300, 600, 900, 1200, 1500]
    assert targets.active_vm_count.tolist() == [2] * 6
    assert targets.aggregate_cpu_utilization.tolist() == [20., 21., 22., 23., 24., 25.]


@pytest.mark.parametrize("defect", ["duplicate", "missing", "infinite", "range", "order", "gap", "coverage"])
def test_source_defects_require_explicit_preprocessing(readings, defect):
    if defect == "duplicate":
        readings = pd.concat([readings, readings.iloc[[0]]], ignore_index=True)
    elif defect == "missing":
        readings.loc[0, "avg_cpu"] = np.nan
    elif defect == "infinite":
        readings.loc[0, "max_cpu"] = np.inf
    elif defect == "range":
        readings.loc[0, "max_cpu"] = 101
    elif defect == "order":
        readings.loc[0, "min_cpu"] = 20
    elif defect == "gap":
        readings = readings[readings.timestamp != 600]
    elif defect == "coverage":
        readings = readings.iloc[1:]
    with pytest.raises(ValueError):
        validate_readings(readings)


def test_split_returns_no_holdout_labels(readings):
    targets = build_targets(readings)
    train, template = split_targets(targets, train_size=4, horizon=2)
    assert train.timestamp.tolist() == [0, 300, 600, 900]
    assert template.columns.tolist() == ["timestamp"]
    assert template.timestamp.tolist() == [1200, 1500]
    with pytest.raises(ValueError, match="Target length"):
        split_targets(targets)


def test_holdout_cpu_changes_cannot_change_training_analysis(readings, tmp_path, monkeypatch):
    from cwfc import audit

    # Test the analysis boundary, without spending time rendering plots here.
    monkeypatch.setattr(audit, "save_plots", lambda *args: None)
    metadata = pd.DataFrame([
        {"vm_id": vm, "subscription_id": "s", "deployment_id": "d",
         "timestamp_vm_created": 0, "timestamp_vm_deleted": 9000,
         "max_cpu": 100., "avg_cpu": 50., "p95_max_cpu": 90.,
         "vm_category": category, "virtual_core_bucket": core,
         "memory_gb_bucket": 4}
        for vm, category, core in [("small", "Interactive", 2), ("large", "Unknown", 8)]
    ])
    results = []
    outputs = []
    for variant in range(2):
        data = tmp_path / f"data{variant}"
        output = tmp_path / f"report{variant}"
        data.mkdir()
        changed = readings.copy()
        if variant:
            changed.loc[changed.timestamp >= 1200, "avg_cpu"] = 99.
        changed.to_csv(data / "vm_cpu_readings.csv.gz", index=False)
        metadata.to_csv(data / "vm_metadata.csv.gz", index=False)
        results.append(audit.run_audit(data, output, train_size=4, horizon=2))
        outputs.append(output)
    assert results[0] == results[1]
    for filename in ["training_targets.csv", "training_segments.csv", "holdout_timestamps.csv", "data_audit.md"]:
        assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()
