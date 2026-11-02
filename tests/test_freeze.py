import json

import pandas as pd
import pytest

from cwfc import freeze


@pytest.fixture
def source(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    readings = pd.DataFrame([
        {"timestamp": step * 300, "vm_id": vm, "min_cpu": 0.,
         "avg_cpu": float(10 + (step % 9) * .5 + offset), "max_cpu": 100.}
        for step in range(134) for vm, offset in [("a", 0), ("b", 5)]
    ])
    readings.to_csv(data / "vm_cpu_readings.csv.gz", index=False)
    return data, readings


def test_freeze_is_verifiable_idempotent_and_does_not_generate_final_predictions(source, tmp_path):
    data, _ = source
    path = tmp_path / "config" / "frozen_method.json"
    output = tmp_path / "reports"
    method = freeze.freeze_method(data, output, path, "2026-11-02")
    assert freeze.load_frozen_method(path) == method
    before = path.read_bytes()
    assert freeze.freeze_method(data, output, path, "2026-11-02") == method
    assert path.read_bytes() == before
    assert method["targets"]["aggregate_cpu_utilization"]["window"] == 12
    assert method["headroom_policy"]["calibration_residuals"] == 72
    assert method["holdout_labels_used"] is False
    assert not list(tmp_path.rglob("predictions.csv"))
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["headroom_policy"]["margin_percentage_points"] += .1
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        freeze.load_frozen_method(path)


def test_frozen_method_refuses_changed_training_and_changed_code(source, tmp_path, monkeypatch):
    data, readings = source
    path = tmp_path / "frozen_method.json"
    freeze.freeze_method(data, tmp_path / "reports", path, "2026-11-02")
    before = path.read_bytes()
    readings.loc[0, "avg_cpu"] += 1
    readings.to_csv(data / "vm_cpu_readings.csv.gz", index=False)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        freeze.freeze_method(data, tmp_path / "reports", path, "2026-11-02")
    assert path.read_bytes() == before
    monkeypatch.setattr(freeze, "source_fingerprints", lambda: {"changed": "source"})
    with pytest.raises(ValueError, match="Source code differs"):
        freeze.load_frozen_method(path)


def test_holdout_values_cannot_change_method_or_margin(source, tmp_path):
    data, readings = source
    original_path = tmp_path / "original.json"
    original = freeze.freeze_method(data, tmp_path / "original_report", original_path, "2026-11-02")
    readings.loc[readings.timestamp >= 33000, "avg_cpu"] = 99.
    readings.to_csv(data / "vm_cpu_readings.csv.gz", index=False)
    changed_path = tmp_path / "changed.json"
    changed = freeze.freeze_method(data, tmp_path / "changed_report", changed_path, "2026-11-02")
    assert original == changed
    assert original_path.read_bytes() == changed_path.read_bytes()
    for report in (tmp_path / "original_report").iterdir():
        assert report.read_bytes() == (tmp_path / "changed_report" / report.name).read_bytes()
