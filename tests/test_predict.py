from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from cwfc.freeze import load_frozen_method
from cwfc.predict import DEFAULT_METHOD, forecast_from_training, run_prediction, validate_predictions


def write_source(path: Path, timestamps: int = 134, vms: int = 2):
    path.mkdir()
    readings = pd.DataFrame([
        {"timestamp": step * 300, "vm_id": f"vm{vm}", "min_cpu": 0.,
         "avg_cpu": 10 + step * .2 + vm * 2, "max_cpu": 100.}
        for step in range(timestamps) for vm in range(vms)
    ])
    readings.to_csv(path / "vm_cpu_readings.csv.gz", index=False)
    return readings


def test_frozen_forecast_matches_known_training_values_with_nonzero_template_index():
    method = load_frozen_method(DEFAULT_METHOD)
    train = pd.DataFrame({"timestamp": np.arange(20) * 300,
                          "active_vm_count": 3, "aggregate_cpu_utilization": 10. + np.arange(20)})
    template = pd.DataFrame({"timestamp": np.arange(20, 24) * 300}, index=[20, 21, 22, 23])
    result = forecast_from_training(train, template, method)
    assert result.active_vm_count.tolist() == [3.] * 4
    assert result.aggregate_cpu_utilization.tolist() == [23.5] * 4
    assert result.timestamp.tolist() == [6000, 6300, 6600, 6900]
    with pytest.raises(ValueError, match="timestamps only"):
        forecast_from_training(train, template.assign(aggregate_cpu_utilization=99), method)


@pytest.mark.parametrize("defect", ["extra_column", "row_count", "duplicate", "wrong_timestamp",
                                    "missing", "negative_count", "cpu_range"])
def test_submission_validation_rejects_contract_violations(defect):
    frame = pd.DataFrame({"timestamp": [6000, 6300], "active_vm_count": [3., 3.],
                          "aggregate_cpu_utilization": [20., 20.]})
    if defect == "extra_column":
        frame["index"] = [0, 1]
    elif defect == "row_count":
        frame = frame.iloc[:1]
    elif defect == "duplicate":
        frame.loc[1, "timestamp"] = 6000
    elif defect == "wrong_timestamp":
        frame.loc[1, "timestamp"] = 6600
    elif defect == "missing":
        frame.loc[0, "aggregate_cpu_utilization"] = np.nan
    elif defect == "negative_count":
        frame.loc[0, "active_vm_count"] = -1
    elif defect == "cpu_range":
        frame.loc[0, "aggregate_cpu_utilization"] = 101
    with pytest.raises(ValueError):
        validate_predictions(frame, [6000, 6300])


def test_default_command_supports_a_small_vm_population_without_fixed_forecast_values(tmp_path):
    source = tmp_path / "data"
    write_source(source, vms=3)
    output = tmp_path / "predictions.csv"
    reports = tmp_path / "reports"
    frozen_before = DEFAULT_METHOD.read_bytes()
    result = run_prediction(source, output, reports)
    predictions = pd.read_csv(output)
    assert len(predictions) == 24
    assert predictions.timestamp.tolist() == list(range(33000, 40200, 300))
    assert predictions.active_vm_count.eq(3).all()
    np.testing.assert_allclose(predictions.aggregate_cpu_utilization, 32.7)
    assert result["matches_frozen_training"] is False
    assert result["matches_frozen_split"] is True
    assert result["headroom_calibration_matches_input_training"] is False
    plan = pd.read_csv(reports / "operational_plan.csv")
    margin = load_frozen_method(DEFAULT_METHOD)["headroom_policy"]["margin_percentage_points"]
    np.testing.assert_allclose(plan.cpu_planning_ceiling_pct, 32.7 + margin)
    assert DEFAULT_METHOD.read_bytes() == frozen_before


def test_short_smoke_trace_keeps_the_same_window_and_can_be_reproduced(tmp_path):
    source = tmp_path / "data"
    write_source(source, timestamps=28)
    output, reports = tmp_path / "predictions.csv", tmp_path / "reports"
    summary = run_prediction(source, output, reports, horizon=4)
    before = {p.name: p.read_bytes() for p in reports.iterdir()}
    prediction_bytes = output.read_bytes()
    again = run_prediction(source, output, reports, horizon=4)
    assert summary == again
    assert summary["cpu_window"] == 12
    assert summary["training_observations"] == 24
    np.testing.assert_allclose(pd.read_csv(output).aggregate_cpu_utilization, 14.5)
    assert output.read_bytes() == prediction_bytes
    assert before == {p.name: p.read_bytes() for p in reports.iterdir()}


def test_holdout_changes_cannot_change_point_predictions_or_operational_plan(tmp_path):
    source = tmp_path / "data"
    readings = write_source(source)
    output, reports = tmp_path / "predictions.csv", tmp_path / "reports"
    first = run_prediction(source, output, reports)
    original, plan = output.read_bytes(), (reports / "operational_plan.csv").read_bytes()
    readings.loc[readings.timestamp >= 33000, "avg_cpu"] = 99.
    readings.to_csv(source / "vm_cpu_readings.csv.gz", index=False)
    changed = run_prediction(source, output, reports)
    assert output.read_bytes() == original
    assert (reports / "operational_plan.csv").read_bytes() == plan
    assert first["training_targets_sha256"] == changed["training_targets_sha256"]
    assert first["source_cpu_sha256"] != changed["source_cpu_sha256"]
    assert changed["holdout_labels_used_for_forecasting"] is False
    assert changed["holdout_scored"] is False


def test_configuration_tampering_blocks_prediction_before_writing(tmp_path):
    source = tmp_path / "data"
    write_source(source)
    method = tmp_path / "method.json"
    shutil.copyfile(DEFAULT_METHOD, method)
    shutil.copyfile(DEFAULT_METHOD.with_suffix(".json.sha256"), method.with_suffix(".json.sha256"))
    method.write_text(method.read_text().replace('"window": 12', '"window": 24'))
    output = tmp_path / "predictions.csv"
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_prediction(source, output, tmp_path / "reports", method_path=method)
    assert not output.exists()
