import json

import numpy as np
import pandas as pd
import pytest

from cwfc import evaluate
from cwfc.freeze import load_frozen_method
from cwfc.policy import planning_ceiling
from cwfc.predict import DEFAULT_METHOD, forecast_from_training, run_prediction


def known_example():
    method = load_frozen_method(DEFAULT_METHOD)
    train = pd.DataFrame({"timestamp": np.arange(110) * 300, "active_vm_count": 2,
                          "aggregate_cpu_utilization": 11 + np.arange(110) * .2})
    actual = pd.DataFrame({"timestamp": np.arange(110, 134) * 300, "active_vm_count": 2,
                           "aggregate_cpu_utilization": 11 + np.arange(110, 134) * .2})
    predictions = forecast_from_training(train, actual[["timestamp"]], method)
    margin = method["headroom_policy"]["margin_percentage_points"]
    plan = pd.DataFrame({
        "timestamp": predictions.timestamp, "active_vm_count_forecast": predictions.active_vm_count,
        "cpu_point_forecast_pct": predictions.aggregate_cpu_utilization,
        "cpu_planning_ceiling_pct": planning_ceiling(predictions.aggregate_cpu_utilization, margin),
        "headroom_percentage_points": margin, "consecutive_breaches_for_review": 2,
    })
    return method, train, actual, predictions, plan


def test_holdout_metrics_match_known_errors_and_reveal_policy_failure():
    method, train, actual, predictions, plan = known_example()
    results = evaluate.evaluate_saved_forecasts(train, actual, predictions, plan, method)
    cpu = results["metrics"].query("target == 'aggregate_cpu_utilization' and model == 'frozen_submission'").iloc[0]
    assert cpu.mae == pytest.approx(3.6)
    assert cpu.bias == pytest.approx(-3.6)
    assert cpu.rmse == pytest.approx(np.sqrt(14.876666666666667))
    assert cpu.nmae_pct_training_mean == pytest.approx(100 * 3.6 / 21.9)
    assert cpu.mae_improvement_pct == pytest.approx(-44)
    assert cpu.peak_n == 24
    assert cpu.peak_mae == pytest.approx(3.6)
    assert results["policy_metrics"]["breaches"] == 24
    assert results["policy_metrics"]["point_coverage"] == 0
    assert results["policy_metrics"]["entire_horizon_covered"] is False
    assert results["policy_metrics"]["alert_events"] == 1
    assert results["policy_metrics"]["alert_timestamps"] == [33300]
    count = results["metrics"].query("target == 'active_vm_count'").iloc[0]
    assert count.mae == 0
    assert np.isnan(count.mae_improvement_pct)


def test_changed_holdout_labels_change_scores_without_changing_forecasts():
    method, train, actual, predictions, plan = known_example()
    original_predictions, original_plan = predictions.copy(), plan.copy()
    before = evaluate.evaluate_saved_forecasts(train, actual, predictions, plan, method)
    actual["aggregate_cpu_utilization"] = 5.
    after = evaluate.evaluate_saved_forecasts(train, actual, predictions, plan, method)
    pd.testing.assert_frame_equal(predictions, original_predictions)
    pd.testing.assert_frame_equal(plan, original_plan)
    assert not before["metrics"].equals(after["metrics"])
    assert after["policy_metrics"]["point_coverage"] == 1
    assert before["policy_metrics"]["peak_threshold_pct"] == after["policy_metrics"]["peak_threshold_pct"]


def test_modified_policy_or_forecast_is_rejected():
    method, train, actual, predictions, plan = known_example()
    changed = plan.copy()
    changed["cpu_planning_ceiling_pct"] += .1
    changed["headroom_percentage_points"] += .1
    with pytest.raises(ValueError, match="frozen policy"):
        evaluate.evaluate_saved_forecasts(train, actual, predictions, changed, method)
    predictions["aggregate_cpu_utilization"] += .1
    with pytest.raises(ValueError, match="frozen training-only forecast"):
        evaluate.evaluate_saved_forecasts(train, actual, predictions, plan, method)


@pytest.fixture
def generated(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "plot_evaluation", lambda *args: None)
    data, generation_reports = tmp_path / "data", tmp_path / "generation"
    data.mkdir()
    readings = pd.DataFrame([
        {"timestamp": step * 300, "vm_id": vm, "min_cpu": 0.,
         "avg_cpu": 10 + step * .2 + offset, "max_cpu": 100.}
        for step in range(134) for vm, offset in [("a", 0), ("b", 2)]
    ])
    readings.to_csv(data / "vm_cpu_readings.csv.gz", index=False)
    predictions = tmp_path / "predictions.csv"
    run_prediction(data, predictions, generation_reports)
    return data, predictions, generation_reports


def test_evaluation_preserves_predictions_plan_freeze_and_generation_record(generated, tmp_path):
    data, predictions, generation_reports = generated
    protected = [predictions, generation_reports / "operational_plan.csv",
                 generation_reports / "prediction_run.json", DEFAULT_METHOD,
                 DEFAULT_METHOD.with_suffix(".json.sha256")]
    before = {str(path): path.read_bytes() for path in protected}
    output = tmp_path / "evaluation"
    summary = evaluate.run_evaluation(data, predictions, output, generation_reports=generation_reports)
    assert before == {str(path): path.read_bytes() for path in protected}
    assert summary["holdout_labels_used_for_scoring"] is True
    assert summary["parameters_or_margin_changed"] is False
    assert summary["protected_artifacts_unchanged"] is True
    generation = json.loads((generation_reports / "prediction_run.json").read_text())
    assert generation["holdout_scored"] is False  # Historical generation status is preserved.
    assert evaluate.run_evaluation(data, predictions, output, generation_reports=generation_reports) == summary


@pytest.mark.parametrize("artifact", ["predictions", "plan", "source"])
def test_pre_evaluation_hash_checks_reject_changed_inputs(generated, tmp_path, artifact):
    data, predictions, generation_reports = generated
    path = {"predictions": predictions, "plan": generation_reports / "operational_plan.csv",
            "source": data / "vm_cpu_readings.csv.gz"}[artifact]
    with path.open("ab") as handle:
        handle.write(b"changed")
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError):
        evaluate.run_evaluation(data, predictions, output, generation_reports=generation_reports)
    assert not (output / "holdout_evaluation.json").exists()
