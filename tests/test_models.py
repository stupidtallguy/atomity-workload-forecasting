import numpy as np
import pandas as pd
import pytest

from cwfc import compare
from cwfc.backtest import evaluate_baselines
from cwfc.models import (
    CPU_MODEL_CONFIG, autoregressive_one, damped_trend,
    exponential_smoothing, forecast_cpu,
)


def training_frame():
    return pd.DataFrame({
        "timestamp": np.arange(110) * 300,
        "active_vm_count": 2,
        "aggregate_cpu_utilization": 20 + np.sin(np.arange(110) / 8),
    })


def test_exponential_smoothing_matches_hand_calculation():
    np.testing.assert_allclose(exponential_smoothing([10, 20, 30], 3, alpha=.5), [22.5] * 3)


def test_local_trend_is_damped_at_each_future_lead():
    result = damped_trend(np.arange(1, 13), 3, window=12, damping=.5)
    np.testing.assert_allclose(result, [12.5, 12.75, 12.875])


def test_ar1_recursively_forecasts_known_process():
    result = autoregressive_one([2, 6, 8, 9, 9.5], 3)
    np.testing.assert_allclose(result, [9.75, 9.875, 9.9375])


@pytest.mark.parametrize("model", CPU_MODEL_CONFIG)
def test_all_candidates_handle_constant_histories(model):
    np.testing.assert_allclose(forecast_cpu(np.repeat(7., 38), 24, model), np.repeat(7., 24))


def test_ar1_constraint_prevents_explosive_extrapolation():
    result = autoregressive_one(np.geomspace(1, 80, 38), 240)
    assert np.isfinite(result).all()
    assert ((result >= 0) & (result <= 100)).all()
    assert abs(result[-1] - result[-2]) < 1e-3


def test_trend_obeys_physical_cpu_bounds():
    np.testing.assert_allclose(damped_trend(np.linspace(0, 100, 12), 24), np.repeat(100., 24))
    np.testing.assert_allclose(damped_trend(np.linspace(100, 0, 12), 24), np.zeros(24))


@pytest.mark.parametrize("values", [[], [1, np.nan], [1, np.inf], [-1, 2], [1, 101]])
def test_invalid_history_is_rejected(values):
    with pytest.raises(ValueError):
        exponential_smoothing(values, 24)


def test_comparison_reproduces_all_original_baseline_results():
    train = training_frame()
    baseline = evaluate_baselines(train)
    combined = compare.evaluate_cpu_models(train)
    for name in baseline:
        frame = combined[name]
        if "model" in frame:
            frame = frame[~frame.model.isin(CPU_MODEL_CONFIG)].reset_index(drop=True)
        pd.testing.assert_frame_equal(baseline[name], frame)


def test_candidate_predictions_and_peak_cutoffs_ignore_their_validation_labels():
    train = training_frame()
    original = compare.evaluate_cpu_models(train)["predictions"].query("fold == 1")
    train.loc[38:61, "aggregate_cpu_utilization"] = 99.
    changed = compare.evaluate_cpu_models(train)["predictions"].query("fold == 1")
    columns = ["model", "target", "origin_timestamp", "timestamp", "lead",
               "prediction", "training_mean_scale", "peak_threshold"]
    pd.testing.assert_frame_equal(original[columns], changed[columns])
    candidates = original[original.model.isin(CPU_MODEL_CONFIG)]
    assert set(candidates.target) == {"aggregate_cpu_utilization"}
    assert candidates.groupby("model").size().eq(24).all()


def test_raw_holdout_changes_cannot_change_comparison_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "plot_comparison", lambda *args: None)
    readings = pd.DataFrame([
        {"timestamp": step * 300, "vm_id": vm, "min_cpu": 0.,
         "avg_cpu": float(10 + step + offset), "max_cpu": 100.}
        for step in range(28) for vm, offset in [("a", 0), ("b", 5)]
    ])
    outputs, summaries = [], []
    for variant in range(2):
        data, output = tmp_path / f"data{variant}", tmp_path / f"out{variant}"
        data.mkdir()
        changed = readings.copy()
        if variant:
            changed.loc[changed.timestamp >= 24 * 300, "avg_cpu"] = 99.
        changed.to_csv(data / "vm_cpu_readings.csv.gz", index=False)
        summaries.append(compare.run_comparison(data, output, train_size=24, horizon=4,
                                                 min_train_size=16, windows=(3, 6, 12)))
        outputs.append(output)
    assert summaries[0] == summaries[1]
    for original in outputs[0].iterdir():
        assert original.read_bytes() == (outputs[1] / original.name).read_bytes()
