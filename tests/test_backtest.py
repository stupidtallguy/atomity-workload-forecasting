import numpy as np
import pandas as pd
import pytest

from cwfc import backtest
from cwfc.baselines import moving_average_forecast


def training_frame(n=110):
    return pd.DataFrame({
        "timestamp": np.arange(n) * 300,
        "active_vm_count": np.repeat(2, n),
        "aggregate_cpu_utilization": 20 + np.sin(np.arange(n) / 8),
    })


def test_moving_average_uses_observed_history_and_ignores_template_values():
    history = pd.DataFrame({"cpu": [10., 20., 30.]})
    template = pd.DataFrame({"cpu": [99., 98.]}, index=[6, 7])
    result = moving_average_forecast(history, template, "cpu", window=2)
    assert result.index.tolist() == [6, 7]
    assert result.tolist() == [25., 25.]
    with pytest.raises(ValueError, match="not enough"):
        moving_average_forecast(history, template, "cpu", window=4)


def test_pooled_errors_match_known_multistep_forecasts():
    frame = training_frame(10)
    frame["aggregate_cpu_utilization"] = np.arange(10) + 20.
    results = backtest.evaluate_baselines(frame, horizon=3, min_train_size=4, windows=(2,))
    cpu = results["metrics"].query("target == 'aggregate_cpu_utilization'").set_index("model")
    assert cpu.loc["persistence", "mae"] == pytest.approx(2.)
    assert cpu.loc["persistence", "rmse"] == pytest.approx(np.sqrt(14 / 3))
    assert cpu.loc["persistence", "bias"] == pytest.approx(-2.)
    assert cpu.loc["moving_average_2", "mae"] == pytest.approx(2.5)
    assert cpu.loc["moving_average_2", "mae_improvement_pct"] == pytest.approx(-25.)


def test_default_folds_are_nonoverlapping_and_forecasts_are_fixed_origin():
    results = backtest.evaluate_baselines(training_frame())
    assert results["folds"].train_n.tolist() == [38, 62, 86]
    predictions = results["predictions"]
    for _, group in predictions.groupby(["model", "target"]):
        assert len(group) == 72
        assert group.timestamp.nunique() == 72
        assert (group.timestamp > group.origin_timestamp).all()
    assert predictions.groupby(["fold", "model", "target"]).prediction.nunique().eq(1).all()
    count = results["metrics"].query("target == 'active_vm_count'")
    assert count.mae.eq(0).all()
    assert count.mae_improvement_pct.isna().all()


def test_validation_values_cannot_change_predictions_at_their_own_origin():
    original = training_frame()
    changed = original.copy()
    changed.loc[38:61, "aggregate_cpu_utilization"] = 99.
    before = backtest.evaluate_baselines(original)["predictions"].query("fold == 1")
    after = backtest.evaluate_baselines(changed)["predictions"].query("fold == 1")
    columns = ["model", "target", "timestamp", "prediction", "training_mean_scale", "peak_threshold"]
    pd.testing.assert_frame_equal(before[columns], after[columns])
    assert not before.actual.equals(after.actual)


def test_zero_training_scale_does_not_create_percentage_infinities():
    frame = training_frame(10)
    frame["aggregate_cpu_utilization"] = 0.
    result = backtest.evaluate_baselines(frame, horizon=3, min_train_size=4, windows=(2,))
    cpu = result["metrics"].query("target == 'aggregate_cpu_utilization'")
    assert cpu.mae.eq(0).all()
    assert cpu.nmae_pct_training_mean.isna().all()
    assert cpu.mae_ratio_to_persistence.isna().all()


def test_raw_holdout_values_cannot_change_backtest_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(backtest, "plot_backtests", lambda *args: None)
    readings = pd.DataFrame([
        {"timestamp": step * 300, "vm_id": vm, "min_cpu": 0.,
         "avg_cpu": float(10 + step + offset), "max_cpu": 100.}
        for step in range(13) for vm, offset in [("a", 0), ("b", 5)]
    ])
    summaries, outputs = [], []
    for variant in range(2):
        source, output = tmp_path / f"data{variant}", tmp_path / f"out{variant}"
        source.mkdir()
        changed = readings.copy()
        if variant:
            changed.loc[changed.timestamp >= 3000, "avg_cpu"] = 99.
        changed.to_csv(source / "vm_cpu_readings.csv.gz", index=False)
        summaries.append(backtest.run_backtest(source, output, train_size=10, horizon=3,
                                               min_train_size=4, windows=(2, 4)))
        outputs.append(output)
    assert summaries[0] == summaries[1]
    for original in outputs[0].iterdir():
        assert original.read_bytes() == (outputs[1] / original.name).read_bytes()
