import numpy as np
import pandas as pd
import pytest

from cwfc.policy import breach_signals, calibrate_margin, evaluate_headroom, planning_ceiling


def test_one_sided_margin_uses_signed_errors_and_higher_quantile():
    assert calibrate_margin([-2, -1, 1, 4], quantile=.5) == 1
    assert calibrate_margin([-3, -2, -1]) == 0
    assert calibrate_margin([1, 2, 3, 4]) == 4


def test_ceiling_adds_percentage_points_and_respects_physical_bounds():
    np.testing.assert_allclose(planning_ceiling([10, 99.8], .5), [10.5, 100])
    with pytest.raises(ValueError):
        planning_ceiling([10], -.5)


def test_review_triggers_on_second_breach_once_per_run_and_resets_on_equality():
    result = breach_signals([11, 12, 13, 10, 11, 12], [10] * 6)
    assert result.alert_event.tolist() == [False, True, False, False, False, True]
    assert result.review_active.tolist() == [False, True, True, False, False, True]
    assert result.breach_streak.tolist() == [1, 2, 3, 0, 1, 2]


def test_sequential_margin_uses_only_previously_observed_residuals():
    validation = pd.DataFrame({
        "fold": [1, 1, 2, 2, 3, 3], "timestamp": [300, 600, 900, 1200, 1500, 1800],
        "origin_timestamp": [0, 0, 600, 600, 1200, 1200],
        "actual": [11., 12., 13., 14., 15., 16.],
        "prediction": [10.] * 6, "is_peak": [True] * 6,
    })
    original = evaluate_headroom(validation)
    assert original["warmup_folds"] == [1]
    assert original["sequential_folds"].margin_pp.tolist() == [2., 4.]
    assert original["sequential_folds"].alert_events.tolist() == [1, 1]
    validation.loc[validation.fold == 2, "actual"] = 99.
    changed = evaluate_headroom(validation)
    assert changed["sequential_folds"].margin_pp.tolist() == [2., 89.]
    np.testing.assert_array_equal(
        original["sequential_predictions"].query("fold == 2").planning_ceiling_pct,
        changed["sequential_predictions"].query("fold == 2").planning_ceiling_pct,
    )


@pytest.mark.parametrize("errors", [[], [np.nan], [np.inf], [[1, 2]]])
def test_invalid_calibration_errors_are_rejected(errors):
    with pytest.raises(ValueError):
        calibrate_margin(errors)
