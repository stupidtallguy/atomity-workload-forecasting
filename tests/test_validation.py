import pandas as pd
import pytest
from cwfc.validation import rolling_origin_splits


def test_rolling_origin_splits_are_temporal():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC")
    })
    splits = rolling_origin_splits(frame, horizon=4, min_train_size=8, step=4)
    assert len(splits) == 3
    assert splits[0].train_end < splits[0].validation_start
    assert splits[0].validation_end < splits[1].validation_start


def test_numeric_timestamps_remain_source_relative_seconds():
    frame = pd.DataFrame({"timestamp": [i * 300 for i in range(110)]})
    splits = rolling_origin_splits(frame, horizon=24, min_train_size=38)
    assert [(s.train_end, s.validation_start, s.validation_end) for s in splits] == [
        (11100, 11400, 18300), (18300, 18600, 25500), (25500, 25800, 32700),
    ]


@pytest.mark.parametrize("step", [0, -1])
def test_nonpositive_step_is_rejected(step):
    with pytest.raises(ValueError, match="positive"):
        rolling_origin_splits(pd.DataFrame({"timestamp": [0, 300, 600]}),
                              horizon=1, min_train_size=1, step=step)


@pytest.mark.parametrize("timestamps", [[0, 300, 300], [0, 600, 300], [0, 300, 900]])
def test_ambiguous_time_grids_are_rejected(timestamps):
    with pytest.raises(ValueError):
        rolling_origin_splits(pd.DataFrame({"timestamp": timestamps}),
                              horizon=1, min_train_size=1)
