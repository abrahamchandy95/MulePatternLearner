"""The split predictions a finished run writes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mule_pattern_learner.training.schedule import EvaluationSample
from mule_pattern_learner.training.summary import prediction_frame


def test_prediction_frames_keep_the_accepted_rows_in_sample_order() -> None:
    accounts = pd.DataFrame({"account_id": ["a", "b", "c", "d"], "group_id": [0, 1, 2, 3]})
    samples = [
        EvaluationSample("2024-01-01", np.array([2, 0]), np.array([True, False])),
        EvaluationSample("2024-02-01", np.array([3]), np.array([False])),
    ]
    scores = np.array([0.9, np.nan, 0.2])
    frame = prediction_frame(accounts, samples, scores, np.array([True, False, True]))
    assert frame.account_id.tolist() == ["c", "d"]
    assert frame.date.tolist() == ["2024-01-01", "2024-02-01"]
    assert frame.observed_label.tolist() == [1, 0] and frame.score.tolist() == [0.9, 0.2]
    assert frame.observed_label.dtype == np.int64
