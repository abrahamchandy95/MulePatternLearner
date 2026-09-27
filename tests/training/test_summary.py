"""The provenance and the split predictions a run writes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.training import summary
from mule_pattern_learner.training.schedule import EvaluationSample
from mule_pattern_learner.training.summary import PACKAGES, prediction_frame, provenance


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


def test_provenance_names_the_code_versions_host_and_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded = provenance(torch.device("cpu"), "torch", "abc")
    assert (recorded["device"], recorded["sampler_backend"]) == ("cpu", "torch")
    assert recorded["dataset_id"] == "abc" and set(recorded["versions"]) == set(PACKAGES)
    assert datetime.fromisoformat(recorded["started"]).tzinfo is not None
    # Outside a git repository the commit and dirty flag are unknown, not an error.
    monkeypatch.setattr(summary, "REPOSITORY_ROOT", tmp_path)
    outside = provenance(torch.device("cpu"), "torch", "abc")
    assert outside["git_commit"] is None and outside["git_dirty"] is None
