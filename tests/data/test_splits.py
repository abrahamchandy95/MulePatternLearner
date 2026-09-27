"""Split dates must run forward."""

from __future__ import annotations

import pytest

from mule_pattern_learner.data.splits import validate_dates


def test_dates_must_have_forward_chronological_splits() -> None:
    config = {
        "dates": {"train": ["2024-07-01"], "validation": ["2024-06-01"], "test": ["2025-01-01"]}
    }
    with pytest.raises(ValueError, match="overlap"):
        validate_dates(config)
