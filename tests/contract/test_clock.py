"""Calendar dates as the graph's millisecond clocks."""

from __future__ import annotations

from mule_pattern_learner.contract.clock import cutoff_ms, timestamp


def test_dates_are_utc_unless_they_name_a_zone_and_cutoffs_end_before_them() -> None:
    assert timestamp("2024-01-01") == 1_704_067_200_000
    assert timestamp("2024-01-01T01:00:00+01:00") == timestamp("2024-01-01")
    assert cutoff_ms("2024-01-01") == timestamp("2024-01-01") - 1
