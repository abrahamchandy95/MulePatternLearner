"""Grouped bootstrap intervals resample whole groups."""


def test_grouped_ap_interval_bootstraps_whole_groups() -> None:
    import numpy as np

    from mule_pattern_learner.metrics import grouped_ap_interval

    y = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=np.int64)
    scores = np.array([0.9, 0.2, 0.1, 0.8, 0.3, 0.4, 0.2, 0.7])
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    low, high = grouped_ap_interval(y, scores, groups, seed=7, draws=100) or (None, None)
    assert low is not None and high is not None and 0.0 <= low <= high <= 1.0
    # Same seed, same interval; one group or no positive has no interval.
    assert grouped_ap_interval(y, scores, groups, seed=7, draws=100) == [low, high]
    assert grouped_ap_interval(y, scores, np.zeros(8, dtype=np.int64)) is None
    assert grouped_ap_interval(np.zeros(8, dtype=np.int64), scores, groups) is None
