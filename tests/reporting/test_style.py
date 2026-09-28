"""The figures' fixed look: the colours that never change, the resolution and the numbers."""

from __future__ import annotations

from mule_pattern_learner.reporting import style


def test_the_mule_non_mule_and_baseline_colours_never_change() -> None:
    assert (style.MULE, style.NON_MULE, style.BASELINE) == ("#eb6834", "#2a78d6", "#0b0b0b")
    assert style.SPLIT_COLOURS == {"validation": "#4a3aa7", "test": "#008300"}
    # No other series takes the colour of mules, non-mules, a split or the baseline.
    fixed = {style.MULE, style.NON_MULE, style.BASELINE, *style.SPLIT_COLOURS.values()}
    assert not fixed & set(style.MEASURES)
    assert len(set(style.MEASURES)) == len(style.MEASURE_LINES) == len(style.MEASURES)
    assert style.DPI == 150


def test_numbers_print_as_counts_shares_and_intervals() -> None:
    assert [style.number(v) for v in (None, 47_749, 0.13421, 0.00084, 0.0, True)] == [
        "n/a",
        "47,749",
        "0.134",
        "0.00084",
        "0",
        "True",
    ]
    assert style.estimate(0.134, [0.081, 0.212]) == "0.134 (0.081 to 0.212)"
    assert style.estimate(0.134, None) == "0.134"
