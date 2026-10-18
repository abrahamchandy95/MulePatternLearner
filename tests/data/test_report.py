"""What mule prepare reports of a world, worked out by hand, and the refusal of a selection
world with too few identified active mules (docs/design.md, "Training world")."""

from __future__ import annotations

import pytest

from mule_pattern_learner.data.examples import Draw, Tally, training_frame
from mule_pattern_learner.data.labels import Labels, Unscored, label_identity
from mule_pattern_learner.data.report import (
    REPORT_SCHEMA,
    build_report,
    lines,
    refuse_selection,
)
from mule_pattern_learner.tigergraph.queries.read_labels import IdentifiedMule

W = 7 * 86_400_000
T1 = 1_719_792_000_000  # 2024-07-01 00:00 UTC, a Monday
T2, T3 = T1 + W, T1 + 2 * W
# a is active on T1 and T2, b on T2 alone, c on none of them.
MULES = (
    IdentifiedMule("a", T1 - 1, T1),
    IdentifiedMule("b", T2 - 1, T2 - 1),
    IdentifiedMule("c", T1 - 5 * W, T1 - 4 * W),
)
LABELS = Labels(MULES, label_identity(MULES))
TALLIES = [Tally(T1, 10, ("a",)), Tally(T2, 20, ("a", "b")), Tally(T3, 0, ())]
DRAWS = [Draw(T1, 9, ("u1", "u2", "u3")), Draw(T2, 18, ("u1", "u4")), Draw(T3, 0, ())]
UNSCORED = Unscored(("c",), ())


def test_the_report_of_a_training_world() -> None:
    training = training_frame(TALLIES, DRAWS)
    report = build_report(TALLIES, LABELS, UNSCORED, DRAWS, training)
    frame = report.mondays
    assert list(frame.columns) == REPORT_SCHEMA.names
    assert frame["scorable"].tolist() == [10, 20, 0]
    assert frame["positives"].tolist() == [1, 2, 0]
    # Dropped: the identified mules not active then: b and c, then c, then all three.
    assert frame["dropped"].tolist() == [2, 1, 3]
    assert frame["pool"].tolist() == [9, 18, 0]
    assert frame["unlabelled"].tolist() == [3, 2, 0]
    # Positives: a 1/2 twice and b 1, mean 2/3, so a's rows weigh 3/4 and b's 3/2. Unlabelled:
    # 3 at 9/3 and 2 at 18/2, mean 27/5, so they weigh 5/9 and 5/3.
    assert frame["positive_weight"].tolist() == pytest.approx([0.75, 2.25, 0.0])
    assert frame["unlabelled_weight"].tolist() == pytest.approx([5 / 3, 10 / 3, 0.0])
    assert frame["ratio"].tolist() == pytest.approx([0.45, 0.675, 0.0])
    assert report.identified == 3 and report.identified_active == 2
    # (1/10 + 2/20) / 2, over the Mondays with a scorable account.
    assert report.identified_active_share == pytest.approx(0.1)
    assert report.unscored == UNSCORED


def test_the_report_of_a_world_that_does_not_train() -> None:
    report = build_report(TALLIES, LABELS, UNSCORED)
    frame = report.mondays
    assert frame[["pool", "unlabelled"]].eq(0).all().all()
    assert frame[["positive_weight", "unlabelled_weight", "ratio"]].eq(0.0).all().all()
    assert report.identified_active == 2
    empty = build_report([], LABELS, UNSCORED)
    assert empty.mondays.empty and empty.identified_active_share == 0.0


def test_a_selection_world_needs_enough_identified_active_mules() -> None:
    report = build_report(TALLIES, LABELS, UNSCORED)
    refuse_selection(report, 2)
    with pytest.raises(ValueError, match="2 identified active mules, fewer than the 5"):
        refuse_selection(report, 5)


def test_the_report_reads_as_lines() -> None:
    found = lines(build_report(TALLIES, LABELS, Unscored(("c",), ("d",))))
    assert (
        found[0]
        == "3 identified mules, 2 identified active mules; identified active share 0.100000"
    )
    assert found[1] == "3 positives and 0 unlabelled examples over 3 Mondays"
    assert found[2] == "1 identified mule active on no scored Monday: c"
    assert found[3] == "1 identified mule never scorable on a Monday of its activity: d"
