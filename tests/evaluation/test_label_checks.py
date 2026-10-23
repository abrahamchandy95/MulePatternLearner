"""mule audit's label checks, each failing on its planted case, its reports, and the
identification horizon (docs/design.md, "Checks on every world" and "Identification
horizon")."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.evaluation.answer_key import AnswerKey
from mule_pattern_learner.evaluation.label_checks import (
    IDENTIFICATION_HORIZON,
    IDENTIFIED_NOT_MULE,
    JOINED_COLUMNS,
    LABEL_CHECKS,
    MULE_DATES_AND_RING,
    SCORABLE_WITHOUT_TRUTH,
    TRUTH_JOIN,
    binomial_band,
    check_labels,
    horizon,
    scored_months,
)
from mule_pattern_learner.testing.world import T
from mule_pattern_learner.tigergraph.queries.read_answer_key import AccountTruth

D = 86_400_000
W = 7 * D
LEVEL = DEFAULT_CONFIG.audit.horizon_level
MONDAYS = (T, T + W)
YEAR = DEFAULT_CONFIG.calendar.mondays()


def _truth(
    account: str,
    mule: int = 0,
    ring: int = -1,
    onset: int = 0,
    last: int = 0,
    account_type: str = "deposit",
    owned: bool = True,
) -> AccountTruth:
    return AccountTruth(account, mule, ring, onset, last, T - 400 * D, account_type, owned)


KEY = AnswerKey(
    (
        _truth("a_card", 1, 4, T - 9 * D, T - 8 * D, account_type="credit"),
        _truth("a_m1", 1, 3, T - 2 * D, T + 2 * D),
        _truth("a_m2", 1, -1, T - 2 * D, T + 2 * D, owned=False),
        _truth("a_n1"),
        _truth("a_n2"),
    )
)
POPULATIONS = {T: ["a_m1", "a_n1", "a_n2"], T + W: ["a_m1", "a_n1", "a_n2"]}


def test_a_world_with_sound_labels_passes_every_label_check() -> None:
    found = check_labels(KEY, frozenset({"a_m1"}), POPULATIONS, MONDAYS, LEVEL)
    assert all(found.results[name] for name in LABEL_CHECKS)
    assert all(not found.breaking[name] for name in LABEL_CHECKS)


def test_each_label_check_fails_on_its_planted_case() -> None:
    planted = AnswerKey((*KEY.truths, _truth("a_bad", 1, 2, 0, T)))
    populations = {**POPULATIONS, T: [*POPULATIONS[T], "a_ghost"]}
    # What evaluation joins with the truth gains a column the truth holds.
    found = check_labels(
        planted,
        frozenset({"a_m1", "a_n2", "a_gone"}),
        populations,
        MONDAYS,
        LEVEL,
        joined=(*JOINED_COLUMNS, "is_mule"),
    )
    assert dict(found.breaking) == {
        IDENTIFIED_NOT_MULE: ("a_gone", "a_n2"),
        MULE_DATES_AND_RING: ("a_bad",),
        SCORABLE_WITHOUT_TRUTH: ("a_ghost",),
        TRUTH_JOIN: ("is_mule",),
    }
    assert not any(found.results[name] for name in LABEL_CHECKS)


def test_mules_are_counted_by_type_and_ownership_and_unscored_ones_reported() -> None:
    found = check_labels(KEY, frozenset({"a_m1"}), POPULATIONS, MONDAYS, LEVEL)
    counts = {
        (str(kind), bool(owned)): int(mules)
        for kind, owned, mules in found.counts.itertuples(index=False, name=None)
    }
    assert counts == {("credit", True): 1, ("deposit", False): 1, ("deposit", True): 1}
    # a_m2 is active at T and T + 1 week but never scorable (it has no owner); a_card's
    # activity ends 8 days before T, so it is active at neither.
    assert found.unscored.values.tolist() == [[T, "a_m2"], [T + W, "a_m2"]]


def test_a_binomial_band_by_hand() -> None:
    # Binomial(10, 0.5): P(X <= 1) = 0.0107 and P(X <= 2) = 0.0547, so the 5% quantile is 2;
    # P(X <= 7) = 0.9453 and P(X <= 8) = 0.9893, so the 95% quantile is 8.
    assert binomial_band(10, 0.5, 0.9) == (2, 8)
    assert binomial_band(0, 0.5, 0.9) == (0, 0) and binomial_band(4, 1.0, 0.9) == (4, 4)


def _ms(year: int, month: int) -> int:
    return int(datetime(year, month, 10, tzinfo=UTC).timestamp() * 1000)


def _months_key(identified_per_month: list[int]) -> tuple[AnswerKey, frozenset[str]]:
    """Ten true mules with onset in each month of the default scored year, the first n of
    each month identified."""
    months = [(2024, month) for month in range(4, 13)] + [(2025, month) for month in range(1, 4)]
    truths: list[AccountTruth] = []
    identified: set[str] = set()
    for (year, month), n in zip(months, identified_per_month, strict=True):
        for index in range(10):
            account = f"a_{year}_{month}_{index}"
            truths.append(_truth(account, 1, index, _ms(year, month), _ms(year, month) + D))
            if index < n:
                identified.add(account)
    return AnswerKey(tuple(truths)), frozenset(identified)


def test_the_scored_year_runs_from_april_to_march() -> None:
    months = scored_months(YEAR)
    assert months[0] == "2024-04" and months[-1] == "2025-03" and len(months) == 12


def test_the_horizon_passes_when_the_identified_share_is_flat() -> None:
    key, identified = _months_key([5] * 12)
    table = horizon(key, identified, YEAR, LEVEL)
    assert table["mules"].tolist() == [10] * 12 and table["inside"].all()
    found = check_labels(key, identified, {}, YEAR, LEVEL)
    assert found.results[IDENTIFICATION_HORIZON]


def test_the_horizon_fails_when_one_month_is_identified_less() -> None:
    # The last month's mules are never identified, as if identification could not see past
    # the export: the pooled share is 110 / 120, and 0 of 10 lies far below its band.
    key, identified = _months_key([10] * 11 + [0])
    table = horizon(key, identified, YEAR, LEVEL)
    assert table["inside"].tolist() == [True] * 11 + [False]
    assert not check_labels(key, identified, {}, YEAR, LEVEL).results[IDENTIFICATION_HORIZON]


def test_the_horizon_fails_without_a_mule_in_the_scored_year() -> None:
    assert not check_labels(AnswerKey(()), frozenset(), {}, YEAR, LEVEL).results[
        IDENTIFICATION_HORIZON
    ]


def test_the_join_of_the_keys_and_scores_with_the_truth_shares_the_account_id_alone() -> None:
    # Evaluation joins a Monday, an account, a weight and a logit with the truth on the account;
    # the truth holds none of the other three.
    assert JOINED_COLUMNS == ("t_ms", "account_id", "weight", "logit")
    found = check_labels(KEY, frozenset(), POPULATIONS, MONDAYS, LEVEL)
    assert found.results[TRUTH_JOIN] and found.breaking[TRUTH_JOIN] == ()


@pytest.mark.parametrize("column", ["t_ms", "weight", "logit"])
def test_a_column_the_truth_gains_that_the_join_holds_fails_the_check(
    column: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = AnswerKey.frame

    def with_column(key: AnswerKey) -> pd.DataFrame:
        return frame(key).assign(**{column: 0})

    monkeypatch.setattr(AnswerKey, "frame", with_column)
    found = check_labels(KEY, frozenset(), POPULATIONS, MONDAYS, LEVEL)
    assert found.breaking[TRUTH_JOIN] == (column,) and not found.results[TRUTH_JOIN]
