"""The training, validation and test examples of the reference graph's world, worked out by
hand from docs/design.md.

In the standard world (testing.world) Mia was found 5 days before T (2024-07-01, a Monday),
60 days after her onset; Max was found 20 days after T and Roy 30 days after it, and Sue
is never found. Each found mule's account closed the day it was found. Hal is held out,
Quy has no payments, and X_CASH and G_FEES belong to no customer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd
import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, DataConfig
from mule_pattern_learner.data.answer_key import read_answer_key
from mule_pattern_learner.data.examples import (
    COLUMNS,
    POSITIVE,
    UNLABELLED,
    evaluation_examples,
    read_population,
    training_examples,
    validation_examples,
)
from mule_pattern_learner.data.grid import DAY_MS, windows
from mule_pattern_learner.testing.reference_graph import ReferenceGraph
from mule_pattern_learner.testing.world import Mule, T, WorldBuilder, standard_world
from mule_pattern_learner.tigergraph.queries import (
    LIST_POPULATION,
    READ_ANSWER_KEY,
    READ_SUPERVISION,
    Phase,
)

D = DAY_MS
CONFIG = DEFAULT_CONFIG.data
# Training as-of dates up to the labels' cutoff.
NO_GAP = replace(CONFIG, maturity_gap_days=0)
# Mia's positive days after T: the Mondays from her onset (T - 60 days) to her discovery.
MIA_DAYS = [-56, -49, -42, -35, -28, -21, -14, -7]
# The test window's Mondays, 2024-10-07 to 2024-12-30, in days after T.
TEST_DAYS = list(range(98, 183, 7))
# The training customers' accounts open at T.
OPEN_AT_T = ["A_ANN", "A_ANN_CARD", "A_BEN", "A_MAX", "A_QUY", "A_ROY", "A_SUE", "A_VIC"]


class _NoAnswerKey:
    """The reference graph, refusing read_answer_key and keeping the names of the queries
    it ran. The import contracts stop only imports of data.answer_key, while any code may
    import the query's name from tigergraph.queries."""

    def __init__(self, graph: ReferenceGraph) -> None:
        self.graph = graph
        self.ran: set[str] = set()

    def run(
        self,
        name: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
        attempts: int | None = None,
        timeout_retries: int = 1,
    ) -> list[dict[str, Any]]:
        if name == READ_ANSWER_KEY:
            raise AssertionError(f"{READ_ANSWER_KEY} ran outside evaluation")
        self.ran.add(name)
        return self.graph.run(
            name,
            params,
            timeout_s=timeout_s,
            attempts=attempts,
            timeout_retries=timeout_retries,
        )


def _dated(examples: pd.DataFrame) -> pd.DataFrame:
    """examples, typed as COLUMNS says, with each as-of time in days after T."""
    assert {str(name): str(kind) for name, kind in examples.dtypes.items()} == COLUMNS
    return examples.assign(day=(examples.as_of_ms - T) // D)


def _training(graph: ReferenceGraph, config: DataConfig = NO_GAP) -> pd.DataFrame:
    return _dated(training_examples(graph, windows(config).training, config))


def _days(examples: pd.DataFrame, account_id: str, label: int) -> list[int]:
    """The days after T on which the account is an example with label."""
    chosen = examples[(examples.account_id == account_id) & (examples.label == label)]
    return list(chosen.day)


def _accounts(examples: pd.DataFrame, day: int, label: int) -> list[str]:
    """The accounts that are examples with label on the day after T."""
    return list(examples.account_id[(examples.day == day) & (examples.label == label)])


def _weights(examples: pd.DataFrame, label: int) -> pd.Series:
    """The total weight of label's examples on each day."""
    return examples[examples.label == label].groupby("day").weight.sum()


def _world_with(account_id: str, opened: int, closed: int, mule: Mule) -> WorldBuilder:
    """The standard world and one more customer whose account is mule."""
    world = standard_world()
    party = account_id.replace("A_", "P_")
    world.party(party, opened)
    world.account(account_id, party, opened, closed_ms=closed, mule=mule)
    return world


def test_the_population_is_every_open_account_of_the_phases_customers() -> None:
    graph = standard_world().graph()
    population = read_population(graph, T, Phase.TRAINING)
    assert list(population.account_id) == OPEN_AT_T
    assert [str(kind) for kind in population.dtypes] == ["str", "str", "int64", "int64"]
    assert set(read_population(graph, T, Phase.TEST).account_id) == {*OPEN_AT_T, "A_HAL"}


def test_the_default_gap_keeps_out_a_mule_too_new_to_have_been_found() -> None:
    """With the 60-day gap the last training date is T - 63 days, before Mia's onset."""
    assert _training(standard_world().graph(), CONFIG).empty


def test_a_revealed_mule_is_positive_from_onset_to_discovery_among_its_dates_accounts() -> None:
    """Every other open training account is drawn, as there are fewer than 15 per positive:
    Quy with no payments, and Max, Roy and Sue, mules not yet found. Hal, held out, and
    the accounts of no customer never are."""
    examples = _training(standard_world().graph())
    assert sorted(set(examples.day)) == MIA_DAYS
    assert _days(examples, "A_MIA", POSITIVE) == MIA_DAYS
    assert _days(examples, "A_MIA", UNLABELLED) == []
    others = [account for account in OPEN_AT_T if account != "A_MIA"]
    assert _accounts(examples, -7, UNLABELLED) == others
    # Old's account closed 20 days before T, so it is open on the day 21 days before.
    assert _accounts(examples, -21, UNLABELLED) == sorted([*others, "A_OLD"])
    assert not {"A_HAL", "X_CASH", "G_FEES"} & set(examples.account_id)


def test_each_mule_weighs_one_and_each_dates_unlabelled_weigh_what_its_positives_do() -> None:
    examples = _training(standard_world().graph())
    mia = examples[examples.account_id == "A_MIA"]
    assert list(mia.weight) == [1 / 8] * 8
    unlabelled = examples[examples.label == UNLABELLED]
    assert set(unlabelled.weight[unlabelled.day == -7]) == {1 / 8 / 8}
    assert set(unlabelled.weight[unlabelled.day == -21]) == {1 / 8 / 9}
    pd.testing.assert_series_equal(_weights(examples, POSITIVE), _weights(examples, UNLABELLED))


def test_a_revealed_mules_dates_in_the_30_days_before_its_onset_are_dropped() -> None:
    """Long, found 2 days before T after 98 days as a mule, is positive on 14 dates. On the
    first two Mia is still more than 30 days from her onset, so she is unlabelled; on the
    next four she is within 30 days of it, so she is no example."""
    long = Mule(2, "first_layer", "collector", T - 100 * D, T - 3 * D, T - 2 * D, "monitoring")
    world = _world_with("A_LONG", T - 200 * D, T - 2 * D, long)
    examples = _training(world.graph())
    long_days = list(range(-98, 0, 7))
    assert _days(examples, "A_LONG", POSITIVE) == long_days
    assert set(examples.weight[examples.account_id == "A_LONG"]) == {1 / 14}
    assert _days(examples, "A_MIA", UNLABELLED) == [-98, -91]
    assert _days(examples, "A_MIA", POSITIVE) == MIA_DAYS
    assert sorted(set(examples.day)) == long_days
    weights = examples[examples.label == POSITIVE].groupby("account_id").weight.sum()
    assert weights.to_dict() == pytest.approx({"A_LONG": 1.0, "A_MIA": 1.0})
    pd.testing.assert_series_equal(_weights(examples, POSITIVE), _weights(examples, UNLABELLED))


def test_the_unlabelled_are_a_fixed_draw_of_so_many_per_positive() -> None:
    graph = standard_world().graph()
    config = replace(NO_GAP, unlabelled_per_positive=2)
    examples = _training(graph, config)
    assert _training(graph, config).equals(examples)
    unlabelled = examples[examples.label == UNLABELLED]
    assert set(unlabelled.groupby("day").size()) == {2}
    assert set(unlabelled.weight) == {1 / 8 / 2}
    assert set(unlabelled.account_id) <= {*OPEN_AT_T, "A_OLD"} - {"A_MIA"}
    # Each date draws apart, and another seed draws other accounts.
    draws = {tuple(_accounts(examples, day, UNLABELLED)) for day in MIA_DAYS}
    assert len(draws) > 1
    assert not _training(graph, replace(config, seed=1)).equals(examples)


def test_training_and_validation_read_supervision_and_never_the_answer_key() -> None:
    graph = _NoAnswerKey(standard_world().graph())
    window = windows(NO_GAP)
    assert not training_examples(graph, window.training, NO_GAP).empty
    assert not validation_examples(graph, window.validation, NO_GAP).empty
    assert graph.ran == {READ_SUPERVISION, LIST_POPULATION}
    # The guard itself: reading the answer key through it fails.
    with pytest.raises(AssertionError, match=READ_ANSWER_KEY):
        read_answer_key(graph)


def test_validation_ranks_the_revealed_mules_active_each_date_among_the_others() -> None:
    """Max (found 20 days after T) and Roy (30 days after) are revealed before the window's
    cutoff; Sue is never found, so she is among the others. Ann's new account opens a week
    after T and New's two weeks after it."""
    graph = standard_world().graph()
    examples = _dated(validation_examples(graph, windows(CONFIG).validation, CONFIG))
    assert sorted(set(examples.day)) == list(range(0, 92, 7))
    assert _days(examples, "A_MAX", POSITIVE) == [0, 7, 14]
    assert _days(examples, "A_ROY", POSITIVE) == [0, 7, 14, 21, 28]
    assert set(examples.account_id[examples.label == POSITIVE]) == {"A_MAX", "A_ROY"}
    others = ["A_ANN", "A_ANN_CARD", "A_BEN", "A_QUY", "A_SUE", "A_VIC"]
    assert _accounts(examples, 0, UNLABELLED) == others
    assert _accounts(examples, 35, UNLABELLED) == sorted([*others, "A_ANN_NEW", "A_NEW"])
    assert set(examples.weight) == {1.0} and set(examples.split) == {0}


def test_the_others_are_a_draw_each_weighing_the_accounts_it_stands_for() -> None:
    graph = standard_world().graph()
    config = replace(CONFIG, scored_sample=3)
    examples = _dated(validation_examples(graph, windows(config).validation, config))
    assert _accounts(examples, 0, POSITIVE) == ["A_MAX", "A_ROY"]
    others = examples[examples.label == UNLABELLED]
    assert set(others.groupby("day").size()) == {3}
    # Six others at T and eight five weeks later.
    assert set(others.weight[others.day == 0]) == {2.0}
    assert set(others.weight[others.day == 35]) == {8 / 3}
    assert validation_examples(graph, windows(config).validation, config).equals(
        examples.drop(columns="day")
    )


def test_the_test_window_ranks_every_true_mule_active_each_date_and_every_customer() -> None:
    """Sue, never found, is the only mule active in the test window; Hal, held out, is
    ranked with his split."""
    graph = standard_world().graph()
    window = windows(CONFIG).test
    examples = _dated(evaluation_examples(graph, window, read_answer_key(graph), CONFIG))
    assert sorted(set(examples.day)) == TEST_DAYS
    assert _days(examples, "A_SUE", POSITIVE) == TEST_DAYS
    assert set(examples.account_id[examples.label == POSITIVE]) == {"A_SUE"}
    assert _accounts(examples, 98, UNLABELLED) == [
        "A_ANN",
        "A_ANN_CARD",
        "A_ANN_NEW",
        "A_BEN",
        "A_HAL",
        "A_NEW",
        "A_QUY",
        "A_VIC",
    ]
    assert set(examples.split[examples.account_id == "A_HAL"]) == {1}
    assert set(examples.weight) == {1.0}


def test_a_mule_is_active_from_its_onset_and_not_ranked_once_found() -> None:
    """Kept was found 100 days after T, yet its account stayed open; Late's onset is 120
    days after T and it is never found."""
    kept = Mule(3, "first_layer", "pass_through", T - 20 * D, T + 90 * D, T + 100 * D, "trace")
    world = _world_with("A_KEPT", T - 100 * D, 0, kept)
    world.party("P_LATE", T - 100 * D)
    late = Mule(4, "onward_layer", "collector", T + 120 * D, T + 150 * D)
    world.account("A_LATE", "P_LATE", T - 100 * D, mule=late)
    graph = world.graph()
    window = windows(CONFIG).test
    examples = _dated(evaluation_examples(graph, window, read_answer_key(graph), CONFIG))
    assert _days(examples, "A_KEPT", POSITIVE) == [98]
    assert _days(examples, "A_KEPT", UNLABELLED) == []
    assert _days(examples, "A_LATE", UNLABELLED) == [98, 105, 112, 119]
    assert _days(examples, "A_LATE", POSITIVE) == list(range(126, 183, 7))


def test_an_unknown_truth_or_a_mule_without_an_onset_is_refused() -> None:
    graph = standard_world().graph()
    window = windows(CONFIG).test
    truth = read_answer_key(graph)
    with pytest.raises(ValueError, match="does not know"):
        evaluation_examples(graph, window, truth[truth.account_id != "A_ANN"], CONFIG)
    unbegun = truth.assign(mule_onset_ms=truth.mule_onset_ms.where(truth.account_id != "A_SUE", 0))
    with pytest.raises(ValueError, match="A_SUE"):
        evaluation_examples(graph, window, unbegun, CONFIG)
    # A customer's account whose truth is unknown (-1).
    world = standard_world()
    world.account("A_ODD", "P_ANN", T - 10 * D, external=True)
    odd = world.graph()
    with pytest.raises(ValueError, match="does not know"):
        evaluation_examples(odd, window, read_answer_key(odd), CONFIG)
    # The test window has no labels' cutoff to train or validate on.
    with pytest.raises(ValueError, match="cutoff"):
        training_examples(graph, window, CONFIG)
    with pytest.raises(ValueError, match="cutoff"):
        validation_examples(graph, window, CONFIG)
