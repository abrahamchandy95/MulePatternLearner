"""The projection of the proof of concept's run (docs/design.md, "Proof-of-concept settings (25
October 2026)"): the seconds of each stage of generating, loading and fetching three worlds of
30,000 people and training one rotation on them, from the rates of the benchmark of world T (24
October 2026: `mule bench` and the 5,000-person test) scaled to the sizes the settings of
config.py give. The run must come to about 7 hours at most, and the audit's sample of non-mule
account-Mondays must lie inside the fallback samples that the selection and test worlds already
fetch, so that the audit fetches no non-mule account-Monday twice.

The numbers of the design's table are the output of `stage_table`, which this module prints under
`pytest -s`."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import pytest

from mule_pattern_learner.config import (
    DEFAULT_CONFIG,
    FULL_DESIGN_CONFIG,
    RunConfig,
)

HOUR = 3_600.0
# The people of the benchmark's world T and of each development world, and the scored Mondays.
T_PEOPLE, D_PEOPLE, MONDAYS = 5_000, 30_000, 52
# The worlds of the rotation: the training, selection and test world.
WORLDS = 3
# Scorable accounts per person on a Monday (mule bench, selection section).
PER_PERSON = 1.577533
# Accounts a second of each query at 1 and at 2 concurrent requests of 32 accounts (mule bench, fetch
# section, accounts_per_second), and the fill of 192 account-Mondays at 1 (fetch section): 584.4 s.
RATE_1 = {"b1": 77.9688, "b2": 90.192, "b3": 76.7793, "b4": 3.14518, "b5": 11.4117, "b6": 26.0685}
RATE_2 = {"b1": 108.744, "b2": 117.603, "b3": 124.962, "b4": 10.6964, "b5": 14.7917, "b6": 41.0114}
RATE_4 = {"b1": 122.486, "b2": 152.573, "b3": 144.642, "b4": 10.8552, "b5": 24.5439, "b6": 42.563}
RATES = {1: RATE_1, 2: RATE_2, 4: RATE_4}
FILL_SECONDS, FILL_ACCOUNTS = 584.403, 192
SCORED_QUERIES, NODE_QUERIES = ("b4", "b5", "b6"), ("b1", "b2", "b3")
# From the 5,000-person test (24 October 2026), in seconds at 5,000 people: PhantomLedger's
# generation of 455 days, mule-load push, mule prepare, the readiness check every command that
# reads the world runs and the read_world inside it, read_hubs for one Monday, and mule-load
# clear; and the first wipe. A world of the design simulates 730 days.
GENERATE, PUSH, PREPARE, READY, READ_WORLD, READ_HUBS = 40.375, 529.17, 113.0, 34.0, 15.62, 1.6
CLEAR, WIPE = 60.0, 100.0
SIMULATED_DAYS = 730 / 455
# Allowances for what the benchmark did not time: what mule audit reads besides its fetch (the
# answer key and the labels), its computing (the feature audit's 1,000 replicates and the clock
# probe), and mule train with mule evaluate for one seed (building every example once, at most 30
# epochs, two XGBoost comparisons, the paired bootstrap).
AUDIT_SETUP, AUDIT_COMPUTING, TRAIN_EVALUATE = 60.0, 300.0, 900.0
# The audit's non-mule sample against the training world's draw, which takes the lowest hashes of
# each Monday and not of the year: about a quarter of the sample is fetched again. A Monday draws
# one more unlabelled account for each positive above the minimum, about a tenth more in all.
AUDIT_EXTRA, DRAW_EXTRA = 0.25, 1.1
# The true-mule account-Mondays of a world, which mule audit fetches every one of and the test
# world's fallback list holds every one of: about 700 where the design's calibration holds (75
# identified customers in a 30,000-person world, about half of the true mules, 3.9 active Mondays
# each and an early-warning Monday), 900 as the high case, and 200 at the rate of world T.
EXPECTED, HIGH, WORLD_T = 700, 900, 200


def seconds_per_account_monday(rate: dict[str, float]) -> float:
    """The seconds to fetch one sparse account-Monday at the rates of rate: B4 to B6 for the scored
    account, and B1 to B3 for each of the account nodes of its subgraph, whose number is what the
    fill of 192 account-Mondays at one request at a time leaves after B4 to B6 (70.4 a subgraph)."""
    scored_1 = sum(1 / RATE_1[query] for query in SCORED_QUERIES)
    per_node_1 = sum(1 / RATE_1[query] for query in NODE_QUERIES)
    nodes = (FILL_SECONDS / FILL_ACCOUNTS - scored_1) / per_node_1
    return sum(1 / rate[query] for query in SCORED_QUERIES) + nodes * sum(
        1 / rate[query] for query in NODE_QUERIES
    )


@dataclass(frozen=True)
class Projection:
    """The seconds of each stage of the run, and its account-Mondays fetched."""

    stages: dict[str, float]
    account_mondays: float

    @property
    def total(self) -> float:
        return sum(self.stages.values())


def project(config: RunConfig, mules: float, mule_cost: float = 1.0) -> Projection:
    """The run of the three worlds of the first rotation, with mules true-mule account-Mondays in
    each, each costing mule_cost sparse account-Mondays to fetch."""
    scale = D_PEOPLE / T_PEOPLE
    population = PER_PERSON * D_PEOPLE * MONDAYS
    data, audit = config.data, config.audit
    held = mules * mule_cost
    sample = population / data.fallback_weight
    training = (
        held + DRAW_EXTRA * MONDAYS * data.min_unlabelled + AUDIT_EXTRA * audit.nonmule_sample
    )
    count = training + 2 * (held + sample)
    ready = READY - READ_WORLD + READ_WORLD * scale
    stages = {
        "generate": WORLDS * GENERATE * scale * SIMULATED_DAYS,
        "push": WORLDS * PUSH * scale,
        "wipe and clear": WIPE + WORLDS * CLEAR,
        "prepare": WORLDS * PREPARE * scale,
        "fetch: readiness": WORLDS * ready,
        "fetch: hubs": WORLDS * MONDAYS * READ_HUBS * scale,
        "fetch: accounts": count
        * seconds_per_account_monday(RATES[config.transport.concurrent_requests]),
        "audit: readiness and answer key": WORLDS * (ready + AUDIT_SETUP),
        "audit: computing": WORLDS * AUDIT_COMPUTING,
        "train and evaluate": TRAIN_EVALUATE,
    }
    return Projection(stages, count)


def stage_table(config: RunConfig) -> str:
    """The stages of the three cases as hours, a row a stage, as the design's table has them."""
    cases = {
        name: project(config, mules)
        for name, mules in (("T", WORLD_T), ("expected", EXPECTED), ("high", HIGH))
    }
    rows = [
        f"{stage}: " + ", ".join(f"{case.stages[stage] / HOUR:.2f}" for case in cases.values())
        for stage in cases["T"].stages
    ]
    totals = ", ".join(f"{case.total / HOUR:.2f}" for case in cases.values())
    counts = ", ".join(f"{case.account_mondays:,.0f}" for case in cases.values())
    return "\n".join([*rows, f"total: {totals}", f"account-Mondays fetched: {counts}"])


def test_the_run_projects_to_about_seven_hours_at_most() -> None:
    print(stage_table(DEFAULT_CONFIG))
    high = project(DEFAULT_CONFIG, HIGH)
    expected = project(DEFAULT_CONFIG, EXPECTED)
    assert high.total <= 7.05 * HOUR
    assert expected.total <= 6.8 * HOUR
    assert project(DEFAULT_CONFIG, WORLD_T).total < expected.total < high.total


def test_the_fetch_of_a_sparse_account_monday_is_two_seconds_at_two_requests_at_once() -> None:
    assert seconds_per_account_monday(RATE_1) == pytest.approx(FILL_SECONDS / FILL_ACCOUNTS)
    assert seconds_per_account_monday(RATE_2) == pytest.approx(1.99, abs=0.01)


def test_the_requests_at_once_are_the_ones_the_benchmark_measured() -> None:
    """The projection reads the rates at the setting's number of concurrent requests (the
    benchmark measured 1, 2 and 4); one at a time is the full design's, and costs 3.04 s a sparse
    account-Monday, 1.5 times as much as two."""
    one = project(replace(DEFAULT_CONFIG, transport=FULL_DESIGN_CONFIG.transport), EXPECTED)
    two = project(DEFAULT_CONFIG, EXPECTED)
    ratio = one.stages["fetch: accounts"] / two.stages["fetch: accounts"]
    assert ratio == pytest.approx(1.53, abs=0.01)
    unmeasured = replace(DEFAULT_CONFIG.transport, concurrent_requests=3)
    with pytest.raises(KeyError):
        project(replace(DEFAULT_CONFIG, transport=unmeasured), EXPECTED)


def test_the_audit_sample_lies_inside_the_fallback_samples_of_the_selection_and_test_worlds() -> (
    None
):
    sample = PER_PERSON * D_PEOPLE * MONDAYS / DEFAULT_CONFIG.data.fallback_weight
    # About 410 account-Mondays a world, which the audit's 300 lie inside by four standard
    # deviations, so the lowest 300 of the year's hashes are all in the fallback sample.
    assert 400 < sample < 420
    assert DEFAULT_CONFIG.audit.nonmule_sample + 4 * math.sqrt(sample) <= sample


def test_a_slower_mule_costs_what_its_account_mondays_cost_more() -> None:
    base = project(DEFAULT_CONFIG, EXPECTED)
    slow = project(DEFAULT_CONFIG, EXPECTED, mule_cost=1.5)
    extra = 0.5 * WORLDS * EXPECTED * seconds_per_account_monday(RATE_2)
    assert slow.total - base.total == pytest.approx(extra)
