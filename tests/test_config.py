"""The configuration sections: their defaults, which are the design's, and the values they
refuse."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, date, datetime
import hashlib

import pytest

from mule_pattern_learner.config import (
    DEFAULT_CONFIG,
    FULL_DESIGN_CONFIG,
    PROOF_OF_CONCEPT_ROTATIONS,
    ROTATIONS,
    WORLDS,
    AuditConfig,
    BenchConfig,
    CalendarConfig,
    DataConfig,
    ModelConfig,
    ParityConfig,
    Rotation,
    RunConfig,
    ScoreConfig,
    SubgraphConfig,
    TrainingConfig,
    TransportConfig,
    WorldConfig,
    WorldsConfig,
)


def _day(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, UTC)


def test_the_defaults_are_the_designs_settings_but_the_proof_of_concepts() -> None:
    """The full design's value of every setting the proof of concept changes is in
    FULL_DESIGN_CONFIG, which this test reads them from."""
    assert RunConfig() == DEFAULT_CONFIG
    calendar = DEFAULT_CONFIG.calendar
    assert calendar.export_start == date(2024, 1, 1)
    assert (calendar.history_days, calendar.scored_weeks, calendar.lookback_days) == (90, 52, 90)
    # 90 days of history, then 365 scored days.
    assert calendar.export_days == 455
    transport = DEFAULT_CONFIG.transport
    assert (transport.max_query_attempts, transport.max_outage_s) == (6, 900)
    # A query may run for 24 hours.
    assert transport.query_timeout_s == 86_400
    # 32 accounts a request, sized under half of the 64,000,000-byte response limit.
    assert (transport.accounts_per_request, transport.response_budget_bytes) == (32, 32_000_000)
    # A part of the cache holds 1,024 accounts of a scoring time.
    assert transport.accounts_per_part == 1_024
    subgraph = DEFAULT_CONFIG.subgraph
    assert (subgraph.lookback_days, subgraph.recent_payments) == (90, 256)
    caps = (subgraph.scored_payments, subgraph.counterparty_payments, subgraph.own_accounts)
    assert caps == (32, 16, 8)
    identity = (subgraph.devices_ips, subgraph.co_users, subgraph.community)
    assert identity == (8, 48, 64)
    assert (subgraph.hub_threshold, subgraph.crowd_threshold) == (200, 8)
    assert (subgraph.new_windows, subgraph.sharing_days) == ((7, 30), 30)
    data = FULL_DESIGN_CONFIG.data
    assert (data.per_positive, data.min_unlabelled, data.min_selection_identified) == (15, 500, 5)
    assert data.draw_key == b"mule-draw-v1\x00\x00\x00\x00" and len(data.draw_key) == 16
    assert (data.scoring, data.fallback_weight) == ("whole", 10)
    model = DEFAULT_CONFIG.model
    assert (model.hidden, model.heads, model.dropout) == (64, 4, 0.2)
    assert (model.layers, model.head_hidden, model.community) == (2, 32, True)
    training = FULL_DESIGN_CONFIG.training
    optimiser = (training.learning_rate, training.weight_decay, training.batch_size)
    assert optimiser == (0.001, 0.0001, 128)
    assert (training.max_epochs, training.patience, training.seeds) == (30, 5, (0, 1, 2, 3, 4))
    assert (training.prior_multiples, training.fixed_prior) == ((1, 3, 10), None)
    assert training.ablations is True
    assert (training.shuffle_seed, training.bootstrap_seed) == (0, 0)
    audit = DEFAULT_CONFIG.audit
    assert (audit.clock_gain_upper, audit.clock_invariance_margin) == (0.02, 0.05)
    assert audit.clock_min_mule_clusters == 30
    assert audit.separation_auc == 0.95
    near_pure = (audit.near_pure_tail, audit.near_pure_min_mule_share, audit.near_pure_purity)
    assert near_pure == (0.001, 0.01, 0.5)
    assert (audit.report_budget, audit.replicates, audit.interval) == (0.05, 1000, 0.95)
    assert (FULL_DESIGN_CONFIG.audit.nonmule_sample, audit.horizon_level) == (
        200_000,
        1 - 0.01 / 12,
    )
    # mule parity: 8 weeks and 200 accounts each, a quarter of them true mules, an account
    # alone checked against itself among 31 others.
    parity = DEFAULT_CONFIG.parity
    assert (parity.weeks, parity.accounts, parity.mule_share) == (8, 200, 0.25)
    assert (parity.together, parity.alone, parity.seed) == (32, 2, 0)
    assert parity.tolerance == 1e-12
    # mule bench: three weeks, 1, 2 and 4 concurrent requests, projected to 30,000 and
    # 300,000 people.
    bench = DEFAULT_CONFIG.bench
    assert (bench.weeks, bench.unlabelled, bench.positives) == (3, 64, 32)
    assert (bench.concurrency, bench.projected_people) == ((1, 2, 4), (30_000, 300_000))
    assert (bench.bootstrap_rows, bench.bootstrap_replicates, bench.handoff_rows) == (
        1_000_000,
        3,
        100_000,
    )
    assert DEFAULT_CONFIG.score == ScoreConfig((), (), "")


def test_the_proof_of_concept_settings() -> None:
    """The settings of docs/design.md, "Proof-of-concept settings (25 October 2026)", each with the
    full design's value beside it; going back to the full design is putting these back."""
    # Build each example once per world and reuse it (full design: False).
    assert DEFAULT_CONFIG.training.reuse_built_examples is True
    # Two requests of a query at once (full design: 1).
    assert DEFAULT_CONFIG.transport.concurrent_requests == 2
    # Sampled worlds: the training draw is 1 per positive and at least 8 a week (full design: 15
    # and 500), the selection and test worlds are scored on the fallback (whole), each sampled
    # account standing for 6,000 (10), and the audit samples 300 non-mule account-weeks
    # (200,000).
    data = DEFAULT_CONFIG.data
    assert (data.per_positive, data.min_unlabelled, data.min_selection_identified) == (1, 8, 5)
    assert (data.scoring, data.fallback_weight) == ("fallback", 6_000)
    assert DEFAULT_CONFIG.audit.nonmule_sample == 300
    # Train less: one seed, no ablation, the prior fixed at 1 and no grid, and the development
    # rotation D1, D2, D3 alone (full design: five seeds, the four ablations, the grid of 1, 3 and
    # 10 times, and all four rotations).
    training = DEFAULT_CONFIG.training
    assert training.seeds == (0,) and training.ablations is False and training.fixed_prior == 1.0
    assert PROOF_OF_CONCEPT_ROTATIONS == (Rotation("D1", "D2", "D3"),)
    assert DEFAULT_CONFIG.worlds.rotations == PROOF_OF_CONCEPT_ROTATIONS
    # A development world's failed or missing audit is reported and training goes on (full
    # design: it stops training).
    assert training.development_audit_stops is False


def _differences(left: RunConfig, right: RunConfig) -> set[str]:
    """The settings, as "section.setting", in which two configs differ."""
    return {
        f"{section.name}.{setting.name}"
        for section in fields(left)
        for setting in fields(getattr(left, section.name))
        if getattr(getattr(left, section.name), setting.name)
        != getattr(getattr(right, section.name), setting.name)
    }


def test_the_full_design_config_differs_from_the_defaults_in_the_proof_of_concepts_settings() -> (
    None
):
    """Every setting the proof of concept changes has the full design's value in
    FULL_DESIGN_CONFIG, and no other setting differs."""
    assert _differences(FULL_DESIGN_CONFIG, DEFAULT_CONFIG) == {
        "transport.concurrent_requests",
        "training.reuse_built_examples",
        "data.per_positive",
        "data.min_unlabelled",
        "data.scoring",
        "data.fallback_weight",
        "audit.nonmule_sample",
        "training.seeds",
        "training.fixed_prior",
        "training.ablations",
        "training.development_audit_stops",
        "worlds.rotations",
    }
    assert FULL_DESIGN_CONFIG.transport.concurrent_requests == 1
    assert FULL_DESIGN_CONFIG.training.reuse_built_examples is False
    assert (
        FULL_DESIGN_CONFIG.data.scoring == "whole" and FULL_DESIGN_CONFIG.data.fallback_weight == 10
    )
    assert FULL_DESIGN_CONFIG.training.seeds == (0, 1, 2, 3, 4)
    assert FULL_DESIGN_CONFIG.training.fixed_prior is None and FULL_DESIGN_CONFIG.training.ablations


def test_the_worlds_are_the_fixed_seeds_and_sizes() -> None:
    worlds = {world.name: (world.seed, world.people, world.kind) for world in WORLDS}
    assert worlds == {
        "A": (2001, 300_000, "reported"),
        "B": (2002, 300_000, "reported"),
        "C": (2003, 300_000, "reported"),
        "D1": (1001, 30_000, "development"),
        "D2": (1002, 30_000, "development"),
        "D3": (1003, 30_000, "development"),
        "T": (3001, 5_000, "test"),
        "P100": (4001, 100_000, "bench"),
    }
    assert FULL_DESIGN_CONFIG.worlds == WorldsConfig(WORLDS, ROTATIONS)
    assert DEFAULT_CONFIG.worlds == WorldsConfig(WORLDS, PROOF_OF_CONCEPT_ROTATIONS)


def test_each_reported_world_plays_each_role_once() -> None:
    kinds = {world.name: world.kind for world in WORLDS}
    reported = [rotation for rotation in ROTATIONS if kinds[rotation.train] == "reported"]
    assert reported == [Rotation("A", "B", "C"), Rotation("B", "C", "A"), Rotation("C", "A", "B")]
    for role in ("train", "select", "test"):
        assert sorted(getattr(rotation, role) for rotation in reported) == ["A", "B", "C"]
    # Development uses one rotation of its own worlds.
    assert [rotation for rotation in ROTATIONS if rotation not in reported] == [
        Rotation("D1", "D2", "D3")
    ]


def test_the_scored_weeks_start_on_the_52_mondays_from_day_90() -> None:
    starts = DEFAULT_CONFIG.calendar.week_starts()
    assert len(starts) == 52 and list(starts) == sorted(set(starts))
    # Day 90 is Sunday 2024-03-31, so the first Monday is day 91.
    assert _day(starts[0]) == datetime(2024, 4, 1, tzinfo=UTC)
    assert _day(starts[-1]) == datetime(2025, 3, 24, tzinfo=UTC)
    for start in starts:
        day = _day(start)
        assert day.weekday() == 0 and (day.hour, day.minute, day.second) == (0, 0, 0)
        assert start % 86_400_000 == 0
    assert all(b - a == 7 * 86_400_000 for a, b in zip(starts, starts[1:], strict=False))
    # The last week ends exactly when the export does, on day 455, 2025-03-31.
    assert _day(starts[-1] + 7 * 86_400_000) == datetime(2025, 3, 31, tzinfo=UTC)
    # When day history_days is itself a Monday, it is the first scored week's start.
    monday_start = CalendarConfig(export_start=date(2024, 1, 3), history_days=96)
    assert _day(monday_start.week_starts()[0]) == datetime(2024, 4, 8, tzinfo=UTC)


def _offset_seconds(seed: int, week: int) -> int:
    """A scoring time's offset into its week, worked out from the rule of the design apart from
    the calendar's code: the first 8 bytes of the blake2b digest of the key, little-endian,
    scaled to the seconds of a week."""
    digest = hashlib.blake2b(f"scoring-time|{seed}|{week}".encode()).digest()
    return int.from_bytes(digest[:8], "little") * 604_800 >> 64


def test_the_scoring_times_follow_the_rule_of_the_design_and_are_pinned() -> None:
    calendar = DEFAULT_CONFIG.calendar
    starts = calendar.week_starts()
    for seed in (0, 1001, 3001, 2**63 - 1):
        found = calendar.scoring_times(seed)
        assert found == tuple(
            start + _offset_seconds(seed, week) * 1_000 for week, start in enumerate(starts)
        )
    # The first four offsets of two seeds, in seconds: a change of the hash fails here.
    for seed, offsets in (
        (1001, [499_489, 439_790, 59_519, 458_610]),
        (1002, [544_581, 110_372, 318_891, 440_094]),
    ):
        found = calendar.scoring_times(seed)[:4]
        assert [(t - start) // 1_000 for t, start in zip(found, starts, strict=False)] == offsets


def test_the_scoring_times_are_one_whole_second_inside_each_week() -> None:
    calendar = DEFAULT_CONFIG.calendar
    starts = calendar.week_starts()
    found = calendar.scoring_times(1001)
    assert len(found) == len(starts) == 52
    for start, t_ms in zip(starts, found, strict=True):
        assert start <= t_ms < start + 7 * 86_400_000
        assert t_ms % 1_000 == 0
    assert list(found) == sorted(set(found))
    # The offsets cover the week: they are not all on one weekday or hour.
    assert len({(t - start) // 86_400_000 for t, start in zip(found, starts, strict=True)}) > 4


def test_the_scoring_times_are_the_seeds_alone() -> None:
    calendar = DEFAULT_CONFIG.calendar
    assert calendar.scoring_times(1001) == calendar.scoring_times(1001)
    assert calendar.scoring_times(1001) != calendar.scoring_times(1002)
    # A shorter calendar draws the first weeks the longer one does, since week k's offset is the
    # hash of its own key.
    assert replace(calendar, scored_weeks=3).scoring_times(1001) == calendar.scoring_times(1001)[:3]


@pytest.mark.parametrize("seed", [-1, 2**63, True, 1.5, "1001"])
def test_the_scoring_times_refuse_a_seed_that_is_no_world_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        DEFAULT_CONFIG.calendar.scoring_times(seed)  # pyright: ignore[reportArgumentType]


def test_the_last_scored_week_must_end_by_the_end_of_the_export() -> None:
    # 52 weeks end exactly on day 455 of the default calendar, so one more does not fit.
    assert CalendarConfig(scored_weeks=52).week_starts()[-1] + 7 * 86_400_000 == (
        (date(2025, 3, 31) - date(1970, 1, 1)).days * 86_400_000
    )
    # Starting on day 92, the Monday of week 52 is day 455, inside the 457 days of the export,
    # but its week ends on day 462, after it.
    with pytest.raises(ValueError, match=r"calendar.scored_weeks \(52\) do not fit"):
        CalendarConfig(history_days=92)
    # With one week fewer, the last week ends on day 455, inside the export.
    assert (
        CalendarConfig(history_days=92, scored_weeks=51).week_starts()[-1]
        == ((date(2024, 1, 1) - date(1970, 1, 1)).days + 448) * 86_400_000
    )


def test_sections_refuse_values_outside_their_ranges() -> None:
    bad: list[tuple[Callable[[], object], str]] = [
        (lambda: CalendarConfig(export_start=datetime(2024, 1, 1)), "calendar.export_start"),
        (lambda: CalendarConfig(export_start=date(1970, 1, 1)), "calendar.export_start"),
        (lambda: CalendarConfig(history_days=0), "calendar.history_days"),
        (lambda: CalendarConfig(scored_weeks=0), "calendar.scored_weeks"),
        (lambda: CalendarConfig(scored_weeks=53), "calendar.scored_weeks"),
        # The lookback must exceed the 30-day "new" window.
        (lambda: CalendarConfig(lookback_days=30), "calendar.lookback_days"),
        (lambda: CalendarConfig(lookback_days=120), "calendar.history_days"),
        (lambda: RunConfig(calendar="2024"), "calendar must be"),  # pyright: ignore[reportArgumentType]
        (lambda: WorldConfig("A b", 1, 1, "reported"), "world.name"),
        (lambda: WorldConfig("A", -1, 1, "reported"), "world A.seed"),
        (lambda: WorldConfig("A", 1, 0, "reported"), "world A.people"),
        (lambda: WorldConfig("A", 1, 1, "spare"), "world A.kind"),
        (lambda: Rotation("A", "B", "A"), "three distinct worlds"),
        (lambda: WorldsConfig(worlds=[*WORLDS]), "worlds.worlds must be a tuple"),  # pyright: ignore[reportArgumentType]
        (lambda: WorldsConfig(worlds=(*WORLDS, WORLDS[0])), "worlds.worlds repeats"),
        (
            lambda: WorldsConfig(worlds=(*WORLDS, WorldConfig("E", 2001, 1, "development"))),
            "own name and seed",
        ),
        (lambda: WorldsConfig(rotations=(Rotation("A", "B", "Z"),)), "unknown worlds"),
        (lambda: WorldsConfig(rotations=(Rotation("A", "B", "D1"),)), "one kind"),
        (lambda: WorldsConfig(rotations=(Rotation("T", "P100", "A"),)), "one kind"),
        (lambda: RunConfig(worlds=WORLDS), "worlds must be"),  # pyright: ignore[reportArgumentType]
        (lambda: TransportConfig(max_query_attempts=0), "transport.max_query_attempts"),
        (lambda: TransportConfig(max_query_attempts=21), "transport.max_query_attempts"),
        (lambda: TransportConfig(max_query_attempts=True), "transport.max_query_attempts"),
        (lambda: TransportConfig(max_outage_s=-1), "transport.max_outage_s"),
        (lambda: TransportConfig(max_outage_s=1.5), "transport.max_outage_s"),  # pyright: ignore[reportArgumentType]
        (lambda: TransportConfig(query_timeout_s=0), "transport.query_timeout_s"),
        (lambda: TransportConfig(query_timeout_s=86_401), "transport.query_timeout_s"),
        (lambda: TransportConfig(accounts_per_request=0), "transport.accounts_per_request"),
        (lambda: TransportConfig(concurrent_requests=0), "transport.concurrent_requests"),
        (lambda: TransportConfig(concurrent_requests=True), "transport.concurrent_requests"),  # pyright: ignore[reportArgumentType]
        (lambda: TransportConfig(accounts_per_part=0), "transport.accounts_per_part"),
        (lambda: TransportConfig(accounts_per_part=100_001), "transport.accounts_per_part"),
        # Half of the response limit at most.
        (
            lambda: TransportConfig(response_budget_bytes=32_000_001),
            "transport.response_budget_bytes",
        ),
        (lambda: RunConfig(transport="fast"), "transport must be"),  # pyright: ignore[reportArgumentType]
        (lambda: SubgraphConfig(lookback_days=30), "subgraph.lookback_days"),
        (lambda: SubgraphConfig(recent_payments=0), "subgraph.recent_payments"),
        (lambda: SubgraphConfig(scored_payments=1_025), "subgraph.scored_payments"),
        (lambda: SubgraphConfig(co_users=0), "subgraph.co_users"),
        (lambda: SubgraphConfig(hub_threshold=0), "subgraph.hub_threshold"),
        (lambda: SubgraphConfig(crowd_threshold=200), "subgraph.crowd_threshold"),
        (lambda: SubgraphConfig(new_windows=()), "subgraph.new_windows"),
        (lambda: SubgraphConfig(new_windows=(30, 7)), "subgraph.new_windows"),
        (lambda: SubgraphConfig(new_windows=(7, 90)), "subgraph.new_windows"),
        (lambda: SubgraphConfig(sharing_days=90), "subgraph.sharing_days"),
        # The current test is fixed at 30 days, as the queries write it.
        (lambda: SubgraphConfig(sharing_days=20), r"subgraph.sharing_days must be .* \[30, 30\]"),
        (lambda: RunConfig(subgraph="wide"), "subgraph must be"),  # pyright: ignore[reportArgumentType]
        (
            lambda: RunConfig(subgraph=SubgraphConfig(lookback_days=60)),
            "must equal subgraph.lookback_days",
        ),
        (lambda: DataConfig(per_positive=0), "data.per_positive"),
        (lambda: DataConfig(min_unlabelled=0), "data.min_unlabelled"),
        (lambda: DataConfig(draw_key="mule-draw-v1"), "data.draw_key"),  # pyright: ignore[reportArgumentType]
        (lambda: DataConfig(draw_key=b""), "data.draw_key"),
        (lambda: DataConfig(min_selection_identified=0), "data.min_selection_identified"),
        (lambda: DataConfig(scoring="quarter"), "data.scoring"),
        (lambda: DataConfig(fallback_weight=0), "data.fallback_weight"),
        (lambda: RunConfig(data=None), "data must be"),  # pyright: ignore[reportArgumentType]
        (lambda: ModelConfig(hidden=0), "model.hidden"),
        (lambda: ModelConfig(heads=0), "model.heads"),
        (lambda: ModelConfig(hidden=64, heads=5), "split evenly"),
        (lambda: ModelConfig(dropout=1.0), "model.dropout"),
        (lambda: ModelConfig(dropout=True), "model.dropout"),  # pyright: ignore[reportArgumentType]
        (lambda: ModelConfig(layers=0), "model.layers"),
        (lambda: ModelConfig(head_hidden=0), "model.head_hidden"),
        (lambda: ModelConfig(community=1), "model.community"),  # pyright: ignore[reportArgumentType]
        (lambda: RunConfig(model="wide"), "model must be"),  # pyright: ignore[reportArgumentType]
        (lambda: TrainingConfig(learning_rate=0), "training.learning_rate"),
        (lambda: TrainingConfig(weight_decay=-1e-4), "training.weight_decay"),
        (lambda: TrainingConfig(batch_size=0), "training.batch_size"),
        (lambda: TrainingConfig(max_epochs=0), "training.max_epochs"),
        (lambda: TrainingConfig(patience=0), "training.patience"),
        (lambda: TrainingConfig(seeds=()), "training.seeds"),
        (lambda: TrainingConfig(seeds=5), "training.seeds"),  # pyright: ignore[reportArgumentType]
        (lambda: TrainingConfig(seeds=(0, 0)), "training.seeds repeats"),
        (lambda: TrainingConfig(seeds=(-1,)), "training.seeds"),
        (lambda: TrainingConfig(shuffle_seed=-1), "training.shuffle_seed"),
        (lambda: TrainingConfig(shuffle_seed=True), "training.shuffle_seed"),  # pyright: ignore[reportArgumentType]
        (lambda: TrainingConfig(bootstrap_seed=-1), "training.bootstrap_seed"),
        (lambda: TrainingConfig(bootstrap_seed=0.5), "training.bootstrap_seed"),  # pyright: ignore[reportArgumentType]
        (lambda: TrainingConfig(prior_multiples=()), "training.prior_multiples"),
        (lambda: TrainingConfig(prior_multiples=(1, 0)), "training.prior_multiples"),
        (lambda: TrainingConfig(prior_multiples=[1, 3]), "training.prior_multiples"),  # pyright: ignore[reportArgumentType]
        (lambda: TrainingConfig(fixed_prior=0), "training.fixed_prior"),
        (lambda: TrainingConfig(ablations=0), "training.ablations"),  # pyright: ignore[reportArgumentType]
        (lambda: TrainingConfig(reuse_built_examples=1), "training.reuse_built_examples"),  # pyright: ignore[reportArgumentType]
        (lambda: TrainingConfig(development_audit_stops=1), "training.development_audit_stops"),  # pyright: ignore[reportArgumentType]
        (lambda: RunConfig(training=0), "training must be"),  # pyright: ignore[reportArgumentType]
        (lambda: AuditConfig(clock_gain_upper=0), "audit.clock_gain_upper"),
        (lambda: AuditConfig(clock_gain_upper=1), "audit.clock_gain_upper"),
        (lambda: AuditConfig(clock_gain_upper=True), "audit.clock_gain_upper"),  # pyright: ignore[reportArgumentType]
        (lambda: AuditConfig(clock_invariance_margin=0), "audit.clock_invariance_margin"),
        (lambda: AuditConfig(clock_invariance_margin=1), "audit.clock_invariance_margin"),
        (lambda: AuditConfig(clock_invariance_margin="0.05"), "audit.clock_invariance_margin"),  # pyright: ignore[reportArgumentType]
        (lambda: AuditConfig(clock_min_mule_clusters=0), "audit.clock_min_mule_clusters"),
        (lambda: AuditConfig(clock_min_mule_clusters=-30), "audit.clock_min_mule_clusters"),
        (lambda: AuditConfig(clock_min_mule_clusters=30.0), "audit.clock_min_mule_clusters"),  # pyright: ignore[reportArgumentType]
        (lambda: AuditConfig(clock_min_mule_clusters=True), "audit.clock_min_mule_clusters"),  # pyright: ignore[reportArgumentType]
        (lambda: AuditConfig(separation_auc=1.0), "audit.separation_auc"),
        (lambda: AuditConfig(near_pure_tail=0.5), "audit.near_pure_tail"),
        (lambda: AuditConfig(near_pure_min_mule_share=0), "audit.near_pure_min_mule_share"),
        (lambda: AuditConfig(near_pure_purity=1), "audit.near_pure_purity"),
        (lambda: AuditConfig(report_budget=0), "audit.report_budget"),
        (lambda: AuditConfig(replicates=0), "audit.replicates"),
        (lambda: AuditConfig(interval=1), "audit.interval"),
        (lambda: AuditConfig(nonmule_sample=0), "audit.nonmule_sample"),
        (lambda: AuditConfig(horizon_level=1), "audit.horizon_level"),
        (lambda: RunConfig(audit=0), "audit must be"),  # pyright: ignore[reportArgumentType]
        (lambda: ParityConfig(weeks=0), "parity.weeks"),
        (lambda: ParityConfig(accounts=0), "parity.accounts"),
        (lambda: ParityConfig(mule_share=0), "parity.mule_share"),
        (lambda: ParityConfig(mule_share=1), "parity.mule_share"),
        # An account is checked alone against itself among others, so a request holds two.
        (lambda: ParityConfig(together=1), "parity.together"),
        (lambda: ParityConfig(alone=0), "parity.alone"),
        (lambda: ParityConfig(seed=-1), "parity.seed"),
        (lambda: ParityConfig(tolerance=0), "parity.tolerance"),
        (lambda: ParityConfig(tolerance=1), "parity.tolerance"),
        (lambda: RunConfig(parity=0), "parity must be"),  # pyright: ignore[reportArgumentType]
        (lambda: BenchConfig(weeks=0), "bench.weeks"),
        (lambda: BenchConfig(unlabelled=0), "bench.unlabelled"),
        (lambda: BenchConfig(positives=0), "bench.positives"),
        (lambda: BenchConfig(concurrency=()), "bench.concurrency"),
        (lambda: BenchConfig(concurrency=(1, 1)), "bench.concurrency repeats"),
        (lambda: BenchConfig(concurrency=(0,)), "bench.concurrency"),
        (lambda: BenchConfig(concurrency=[1, 2]), "bench.concurrency"),  # pyright: ignore[reportArgumentType]
        (lambda: BenchConfig(bootstrap_rows=0), "bench.bootstrap_rows"),
        (lambda: BenchConfig(bootstrap_replicates=0), "bench.bootstrap_replicates"),
        (lambda: BenchConfig(handoff_rows=1), "bench.handoff_rows"),
        (lambda: BenchConfig(projected_people=()), "bench.projected_people"),
        (lambda: BenchConfig(projected_people=(0,)), "bench.projected_people"),
        (lambda: RunConfig(bench=0), "bench must be"),  # pyright: ignore[reportArgumentType]
        (lambda: ScoreConfig(accounts=("a", "a")), "score.accounts repeats"),
        (lambda: ScoreConfig(accounts=("",)), "score.accounts"),
        (lambda: ScoreConfig(dates=(date(2024, 4, 2), date(2024, 4, 2))), "score.dates repeats"),
        (lambda: ScoreConfig(dates=("2024-04-01",)), "score.dates must be a date"),  # pyright: ignore[reportArgumentType]
        (lambda: ScoreConfig(run=1), "score.run must be a str"),  # pyright: ignore[reportArgumentType]
        (lambda: RunConfig(score=()), "score must be"),  # pyright: ignore[reportArgumentType]
    ]
    for build, name in bad:
        with pytest.raises(ValueError, match=name):
            build()


def test_the_settings_a_user_may_change_are_accepted() -> None:
    assert TrainingConfig(fixed_prior=1).fixed_prior == 1
    assert DataConfig(scoring="fallback").scoring == "fallback"
    assert ScoreConfig(("a_01",), (date(2024, 4, 1),)).dates == (date(2024, 4, 1),)
    # Any date is scored, at 00:00 UTC, a Tuesday or a Sunday as well as a Monday.
    days = (date(2024, 4, 2), date(2024, 4, 7))
    assert ScoreConfig(("a_01",), days).dates == days
    assert not ModelConfig(community=False).community


def test_settings_are_frozen_and_change_by_replace() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_CONFIG.transport.max_outage_s = 0  # pyright: ignore[reportAttributeAccessIssue]
    changed = replace(DEFAULT_CONFIG, transport=TransportConfig(max_outage_s=0))
    assert changed.transport.max_outage_s == 0 and DEFAULT_CONFIG.transport.max_outage_s == 900
