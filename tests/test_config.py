"""The configuration sections: their defaults, which are the design's, and the values they
refuse."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime

import pytest

from mule_pattern_learner.config import (
    DEFAULT_CONFIG,
    ROTATIONS,
    WORLDS,
    AuditConfig,
    CalendarConfig,
    DataConfig,
    ModelConfig,
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


def test_the_defaults_are_the_designs_settings() -> None:
    assert RunConfig() == DEFAULT_CONFIG
    calendar = DEFAULT_CONFIG.calendar
    assert calendar.export_start == date(2024, 1, 1)
    assert (calendar.history_days, calendar.scored_mondays, calendar.lookback_days) == (90, 52, 90)
    # 90 days of history, then 365 scored days.
    assert calendar.export_days == 455
    transport = DEFAULT_CONFIG.transport
    assert (transport.max_query_attempts, transport.max_outage_s) == (6, 900)
    # 32 accounts a request, sized under half of the 64,000,000-byte response limit.
    assert (transport.accounts_per_request, transport.response_budget_bytes) == (32, 32_000_000)
    subgraph = DEFAULT_CONFIG.subgraph
    assert (subgraph.lookback_days, subgraph.recent_payments) == (90, 256)
    caps = (subgraph.scored_payments, subgraph.counterparty_payments, subgraph.own_accounts)
    assert caps == (32, 16, 8)
    identity = (subgraph.devices_ips, subgraph.co_users, subgraph.community)
    assert identity == (8, 48, 64)
    assert (subgraph.hub_threshold, subgraph.crowd_threshold) == (200, 8)
    assert (subgraph.new_windows, subgraph.sharing_days) == ((7, 30), 30)
    data = DEFAULT_CONFIG.data
    assert (data.per_positive, data.min_unlabelled, data.min_selection_identified) == (15, 500, 5)
    assert data.draw_key == b"mule-draw-v1\x00\x00\x00\x00" and len(data.draw_key) == 16
    assert (data.scoring, data.fallback_weight) == ("whole", 10)
    model = DEFAULT_CONFIG.model
    assert (model.hidden, model.heads, model.dropout) == (64, 4, 0.2)
    assert (model.layers, model.head_hidden, model.community) == (2, 32, True)
    training = DEFAULT_CONFIG.training
    optimiser = (training.learning_rate, training.weight_decay, training.batch_size)
    assert optimiser == (0.001, 0.0001, 128)
    assert (training.max_epochs, training.patience, training.seeds) == (30, 5, (0, 1, 2, 3, 4))
    assert (training.prior_multiples, training.fixed_prior) == ((1, 3, 10), None)
    audit = DEFAULT_CONFIG.audit
    assert (audit.clock_probe_upper, audit.separation_auc) == (0.55, 0.95)
    near_pure = (audit.near_pure_tail, audit.near_pure_min_mule_share, audit.near_pure_purity)
    assert near_pure == (0.001, 0.01, 0.5)
    assert (audit.report_budget, audit.replicates, audit.interval) == (0.05, 1000, 0.95)
    assert (audit.nonmule_sample, audit.horizon_level) == (200_000, 1 - 0.01 / 12)
    assert DEFAULT_CONFIG.score == ScoreConfig((), ())


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
    assert DEFAULT_CONFIG.worlds == WorldsConfig(WORLDS, ROTATIONS)


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


def test_the_scored_mondays_are_the_52_mondays_from_day_90() -> None:
    mondays = DEFAULT_CONFIG.calendar.mondays()
    assert len(mondays) == 52 and list(mondays) == sorted(set(mondays))
    # Day 90 is Sunday 2024-03-31, so the first Monday is day 91.
    assert _day(mondays[0]) == datetime(2024, 4, 1, tzinfo=UTC)
    assert _day(mondays[-1]) == datetime(2025, 3, 24, tzinfo=UTC)
    for monday in mondays:
        day = _day(monday)
        assert day.weekday() == 0 and (day.hour, day.minute, day.second) == (0, 0, 0)
        assert monday % 86_400_000 == 0
    assert all(b - a == 7 * 86_400_000 for a, b in zip(mondays, mondays[1:], strict=False))
    # The last Monday falls before the export ends on day 455, 2025-03-31.
    assert _day(mondays[-1]) < datetime(2025, 3, 31, tzinfo=UTC)
    # When day history_days is itself a Monday, it is the first scored Monday.
    monday_start = CalendarConfig(export_start=date(2024, 1, 3), history_days=96)
    assert _day(monday_start.mondays()[0]) == datetime(2024, 4, 8, tzinfo=UTC)


def test_sections_refuse_values_outside_their_ranges() -> None:
    bad: list[tuple[Callable[[], object], str]] = [
        (lambda: CalendarConfig(export_start=datetime(2024, 1, 1)), "calendar.export_start"),
        (lambda: CalendarConfig(export_start=date(1970, 1, 1)), "calendar.export_start"),
        (lambda: CalendarConfig(history_days=0), "calendar.history_days"),
        (lambda: CalendarConfig(scored_mondays=0), "calendar.scored_mondays"),
        (lambda: CalendarConfig(scored_mondays=53), "calendar.scored_mondays"),
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
        (lambda: TransportConfig(accounts_per_request=0), "transport.accounts_per_request"),
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
        (lambda: TrainingConfig(prior_multiples=()), "training.prior_multiples"),
        (lambda: TrainingConfig(prior_multiples=(1, 0)), "training.prior_multiples"),
        (lambda: TrainingConfig(prior_multiples=[1, 3]), "training.prior_multiples"),  # pyright: ignore[reportArgumentType]
        (lambda: TrainingConfig(fixed_prior=0), "training.fixed_prior"),
        (lambda: RunConfig(training=0), "training must be"),  # pyright: ignore[reportArgumentType]
        (lambda: AuditConfig(clock_probe_upper=0.5), "audit.clock_probe_upper"),
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
        (lambda: ScoreConfig(accounts=("a", "a")), "score.accounts repeats"),
        (lambda: ScoreConfig(accounts=("",)), "score.accounts"),
        (lambda: ScoreConfig(dates=(date(2024, 4, 2),)), "score.dates must be Mondays"),
        (lambda: ScoreConfig(dates=("2024-04-01",)), "score.dates must be a date"),  # pyright: ignore[reportArgumentType]
        (lambda: RunConfig(score=()), "score must be"),  # pyright: ignore[reportArgumentType]
    ]
    for build, name in bad:
        with pytest.raises(ValueError, match=name):
            build()


def test_the_settings_a_user_may_change_are_accepted() -> None:
    assert TrainingConfig(fixed_prior=1).fixed_prior == 1
    assert DataConfig(scoring="fallback").scoring == "fallback"
    assert ScoreConfig(("a_01",), (date(2024, 4, 1),)).dates == (date(2024, 4, 1),)
    assert not ModelConfig(community=False).community


def test_settings_are_frozen_and_change_by_replace() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_CONFIG.transport.max_outage_s = 0  # pyright: ignore[reportAttributeAccessIssue]
    changed = replace(DEFAULT_CONFIG, transport=TransportConfig(max_outage_s=0))
    assert changed.transport.max_outage_s == 0 and DEFAULT_CONFIG.transport.max_outage_s == 900
