"""The configuration sections: their defaults, and the values they refuse."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime

import pytest

from mule_pattern_learner.config import (
    DEFAULT_CONFIG,
    DataConfig,
    ModelConfig,
    RunConfig,
    SubgraphConfig,
    TrainingConfig,
    TransportConfig,
)


def test_the_defaults_are_the_built_in_settings() -> None:
    assert RunConfig() == DEFAULT_CONFIG
    transport = DEFAULT_CONFIG.transport
    assert (transport.max_query_attempts, transport.max_outage_s) == (6, 900)
    assert transport.seeds_per_request == 32
    # The design's defaults for the subgraph query and the date tables.
    subgraph = DEFAULT_CONFIG.subgraph
    caps = (subgraph.seed_payments, subgraph.other_payments, subgraph.owners, subgraph.siblings)
    identity = (subgraph.enrolled_devices, subgraph.enrolled_ips)
    sent = (subgraph.sent_devices, subgraph.sent_ips)
    assert (caps, identity, sent) == ((32, 16, 1, 8), (8, 8), (8, 8))
    assert (subgraph.reverse_accounts, subgraph.identity_accounts) == (16, 48)
    assert (subgraph.community_parties, subgraph.recent_payments) == (64, 256)
    assert (subgraph.hub_threshold, subgraph.observed_threshold) == (200, 8)
    assert subgraph.lookback_days == 365
    assert not hasattr(subgraph, "addresses")
    # The design's windows, gap and unlabelled count, and a scored sample and seed it leaves.
    data = DEFAULT_CONFIG.data
    assert (data.training, data.validation, data.test) == (
        (date(2024, 1, 1), date(2024, 6, 30)),
        (date(2024, 7, 1), date(2024, 9, 30)),
        (date(2024, 10, 1), date(2024, 12, 31)),
    )
    assert (data.maturity_gap_days, data.unlabelled_per_positive) == (60, 15)
    assert (data.scored_sample, data.seed) == (2000, 0)
    # The design's model and training defaults, and a batch size it leaves.
    model = DEFAULT_CONFIG.model
    assert (model.hidden, model.heads, model.dropout, model.label_mask) == (64, 4, 0.2, 0.5)
    training = DEFAULT_CONFIG.training
    optimiser = (training.learning_rate, training.weight_decay)
    assert (optimiser, training.max_epochs, training.patience) == ((0.001, 0.0001), 30, 5)
    assert (training.priors, training.batch_size, training.seeds) == ((0.001, 0.003, 0.01), 128, 5)


def test_sections_refuse_values_outside_their_ranges() -> None:
    bad: list[tuple[Callable[[], object], str]] = [
        (lambda: TransportConfig(max_query_attempts=0), "transport.max_query_attempts"),
        (lambda: TransportConfig(max_query_attempts=21), "transport.max_query_attempts"),
        (lambda: TransportConfig(max_query_attempts=True), "transport.max_query_attempts"),
        (lambda: TransportConfig(max_outage_s=-1), "transport.max_outage_s"),
        (lambda: TransportConfig(max_outage_s=1.5), "transport.max_outage_s"),  # pyright: ignore[reportArgumentType]
        (lambda: TransportConfig(seeds_per_request=0), "transport.seeds_per_request"),
        (lambda: TransportConfig(seeds_per_request=1_025), "transport.seeds_per_request"),
        (lambda: RunConfig(transport="fast"), "transport must be"),  # pyright: ignore[reportArgumentType]
        (lambda: SubgraphConfig(seed_payments=0), "subgraph.seed_payments"),
        (lambda: SubgraphConfig(owners=0), "subgraph.owners"),
        (lambda: SubgraphConfig(identity_accounts=2_000), "subgraph.identity_accounts"),
        (lambda: SubgraphConfig(community_parties=0), "subgraph.community_parties"),
        (lambda: SubgraphConfig(recent_payments=1_025), "subgraph.recent_payments"),
        (lambda: SubgraphConfig(hub_threshold=0), "subgraph.hub_threshold"),
        (lambda: SubgraphConfig(observed_threshold=0), "subgraph.observed_threshold"),
        (lambda: SubgraphConfig(observed_threshold=True), "subgraph.observed_threshold"),
        (lambda: SubgraphConfig(lookback_days=29), "subgraph.lookback_days"),
        (lambda: SubgraphConfig(lookback_days=3_651), "subgraph.lookback_days"),
        (lambda: RunConfig(subgraph="wide"), "subgraph must be"),  # pyright: ignore[reportArgumentType]
        (lambda: DataConfig(training=(date(2024, 6, 30), date(2024, 1, 1))), "data.training"),
        (lambda: DataConfig(test=(date(2024, 10, 1),)), "data.test"),  # pyright: ignore[reportArgumentType]
        (
            lambda: DataConfig(validation=(datetime(2024, 7, 1), date(2024, 9, 30))),
            "data.validation",
        ),
        (lambda: DataConfig(validation=(date(2024, 6, 30), date(2024, 9, 30))), "overlap"),
        (lambda: DataConfig(test=(date(2024, 9, 30), date(2024, 12, 31))), "overlap"),
        (lambda: DataConfig(maturity_gap_days=-1), "data.maturity_gap_days"),
        (lambda: DataConfig(maturity_gap_days=366), "data.maturity_gap_days"),
        (lambda: DataConfig(unlabelled_per_positive=0), "data.unlabelled_per_positive"),
        (lambda: DataConfig(scored_sample=0), "data.scored_sample"),
        (lambda: DataConfig(seed=-1), "data.seed"),
        (lambda: RunConfig(data=None), "data must be"),  # pyright: ignore[reportArgumentType]
        (lambda: ModelConfig(hidden=0), "model.hidden"),
        (lambda: ModelConfig(heads=0), "model.heads"),
        (lambda: ModelConfig(hidden=64, heads=5), "split evenly"),
        (lambda: ModelConfig(dropout=1.0), "model.dropout"),
        (lambda: ModelConfig(dropout=True), "model.dropout"),  # pyright: ignore[reportArgumentType]
        (lambda: ModelConfig(label_mask=-0.1), "model.label_mask"),
        (lambda: RunConfig(model="wide"), "model must be"),  # pyright: ignore[reportArgumentType]
        (lambda: TrainingConfig(learning_rate=0), "training.learning_rate"),
        (lambda: TrainingConfig(weight_decay=-1e-4), "training.weight_decay"),
        (lambda: TrainingConfig(max_epochs=0), "training.max_epochs"),
        (lambda: TrainingConfig(patience=0), "training.patience"),
        (lambda: TrainingConfig(priors=()), "training.priors"),
        (lambda: TrainingConfig(priors=[0.01]), "training.priors"),  # pyright: ignore[reportArgumentType]
        (lambda: TrainingConfig(priors=(0.01, 1.0)), "training.priors"),
        (lambda: TrainingConfig(priors=(0.01, 0.01)), "repeats"),
        (lambda: TrainingConfig(batch_size=0), "training.batch_size"),
        (lambda: TrainingConfig(seeds=0), "training.seeds"),
        (lambda: TrainingConfig(seeds=1_001), "training.seeds"),
        (lambda: RunConfig(training=0), "training must be"),  # pyright: ignore[reportArgumentType]
    ]
    for build, name in bad:
        with pytest.raises(ValueError, match=name):
            build()


def test_settings_are_frozen_and_change_by_replace() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_CONFIG.transport.max_outage_s = 0  # pyright: ignore[reportAttributeAccessIssue]
    changed = replace(DEFAULT_CONFIG, transport=TransportConfig(max_outage_s=0))
    assert changed.transport.max_outage_s == 0 and DEFAULT_CONFIG.transport.max_outage_s == 900
