"""The configuration sections: their defaults, and the values they refuse."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace

import pytest

from mule_pattern_learner.config import (
    DEFAULT_CONFIG,
    ModelConfig,
    RunConfig,
    TrainingConfig,
    TransportConfig,
)


def test_the_defaults_are_the_built_in_settings() -> None:
    assert RunConfig() == DEFAULT_CONFIG
    transport = DEFAULT_CONFIG.transport
    assert (transport.max_query_attempts, transport.max_outage_s) == (6, 900)
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
        (lambda: RunConfig(transport="fast"), "transport must be"),  # pyright: ignore[reportArgumentType]
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
