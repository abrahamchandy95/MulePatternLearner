"""The configuration sections: their defaults, and the values they refuse."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, RunConfig, TransportConfig


def test_the_defaults_are_the_built_in_settings() -> None:
    assert RunConfig() == DEFAULT_CONFIG
    transport = DEFAULT_CONFIG.transport
    assert (transport.max_query_attempts, transport.max_outage_s) == (6, 900)


def test_sections_refuse_values_outside_their_ranges() -> None:
    bad: list[tuple[Callable[[], object], str]] = [
        (lambda: TransportConfig(max_query_attempts=0), "transport.max_query_attempts"),
        (lambda: TransportConfig(max_query_attempts=21), "transport.max_query_attempts"),
        (lambda: TransportConfig(max_query_attempts=True), "transport.max_query_attempts"),
        (lambda: TransportConfig(max_outage_s=-1), "transport.max_outage_s"),
        (lambda: TransportConfig(max_outage_s=1.5), "transport.max_outage_s"),  # pyright: ignore[reportArgumentType]
        (lambda: RunConfig(transport="fast"), "transport must be"),  # pyright: ignore[reportArgumentType]
    ]
    for build, name in bad:
        with pytest.raises(ValueError, match=name):
            build()


def test_settings_are_frozen_and_change_by_replace() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_CONFIG.transport.max_outage_s = 0  # pyright: ignore[reportAttributeAccessIssue]
    changed = replace(DEFAULT_CONFIG, transport=TransportConfig(max_outage_s=0))
    assert changed.transport.max_outage_s == 0 and DEFAULT_CONFIG.transport.max_outage_s == 900
