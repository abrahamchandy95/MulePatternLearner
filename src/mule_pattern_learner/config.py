"""The settings as frozen dataclasses; DEFAULT_CONFIG holds every default.

RunConfig has one section per concern, and each component receives only its own
section. A section checks its values when it is built, so a bad setting fails before any
database work. `dataclasses.replace` changes a setting.
"""

from __future__ import annotations

from dataclasses import dataclass

# The ranges of the transport settings; the executor checks the ones it is given too.
QUERY_ATTEMPTS = range(1, 21)
OUTAGE_SECONDS = range(0, 86_401)


def _integer(name: str, value: object, bounds: range) -> int:
    """value when it is an integer within bounds (a boolean is not one)."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in bounds:
        raise ValueError(
            f"{name} must be an integer in [{bounds.start}, {bounds[-1]}], got {value!r}"
        )
    return value


def _instance(name: str, value: object, kind: type) -> None:
    if not isinstance(value, kind):
        raise ValueError(f"{name} must be a {kind.__name__}, got {value!r}")


@dataclass(frozen=True)
class TransportConfig:
    """The retry budgets of every TigerGraph request. Changing them never changes results."""

    # Attempts of one request that count toward its retries (tigergraph.executor).
    max_query_attempts: int = 6
    # Wall-clock seconds one operation keeps retrying while TigerGraph is unavailable.
    max_outage_s: int = 900

    def __post_init__(self) -> None:
        _integer("transport.max_query_attempts", self.max_query_attempts, QUERY_ATTEMPTS)
        _integer("transport.max_outage_s", self.max_outage_s, OUTAGE_SECONDS)


@dataclass(frozen=True)
class RunConfig:
    """Every setting; the defaults are DEFAULT_CONFIG."""

    transport: TransportConfig = TransportConfig()

    def __post_init__(self) -> None:
        _instance("transport", self.transport, TransportConfig)


DEFAULT_CONFIG = RunConfig()
