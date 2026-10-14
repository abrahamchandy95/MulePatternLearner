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
# The ranges of the subgraph settings; fetch_subgraphs refuses a lookback outside its range.
CAPS = range(1, 1_025)
HUB_THRESHOLDS = range(1, 1_000_001)
LOOKBACK_DAYS = range(30, 3_651)


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
class SubgraphConfig:
    """The caps and windows of the subgraph query (docs/design.md, "The subgraph query").

    Each field is the fetch_subgraphs parameter of the same name. The caps and the lookback
    are the design's defaults; the design leaves the hub threshold to be measured.
    """

    # Payments per direction: a seed's newest, and those of every other expanded account.
    seed_payments: int = 32
    other_payments: int = 16
    # The seed owner's accounts.
    siblings: int = 8
    # An account's links through its owner, and the nodes observed on its own payments.
    enrolled_devices: int = 8
    enrolled_ips: int = 8
    addresses: int = 4
    sent_devices: int = 8
    sent_ips: int = 8
    # Other customers' accounts per identity node of a seed, and in all per seed.
    reverse_accounts: int = 16
    identity_accounts: int = 48
    # Distinct visible owners above which a node is a hub and is not expanded. A placeholder
    # until it is measured: above PhantomLedger's largest ring (150 members).
    hub_threshold: int = 200
    # Days of payment history before the as-of time.
    lookback_days: int = 365

    def __post_init__(self) -> None:
        for name in (
            "seed_payments",
            "other_payments",
            "siblings",
            "enrolled_devices",
            "enrolled_ips",
            "addresses",
            "sent_devices",
            "sent_ips",
            "reverse_accounts",
            "identity_accounts",
        ):
            _integer(f"subgraph.{name}", getattr(self, name), CAPS)
        _integer("subgraph.hub_threshold", self.hub_threshold, HUB_THRESHOLDS)
        _integer("subgraph.lookback_days", self.lookback_days, LOOKBACK_DAYS)


@dataclass(frozen=True)
class RunConfig:
    """Every setting; the defaults are DEFAULT_CONFIG."""

    transport: TransportConfig = TransportConfig()
    subgraph: SubgraphConfig = SubgraphConfig()

    def __post_init__(self) -> None:
        _instance("transport", self.transport, TransportConfig)
        _instance("subgraph", self.subgraph, SubgraphConfig)


DEFAULT_CONFIG = RunConfig()
