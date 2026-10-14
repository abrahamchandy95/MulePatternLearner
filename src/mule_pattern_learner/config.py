"""The settings as frozen dataclasses; DEFAULT_CONFIG holds every default.

RunConfig has one section per concern, and each component receives only its own
section. A section checks its values when it is built, so a bad setting fails before any
database work. `dataclasses.replace` changes a setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# The ranges of the transport settings; the executor checks the ones it is given too.
QUERY_ATTEMPTS = range(1, 21)
OUTAGE_SECONDS = range(0, 86_401)
# The ranges of the subgraph settings; fetch_subgraphs refuses a lookback outside its range.
CAPS = range(1, 1_025)
HUB_THRESHOLDS = range(1, 1_000_001)
LOOKBACK_DAYS = range(30, 3_651)
# The ranges of the data settings.
GAP_DAYS = range(0, 366)
PER_POSITIVE = range(1, 1_001)
SAMPLE_SIZES = range(1, 1_000_001)
SEEDS = range(0, 2**63)


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


def _window(name: str, value: object) -> tuple[date, date]:
    """value when it is a (first, last) pair of dates, not datetimes, the first not later."""
    if not (
        isinstance(value, tuple)
        and len(value) == 2
        and all(type(day) is date for day in value)
        and value[0] <= value[1]
    ):
        raise ValueError(f"{name} must be a (first, last) pair of dates in order, got {value!r}")
    return value


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
    # An expanded account's owners, by its newest ownership links, and each seed owner's
    # accounts.
    owners: int = 1
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
            "owners",
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
class DataConfig:
    """The windows, the customer hold-out and the examples (docs/design.md, "Splits and
    hidden labels" and "Training examples").

    Each window is its first and last as-of day, and they follow one another. The training
    and validation windows may use the mules discovered before the day after their last
    day; the test window uses the full answer key. The windows, the gap and the unlabelled
    count are the design's defaults; the design gives no scored sample size or seed.
    """

    training: tuple[date, date] = (date(2024, 1, 1), date(2024, 6, 30))
    validation: tuple[date, date] = (date(2024, 7, 1), date(2024, 9, 30))
    test: tuple[date, date] = (date(2024, 10, 1), date(2024, 12, 31))
    # Days before the training labels' cutoff that no training as-of date comes nearer.
    maturity_gap_days: int = 60
    # Unlabelled accounts drawn on a training date per positive on that date.
    unlabelled_per_positive: int = 15
    # Accounts other than the active mules drawn on each validation and test date, as many
    # as the audit sample kept from the earlier code draws.
    scored_sample: int = 2000
    # Keys the hold-out hash and seeds every draw of examples.
    seed: int = 0

    def __post_init__(self) -> None:
        training = _window("data.training", self.training)
        validation = _window("data.validation", self.validation)
        test = _window("data.test", self.test)
        if not training[1] < validation[0] or not validation[1] < test[0]:
            raise ValueError("data.training, data.validation and data.test must not overlap")
        _integer("data.maturity_gap_days", self.maturity_gap_days, GAP_DAYS)
        _integer("data.unlabelled_per_positive", self.unlabelled_per_positive, PER_POSITIVE)
        _integer("data.scored_sample", self.scored_sample, SAMPLE_SIZES)
        _integer("data.seed", self.seed, SEEDS)


@dataclass(frozen=True)
class RunConfig:
    """Every setting; the defaults are DEFAULT_CONFIG."""

    transport: TransportConfig = TransportConfig()
    subgraph: SubgraphConfig = SubgraphConfig()
    data: DataConfig = DataConfig()

    def __post_init__(self) -> None:
        _instance("transport", self.transport, TransportConfig)
        _instance("subgraph", self.subgraph, SubgraphConfig)
        _instance("data", self.data, DataConfig)


DEFAULT_CONFIG = RunConfig()
