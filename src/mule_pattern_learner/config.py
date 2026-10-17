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
# The ranges of the model and training settings.
HIDDEN = range(1, 4_097)
HEADS = range(1, 65)
EPOCHS = range(1, 10_001)
BATCH_SIZES = range(1, 100_001)
SEED_COUNTS = range(1, 1_001)


def _integer(name: str, value: object, bounds: range) -> int:
    """value when it is an integer within bounds (a boolean is not one)."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in bounds:
        raise ValueError(
            f"{name} must be an integer in [{bounds.start}, {bounds[-1]}], got {value!r}"
        )
    return value


def _fraction(name: str, value: object, *, zero: bool) -> float:
    """value when it is a real number below 1 and above 0, or 0 itself when zero allows it
    (a boolean is not one)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not (0 <= value < 1 if zero else 0 < value < 1)
    ):
        interval = "[0, 1)" if zero else "(0, 1)"
        raise ValueError(f"{name} must be a number in {interval}, got {value!r}")
    return float(value)


def _priors(name: str, value: object) -> None:
    """Checks that value is a tuple of distinct class priors, each in (0, 1)."""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} must be a tuple of priors, got {value!r}")
    for prior in value:
        _fraction(name, prior, zero=False)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} repeats a prior: {value!r}")


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
class ModelConfig:
    """The heterogeneous attention model (docs/design.md, "The model"). The hidden size, the
    heads, the dropout and the label mask are the design's defaults."""

    # The width of every node's representation after each layer, split among the heads.
    hidden: int = 64
    heads: int = 4
    # The share of each node's representation zeroed after each layer while training.
    dropout: float = 0.2
    # The label-input variant: the chance that a training step hides a confirmed neighbour.
    label_mask: float = 0.5

    def __post_init__(self) -> None:
        _integer("model.hidden", self.hidden, HIDDEN)
        _integer("model.heads", self.heads, HEADS)
        if self.hidden % self.heads:
            raise ValueError(f"model.hidden ({self.hidden}) must split evenly among model.heads")
        _fraction("model.dropout", self.dropout, zero=True)
        _fraction("model.label_mask", self.label_mask, zero=True)


@dataclass(frozen=True)
class TrainingConfig:
    """The optimiser, the epochs, the class priors tried and the seeds (docs/design.md,
    "Training"). All are the design's defaults but the batch size, which it leaves."""

    # Adam's learning rate and weight decay.
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    # Epochs at most per prior, and epochs without a better validation score before it stops.
    max_epochs: int = 30
    patience: int = 5
    # The nnPU class priors tried; validation chooses one.
    priors: tuple[float, ...] = (0.001, 0.003, 0.01)
    # Examples per optimiser step.
    batch_size: int = 128
    # Models trained per world, each from its own seed.
    seeds: int = 5

    def __post_init__(self) -> None:
        _fraction("training.learning_rate", self.learning_rate, zero=False)
        _fraction("training.weight_decay", self.weight_decay, zero=True)
        _integer("training.max_epochs", self.max_epochs, EPOCHS)
        _integer("training.patience", self.patience, EPOCHS)
        _priors("training.priors", self.priors)
        _integer("training.batch_size", self.batch_size, BATCH_SIZES)
        _integer("training.seeds", self.seeds, SEED_COUNTS)


@dataclass(frozen=True)
class RunConfig:
    """Every setting; the defaults are DEFAULT_CONFIG."""

    transport: TransportConfig = TransportConfig()
    subgraph: SubgraphConfig = SubgraphConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

    def __post_init__(self) -> None:
        _instance("transport", self.transport, TransportConfig)
        _instance("subgraph", self.subgraph, SubgraphConfig)
        _instance("data", self.data, DataConfig)
        _instance("model", self.model, ModelConfig)
        _instance("training", self.training, TrainingConfig)


DEFAULT_CONFIG = RunConfig()
