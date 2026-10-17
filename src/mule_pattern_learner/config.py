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
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

    def __post_init__(self) -> None:
        _instance("transport", self.transport, TransportConfig)
        _instance("model", self.model, ModelConfig)
        _instance("training", self.training, TrainingConfig)


DEFAULT_CONFIG = RunConfig()
