"""The settings of a run as frozen dataclasses; DEFAULT_CONFIG is the built-in run.

RunConfig has one section per concern: the frozen TigerGraph scope (scope), the
prepared dataset (dataset), neighbour sampling (sampler, which is
contract.sampler_plan.SamplerPlan itself), the model's feature groups (features), the
model (model), the nnPU loss (loss), optimisation (training), the connection to
TigerGraph (transport) and the host (runtime). Every default is the run `mule
train` performs, and each component receives only its own section. A section checks
its values when it is built, with the ranges of contract.bounds, so a bad setting
fails before any database work. `dataclasses.replace` changes a setting.

fingerprint() names the settings that can change a run's results. to_dict and
from_dict map a configuration to JSON values and back, for the files a run writes, and
with_changes changes the settings a table of the same shape names.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime
import math
from typing import Any

from .contract.bounds import (
    BATCH_ROOTS,
    CONTEXT_LRU_CAPACITY,
    ENCODING_CHECK_EVERY,
    HEADS,
    HIDDEN,
    OUTAGE_SECONDS,
    PREFETCH_BATCHES,
    QUERY_ATTEMPTS,
    QUERY_CONCURRENCY,
    REQUEST_KEYS,
    REVEAL_PER_SPLIT,
    SCOPE_ID_BYTES,
    SEED_LIMIT,
)
from .contract.clock import timestamp
from .contract.feature_groups import ARCHITECTURES, BUILT_IN_GROUPS, FEATURE_GROUPS, FeaturePlan
from .contract.fingerprints import fingerprint
from .contract.graph_schema import SPLITS
from .contract.sampler_plan import PoolPlan, SamplerPlan


def _integer(name: str, value: object, low: int = 0) -> int:
    """value when it is an integer of at least low (a boolean is not one)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{name} must be an integer >= {low}, got {value!r}")
    return value


def _number(
    name: str,
    value: object,
    low: float,
    high: float = math.inf,
    *,
    open_low: bool = False,
    open_high: bool = False,
) -> float:
    """value as a float when it is a number from low to high; open ends exclude the bound."""
    number = math.nan
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
    above = number > low if open_low else number >= low
    below = number < high if open_high else number <= high
    if not (math.isfinite(number) and above and below):
        closing = ")" if open_high or math.isinf(high) else "]"
        interval = f"{'(' if open_low else '['}{low}, {high}{closing}"
        raise ValueError(f"{name} must be a number in {interval}, got {value!r}")
    return number


def _choice(name: str, value: object, choices: Sequence[object]) -> None:
    """value must be one of choices, of the same type (True is not 1)."""
    if not any(type(value) is type(choice) and value == choice for choice in choices):
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")


def _flag(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, got {value!r}")


def _text(name: str, value: object, max_bytes: int | None = None) -> None:
    """value must be a nonempty string, of at most max_bytes bytes when that is given."""
    if (
        not isinstance(value, str)
        or not value
        or (max_bytes is not None and len(value.encode()) > max_bytes)
    ):
        longest = "" if max_bytes is None else f" of at most {max_bytes} bytes"
        raise ValueError(f"{name} must be a nonempty string{longest}, got {value!r}")


def _instance(name: str, value: object, kind: type) -> None:
    if not isinstance(value, kind):
        raise ValueError(f"{name} must be a {kind.__name__}, got {value!r}")


def _names(name: str, value: object) -> tuple[str, ...]:
    """value as a tuple of strings, from any sequence of them but a string."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a list of names, got {value!r}")
    names = tuple(value)
    if not all(isinstance(item, str) for item in names):
        raise ValueError(f"{name} must be a list of names, got {value!r}")
    return names


def _set(section: object, name: str, value: object) -> None:
    """Normalise a field of a frozen section while it is checked."""
    object.__setattr__(section, name, value)


@dataclass(frozen=True)
class ScopeConfig:
    """The frozen scope the splits come from, and the first run's label reveal."""

    id: str = "strict_mule_v2"
    # The first run creates a missing scope; false forbids that write.
    create: bool = True
    # The accounts no party owns: "independent", "shared" or "linked" (tigergraph.scope).
    unowned: str = "linked"
    # Known mules the first run reveals per split, among those a bank would have
    # discovered before the split's cutoff (gsql/queries/label_reveal.gsql).
    reveal_per_split: int = 20
    # Seed of the reveal's deterministic draws.
    reveal_salt: int = 42

    def __post_init__(self) -> None:
        _text("scope.id", self.id, SCOPE_ID_BYTES)
        _flag("scope.create", self.create)
        _choice("scope.unowned", self.unowned, ("independent", "shared", "linked"))
        REVEAL_PER_SPLIT.check("scope.reveal_per_split", self.reveal_per_split)
        _integer("scope.reveal_salt", self.reveal_salt)


@dataclass(frozen=True)
class SplitDates:
    """The cutoff dates of each split, as ISO dates; the splits follow one another."""

    train: tuple[str, ...] = ("2024-07-01",)
    validation: tuple[str, ...] = ("2024-10-01",)
    test: tuple[str, ...] = ("2025-01-01",)

    def __post_init__(self) -> None:
        for split in SPLITS:
            dates = _names(f"dataset.dates.{split}", getattr(self, split))
            if not dates:
                raise ValueError(f"dataset.dates.{split} must name at least one date")
            for date in dates:
                try:
                    datetime.fromisoformat(date)
                except ValueError:
                    raise ValueError(f"dataset.dates.{split}: not an ISO date: {date!r}") from None
            _set(self, split, dates)
        if not (
            max(map(timestamp, self.train))
            < min(map(timestamp, self.validation))
            <= max(map(timestamp, self.validation))
            < min(map(timestamp, self.test))
        ):
            raise ValueError("Train, validation and test cutoffs overlap or are out of order")

    def __getitem__(self, split: str) -> tuple[str, ...]:
        if split not in SPLITS:
            raise KeyError(split)
        return getattr(self, split)

    def all(self) -> list[str]:
        """Every date of every split, sorted."""
        return sorted({date for split in SPLITS for date in self[split]})


@dataclass(frozen=True)
class SeedLimits:
    """The size of each split's label-blind seed reservoir."""

    train: int = 20_000
    validation: int = 2_000
    test: int = 2_000

    def __post_init__(self) -> None:
        for split in SPLITS:
            SEED_LIMIT.check(f"dataset.seed_limits.{split}", getattr(self, split))

    def __getitem__(self, split: str) -> int:
        if split not in SPLITS:
            raise KeyError(split)
        return getattr(self, split)


@dataclass(frozen=True)
class DatasetConfig:
    """What preparation stages inside the scope: the split cutoffs and seed reservoirs."""

    dates: SplitDates = SplitDates()
    seed_limits: SeedLimits = SeedLimits()
    # Seed of the label-blind seed reservoirs (data.accounts).
    seed: int = 42
    # Seed of the scope's partition into splits and of the evaluation samples.
    split_seed: int = 42

    def __post_init__(self) -> None:
        _instance("dataset.dates", self.dates, SplitDates)
        _instance("dataset.seed_limits", self.seed_limits, SeedLimits)
        _integer("dataset.seed", self.seed)
        _integer("dataset.split_seed", self.split_seed)


# The sampler of the built-in run: the candidate pools TigerGraph returns per hop and
# the client's resampling into the fan-out slots.
BUILT_IN_SAMPLER = SamplerPlan(
    fanouts=(16, 4),
    roots=PoolPlan(recent=8, older=4, distinct=4, associations=2, max_history=2048),
    children=PoolPlan(recent=4, older=2, distinct=2, associations=0, max_history=2048),
    relation_fanouts=(8, 4),
    association_fanout=1,
    association_slots=2,
    # cuGraph on CUDA when its functional probe passes, otherwise the torch sampler.
    backend="auto",
    evaluation_seed=0,
)


@dataclass(frozen=True)
class ModelConfig:
    """The model: its architecture, width, attention heads, dropout and slot sum."""

    # "tgat" attends over sampled neighbours; "summary" reads only the root's inputs.
    architecture: str = "tgat"
    hidden: int = 64
    heads: int = 4
    dropout: float = 0.15
    # Beside attention, feed the head a small MLP of each of the root's hop-1 slots, summed
    # and divided by the hop-1 fan-out. Attention averages linear projections of the
    # slots; the MLP can test a combined condition on each slot before pooling, so the sum
    # counts the slots that meet it (Xu, Hu, Leskovec and Jegelka, "How Powerful are Graph
    # Neural Networks?", ICLR 2019). Most roots fill all 16 slots, so this is mostly the
    # share of such slots. Provisional: not yet measured in a run on the reference graph. The
    # summary architecture has no slots and ignores it.
    slot_sum: bool = True

    def __post_init__(self) -> None:
        _choice("model.architecture", self.architecture, ARCHITECTURES)
        HIDDEN.check("model.hidden", self.hidden)
        HEADS.check("model.heads", self.heads)
        if self.hidden % self.heads:
            raise ValueError(
                f"model.hidden ({self.hidden}) must be divisible by model.heads ({self.heads})"
            )
        _set(self, "dropout", _number("model.dropout", self.dropout, 0, 1, open_high=True))
        _flag("model.slot_sum", self.slot_sum)


@dataclass(frozen=True)
class LossConfig:
    """The nnPU loss: the class prior and the weight of the revealed positives."""

    class_prior: float = 0.001
    # Imbalanced nnPU (Su, Chen and Xu, IJCAI 2021): weigh the revealed positives as a
    # balanced problem would. With "prior" (textbook nnPU) the positives carry 0.001 of
    # the loss, and the reference run collapsed to scoring every account near zero. A
    # number in (0, 1) is the weight itself. Provisional: not yet compared with other
    # weights over several seeds.
    positive_weight: str | float = "balanced"

    def __post_init__(self) -> None:
        prior = _number("loss.class_prior", self.class_prior, 0, 1, open_low=True, open_high=True)
        _set(self, "class_prior", prior)
        if isinstance(self.positive_weight, str):
            _choice("loss.positive_weight", self.positive_weight, ("prior", "balanced"))
        else:
            weight = _number(
                "loss.positive_weight", self.positive_weight, 0, 1, open_low=True, open_high=True
            )
            _set(self, "positive_weight", weight)


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation, early stopping and the proxy evaluation on observed labels."""

    seed: int = 42
    epochs: int = 30
    # None trains on every marginal account of an epoch.
    steps_per_epoch: int | None = 100
    batch_size: int = 64
    # Epochs without a better validation AP before training stops; 0 never stops early.
    patience: int = 6
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    # Validate, select and save an exponential moving average of the weights (decay per
    # step, warmed up); training itself is unchanged. With 11 validation positives the
    # raw weights' AP swung 0.011 to 0.096 between epochs of the reference run. 0
    # validates the raw weights.
    weight_average_decay: float = 0.99
    # Unlabeled accounts of each validation and test cutoff that the proxy evaluation
    # scores beside every observed positive; None scores them all.
    proxy_unlabeled_limit: int | None = 2000

    def __post_init__(self) -> None:
        _integer("training.seed", self.seed)
        _integer("training.epochs", self.epochs, 1)
        if self.steps_per_epoch is not None:
            _integer("training.steps_per_epoch", self.steps_per_epoch, 1)
        BATCH_ROOTS.check("training.batch_size", self.batch_size)
        _integer("training.patience", self.patience)
        rate = _number("training.learning_rate", self.learning_rate, 0, open_low=True)
        _set(self, "learning_rate", rate)
        _set(self, "weight_decay", _number("training.weight_decay", self.weight_decay, 0))
        decay = _number(
            "training.weight_average_decay", self.weight_average_decay, 0, 1, open_high=True
        )
        _set(self, "weight_average_decay", decay)
        if self.proxy_unlabeled_limit is not None:
            _integer("training.proxy_unlabeled_limit", self.proxy_unlabeled_limit, 1)


@dataclass(frozen=True)
class TransportConfig:
    """Context requests and their retry budgets. Changing them never changes results."""

    # Measured on the reference graph: 8 contexts per request, 16 in parallel built a 64-root
    # batch in about 11 s, against about 20 s for 16 x 8 and 22 s for 4 x 16.
    request_batch_size: int = 8
    query_concurrency: int = 16
    context_lru_capacity: int = 256
    encoding_check_every: int = 64
    max_query_attempts: int = 6
    # Wall-clock seconds one operation keeps retrying while TigerGraph is unavailable.
    max_outage_s: int = 900

    def __post_init__(self) -> None:
        REQUEST_KEYS.check("transport.request_batch_size", self.request_batch_size)
        QUERY_CONCURRENCY.check("transport.query_concurrency", self.query_concurrency)
        CONTEXT_LRU_CAPACITY.check("transport.context_lru_capacity", self.context_lru_capacity)
        ENCODING_CHECK_EVERY.check("transport.encoding_check_every", self.encoding_check_every)
        QUERY_ATTEMPTS.check("transport.max_query_attempts", self.max_query_attempts)
        OUTAGE_SECONDS.check("transport.max_outage_s", self.max_outage_s)


@dataclass(frozen=True)
class RuntimeConfig:
    """The host: device, threads, determinism, prefetch, checkpoints, logs and rejections."""

    # CUDA when available, then Apple MPS, then CPU; or "cpu", "mps", "cuda".
    device: str = "auto"
    threads: int = 4
    # True: deterministic algorithms, warning on CUDA-only gaps; "strict" fails on them.
    deterministic: bool | str = True
    prefetch_batches: int = 2
    # Also save the resume state every n training steps; 0 saves it once per epoch.
    checkpoint_every_steps: int = 0
    log_every_steps: int = 10
    # Largest fraction of an epoch's or evaluation split's roots TigerGraph may reject.
    # It only decides whether a run may go on, never its numbers.
    max_rejected_root_fraction: float = 0.0

    def __post_init__(self) -> None:
        _text("runtime.device", self.device)
        _integer("runtime.threads", self.threads, 1)
        _choice("runtime.deterministic", self.deterministic, (True, False, "strict"))
        PREFETCH_BATCHES.check("runtime.prefetch_batches", self.prefetch_batches)
        _integer("runtime.checkpoint_every_steps", self.checkpoint_every_steps)
        _integer("runtime.log_every_steps", self.log_every_steps, 1)
        fraction = _number(
            "runtime.max_rejected_root_fraction", self.max_rejected_root_fraction, 0, 1
        )
        _set(self, "max_rejected_root_fraction", fraction)


# The sections fingerprint() leaves out, so a resumed run may change them. The transport
# section and the runtime's prefetch, checkpoint, log and rejection settings never change
# a run's numbers. The device, threads and determinism can change its floating-point
# results: config.json's provenance records those the run started with, and a resumed
# segment that changes them says so in events.jsonl.
RUNTIME_SECTIONS = ("transport", "runtime")


@dataclass(frozen=True)
class RunConfig:
    """Every setting of one run; the defaults are the built-in run."""

    scope: ScopeConfig = ScopeConfig()
    dataset: DatasetConfig = DatasetConfig()
    sampler: SamplerPlan = BUILT_IN_SAMPLER
    # The pool groups feed counts over the root's candidate pool (not all-time totals) to
    # the TGAT model's summary branch. Without them a root's node vector held only its
    # entity type, is_external, is_deposit and history_withheld. In the diagnostic study
    # distinct payers and internal first-time inflows alone ranked test mules at a
    # weighted ROC AUC of 0.88 and 0.92, against the model's 0.78; those counts were
    # chosen after reading the generator's mule typology, and the internal ones
    # (pool_internal_inflows) suit the generator more than a real bank. Computed on the
    # client, so the query is unchanged.
    features: tuple[str, ...] = BUILT_IN_GROUPS
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()
    transport: TransportConfig = TransportConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    def __post_init__(self) -> None:
        sections: dict[str, tuple[object, type]] = {
            "scope": (self.scope, ScopeConfig),
            "dataset": (self.dataset, DatasetConfig),
            "sampler": (self.sampler, SamplerPlan),
            "model": (self.model, ModelConfig),
            "loss": (self.loss, LossConfig),
            "training": (self.training, TrainingConfig),
            "transport": (self.transport, TransportConfig),
            "runtime": (self.runtime, RuntimeConfig),
        }
        for name, (section, kind) in sections.items():
            _instance(name, section, kind)
        groups = _names("features", self.features)
        unknown = sorted(set(groups) - FEATURE_GROUPS.keys())
        if unknown:
            raise ValueError(f"unknown feature groups {unknown}; known: {sorted(FEATURE_GROUPS)}")
        if len(set(groups)) != len(groups):
            raise ValueError(f"features names a group twice: {list(groups)}")
        _set(self, "features", groups)

    def feature_plan(self) -> FeaturePlan:
        """The model's inputs: the feature groups, read by the model's architecture."""
        return FeaturePlan(self.features, self.model.architecture)

    def results_view(self) -> dict[str, Any]:
        """The settings that can change a run's results, as JSON values.

        Every section but transport and runtime, and the sampler without its backend:
        a run checks the backend it samples with on its own, so a resumed run may name
        another one explicitly.
        """
        value = self.to_dict()
        for name in RUNTIME_SECTIONS:
            del value[name]
        del value["sampler"]["backend"]
        return value

    def fingerprint(self) -> str:
        return fingerprint(self.results_view())

    def to_dict(self) -> dict[str, Any]:
        """Every setting as JSON values: sections are tables and tuples are lists."""
        return as_json(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunConfig:
        """The configuration a to_dict table describes; absent keys keep their defaults."""
        return DEFAULT_CONFIG.with_changes(value)

    def with_changes(self, changes: Mapping[str, Any]) -> RunConfig:
        """This configuration with the settings a table of to_dict's shape names changed.

        A nested table changes only the fields it names, so {"training": {"epochs": 3}}
        keeps every other setting. Unknown keys are refused with their dotted names.
        """
        return _replaced(self, changes, "")


def as_json(value: Any) -> Any:
    """A section or setting as JSON values (dataclasses as tables, tuples as lists)."""
    if is_dataclass(value) and not isinstance(value, type):
        return as_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): as_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [as_json(v) for v in value]
    return value


def _replaced(default: Any, value: object, path: str) -> Any:
    """default with the fields a table sets, recursively through nested sections."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.rstrip('.') or 'The configuration'} must be a table")
    names = {field.name for field in fields(default)}
    unknown = sorted(f"{path}{key}" for key in value if key not in names)
    if unknown:
        raise ValueError(f"Unknown configuration key(s): {', '.join(unknown)}")
    changes: dict[str, Any] = {}
    for name, item in value.items():
        current = getattr(default, name)
        if is_dataclass(current):
            changes[name] = _replaced(current, item, f"{path}{name}.")
        elif isinstance(current, tuple) and isinstance(item, list):
            changes[name] = tuple(item)
        else:
            changes[name] = item
    return replace(default, **changes)


def _leaves(value: object, path: str = "") -> Iterator[tuple[str, object]]:
    if isinstance(value, Mapping) and value:
        for key, item in value.items():
            yield from _leaves(item, f"{path}.{key}" if path else str(key))
    else:
        yield path, value


def differing_settings(current: Mapping[str, Any], recorded: Mapping[str, Any]) -> list[str]:
    """The dotted names of the settings whose JSON values differ between two tables."""
    have, want = dict(_leaves(current)), dict(_leaves(recorded))
    return sorted(
        name
        for name in have.keys() | want.keys()
        if fingerprint(have.get(name)) != fingerprint(want.get(name))
    )


DEFAULT_CONFIG = RunConfig()
