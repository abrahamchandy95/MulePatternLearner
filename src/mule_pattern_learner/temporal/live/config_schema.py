"""Schema for live temporal training configuration files.

`validate_config` rejects unknown or mistyped keys before any database work.
Operational keys (transport, runtime and preparation switches) receive their
documented defaults. Modelling keys whose absence has a meaning downstream
(feature_groups, extraction_groups, sampler, observed_labels, fanouts, ...) are
kept absent so the owning component applies its own default (FALLBACKS and
`setting` for the scalar ones); saved checkpoint configurations rely on that.
`run_config` starts from DEFAULT_RUN, so a training run always has the modelling keys
DEFAULT_RUN sets (feature_groups, sampler, fanouts and the model and optimisation keys).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

TRANSPORT_DEFAULTS: dict[str, int] = {
    # Measured on the live graph: 8 contexts per request, 16 in parallel built a 64-root
    # batch in about 11 s, against about 20 s for 16 x 8 and 22 s for 4 x 16.
    "request_batch_size": 8,
    "query_concurrency": 16,
    "context_lru_capacity": 256,
    "encoding_check_every": 64,
    "max_query_attempts": 6,
    # Wall-clock seconds one operation keeps retrying while TigerGraph is unavailable.
    "max_outage_s": 900,
}
OPERATIONAL_DEFAULTS: dict[str, Any] = {
    **TRANSPORT_DEFAULTS,
    "context_storage": "stream",
    "prepare_batch_size": 16,
    # The first run creates a missing scope; false forbids that write.
    "create_scope": True,
    "scope_unowned": "linked",
    "deterministic": True,
    "prefetch_batches": 2,
    "checkpoint_every_steps": 0,
    "log_every_steps": 10,
}
# What a modelling key means when a configuration leaves it absent (or null). Unlike
# OPERATIONAL_DEFAULTS these never enter a validated configuration, so config.json and
# resume fingerprints record only what was written (preparation views resolve
# split_seed, label_policy and the SQLite fanouts through them). Configs from
# run_config set every one of them (DEFAULT_RUN); the fallbacks decide saved and
# hand-written configurations, so they must not change (fanouts stays (8, 4)).
FALLBACKS: dict[str, Any] = {
    "seed": 42,
    "split_seed": 42,
    "label_policy": "observed",
    "fanouts": (8, 4),
    "batch_size": 64,
    "hidden": 64,
    "heads": 4,
    "dropout": 0.15,
    "variant": "temporal",
    "epochs": 30,
    "patience": 6,
    "max_rejected_root_fraction": 0.0,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "positive_weight": "prior",
    "device": "auto",
    "threads": 4,
}
# The run `mule-temporal train` performs with no configuration file: strict inductive
# splits over the frozen scope, labels revealed in the graph (label contract), the v5
# candidate-pool sampler and the model and optimisation settings of the reference run.
# A configuration file only overrides keys (see run_config for how tables merge).
# Identity (dataset_id) is read from the scope or derived from the graph, and the
# prepared cache lives inside the run directory, so none of it is configured.
DEFAULT_RUN: dict[str, Any] = {
    "evaluation_protocol": "strict_inductive",
    "scope_id": "strict_mule_v2",
    "label_policy": "graph_observed",
    # Known mules the first run reveals per split, among those a bank would have
    # discovered before the split's cutoff (gsql/temporal/label_reveal.gsql).
    "reveal_per_split": 20,
    "evaluation_unlabeled_limit": 2000,
    "fanouts": [16, 4],
    "batch_size": 64,
    "epochs": 30,
    "steps_per_epoch": 100,
    # 0 disables early stopping.
    "patience": 6,
    # Largest fraction of an epoch's or evaluation split's roots TigerGraph may reject.
    "max_rejected_root_fraction": 0.0,
    "hidden": 64,
    "heads": 4,
    "dropout": 0.15,
    "variant": "temporal",
    "architecture": "split",
    "feature_groups": [
        "entity_meta",
        "hub_indicator",
        "message_core",
        "time_encoding",
        "pair_history",
        "flow_timing",
    ],
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "class_prior": 0.001,
    # Imbalanced nnPU (Su, Chen and Xu, IJCAI 2021): weigh the revealed positives as a
    # balanced problem would. With "prior" (textbook nnPU) the positives carry 0.001 of
    # the loss, and the reference run collapsed to scoring every account near zero.
    # Provisional: not yet compared with other weights over several seeds.
    "positive_weight": "balanced",
    "seed": 42,
    "split_seed": 42,
    # CUDA when available, then Apple MPS, then CPU.
    "device": "auto",
    "threads": 4,
    "dates": {"train": ["2024-07-01"], "validation": ["2024-10-01"], "test": ["2025-01-01"]},
    "seed_limits": {"train": 20000, "validation": 2000, "test": 2000},
    "sampler": {
        "policy": "resample",
        "recent": 8,
        "older": 4,
        "distinct": 4,
        "associations": 2,
        "max_history": 2048,
        "relation_fanouts": [8, 4],
        "association_fanout": 1,
        "association_slots": 2,
        # cuGraph on CUDA when its functional probe passes, otherwise the torch sampler.
        "backend": "auto",
        "evaluation_seed": 0,
        "children": {
            "recent": 4,
            "older": 2,
            "distinct": 2,
            "associations": 0,
            "max_history": 2048,
        },
    },
}
# Identifiers that become directory names or server-side scope metadata.
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

Positive = Annotated[int, Field(ge=1)]
NonNegative = Annotated[int, Field(ge=0)]


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class PoolConfig(_Strict):
    """One candidate pool; ranges mirror `PoolPlan`."""

    recent: Annotated[int, Field(ge=1, le=32)] | None = None
    older: Annotated[int, Field(ge=0, le=16)] | None = None
    distinct: Annotated[int, Field(ge=0, le=16)] | None = None
    associations: Annotated[int, Field(ge=0, le=8)] | None = None
    max_history: Annotated[int, Field(ge=32, le=4096)] | None = None


class SamplerConfig(PoolConfig):
    """`[sampler]`: flat keys are the roots pool, `[sampler.children]` the second hop."""

    policy: Literal["recent", "stratified", "resample"] | None = None
    children: PoolConfig | None = None
    relation_fanouts: (
        Annotated[list[Annotated[int, Field(ge=1, le=64)]], Field(min_length=2, max_length=2)]
        | None
    ) = None
    association_fanout: Annotated[int, Field(ge=0, le=8)] | None = None
    association_slots: Annotated[int, Field(ge=0, le=16)] | None = None
    backend: Literal["auto", "cugraph", "torch"] | None = None
    evaluation_seed: NonNegative | None = None


class DatesConfig(_Strict):
    train: Annotated[list[str], Field(min_length=1)]
    validation: Annotated[list[str], Field(min_length=1)]
    test: Annotated[list[str], Field(min_length=1)]

    @field_validator("train", "validation", "test")
    @classmethod
    def _iso_dates(cls, values: list[str]) -> list[str]:
        for value in values:
            try:
                datetime.fromisoformat(value)
            except ValueError as error:
                raise ValueError(f"not an ISO date: {value!r}") from error
        return values


class SeedLimitsConfig(_Strict):
    train: Annotated[int, Field(ge=1, le=20000)]
    validation: Annotated[int, Field(ge=1, le=20000)]
    test: Annotated[int, Field(ge=1, le=20000)]


class LiveConfig(_Strict):
    """Every key is optional here; preparation and training require what they use.

    Operational keys default to OPERATIONAL_DEFAULTS, which validate_config also fills.
    """

    # Identity and preparation.
    dataset_id: str | None = None
    prepared_id: str | None = None
    evaluation_protocol: Literal["strict_inductive", "shared_history"] | None = None
    scope_id: str | None = None
    scope_unowned: Literal["independent", "shared", "linked"] = OPERATIONAL_DEFAULTS[
        "scope_unowned"
    ]
    create_scope: bool = OPERATIONAL_DEFAULTS["create_scope"]
    label_policy: Literal["observed", "graph_observed"] | None = None
    observed_labels: str | None = None
    # First-run label reveal (graph_observed): at most this many known mules per split.
    reveal_per_split: Annotated[int, Field(ge=0, le=1000)] | None = None
    # Seed of the reveal's deterministic draws; defaults to `seed`.
    reveal_salt: int | None = None
    context_storage: Literal["stream", "sqlite"] = OPERATIONAL_DEFAULTS["context_storage"]
    prepare_batch_size: Annotated[int, Field(ge=1, le=128)] = OPERATIONAL_DEFAULTS[
        "prepare_batch_size"
    ]
    dates: DatesConfig | None = None
    seed_limits: SeedLimitsConfig | None = None
    seed: int | None = None
    # Reservoir seed of the prepared cohort; defaults to `seed`.
    cohort_seed: int | None = None
    split_seed: int | None = None
    evaluation_unlabeled_limit: NonNegative | None = None
    # Features, sampling and model.
    feature_groups: list[str] | None = None
    extraction_groups: list[str] | None = None
    variant: Literal["temporal", "no_fourier", "tabular"] | None = None
    architecture: Literal["single", "split", "summary"] | None = None
    fanouts: (
        Annotated[list[Annotated[int, Field(ge=1, le=64)]], Field(min_length=2, max_length=2)]
        | None
    ) = None
    per_relation: Annotated[int, Field(ge=1, le=32)] | None = None
    sampler: SamplerConfig | None = None
    hidden: Positive | None = None
    heads: Positive | None = None
    dropout: Annotated[float, Field(ge=0, lt=1)] | None = None
    # Optimisation.
    epochs: Positive | None = None
    steps_per_epoch: Positive | None = None
    patience: NonNegative | None = None
    # Largest fraction of an epoch's or evaluation split's roots TigerGraph may reject.
    max_rejected_root_fraction: Annotated[float, Field(ge=0, le=1)] | None = None
    batch_size: Annotated[int, Field(ge=1, le=128)] | None = None
    learning_rate: Annotated[float, Field(gt=0)] | None = None
    weight_decay: Annotated[float, Field(ge=0)] | None = None
    class_prior: Annotated[float, Field(gt=0, lt=1)] | None = None
    positive_weight: Literal["prior", "balanced"] | Annotated[float, Field(gt=0, lt=1)] | None = (
        None
    )
    device: str | None = None
    threads: Positive | None = None
    deterministic: bool | Literal["strict"] = OPERATIONAL_DEFAULTS["deterministic"]
    # Transport and runtime.
    request_batch_size: Annotated[int, Field(ge=1, le=64)] = OPERATIONAL_DEFAULTS[
        "request_batch_size"
    ]
    query_concurrency: Annotated[int, Field(ge=1, le=16)] = OPERATIONAL_DEFAULTS[
        "query_concurrency"
    ]
    context_lru_capacity: Annotated[int, Field(ge=0, le=4096)] = OPERATIONAL_DEFAULTS[
        "context_lru_capacity"
    ]
    encoding_check_every: Annotated[int, Field(ge=1, le=1_000_000)] = OPERATIONAL_DEFAULTS[
        "encoding_check_every"
    ]
    max_query_attempts: Annotated[int, Field(ge=1, le=20)] = OPERATIONAL_DEFAULTS[
        "max_query_attempts"
    ]
    max_outage_s: Annotated[int, Field(ge=0, le=86_400)] = OPERATIONAL_DEFAULTS["max_outage_s"]
    prefetch_batches: Annotated[int, Field(ge=0, le=8)] = OPERATIONAL_DEFAULTS["prefetch_batches"]
    checkpoint_every_steps: NonNegative = OPERATIONAL_DEFAULTS["checkpoint_every_steps"]
    log_every_steps: Positive = OPERATIONAL_DEFAULTS["log_every_steps"]

    @field_validator("dataset_id", "prepared_id")
    @classmethod
    def _identifier(cls, value: str | None) -> str | None:
        if value is not None and not IDENTIFIER.match(value):
            raise ValueError(
                "must be 1-128 characters of letters, digits, '.', '_' or '-' "
                "and start with a letter or digit"
            )
        return value

    @field_validator("feature_groups", "extraction_groups")
    @classmethod
    def _known_groups(cls, groups: list[str] | None) -> list[str] | None:
        from .contract import FEATURE_GROUPS

        unknown = sorted(set(groups or ()) - set(FEATURE_GROUPS))
        if unknown:
            raise ValueError(f"unknown feature groups {unknown}; known: {sorted(FEATURE_GROUPS)}")
        return groups


KNOWN_KEYS = frozenset(LiveConfig.model_fields)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a validated copy with operational defaults; raise ValueError on bad input."""
    if not isinstance(config, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("Configuration must be a table")
    unknown = sorted(set(config) - KNOWN_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown configuration key(s): {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(KNOWN_KEYS))}"
        )
    try:
        model = LiveConfig.model_validate(config)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'config'}: {item['msg']}"
            for item in error.errors()
        )
        raise ValueError(f"Invalid configuration: {problems}") from None
    # Nested tables keep only the keys that were written, like the top level.
    result = model.model_dump(exclude_unset=True)
    for key, value in OPERATIONAL_DEFAULTS.items():
        result.setdefault(key, value)
    return result


def setting(config: dict[str, Any], key: str) -> Any:
    """config[key]; a missing or null key means its FALLBACKS or OPERATIONAL_DEFAULTS value."""
    value = config.get(key)
    if value is not None:
        return value
    return FALLBACKS[key] if key in FALLBACKS else OPERATIONAL_DEFAULTS[key]


def model_seed(config: dict[str, Any]) -> int:
    return int(setting(config, "seed"))


def split_seed(config: dict[str, Any]) -> int:
    return int(setting(config, "split_seed"))


def fanouts(config: dict[str, Any]) -> tuple[int, int]:
    """Children sampled per context at hop 1 and hop 2 (validated configs hold exactly two)."""
    return tuple(int(v) for v in setting(config, "fanouts"))  # type: ignore[return-value]


def merged(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """base with overrides applied: tables merge key by key, lists and scalars replace."""
    result = dict(base)
    for key, value in overrides.items():
        current = result.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            result[key] = merged(cast(dict[str, Any], current), cast(dict[str, Any], value))
        else:
            result[key] = value
    return result


def run_config(path: Path | None = None) -> dict[str, Any]:
    """DEFAULT_RUN with the keys of an optional TOML/JSON overrides file, validated.

    Tables merge recursively, so `[sampler] backend = "torch"` changes only the
    backend and `[dates] train = [...]` keeps the default validation and test dates.
    Lists and scalars replace the default. A `[sampler]` table that names another
    policy replaces the whole default sampler table, because pool settings of one
    policy do not apply to another.
    """
    import copy

    config = copy.deepcopy(DEFAULT_RUN)
    if path is not None:
        from mule_pattern_learner.configuration import load_config

        overrides = load_config(path)
        if overrides.get("observed_labels") and "label_policy" not in overrides:
            # A label file is the "observed" policy; the default reads the graph.
            overrides["label_policy"] = "observed"
        sampler = overrides.get("sampler")
        if isinstance(sampler, dict):
            policy = cast(dict[str, Any], sampler).get("policy", config["sampler"]["policy"])
            if policy != config["sampler"]["policy"]:
                del config["sampler"]
        config = merged(config, overrides)
    return validate_config(config)
