"""Schema for live temporal training configuration files.

`validate_config` rejects unknown or mistyped keys before any database work.
Operational keys (transport, runtime and preparation switches) receive their
documented defaults. Modelling keys whose absence has a meaning downstream
(feature_groups, extraction_groups, sampler, observed_labels, fanouts, ...) are
kept absent so the owning component applies its own default.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Annotated, Any, Literal

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
# How temporal_create_training_scope partitions accounts without an owning Party.
SCOPE_UNOWNED_POLICIES = ("independent", "shared", "linked")
OPERATIONAL_DEFAULTS: dict[str, Any] = {
    **TRANSPORT_DEFAULTS,
    "context_storage": "stream",
    "prepare_batch_size": 16,
    "create_scope": False,
    "scope_unowned": "linked",
    "deterministic": True,
    "prefetch_batches": 2,
    "checkpoint_every_steps": 0,
    "log_every_steps": 10,
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
    """Every key is optional here; preparation and training require what they use."""

    # Identity and preparation.
    dataset_id: str | None = None
    prepared_id: str | None = None
    evaluation_protocol: Literal["strict_inductive", "shared_history"] | None = None
    scope_id: str | None = None
    scope_unowned: Literal["independent", "shared", "linked"] = "linked"
    create_scope: bool = False
    label_policy: Literal["observed", "graph_observed"] | None = None
    observed_labels: str | None = None
    context_storage: Literal["stream", "sqlite"] = "stream"
    prepare_batch_size: Annotated[int, Field(ge=1, le=128)] = 16
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
    positive_weight: Literal["prior"] | Annotated[float, Field(gt=0)] | None = None
    device: str | None = None
    threads: Positive | None = None
    deterministic: bool | Literal["strict"] = True
    # Transport and runtime.
    request_batch_size: Annotated[int, Field(ge=1, le=64)] = 8
    query_concurrency: Annotated[int, Field(ge=1, le=16)] = 16
    context_lru_capacity: Annotated[int, Field(ge=0, le=4096)] = 256
    encoding_check_every: Annotated[int, Field(ge=1, le=1_000_000)] = 64
    max_query_attempts: Annotated[int, Field(ge=1, le=20)] = 6
    max_outage_s: Annotated[int, Field(ge=0, le=86_400)] = 900
    prefetch_batches: Annotated[int, Field(ge=0, le=8)] = 2
    checkpoint_every_steps: NonNegative = 0
    log_every_steps: Positive = 10

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
