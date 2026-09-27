"""The candidate pools TigerGraph returns per hop and the client-side resampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
import operator
from typing import Any

from .fingerprints import fingerprint

POOL_KEYS = ("recent", "older", "distinct", "associations", "max_history")
SAMPLER_BACKENDS = ("auto", "cugraph", "torch")
# The keys of a [sampler] table besides the roots pool and [sampler.children].
SAMPLER_KEYS = (
    "relation_fanouts",
    "association_fanout",
    "association_slots",
    "backend",
    "evaluation_seed",
)
# Version of the resample key scheme (sampling.candidates.selection_keys), part of the
# fingerprint.
# 2: evaluation keys mix the hop in (hop 1 unchanged, hop 2 an independent stream).
SELECTION_KEYS_VERSION = 2


def _bounded(owner: str, name: str, value: object, low: int, high: int) -> int:
    try:
        number = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        number = None
    if isinstance(value, bool) or number is None or not low <= number <= high:
        raise ValueError(f"{owner} {name} must be an integer in [{low},{high}], got {value!r}")
    return number


@dataclass(frozen=True)
class PoolPlan:
    """Bounded, cutoff-safe candidate pool that TigerGraph returns per context and hop.

    Per payment relation: the `recent` most recent visible events, `older` rank
    quantiles and `distinct` events with new peers. Per association relation:
    `associations` valid-time associations. A context whose visible history in one
    relation exceeds `max_history` is rejected (history_capacity_exceeded).
    """

    recent: int = 2
    older: int = 0
    distinct: int = 0
    associations: int = 2
    max_history: int = 2048

    def __post_init__(self) -> None:
        for name, low, high in (
            ("recent", 1, 32),
            ("older", 0, 16),
            ("distinct", 0, 16),
            ("associations", 0, 8),
            ("max_history", 32, 4096),
        ):
            object.__setattr__(self, name, _bounded("Pool", name, getattr(self, name), low, high))

    @property
    def response_bound(self) -> int:
        """Maximum messages in one context: 4 payment and 14 association relations."""
        return 4 * (self.recent + self.older + self.distinct) + 14 * self.associations

    def query_params(self) -> dict[str, int]:
        return {
            "per_relation": self.recent,
            "k_old": self.older,
            "k_div": self.distinct,
            "k_assoc": self.associations,
            "max_history": self.max_history,
        }


def _default_children(roots: PoolPlan) -> PoolPlan:
    # The resampled second hop is payments-only, so children skip association candidates.
    return replace(roots, associations=0)


@dataclass(frozen=True, init=False)
class SamplerPlan:
    """Candidate pools per hop plus the client-side neighbour resampling.

    TigerGraph returns each context's candidate pool (`PoolPlan`, one per hop). The
    client draws, per context and relation, at most `relation_fanouts[hop-1]` payment
    candidates (`association_fanout` per association relation at hop 1) uniformly
    without replacement, then merges them into the fanout slots with at most
    `association_slots` of them associations. Hop 2 is payments-only, so the children
    pool defaults to the roots pool without associations.
    """

    roots: PoolPlan
    children: PoolPlan
    relation_fanouts: tuple[int, int]
    association_fanout: int
    association_slots: int
    backend: str
    evaluation_seed: int

    def __init__(
        self,
        *,
        roots: PoolPlan | None = None,
        children: PoolPlan | None = None,
        relation_fanouts: Sequence[int] = (8, 4),
        association_fanout: int = 1,
        association_slots: int = 2,
        backend: str = "auto",
        evaluation_seed: int = 0,
    ) -> None:
        roots = roots if roots is not None else PoolPlan()
        fanouts = tuple(relation_fanouts)
        if len(fanouts) != 2:
            raise ValueError("Sampler relation_fanouts must have one value per hop")
        if backend not in SAMPLER_BACKENDS:
            raise ValueError(f"Sampler backend must be one of {SAMPLER_BACKENDS}")
        values = {
            "roots": roots,
            "children": children if children is not None else _default_children(roots),
            "relation_fanouts": tuple(
                _bounded("Sampler", "relation_fanouts", v, 1, 64) for v in fanouts
            ),
            "association_fanout": _bounded(
                "Sampler", "association_fanout", association_fanout, 0, 8
            ),
            "association_slots": _bounded("Sampler", "association_slots", association_slots, 0, 16),
            "backend": backend,
            "evaluation_seed": _bounded(
                "Sampler", "evaluation_seed", evaluation_seed, 0, 2**63 - 1
            ),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def pool(self, hop: int) -> PoolPlan:
        if hop == 1:
            return self.roots
        if hop == 2:
            return self.children
        raise ValueError("Hop must be 1 or 2")

    def query_params(self, hop: int = 1) -> dict[str, int]:
        return self.pool(hop).query_params()

    def response_bound(self, hop: int = 1) -> int:
        """Maximum messages in one context at this hop."""
        return self.pool(hop).response_bound

    def to_config(self) -> dict[str, Any]:
        """The `[sampler]` table that `from_config` maps back to this plan."""
        values: dict[str, Any] = asdict(self.roots)
        if self.children != _default_children(self.roots):
            values["children"] = asdict(self.children)
        values |= {
            "association_slots": self.association_slots,
            "relation_fanouts": list(self.relation_fanouts),
            "association_fanout": self.association_fanout,
            "backend": self.backend,
            "evaluation_seed": self.evaluation_seed,
        }
        return values

    def fingerprint(self) -> str:
        """Selection semantics for manifest checks; the execution backend is excluded.

        It includes `selection_keys` (SELECTION_KEYS_VERSION), so manifests and
        checkpoints drawn with an older key scheme are not treated as comparable. The
        value keeps the policy name it had while other policies existed, so recorded
        fingerprints still compare equal.
        """
        value = self.to_config()
        value.pop("backend")
        value["children"] = asdict(self.children)
        value["policy"] = "resample"
        value["selection_keys"] = SELECTION_KEYS_VERSION
        return fingerprint(value)

    def pool_fingerprint(self) -> str:
        """Only what TigerGraph is asked for (preparation and cache identity)."""
        return fingerprint({"roots": self.query_params(1), "children": self.query_params(2)})

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SamplerPlan:
        """Flat `[sampler]` pool keys describe roots; `[sampler.children]` overrides them.

        Configurations saved while other policies existed name this one, `policy =
        "resample"`; any other policy is refused.
        """
        values = dict(config.get("sampler") or {})
        children_values = values.pop("children", None)
        policy = values.pop("policy", "resample")
        if policy != "resample":
            raise ValueError(f"Unknown history sampler {policy!r}: only resample remains")
        unknown = sorted(set(values) - set(POOL_KEYS) - set(SAMPLER_KEYS))
        if unknown:
            raise ValueError(f"Unknown [sampler] key(s): {', '.join(unknown)}")
        roots = PoolPlan(**{k: values.pop(k) for k in POOL_KEYS if k in values})
        children = None
        if children_values is not None:
            if not isinstance(children_values, dict):
                raise ValueError("[sampler.children] must be a table of pool keys")
            unknown = sorted(set(children_values) - set(POOL_KEYS))
            if unknown:
                raise ValueError(f"Unknown [sampler.children] key(s): {', '.join(unknown)}")
            children = replace(_default_children(roots), **children_values)
        return cls(roots=roots, children=children, **values)


def sampler_pools(sampler: SamplerPlan) -> dict[str, dict[str, Any]]:
    """The query-relevant part of a sampler: what TigerGraph returns per hop."""
    return {"roots": sampler.query_params(1), "children": sampler.query_params(2)}
