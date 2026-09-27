"""The candidate pools TigerGraph returns per hop and the client-side resampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from .bounds import ASSOCIATION_FANOUT, ASSOCIATION_SLOTS, EVALUATION_SEED, FANOUT, POOL
from .fingerprints import fingerprint

SAMPLER_BACKENDS = ("auto", "cugraph", "torch")
# Version of the resample key scheme (sampling.candidates.selection_keys), part of the
# fingerprint.
# 2: evaluation keys mix the hop in (hop 1 unchanged, hop 2 an independent stream).
SELECTION_KEYS_VERSION = 2


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
        for name, bound in POOL.items():
            object.__setattr__(self, name, bound.check(f"Pool {name}", getattr(self, name)))

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
    """The neighbours sampled per hop: candidate pools plus the client-side resampling.

    Each context of a batch has `fanouts[hop-1]` neighbour slots. TigerGraph returns
    each context's candidate pool (`PoolPlan`, one per hop). The client draws, per
    context and relation, at most `relation_fanouts[hop-1]` payment candidates
    (`association_fanout` per association relation at hop 1) uniformly without
    replacement, then merges them into the slots with at most `association_slots` of
    them associations. Hop 2 is payments-only, so the children pool defaults to the
    roots pool without associations. It is the sampler section of config.RunConfig.
    """

    fanouts: tuple[int, int]
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
        fanouts: Sequence[int] = (16, 4),
        roots: PoolPlan | None = None,
        children: PoolPlan | None = None,
        relation_fanouts: Sequence[int] = (8, 4),
        association_fanout: int = 1,
        association_slots: int = 2,
        backend: str = "auto",
        evaluation_seed: int = 0,
    ) -> None:
        roots = roots if roots is not None else PoolPlan()
        slots, per_relation = tuple(fanouts), tuple(relation_fanouts)
        if len(slots) != 2:
            raise ValueError("Sampler fanouts must have one value per hop")
        if len(per_relation) != 2:
            raise ValueError("Sampler relation_fanouts must have one value per hop")
        if backend not in SAMPLER_BACKENDS:
            raise ValueError(f"Sampler backend must be one of {SAMPLER_BACKENDS}")
        values = {
            "fanouts": tuple(FANOUT.check("Sampler fanouts", v) for v in slots),
            "roots": roots,
            "children": children if children is not None else _default_children(roots),
            "relation_fanouts": tuple(
                FANOUT.check("Sampler relation_fanouts", v) for v in per_relation
            ),
            "association_fanout": ASSOCIATION_FANOUT.check(
                "Sampler association_fanout", association_fanout
            ),
            "association_slots": ASSOCIATION_SLOTS.check(
                "Sampler association_slots", association_slots
            ),
            "backend": backend,
            "evaluation_seed": EVALUATION_SEED.check("Sampler evaluation_seed", evaluation_seed),
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

    def fingerprint(self) -> str:
        """Selection semantics for recorded comparisons; the execution backend is excluded.

        It includes `selection_keys` (SELECTION_KEYS_VERSION), so models drawn with an
        older key scheme are not treated as comparable. The value keeps the layout and
        the policy name it had while the plan held neither the fan-outs nor other
        policies, so recorded fingerprints still compare equal; a run's fingerprint
        (config.RunConfig.fingerprint) covers the fan-outs.
        """
        return fingerprint(
            {
                **asdict(self.roots),
                "children": asdict(self.children),
                "relation_fanouts": list(self.relation_fanouts),
                "association_fanout": self.association_fanout,
                "association_slots": self.association_slots,
                "evaluation_seed": self.evaluation_seed,
                "policy": "resample",
                "selection_keys": SELECTION_KEYS_VERSION,
            }
        )


def sampler_pools(sampler: SamplerPlan) -> dict[str, dict[str, Any]]:
    """The query-relevant part of a sampler: what TigerGraph returns per hop."""
    return {"roots": sampler.query_params(1), "children": sampler.query_params(2)}
