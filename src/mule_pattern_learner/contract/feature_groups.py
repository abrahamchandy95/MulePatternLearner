"""The feature groups the context query and the batches share, and the feature plan.

FEATURE_GROUPS holds the groups of the built-in run and no others (the owner's decision
in docs/architecture.md), in the order that fixes column order. The pool groups and the hub
indicator are computed by the client; the others come from TigerGraph. The groups
training does not read are analytics (contract.analytics_features).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .fingerprints import fingerprint
from .graph_schema import NODE_TYPES, PAYMENT_RELATIONS, RAILS, RELATIONS
from .server import CONTEXT_CONTRACT
from .time_basis import BASIS_ID


def contract_fingerprint() -> str:
    """The feature, relation and time-basis contract that checkpoints record.

    The pool groups are left out: TigerGraph never sees them, and only models that
    use them depend on their definitions, which `FeaturePlan.fingerprint` covers.
    """
    return fingerprint(
        {
            "version": CONTEXT_CONTRACT,
            "basis": BASIS_ID,
            "relations": RELATIONS,
            "rails": RAILS,
            "groups": {k: vars(v) for k, v in FEATURE_GROUPS.items() if k not in POOL_GROUPS},
            "client_groups": sorted(CLIENT_GROUPS - set(POOL_GROUPS)),
        }
    )


def pool_definition(groups: Sequence[str]) -> dict[str, Any]:
    """What the pool counts of these groups mean, for the input fingerprint."""
    return {
        "version": POOL_ACTIVITY_VERSION,
        "groups": {g: vars(FEATURE_GROUPS[g]) for g in POOL_GROUPS if g in groups},
        "first_inflow_bands": FIRST_INFLOW_BANDS,
        "pass_through": [PASS_THROUGH_SECONDS, *PASS_THROUGH_RATIO],
    }


# The pool groups: counts over the payment messages of the context's own candidate pool
# (at most `recent + older + distinct` per relation), not over the account's whole
# history, computed by the client (batching.pool_counts.pool_activity). The numbers are
# round, but the choice of counts followed a diagnostic study that had read the data
# generator's mule typology and test-split mules, so test audits are optimistic for them.
# The internal-payer counts and their amount bands suit the generator, which places scam
# victims inside the bank, more than a real bank, so they are a group of their own
# (pool_internal_inflows) that an ablation can drop.
FIRST_INFLOW_BANDS = (100, 1000)
# Rapid pass-through: the next outflow after an inflow follows within a day and moves
# 50 to 100 percent of the inflow amount.
PASS_THROUGH_SECONDS = 86_400
PASS_THROUGH_RATIO = (0.5, 1.0)
# Changes whenever batching.pool_counts.pool_activity changes what a count means.
POOL_ACTIVITY_VERSION = 1
POOL_ACTIVITY_FEATURES = tuple(
    f"pool_{r}_{v}" for r in PAYMENT_RELATIONS for v in ("count", "unique")
) + ("pool_in_unique", "pool_out_unique", "pool_first_in", "pool_pass_through_1d")
POOL_INTERNAL_FEATURES = ("pool_first_in_internal",) + tuple(
    f"pool_first_in_internal_{band}" for band in FIRST_INFLOW_BANDS
)
POOL_GROUPS = ("pool_activity", "pool_internal_inflows")


@dataclass(frozen=True)
class FeatureGroup:
    path: str
    names: tuple[str, ...]
    identity: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()


FEATURE_GROUPS = {
    "entity_meta": FeatureGroup(
        "node",
        tuple("type_" + t for t in NODE_TYPES) + ("is_external", "is_deposit"),
        tuple("type_" + t for t in NODE_TYPES) + ("is_external", "is_deposit"),
    ),
    # Client computed from the hub registry; never requested from TigerGraph.
    "hub_indicator": FeatureGroup("node", ("history_withheld",), identity=("history_withheld",)),
    # The pool groups are client computed from the context's payment messages and never
    # requested from TigerGraph. First-time counts read the pair_history fields and
    # pass-through counts the flow_timing fields.
    "pool_activity": FeatureGroup(
        "summary", POOL_ACTIVITY_FEATURES, requires=("pair_history", "flow_timing")
    ),
    "pool_internal_inflows": FeatureGroup(
        "summary", POOL_INTERNAL_FEATURES, requires=("pair_history",)
    ),
    "message_core": FeatureGroup(
        "message", ("amount", "amount_present", "is_event"), ("amount_present", "is_event")
    ),
    "time_encoding": FeatureGroup(
        "message",
        ("gap_present",)
        + tuple(f"age_fourier_{i}" for i in range(64))
        + tuple(f"gap_fourier_{i}" for i in range(64)),
    ),
    "pair_history": FeatureGroup(
        "message",
        ("pair_prior_count", "pair_first_age_seconds", "pair_first_present"),
        ("pair_first_present",),
    ),
    "flow_timing": FeatureGroup(
        "message",
        (
            "flow_delay_seconds",
            "flow_present",
            "flow_censored",
            "flow_observation_seconds",
            "flow_amount_ratio",
            "flow_ratio_present",
            "flow_same_rail",
        ),
        ("flow_present", "flow_censored", "flow_ratio_present", "flow_same_rail"),
    ),
}
CLIENT_GROUPS = frozenset({"hub_indicator", *POOL_GROUPS})
# The built-in run's groups without the pool groups: the model the pool groups were
# added to.
CORE_GROUPS = (
    "entity_meta",
    "hub_indicator",
    "message_core",
    "time_encoding",
    "pair_history",
    "flow_timing",
)
# The groups of the built-in run (config.DEFAULT_CONFIG): the core groups plus the pool
# groups, which are every group of the registry.
BUILT_IN_GROUPS = (*CORE_GROUPS, *POOL_GROUPS)


# The model architectures: "tgat" is the graph model (model.tgat.TGAT), "summary" the
# controls without attention (model.summary_mlp.SummaryMLP).
ARCHITECTURES = ("tgat", "summary")


@dataclass(frozen=True)
class FeaturePlan:
    """The feature groups a model reads and its architecture.

    "tgat" is the graph model: attention over sampled neighbours, with a summary
    branch for the root's summary columns. "summary" reads only the root's node and
    summary columns (the controls without attention).
    """

    groups: tuple[str, ...] = BUILT_IN_GROUPS
    architecture: str = "tgat"

    def __post_init__(self) -> None:
        if len(set(self.groups)) != len(self.groups) or set(self.groups) - FEATURE_GROUPS.keys():
            raise ValueError("Duplicate or unknown feature groups")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"Architecture must be one of {list(ARCHITECTURES)}")
        for name in self.groups:
            if set(FEATURE_GROUPS[name].requires) - set(self.groups):
                raise ValueError(f"Missing dependencies for {name}")
        if self.architecture != "summary" and "message_core" not in self.groups:
            raise ValueError("Graph models require message_core")
        if self.architecture == "summary" and not self.names("node", "summary"):
            raise ValueError("Summary model needs node or summary inputs")

    def names(self, *paths: str) -> tuple[str, ...]:
        # Registry order is canonical, independent of configuration list order.
        return tuple(
            n
            for g, spec in FEATURE_GROUPS.items()
            if g in self.groups and spec.path in paths
            for n in spec.names
        )

    @property
    def node_names(self) -> tuple[str, ...]:
        return self.names("node", "summary")

    @property
    def edge_names(self) -> tuple[str, ...]:
        return self.names("message")

    def fingerprint(self) -> str:
        """The model inputs: contract, groups, architecture and any pool definitions."""
        value: dict[str, Any] = {
            "contract": contract_fingerprint(),
            "groups": sorted(self.groups),
            "architecture": self.architecture,
        }
        # The contract leaves the pool groups out, so a plan with one covers them here.
        if set(POOL_GROUPS) & set(self.groups):
            value["pool"] = pool_definition(self.groups)
        return fingerprint(value)

    def query_flags(self, hop: int = 1) -> dict[str, bool]:
        """GSQL `include_*` parameters for one hop: one per group TigerGraph computes.

        message_core is always computed and client groups are never requested; every
        message carries its channel and sampling stratum. TGAT models read only node
        and message inputs of children, so their second hop would skip a summary group
        TigerGraph computed; none of the training groups is one (the summary groups are
        the client's pool counts), so both hops send the same flags. Summary models
        fetch no children.
        """
        if hop not in (1, 2):
            raise ValueError("Hop must be 1 or 2")
        skip_summary = hop == 2 and self.architecture == "tgat"
        return {
            "include_" + name: name in self.groups and not (skip_summary and spec.path == "summary")
            for name, spec in FEATURE_GROUPS.items()
            if name != "message_core" and name not in CLIENT_GROUPS
        }


def extraction_plan(model: FeaturePlan) -> FeaturePlan:
    """What the context source asks TigerGraph for: the model's groups but the client ones.

    Client groups are computed locally. The architecture is the model's, so a TGAT
    model skips summary groups at hop 2.
    """
    groups = tuple(g for g in model.groups if g not in CLIENT_GROUPS)
    return FeaturePlan(groups, model.architecture)
