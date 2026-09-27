"""Cutoff- and scope-safe registry of hub accounts that are never expanded as child contexts.

A hub is an Account whose history visible before a ROOT context's cutoff exceeds
the context capacity (`visible_history`). Batches replace such children with
local stub contexts (`history_withheld = 1`) instead of fetching them.

Leakage:
- Time: the visible count uses only edges with event_seq < the root cutoff, so
  hub status never depends on events after the prediction time. Nothing else
  decides it (all-time degree is reported for information only).
- Scope: with a scope_id, TigerGraph counts per visibility phase only the
  events whose Account endpoints are all allowed in that phase (the endpoint
  rule of temporal_training_context), and a hub must itself be allowed in the
  phase. Held-out partitions therefore cannot change a phase-1 stub decision.
  Rows carry their phase and `is_stub` is keyed by (root cutoff, phase).
  Without a scope_id (`mule score`) counts are unscoped and
  every row has phase 3. Counts cover all currencies, an upper bound of the
  context query's USD capacity count.

Completeness: a child's cutoff (its connecting event) precedes its root's
cutoff and a child context inherits its root's phase, so a child that is not a
stub has at most `threshold` visible events per relation and never exceeds
max_history.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..artifacts import atomic_write, file_digest
from ..contract.graph_schema import HUB_COLUMNS, HUB_REASONS, context_scope
from ..paths import DatasetPaths
from ..runtime.progress import warn

if TYPE_CHECKING:
    from ..contract.feature_groups import FeaturePlan
    from ..contract.sampler_plan import SamplerPlan


SCOPED_PHASES = (1, 2, 3)
UNSCOPED_PHASES = (3,)


def hub_threshold(sampler: SamplerPlan) -> int:
    """The smallest per-relation history capacity of any fetched context."""
    return min(sampler.roots.max_history, sampler.children.max_history)


def registry_phases(scope_id: str) -> tuple[int, ...]:
    """Phases a registry covers: all three with a scope, only 3 without one."""
    return SCOPED_PHASES if scope_id else UNSCOPED_PHASES


class HubRegistry:
    """Hub accounts per (root cutoff_seq, visibility phase); `is_stub` is the batch lookup."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        cutoff_seqs: Iterable[int] | None,
        threshold: int,
        scope_id: str = "",
    ) -> None:
        if tuple(frame.columns) != tuple(HUB_COLUMNS):
            raise ValueError(f"Hub registry needs columns {tuple(HUB_COLUMNS)}")
        self.frame = (
            frame.astype(
                {name: "int64" if kind is int else str for name, kind in HUB_COLUMNS.items()}
            )
            .sort_values(["cutoff_seq", "visibility_phase", "account_id"])
            .reset_index(drop=True)
        )
        self.cutoff_seqs = None if cutoff_seqs is None else frozenset(map(int, cutoff_seqs))
        self.threshold, self.scope_id = int(threshold), str(scope_id)
        phases = frozenset(registry_phases(self.scope_id))
        # None (only for `empty`) accepts every phase.
        self.phases: frozenset[int] | None = phases
        if not set(self.frame.reason) <= set(HUB_REASONS):
            raise ValueError("Unknown hub reason")
        if self.frame.duplicated(["account_id", "cutoff_seq", "visibility_phase"]).any():
            raise ValueError("Duplicate hub rows")
        if self.cutoff_seqs is not None and not set(self.frame.cutoff_seq) <= self.cutoff_seqs:
            raise ValueError("Hub rows reference cutoffs outside the registry")
        if not set(self.frame.visibility_phase) <= phases:
            raise ValueError("Hub rows reference visibility phases outside the registry")
        index: dict[tuple[int, int], set[str]] = {}
        for account, cutoff, phase in zip(
            self.frame.account_id.tolist(),
            self.frame.cutoff_seq.tolist(),
            self.frame.visibility_phase.tolist(),
            strict=True,
        ):
            index.setdefault((int(cutoff), int(phase)), set()).add(str(account))
        self._index = {key: frozenset(accounts) for key, accounts in index.items()}

    @classmethod
    def empty(cls) -> HubRegistry:
        """No hubs at any cutoff or phase (tests and graphs without hubs)."""
        registry = cls(
            pd.DataFrame({name: [] for name in HUB_COLUMNS}), cutoff_seqs=None, threshold=0
        )
        registry.phases = None
        return registry

    def __len__(self) -> int:
        return len(self.frame)

    def is_stub(self, node_type: str, node_id: str, root_cutoff_seq: int, phase: int = 3) -> bool:
        """True when this child must not be fetched.

        Keyed by the ROOT context's cutoff and the batch's visibility phase (3
        for unscoped batches). A cutoff or phase the registry does not cover is
        an error, never a silent "not a hub".
        """
        if self.cutoff_seqs is not None and root_cutoff_seq not in self.cutoff_seqs:
            raise ValueError(f"Hub registry does not cover root cutoff_seq {root_cutoff_seq}")
        if self.phases is not None and phase not in self.phases:
            covered = ", ".join(map(str, sorted(self.phases)))
            raise ValueError(
                f"Hub registry (scope {self.scope_id!r}) covers visibility phases {covered}, "
                f"not {phase}; a scoped batch needs the scoped registry of its preparation"
            )
        if node_type != "Account":
            return False
        return node_id in self._index.get((int(root_cutoff_seq), int(phase)), frozenset())

    def counts(self) -> dict[str, dict[str, int]]:
        """Hub count per cutoff and phase: {"<cutoff_seq>": {"<phase>": n}}."""
        cutoffs = (
            sorted(self.cutoff_seqs)
            if self.cutoff_seqs is not None
            else sorted({cutoff for cutoff, _ in self._index})
        )
        phases = (
            sorted(self.phases)
            if self.phases is not None
            else sorted({phase for _, phase in self._index})
        )
        return {
            str(cutoff): {str(phase): len(self._index.get((cutoff, phase), ())) for phase in phases}
            for cutoff in cutoffs
        }

    def save(self, path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pa.schema(
            [
                (name, pa.int64() if kind is int else pa.string())
                for name, kind in HUB_COLUMNS.items()
            ]
        )
        table = pa.Table.from_pandas(self.frame, schema=schema, preserve_index=False)
        with atomic_write(path) as pending:
            pq.write_table(table, pending)


def hub_manifest(registry: HubRegistry, path: Path) -> dict[str, Any]:
    """Manifest fields recorded next to a saved registry."""
    return {
        "hubs_sha256": file_digest(path),
        "hub_threshold": registry.threshold,
        "hub_scope_id": registry.scope_id,
        "hub_counts": registry.counts(),
    }


def load_hub_registry(dataset: DatasetPaths, manifest: dict[str, Any]) -> HubRegistry:
    """Load and verify the prepared registry for the dataset's cutoffs and scope."""
    path = dataset.hubs
    if not path.exists() or file_digest(path) != manifest["hubs_sha256"]:
        raise ValueError(f"Prepared hub registry changed or is missing: {path}")
    expected = context_scope(manifest["source"]["scope_id"])
    if manifest["hub_scope_id"] != expected:
        raise ValueError(
            f"Prepared hub registry was computed for scope {manifest['hub_scope_id']!r}, "
            f"but the dataset's contexts use scope {expected!r}; prepare it again"
        )
    registry = HubRegistry(
        pd.read_parquet(path),
        cutoff_seqs=[int(value) for value in manifest["cutoff_seqs"].values()],
        threshold=int(manifest["hub_threshold"]),
        scope_id=str(manifest["hub_scope_id"]),
    )
    if registry.counts() != manifest.get("hub_counts"):
        raise ValueError(
            "Hub registry counts differ from the manifest: "
            + json.dumps({"file": registry.counts(), "manifest": manifest.get("hub_counts")})
        )
    return registry


def warn_hub_stubs(hubs: HubRegistry, plan: FeaturePlan) -> None:
    """Warn when hub children become stubs the model cannot recognise as hubs."""
    if len(hubs) and "hub_indicator" not in plan.groups:
        warn(
            "hub_stubs",
            f"The hub registry lists {len(hubs)} hub rows but the feature plan has no "
            "hub_indicator group: hub children are replaced by stubs without history, which "
            "this model cannot tell apart from dormant accounts. Add hub_indicator to "
            "features to give stubs their history_withheld flag.",
        )
