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
  Without a scope_id (shared_history and score-new) counts are unscoped and
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

from ..common import digest
from .source import CONVERSION_ERRORS, QueryExecutor, checked_rows, run_query

if TYPE_CHECKING:
    from .contract import SamplerPlan

HUB_FILE = "hubs.parquet"
HUB_QUERY = "temporal_hub_registry"
HUB_COLUMNS = (
    "account_id",
    "cutoff_seq",
    "visibility_phase",
    "max_visible",
    "max_degree",
    "reason",
)
HUB_REASONS = ("visible_history",)
MAX_CUTOFFS = 24
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
        if tuple(frame.columns) != HUB_COLUMNS:
            raise ValueError(f"Hub registry needs columns {HUB_COLUMNS}")
        self.frame = (
            frame.astype(
                {
                    "account_id": str,
                    "cutoff_seq": "int64",
                    "visibility_phase": "int64",
                    "max_visible": "int64",
                    "max_degree": "int64",
                    "reason": str,
                }
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
                ("account_id", pa.string()),
                ("cutoff_seq", pa.int64()),
                ("visibility_phase", pa.int64()),
                ("max_visible", pa.int64()),
                ("max_degree", pa.int64()),
                ("reason", pa.string()),
            ]
        )
        table = pa.Table.from_pandas(self.frame, schema=schema, preserve_index=False)
        pending = path.with_suffix(".pending.parquet")
        pq.write_table(table, pending)
        pending.replace(path)


def _parse_hubs(
    rows: list[dict[str, Any]], cutoffs: list[int], threshold: int, scope_id: str
) -> pd.DataFrame:
    checked_rows(rows)
    for row in rows:
        if "scan_cap" in row:
            raise ValueError(
                "The installed temporal_hub_registry still has the scan_cap decision; "
                "run `mule-temporal install`"
            )
        if "threshold" in row and int(row["threshold"]) != threshold:
            raise ValueError(
                f"Hub registry echoed threshold={row['threshold']}, expected {threshold}"
            )
        if "scope_id" in row and str(row["scope_id"]) != scope_id:
            raise ValueError(
                f"Hub registry echoed scope_id={row['scope_id']!r}, expected {scope_id!r}"
            )
        if "cutoff_seqs" in row and sorted(map(int, row["cutoff_seqs"])) != cutoffs:
            raise ValueError("Hub registry echoed different cutoffs")
    pages = [row["hubs"] for row in rows if "hubs" in row]
    if not pages:
        raise ValueError("Hub registry response has no hubs field")
    phases = registry_phases(scope_id)
    records = []
    for page in pages:
        for item in page:
            record = dict(item.get("attributes", item))
            try:
                account = str(record["account_id"])
                cutoff = int(record["cutoff_seq"])
                phase = int(record["visibility_phase"])
                visible_count = int(record["max_visible"])
                degree = int(record["max_degree"])
                reason = str(record["reason"])
            except (KeyError, *CONVERSION_ERRORS):
                raise ValueError(f"Malformed hub registry row: {record}") from None
            if (
                not account
                or cutoff not in cutoffs
                or phase not in phases
                or visible_count <= threshold
                or degree < 0
                or reason != "visible_history"
            ):
                raise ValueError(f"Hub registry row violates the query contract: {record}")
            records.append(
                {
                    "account_id": account,
                    "cutoff_seq": cutoff,
                    "visibility_phase": phase,
                    "max_visible": visible_count,
                    "max_degree": degree,
                    "reason": reason,
                }
            )
    return pd.DataFrame(records, columns=list(HUB_COLUMNS))


def query_hub_registry(
    executor: QueryExecutor,
    cutoff_seqs: Iterable[int],
    *,
    threshold: int,
    scope_id: str = "",
    timeout_s: float = 1800.0,
) -> HubRegistry:
    """Run the read-only temporal_hub_registry query for 1..24 root cutoffs.

    With a scope_id the scope must be ready, and rows cover phases 1, 2 and 3;
    without one the counts are unscoped and rows have phase 3.
    """
    cutoffs = sorted({int(value) for value in cutoff_seqs})
    if not 1 <= len(cutoffs) <= MAX_CUTOFFS or cutoffs[0] <= 0:
        raise ValueError(f"Hub registry needs 1..{MAX_CUTOFFS} positive cutoff sequences")
    if threshold < 1:
        raise ValueError("Hub threshold must be positive")
    rows = run_query(
        executor,
        HUB_QUERY,
        {"cutoff_seqs": cutoffs, "threshold": threshold, "scope_id": scope_id},
        timeout_s=timeout_s,
    )
    return HubRegistry(
        _parse_hubs(rows, cutoffs, threshold, scope_id),
        cutoff_seqs=cutoffs,
        threshold=threshold,
        scope_id=scope_id,
    )


def hub_manifest(registry: HubRegistry, path: Path) -> dict[str, Any]:
    """Manifest fields recorded next to a saved registry."""
    return {
        "hubs_sha256": digest(path),
        "hub_threshold": registry.threshold,
        "hub_scope_id": registry.scope_id,
        "hub_counts": registry.counts(),
    }


def load_hub_registry(dataset: Path, manifest: dict[str, Any]) -> HubRegistry:
    """Load and verify the prepared registry for the dataset's cutoffs and scope."""
    path = dataset / HUB_FILE
    if "hubs_sha256" not in manifest or "hub_scope_id" not in manifest:
        raise ValueError(
            "Prepared dataset has no scoped hub registry; prepare it again (set a new prepared_id)"
        )
    if not path.exists() or digest(path) != manifest["hubs_sha256"]:
        raise ValueError(f"Prepared hub registry changed or is missing: {path}")
    config = manifest.get("config")
    if isinstance(config, dict):
        strict = config.get("evaluation_protocol") == "strict_inductive"
        expected = str(config.get("scope_id") or "") if strict else ""
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
