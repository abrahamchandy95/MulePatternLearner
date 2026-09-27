"""Assemble two-hop batches without collapsing different times for the same node.

Roots are fetched with the roots pool (hop 1) and first-hop children with the
children pool (hop 2). Neighbors are resampled from those pools per step
(`sampling.backend.select_resampled`). Hub children become local stubs instead of
fetches, and a child that TigerGraph rejects is masked out of the first hop. Tensors
are assembled by column gathers; Fourier time features are computed on the target
device from `age_ms` and `gap_ms` and never read from TigerGraph. `build_root_batch`,
which training, preparation and scoring share, drops the roots TigerGraph rejected
first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch

from ..contract.bounds import FANOUT
from ..contract.feature_groups import CLIENT_GROUPS, FEATURE_GROUPS, FeaturePlan
from ..contract.graph_schema import ContextKey
from ..contract.sampler_plan import SamplerPlan
from ..sampling.backend import batch_backend, select_resampled
from ..sampling.candidates import CandidateTable
from .features import DAY_MS, base_matrix, edge_block, node_matrix
from .limits import BatchIndex, BatchLimits
from .time_encoding import fourier64_torch

if TYPE_CHECKING:
    from ..data.contexts import ContextSource


_CLIENT_NAMES = frozenset(n for group in CLIENT_GROUPS for n in FEATURE_GROUPS[group].names)


class HubLookup(Protocol):
    """Hub status from history visible before the root cutoff (see data/hub_registry.py).

    `phase` is the batch's visibility phase (1 train, 2 validation, 3 test); unscoped
    batches (score-new) use 3. Positional-only, so implementations may name them freely.
    """

    def is_stub(
        self, node_type: str, node_id: str, root_cutoff_seq: int, phase: int = 3, /
    ) -> bool: ...


def child_key(message: dict[str, Any], parent: ContextKey | None = None) -> ContextKey:
    return ContextKey(
        message["node_type"],
        message["node_id"],
        int(message["event_seq"]),
        int(message["event_ts_ms"]),
        parent.scope_id if parent else "",
        parent.visibility_phase if parent else 3,
    )


def _fetched(rows: Sequence[dict[str, Any] | None]) -> None:
    """Client-computed features (hub_indicator, the pool groups) never come from TigerGraph."""
    for row in rows:
        found = _CLIENT_NAMES & row["features"].keys() if row is not None else ()
        if found:
            raise ValueError(f"TigerGraph returned a client-only feature {sorted(found)}")


def _stub_row(key: ContextKey, message: dict[str, Any]) -> dict[str, Any]:
    """Local context for a hub child: peer metadata only, history withheld."""
    first_ms = int(message["peer_first_ms"])
    if first_ms <= 0 or first_ms > key.cutoff_ms:
        raise ValueError("Peer metadata is not visible at its historical cutoff")
    return {
        "node_type": key.node_type,
        "node_id": key.node_id,
        "cutoff_seq": key.cutoff_seq,
        "cutoff_ms": key.cutoff_ms,
        "scope_id": key.scope_id,
        "visibility_phase": key.visibility_phase,
        "status": "stub",
        "features": {
            "is_external": float(bool(message["peer_external"])),
            "is_deposit": float(bool(message["peer_deposit"])),
            "age_days": (key.cutoff_ms - first_ms) / DAY_MS,
            "history_withheld": 1.0,
        },
        "messages": [],
    }


def _select(
    keys: Sequence[ContextKey],
    rows: Sequence[dict[str, Any]],
    *,
    hop: int,
    fanout: int,
    sampler: SamplerPlan,
    mode: str,
    step_seed: int,
    backend: str,
    device: torch.device,
) -> list[list[dict[str, Any]]]:
    table = CandidateTable.build(keys, rows)
    slots = select_resampled(
        table,
        hop=hop,
        sampler=sampler,
        fanout=fanout,
        mode=mode,
        step_seed=step_seed,
        backend=backend,
        # Selection drives the child fetch, so the torch sampler runs on the host.
        device=device if backend == "cugraph" else "cpu",
    )
    return [[table.messages[j] for j in row if j >= 0] for row in slots.tolist()]


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(value))
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def make_live_batch(
    store: ContextSource,
    roots: list[ContextKey],
    *,
    fanouts: tuple[int, int] = (8, 4),
    device: str | torch.device = "cpu",
    limits: BatchLimits = BatchLimits(),
    plan: FeaturePlan = FeaturePlan(),
    sampler: SamplerPlan = SamplerPlan(),
    hubs: HubLookup | None = None,
    mode: str = "eval",
    step_seed: int = 0,
    stats: dict[str, Any] | None = None,
    sampler_backend: str | None = None,
) -> dict[str, torch.Tensor]:
    """Two-hop temporal batch; `mode="train"` resamples with `step_seed`.

    Hub status of a child and of an outermost peer is looked up at the earliest
    cutoff among the roots that reach it, so it only uses history visible before
    every such prediction time, and in the batch's visibility phase (3 when the
    roots are unscoped), so it only counts events visible in that phase.

    `sampler_backend` is the run's `resolve_backend` result. Training resolves it
    once on the main thread and passes it here, so every batch of a run samples with
    the same backend; None resolves per call (cached probe).
    """
    if not roots or len(fanouts) != 2 or not all(FANOUT.holds(fanout) for fanout in fanouts):
        raise ValueError(
            f"Nonempty roots and two fanouts in [{FANOUT.low},{FANOUT.high}] are required"
        )
    if mode not in ("train", "eval"):
        raise ValueError("Batch mode must be train or eval")
    limits.validate(len(roots), fanouts, plan, sampler)
    if len({(key.scope_id, key.visibility_phase) for key in roots}) != 1:
        raise ValueError("A batch must have one visibility scope and phase")
    phase = roots[0].batch_phase
    device = torch.device(device)
    backend = batch_backend(sampler, device, mode, sampler_backend)
    counts: dict[str, Any] = {"roots": len(roots), "sampler_backend": backend}
    root_rows = store.fetch(list(roots), hop=1)
    if len(root_rows) != len(roots):
        raise ValueError("Context source returned a different number of rows")
    _fetched(root_rows)
    missing = [key for key, row in zip(roots, root_rows, strict=True) if row is None]
    if missing:
        raise ValueError(
            f"TigerGraph rejected {len(missing)} of {len(roots)} root contexts "
            f"(status counts {dict(store.rejections)}); first {missing[0]}"
        )
    accepted = [row for row in root_rows if row is not None]
    if plan.architecture == "summary":
        if stats is not None:
            stats.update(counts, contexts=len(set(roots)), stub_children=0)
            stats.update(rejected_children=0, first_edges=0, second_edges=0)
        return {
            "root_positions": torch.arange(len(roots), device=device),
            "x": _tensor(node_matrix(accepted, plan), device),
        }

    def select(
        keys: Sequence[ContextKey], rows: Sequence[dict[str, Any]], hop: int
    ) -> list[list[dict[str, Any]]]:
        return _select(
            keys,
            rows,
            hop=hop,
            fanout=fanouts[hop - 1],
            sampler=sampler,
            mode=mode,
            step_seed=step_seed,
            backend=backend,
            device=device,
        )

    first = select(roots, accepted, 1)

    # Children: stub hubs locally, fetch the rest with the children pool.
    slot_keys = [[child_key(m, root) for m in msgs] for root, msgs in zip(roots, first)]
    anchor = {root: root.cutoff_seq for root in roots}
    reached: dict[ContextKey, dict[str, Any]] = {}
    for root, keys, msgs in zip(roots, slot_keys, first, strict=True):
        for key, message in zip(keys, msgs, strict=True):
            anchor[key] = min(anchor.get(key, root.cutoff_seq), root.cutoff_seq)
            reached.setdefault(key, message)
    rows: dict[ContextKey, dict[str, Any]] = dict(zip(roots, accepted, strict=True))
    children = [key for key in reached if key not in rows]
    BatchIndex([*roots, *children], capacity=limits.max_contexts)  # reject before fetching
    stubs = [
        k
        for k in children
        if hubs is not None and hubs.is_stub(k.node_type, k.node_id, anchor[k], phase)
    ]
    for key in stubs:
        rows[key] = _stub_row(key, reached[key])
    fetch = [key for key in children if key not in rows]
    rejected: set[ContextKey] = set()
    child_rows = store.fetch(fetch, hop=2) if fetch else []
    _fetched(child_rows)
    for key, row in zip(fetch, child_rows, strict=True):
        if row is None:
            rejected.add(key)
        else:
            rows[key] = row
    lookup = BatchIndex([*roots, *(k for k in children if k not in rejected)], limits.max_contexts)
    unique = lookup.keys
    contexts = [rows[key] for key in unique]
    second = select(unique, contexts, 2)

    arrays: dict[str, np.ndarray] = {
        "root_positions": np.asarray([lookup[key] for key in roots], dtype=np.int64),
        # The distinct roots lead the contexts; only they get pool counts, and the model
        # reads the summary columns of roots only.
        "x": node_matrix(contexts, plan, pooled=len(set(roots))),
        "neighbor_positions": np.zeros((len(roots), fanouts[0]), dtype=np.int64),
    }
    encodings: dict[str, dict[str, np.ndarray]] = {}
    for prefix, selected, fanout in (
        ("first_", first, fanouts[0]),
        ("second_", second, fanouts[1]),
    ):
        items = [
            (i, j, m)
            for i, msgs in enumerate(selected)
            for j, m in enumerate(msgs)
            if prefix == "second_" or slot_keys[i][j] not in rejected
        ]
        index_i = np.asarray([i for i, _, _ in items], dtype=np.int64)
        index_j = np.asarray([j for _, j, _ in items], dtype=np.int64)
        messages = [m for _, _, m in items]
        block = edge_block(messages, plan)
        shape = (len(selected), fanout)
        arrays[prefix + "edge"] = np.zeros((*shape, len(plan.edge_names)), dtype=np.float32)
        arrays[prefix + "edge"][index_i, index_j] = block["edge"]
        for name in ("relation", "rail", "channel", "stratum"):
            arrays[prefix + name] = np.zeros(shape, dtype=np.int64)
            arrays[prefix + name][index_i, index_j] = block[name]
        arrays[prefix + "mask"] = np.zeros(shape, dtype=bool)
        arrays[prefix + "mask"][index_i, index_j] = True
        if "age_ms" in block:
            encodings[prefix] = {"i": index_i, "j": index_j} | {
                k: block[k] for k in ("age_ms", "age_on", "gap_ms", "gap_on")
            }
        if prefix == "first_":
            arrays["neighbor_positions"][index_i, index_j] = [
                lookup[slot_keys[i][j]] for i, j, _ in items
            ]
        else:
            withheld = np.zeros(len(items), dtype=np.float64)
            if hubs is not None and "hub_indicator" in plan.groups:
                withheld[:] = [
                    hubs.is_stub(m["node_type"], m["node_id"], anchor[unique[i]], phase)
                    for i, _, m in items
                ]
            arrays["second_x"] = np.zeros((*shape, len(plan.names("node"))), dtype=np.float32)
            arrays["second_x"][index_i, index_j] = base_matrix(messages, withheld, plan)
    batch = {name: _tensor(value, device) for name, value in arrays.items()}
    if encodings:
        offset = plan.edge_names.index("age_fourier_0")
        for prefix, parts in encodings.items():
            edge = batch[prefix + "edge"]
            for name, start in (("age", offset), ("gap", offset + 64)):
                on = parts[name + "_on"]
                if on.any():
                    i, j = (_tensor(parts[k][on], device) for k in ("i", "j"))
                    delta = _tensor(parts[name + "_ms"][on], device)
                    # Deltas were checked on the host, so the device never syncs here.
                    edge[i, j, start : start + 64] = fourier64_torch(delta, validate=False)
    if stats is not None:
        stats.update(counts)
        stats.update(
            contexts=len(unique),
            stub_children=len(stubs),
            rejected_children=len(rejected),
            first_edges=int(arrays["first_mask"].sum()),
            second_edges=int(arrays["second_mask"].sum()),
        )
    return batch


class PinnedRoots:
    """Serve already fetched root rows to make_live_batch without a second request.

    Anything else (children, other keys) goes to the wrapped source, so concurrent
    batch builders cannot evict a batch's roots between filtering and assembly. The
    plan, pools and rejection counters are the wrapped source's own objects.
    """

    def __init__(
        self, store: ContextSource, keys: list[ContextKey], rows: list[dict[str, Any]]
    ) -> None:
        self.store = store
        self.rows = dict(zip(keys, rows, strict=True))
        self.plan, self.sampler = store.plan, store.sampler
        self.rejections, self.rejections_by_hop = store.rejections, store.rejections_by_hop

    @property
    def query_calls(self) -> int:
        return self.store.query_calls

    def close(self, *, wait: bool = True) -> None:
        """Nothing to close: the wrapped source's owner closes it."""

    def fetch(self, keys: list[ContextKey], *, hop: int = 1) -> list[dict[str, Any] | None]:
        if hop == 1 and all(key in self.rows for key in keys):
            return [self.rows[key] for key in keys]
        return self.store.fetch(keys, hop=hop)


@dataclass
class RootBatch:
    """One assembled batch for the accepted roots; rejected roots are reported, not scored."""

    requested: list[ContextKey]
    accepted: np.ndarray
    batch: dict[str, torch.Tensor] | None
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def keys(self) -> list[ContextKey]:
        return [k for k, ok in zip(self.requested, self.accepted, strict=True) if ok]

    @property
    def rejected(self) -> list[ContextKey]:
        return [k for k, ok in zip(self.requested, self.accepted, strict=True) if not ok]


def build_root_batch(
    store: ContextSource,
    keys: list[ContextKey],
    *,
    fanouts: tuple[int, int],
    device: str | torch.device,
    plan: FeaturePlan,
    sampler: SamplerPlan,
    hubs: HubLookup,
    mode: str,
    step_seed: int = 0,
    sampler_backend: str | None = None,
) -> RootBatch:
    """Drop roots TigerGraph rejected (per-request status), then assemble the rest.

    Rejections are counted by status on ``store.rejections``; the batch statistics
    carry ``rejected_roots``. A batch with no accepted root has ``batch=None``.
    ``sampler_backend`` is the backend resolved once per run (``resolve_backend``);
    None lets make_live_batch resolve it.
    """
    rows = store.fetch(keys, hop=1)
    accepted = np.fromiter((row is not None for row in rows), dtype=bool, count=len(keys))
    stats: dict[str, Any] = {"rejected_roots": int(len(keys) - accepted.sum())}
    kept = [k for k, ok in zip(keys, accepted, strict=True) if ok]
    if not kept:
        return RootBatch(keys, accepted, None, stats)
    pinned = PinnedRoots(store, kept, [row for row in rows if row is not None])
    batch = make_live_batch(
        pinned,
        kept,
        fanouts=fanouts,
        device=device,
        plan=plan,
        sampler=sampler,
        hubs=hubs,
        mode=mode,
        step_seed=step_seed,
        stats=stats,
        sampler_backend=sampler_backend,
    )
    return RootBatch(keys, accepted, batch, stats)


def batch_counts(stats: Mapping[str, Any]) -> dict[str, int]:
    """The counts among a batch's statistics, as ints; the sampler backend is a name."""
    return {
        k: int(v)
        for k, v in stats.items()
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool)
    }


def batch_device(device: torch.device) -> torch.device:
    """Where the batches of a model on ``device`` are assembled.

    CUDA batches are built (and resampled) on the device by the prefetch workers.
    Other devices build on the CPU and copy on the consuming thread (``to_device``).
    """
    return device if device.type == "cuda" else torch.device("cpu")


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """The batch on the model's device; tensors already there are not copied."""
    return {k: v.to(device) for k, v in batch.items()}


def tensor_digests(batch: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    """Fingerprints of a batch's tensors by name, to compare batches across code versions.

    Every tensor gets its dtype, shape and the sha256 of its little-endian bytes. On one
    machine and device the hashes of two equal batches match exactly. Integer and boolean
    tensors come from integer arithmetic, so their hashes also match across machines.
    Floating tensors do not: math libraries round sin, cos and log differently. They
    also get summaries in float64 that other machines reproduce to within rounding: the
    sum, the sum of absolute values, a sum weighted by position (weights 1 to 7, so a
    moved value changes it) and the elements at four fixed flat positions.
    """
    digests: dict[str, dict[str, Any]] = {}
    for name in sorted(batch):
        tensor = batch[name].detach().cpu().contiguous()
        array = tensor.numpy()
        little = array.astype(array.dtype.newbyteorder("<"), copy=False)
        digest: dict[str, Any] = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(little.tobytes()).hexdigest(),
        }
        if tensor.is_floating_point():
            values = tensor.double().flatten()
            count = len(values)
            weights = torch.arange(count, dtype=torch.float64) % 7 + 1
            positions = sorted({0, count // 3, 2 * count // 3, count - 1}) if count else []
            digest |= {
                "sum": _rounded(values.sum()),
                "abs_sum": _rounded(values.abs().sum()),
                "weighted_sum": _rounded((values * weights).sum()),
                "elements": {str(i): _rounded(values[i]) for i in positions},
            }
        digests[name] = digest
    return digests


def _rounded(value: torch.Tensor) -> float:
    """A float64 scalar to 10 significant digits, far below any cross-machine tolerance."""
    return float(f"{float(value):.10g}")
