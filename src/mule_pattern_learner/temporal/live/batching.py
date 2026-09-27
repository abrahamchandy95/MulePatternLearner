"""Assemble two-hop batches without collapsing different times for the same node.

Roots are fetched with the roots pool (hop 1) and first-hop children with the
children pool (hop 2). Neighbors are resampled from those pools per step
(`sampler.select_resampled`). Hub children become local stubs instead of fetches, and a child that TigerGraph
rejects is masked out of the first hop. Tensors are assembled by column gathers;
Fourier time features are computed on the target device from `age_ms` and
`gap_ms` and never read from TigerGraph. `build_root_batch`, which training,
preparation and scoring share, drops the roots TigerGraph rejected first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch

from ..encoding import fourier64_torch
from .contract import (
    CHANNELS,
    CLIENT_GROUPS,
    FEATURE_GROUPS,
    FIRST_INFLOW_BANDS,
    PASS_THROUGH_RATIO,
    PASS_THROUGH_SECONDS,
    POOL_ACTIVITY_FEATURES,
    POOL_GROUPS,
    POOL_INTERNAL_FEATURES,
    RAILS,
    RELATIONS,
    STRATA,
    ContextKey,
    FeaturePlan,
    SamplerPlan,
)
from .memory import BatchIndex, BatchLimits
from .sampler import CandidateTable, resolve_backend, select_resampled

if TYPE_CHECKING:
    from .source import ContextSource

DAY_MS = 86_400_000
_IDENTITY = frozenset(
    {"gap_present"} | {n for spec in FEATURE_GROUPS.values() for n in spec.identity}
)
_NODE_NAMES = frozenset(
    n for spec in FEATURE_GROUPS.values() if spec.path in ("node", "summary") for n in spec.names
)
_CLIENT_NAMES = frozenset(n for group in CLIENT_GROUPS for n in FEATURE_GROUPS[group].names)
_POOL_NAMES = frozenset(POOL_ACTIVITY_FEATURES + POOL_INTERNAL_FEATURES)
_INCOMING = frozenset({"zelle_in", "payment_in"})
_OUTGOING = frozenset({"zelle_out", "payment_out"})
_RELATION = {name: i for i, name in enumerate(RELATIONS)}
_RAIL = {name: i for i, name in enumerate(RAILS)}
_CHANNEL = {name: i for i, name in enumerate(CHANNELS)}
_STRATUM = {name: i for i, name in enumerate(STRATA)}


class HubLookup(Protocol):
    """Hub status from history visible before the root cutoff (see live/hubs.py).

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


def _log_columns(names: Sequence[str]) -> np.ndarray:
    return np.asarray([n not in _IDENTITY and "_fourier_" not in n for n in names], dtype=bool)


def pool_activity(context: dict[str, Any]) -> dict[str, float]:
    """The pool groups (pool_activity, pool_internal_inflows) of one context.

    These are counts over the payment messages of the context's own candidate pool,
    at most `recent + older + distinct` per relation (`PoolPlan`), not over the account's
    whole history; the `distinct` stratum favours new counterparties. TigerGraph returns
    the messages strictly before the context cutoff and computes their pair and flow
    fields there over the whole visible history, so every value is cutoff-safe. A
    first-time inflow has no earlier payment in its directed pair
    (`pair_prior_count == 0`; the GSQL pair is relation, rail and peer), and an internal
    one has a peer that is not external. A rapid pass-through is an inflow whose next
    outflow in the visible history (the per-message `flow_*` fields; pool events are
    never paired here) follows within PASS_THROUGH_SECONDS and moves PASS_THROUGH_RATIO
    of the inflow amount. Stubs and contexts without payments get zeros.
    """
    peers: dict[str, set[tuple[str, str]]] = {r: set() for r in RELATIONS[:4]}
    counts = dict.fromkeys(RELATIONS[:4], 0)
    bands = dict.fromkeys(FIRST_INFLOW_BANDS, 0)
    first_in = first_internal = pass_through = 0
    low, high = PASS_THROUGH_RATIO
    try:
        for m in context["messages"]:
            relation = m["relation"]
            if relation not in counts:
                continue  # associations
            counts[relation] += 1
            peers[relation].add((m["node_type"], m["node_id"]))
            if relation not in _INCOMING:
                continue
            if int(m["pair_prior_count"]) == 0:
                first_in += 1
                if not m["peer_external"]:
                    first_internal += 1
                    for band in bands:
                        bands[band] += bool(m["amount_present"]) and m["amount"] >= band
            pass_through += bool(
                m["flow_present"]
                and m["flow_ratio_present"]
                and m["flow_delay_seconds"] <= PASS_THROUGH_SECONDS
                and low <= m["flow_amount_ratio"] <= high
            )
    except KeyError as error:
        raise ValueError(f"Message lacks required field {error.args[0]!r}") from None
    values: dict[str, float] = {}
    for relation in RELATIONS[:4]:
        values[f"pool_{relation}_count"] = counts[relation]
        values[f"pool_{relation}_unique"] = len(peers[relation])
    values |= {
        "pool_in_unique": len({peer for r in _INCOMING for peer in peers[r]}),
        "pool_out_unique": len({peer for r in _OUTGOING for peer in peers[r]}),
        "pool_first_in": first_in,
        "pool_pass_through_1d": pass_through,
        "pool_first_in_internal": first_internal,
        **{f"pool_first_in_internal_{band}": n for band, n in bands.items()},
    }
    return values


def _pooled(plan: FeaturePlan) -> bool:
    return any(group in plan.groups for group in POOL_GROUPS)


# Vectorized assembly -------------------------------------------------------------


def _column(items: Sequence[dict[str, Any]], name: str, dtype: Any = np.float64) -> np.ndarray:
    try:
        return np.fromiter((m[name] for m in items), dtype=dtype, count=len(items))
    except KeyError:
        raise ValueError(f"Message lacks required field {name!r}") from None
    except (TypeError, ValueError) as error:  # keep the tuple form for Python 3.12/3.13
        raise ValueError(f"Message field {name!r} is not numeric: {error}") from None


def node_matrix(
    rows: Sequence[dict[str, Any]], plan: FeaturePlan, *, pooled: int | None = None
) -> np.ndarray:
    """The node and summary columns of many contexts (batch_reference.node_features).

    Only the first ``pooled`` rows (all when None) get the client-computed pool counts;
    the rest keep zeros there. Split batches pass their roots, which lead the rows,
    since the split model reads the pool columns of roots only.
    """
    names = plan.node_names
    for row in rows:
        unexpected = set(row["features"]) - _NODE_NAMES
        if unexpected:
            raise ValueError(f"Unrecognized feature fields: {sorted(unexpected)}")
    pools = [pool_activity(row) for row in rows[:pooled]] if _pooled(plan) else []
    values = np.zeros((len(rows), len(names)), dtype=np.float64)
    for j, name in enumerate(names):
        if name in _POOL_NAMES:
            values[: len(pools), j] = [pool[name] for pool in pools]
            continue
        values[:, j] = np.fromiter(
            (
                1.0 if name == "type_" + row["node_type"] else row["features"].get(name, 0)
                for row in rows
            ),
            dtype=np.float64,
            count=len(rows),
        )
    log = _log_columns(names)
    values[:, log] = np.log1p(values[:, log])
    return values.astype(np.float32)


def base_matrix(
    messages: Sequence[dict[str, Any]], withheld: np.ndarray, plan: FeaturePlan
) -> np.ndarray:
    """The base columns of many second-hop messages (batch_reference.base_features)."""
    names = plan.names("node")
    cutoff = _column(messages, "event_ts_ms", np.int64)
    first = _column(messages, "peer_first_ms", np.int64)
    if np.any(first <= 0) or np.any(first > cutoff):
        raise ValueError("Peer metadata is not visible at its historical cutoff")
    values = np.zeros((len(messages), len(names)), dtype=np.float64)
    for j, name in enumerate(names):
        if name.startswith("type_"):
            values[:, j] = [m["node_type"] == name[5:] for m in messages]
        elif name == "is_external":
            values[:, j] = _column(messages, "peer_external")
        elif name == "is_deposit":
            values[:, j] = _column(messages, "peer_deposit")
        elif name == "age_days":
            values[:, j] = (cutoff - first) / DAY_MS
        elif name == "history_withheld":
            values[:, j] = withheld
    log = _log_columns(names)
    values[:, log] = np.log1p(values[:, log])
    return values.astype(np.float32)


def _stratum(message: dict[str, Any]) -> int:
    name = message.get("stratum", "recent" if message["event_id"] else "association")
    try:
        return _STRATUM[name]
    except KeyError:
        raise ValueError(f"Unknown sampling stratum {name!r}") from None


def edge_block(messages: Sequence[dict[str, Any]], plan: FeaturePlan) -> dict[str, np.ndarray]:
    """Edge features and categorical codes for many messages, by column.

    Fourier columns stay zero here; `age_ms`/`gap_ms` and their masks are returned
    so the encoding can run on the target device.
    """
    names = plan.edge_names
    count = len(messages)
    event = np.fromiter((bool(m["event_id"]) for m in messages), dtype=bool, count=count)
    values = np.zeros((count, len(names)), dtype=np.float64)
    for j, name in enumerate(names):
        if "_fourier_" not in name:
            values[:, j] = event if name == "is_event" else _column(messages, name)
    log = _log_columns(names)
    values[:, log] = np.log1p(values[:, log])
    try:
        relation = [_RELATION[m["relation"]] for m in messages]
        rail = [_RAIL[m["rail"]] for m in messages]
    except KeyError as error:
        raise ValueError(f"Unknown relation or rail {error.args[0]!r}") from None
    other = _CHANNEL["other"]
    block = {
        "edge": values.astype(np.float32),
        "relation": np.asarray(relation, dtype=np.int64),
        "rail": np.asarray(rail, dtype=np.int64),
        "channel": np.asarray(
            [_CHANNEL.get(m.get("channel", "unknown"), other) for m in messages], dtype=np.int64
        ),
        "stratum": np.asarray([_stratum(m) for m in messages], dtype=np.int64),
    }
    if "time_encoding" in plan.groups:
        gap_on = event & _column(messages, "gap_present", bool)
        events = [m for m, e in zip(messages, event, strict=True) if e]
        age = np.zeros(count, dtype=np.int64)
        age[event] = _column(events, "age_ms", np.int64)
        gap = np.zeros(count, dtype=np.int64)
        gap[gap_on] = _column(
            [m for m, g in zip(messages, gap_on, strict=True) if g], "gap_ms", np.int64
        )
        if np.any(age < 0) or np.any(gap < 0):
            raise ValueError("Future timestamps cannot be encoded as historical context")
        block |= {"age_ms": age, "age_on": event, "gap_ms": gap, "gap_on": gap_on}
    return block


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


def batch_backend(
    sampler: SamplerPlan, device: torch.device, mode: str, resolved: str | None = None
) -> str:
    """The backend one batch selects with: "torch" or "cugraph".

    `resolved` is the run's `resolve_backend(sampler, device)` result; None resolves
    here (the probe is cached). Evaluation always runs the hash-keyed torch path, so
    there a resolved `cugraph` is accepted on any device and not used.
    """
    if resolved is not None:
        allowed = ("torch", "cugraph") if sampler.backend == "auto" else (sampler.backend,)
        if resolved not in allowed:
            raise ValueError(
                f"sampler_backend {resolved!r} does not fit the sampler backend "
                f"{sampler.backend!r}; expected one of {allowed}"
            )
    if mode == "eval":
        return "torch"
    if resolved is None:
        return resolve_backend(sampler, device)
    if resolved == "cugraph" and device.type != "cuda":
        raise ValueError(f"sampler_backend cugraph needs a CUDA batch device, got {device}")
    return resolved


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
    if not roots or len(fanouts) != 2 or min(fanouts) < 1 or max(fanouts) > 64:
        raise ValueError("Nonempty roots and two fanouts in [1,64] are required")
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
            f"(status counts {dict(getattr(store, 'rejections', {}))}); first {missing[0]}"
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
    batch builders cannot evict a batch's roots between filtering and assembly.
    """

    def __init__(
        self, store: ContextSource, keys: list[ContextKey], rows: list[dict[str, Any]]
    ) -> None:
        self.store = store
        self.rows = dict(zip(keys, rows, strict=True))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)

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
        pinned,  # type: ignore[arg-type]
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
