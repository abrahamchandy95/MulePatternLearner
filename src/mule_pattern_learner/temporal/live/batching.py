"""Assemble two-hop batches without collapsing different times for the same node.

Roots are fetched with the roots pool (hop 1) and first-hop children with the
children pool (hop 2). Neighbors are chosen by a deterministic legacy policy
(`recent`, `stratified`) or resampled per step (`sampler.select_resampled`). Hub
children become local stubs instead of fetches, and a child that TigerGraph
rejects is masked out of the first hop. Tensors are assembled by column gathers;
Fourier time features are computed on the target device from `age_ms` and
`gap_ms` and never read from TigerGraph.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch

from ..encoding import fourier64, fourier64_torch
from .contract import (
    CHANNELS,
    CLIENT_GROUPS,
    FEATURE_GROUPS,
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


def select_messages(
    context: dict[str, Any],
    fanout: int,
    sampler: SamplerPlan = SamplerPlan(),
    *,
    second_hop: bool = False,
) -> list[dict[str, Any]]:
    """Deterministic relation-interleaved recent history; no label-based selection."""
    if fanout < 1:
        raise ValueError("Fanout must be positive")
    if sampler.policy == "resample":
        raise ValueError("The resample policy selects with sampler.select_resampled")
    if sampler.policy == "stratified":
        return stratified_messages(
            context, fanout, second_hop=second_hop, association_slots=sampler.association_slots
        )
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in context["messages"]:
        buckets[message["relation"]].append(message)
    for messages in buckets.values():
        messages.sort(key=lambda m: (-int(m["event_seq"]), m["event_id"], m["node_id"]))
    selected = []
    for position in range(max((len(v) for v in buckets.values()), default=0)):
        for relation in RELATIONS:
            if position < len(buckets[relation]):
                selected.append(buckets[relation][position])
                if len(selected) == fanout:
                    return selected
    return selected


def stratified_messages(
    context: dict[str, Any],
    fanout: int,
    *,
    second_hop: bool = False,
    association_slots: int = 2,
) -> list[dict[str, Any]]:
    """Reserve history strata before backfill; associations cannot crowd out payments."""
    buckets: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for m in context["messages"]:
        buckets[
            (m["relation"], m.get("stratum", "recent" if m["event_id"] else "association"))
        ].append(m)
    for rows in buckets.values():
        rows.sort(key=lambda m: (-int(m["event_seq"]), m["event_id"], m["node_id"]))
    payments: list[dict[str, Any]] = []
    associations: list[dict[str, Any]] = []
    # First one recent per relation, then one older and one diverse, then refill.
    for pos in range(max((len(v) for v in buckets.values()), default=0)):
        for stratum in ("recent", "older", "distinct"):
            for rel in RELATIONS[:4]:
                rows = buckets[(rel, stratum)]
                if pos < len(rows):
                    payments.append(rows[pos])
        for rel in RELATIONS[4:]:
            rows = buckets[(rel, "association")]
            if pos < len(rows):
                associations.append(rows[pos])
    reserve = min(association_slots, len(associations), fanout // 4) if not second_hop else 0
    chosen = payments[: fanout - reserve] + associations[:reserve]
    if not second_hop:
        chosen += (payments[fanout - reserve :] + associations[reserve:])[: fanout - len(chosen)]
    return chosen


def _transform(name: str, value: float) -> float:
    if name in _IDENTITY or "_fourier_" in name:
        return value
    return float(np.log1p(value))


def _log_columns(names: Sequence[str]) -> np.ndarray:
    return np.asarray([n not in _IDENTITY and "_fourier_" not in n for n in names], dtype=bool)


def node_features(context: dict[str, Any], plan: FeaturePlan = FeaturePlan()) -> np.ndarray:
    values = {**context["features"], "type_" + context["node_type"]: 1.0}
    unexpected = set(values) - _NODE_NAMES
    if unexpected:
        raise ValueError(f"Unrecognized feature fields: {sorted(unexpected)}")
    return np.asarray([_transform(n, values.get(n, 0)) for n in plan.node_names], dtype=np.float32)


def base_features(
    message: dict[str, Any], plan: FeaturePlan = FeaturePlan(), *, history_withheld: bool = False
) -> np.ndarray:
    """Outermost peer features known from the message alone (scalar reference)."""
    key = child_key(message)
    first_ms = int(message["peer_first_ms"])
    if first_ms <= 0 or first_ms > key.cutoff_ms:
        raise ValueError("Peer metadata is not visible at its historical cutoff")
    values = {
        "type_" + key.node_type: 1,
        "is_external": message["peer_external"],
        "is_deposit": message["peer_deposit"],
        "age_days": (key.cutoff_ms - first_ms) / DAY_MS,
        "history_withheld": int(history_withheld),
    }
    return np.asarray(
        [_transform(n, values.get(n, 0)) for n in plan.names("node")], dtype=np.float32
    )


def edge_features(message: dict[str, Any], plan: FeaturePlan = FeaturePlan()) -> np.ndarray:
    """Scalar reference for one message; Fourier features come from age_ms/gap_ms."""
    values = {**message, "is_event": bool(message["event_id"])}
    if values["is_event"] and "time_encoding" in plan.groups:
        age = fourier64(np.array([message["age_ms"]], dtype=np.int64))[0]
        values.update({f"age_fourier_{i}": v for i, v in enumerate(age)})
        if message["gap_present"]:
            gap = fourier64(np.array([message["gap_ms"]], dtype=np.int64))[0]
            values.update({f"gap_fourier_{i}": v for i, v in enumerate(gap)})
    missing = [n for n in plan.edge_names if n not in values and "_fourier_" not in n]
    if missing:
        raise ValueError(f"Message lacks required fields: {missing}")
    return np.asarray([_transform(n, values.get(n, 0)) for n in plan.edge_names], dtype=np.float32)


# Vectorized assembly -------------------------------------------------------------


def _column(items: Sequence[dict[str, Any]], name: str, dtype: Any = np.float64) -> np.ndarray:
    try:
        return np.fromiter((m[name] for m in items), dtype=dtype, count=len(items))
    except KeyError:
        raise ValueError(f"Message lacks required field {name!r}") from None
    except (TypeError, ValueError) as error:  # keep the tuple form for Python 3.12/3.13
        raise ValueError(f"Message field {name!r} is not numeric: {error}") from None


def node_matrix(rows: Sequence[dict[str, Any]], plan: FeaturePlan) -> np.ndarray:
    """`node_features` for many contexts, by column."""
    names = plan.node_names
    for row in rows:
        unexpected = set(row["features"]) - _NODE_NAMES
        if unexpected:
            raise ValueError(f"Unrecognized feature fields: {sorted(unexpected)}")
    values = np.zeros((len(rows), len(names)), dtype=np.float64)
    for j, name in enumerate(names):
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
    """`base_features` for many second-hop messages, by column."""
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
    """Client-computed features (hub_indicator) must never come from TigerGraph."""
    for row in rows:
        if row is not None and _CLIENT_NAMES & row["features"].keys():
            raise ValueError("TigerGraph returned a client-only feature (history_withheld)")


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
    if sampler.policy != "resample":
        return [select_messages(row, fanout, sampler, second_hop=hop == 2) for row in rows]
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
    """The backend one batch selects with: "deterministic", "torch" or "cugraph".

    `resolved` is the run's `resolve_backend(sampler, device)` result; None resolves
    here (the probe is cached). Evaluation always runs the hash-keyed torch path, so
    there a resolved `cugraph` is accepted on any device and not used.
    """
    if resolved is not None:
        if sampler.policy != "resample":
            allowed: tuple[str, ...] = ("deterministic",)
        elif sampler.backend == "auto":
            allowed = ("torch", "cugraph")
        else:
            allowed = (sampler.backend,)
        if resolved not in allowed:
            raise ValueError(
                f"sampler_backend {resolved!r} does not fit the {sampler.policy} sampler "
                f"with backend {sampler.backend!r}; expected one of {allowed}"
            )
    if sampler.policy != "resample":
        return "deterministic"
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
    phase = roots[0].visibility_phase if roots[0].scope_id else 3
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
        "x": node_matrix(contexts, plan),
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
