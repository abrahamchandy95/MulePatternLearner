"""Per-step neighbor resampling from bounded, cutoff-safe candidate pools.

TigerGraph returns a bounded candidate pool per context (`PoolPlan`). The `resample`
policy draws, per context and relation, a uniform subset without replacement and
merges it into the fanout slots with the association reserve of the stratified
policy. Two interchangeable subset samplers exist:

- `TorchGroupedSampler`: random keys plus a segmented rank; runs on CPU, MPS or CUDA.
- `CuGraphSampler`: pylibcugraph's heterogeneous temporal sampler on one CUDA GPU.

Evaluation always uses device-independent hash keys on the torch path, so scores do
not depend on the machine or the backend. The hop enters those keys through
`hop_seed(evaluation_seed, hop)`, so a root's hop-2 draw is independent of its hop-1
draw, as in training.

Training draws depend on the backend. The torch sampler takes its keys from a CPU
`torch.Generator` seeded by the step seed, so with `backend = "torch"` a fixed seed
reproduces the same neighborhoods on every device. The cuGraph sampler draws with
cuGraph's own RNG (`random_state` derived from the same step seed): its subsets are
equally distributed (uniform without replacement per context and relation) but
differ from the torch sampler's for the same seed. Compare runs across machines
with `backend = "torch"`, and resolve the backend once per run (`resolve_backend`).

Time keys make cuGraph's single strict comparison match the visibility contract:
payments use `2*event_seq`, associations (emitted at the context cutoff) use
`2*cutoff_seq - 1`, and a context seeds at `2*cutoff_seq`. With
`strictly_decreasing` a payment is eligible iff `event_seq < cutoff_seq`, and every
association of the context is eligible.
"""

from __future__ import annotations

from collections.abc import Sequence
import contextlib
from dataclasses import dataclass
import functools
import hashlib
import threading
from typing import Any, NamedTuple
import warnings

import numpy as np
import torch

from .contract import ContextKey, RELATIONS, SamplerPlan

PAYMENT_RELATIONS = 4
NUM_RELATIONS = len(RELATIONS)
RELATION_INDEX = {name: i for i, name in enumerate(RELATIONS)}
MIN_PYLIBCUGRAPH = (26, 4)  # hop-0 time filter (25.12) and null-label fix (26.04)
PYLIBCUGRAPH_PIN = "pylibcugraph-cu12/cu13==26.8.* (the cuda12 or cuda13 extra)"
PROBE_SEED = 20260924
_MASK64 = (1 << 64) - 1


def stable_hash(text: str) -> int:
    """64-bit hash that is identical across processes, machines and Python versions."""
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def context_hash(key: ContextKey) -> int:
    fields = (key.node_type, key.node_id, key.cutoff_seq, key.cutoff_ms)
    return stable_hash("\x1f".join(map(str, (*fields, key.scope_id, key.visibility_phase))))


def splitmix64(values: np.ndarray | np.integer[Any] | int) -> np.ndarray:
    """Vectorized SplitMix64 finalizer over uint64 (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        z = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def hop_seed(step_seed: int, hop: int) -> int:
    """Hop 1 uses the step seed itself; hop 2 uses an independent derived stream."""
    seed = int(step_seed) & _MASK64
    return seed if hop == 1 else int(splitmix64(np.uint64(seed ^ hop)))


@dataclass(frozen=True)
class CandidateTable:
    """Candidate messages of C contexts, one row each, in canonical order.

    Rows are sorted by (context, relation, -event_seq, event_id, node_id), so keys
    drawn in row order do not depend on the order TigerGraph printed them in.
    """

    keys: tuple[ContextKey, ...]  # [C]
    context: np.ndarray  # int64 [N]
    relation: np.ndarray  # int64 [N], index into RELATIONS
    time_key: np.ndarray  # int64 [N]
    seed_time: np.ndarray  # int64 [C], 2 * cutoff_seq
    messages: tuple[dict[str, Any], ...]

    @property
    def num_contexts(self) -> int:
        return len(self.seed_time)

    def __len__(self) -> int:
        return len(self.context)

    @functools.cached_property
    def context_hash(self) -> np.ndarray:
        """uint64 [C] stable hash of every context key (evaluation keys only)."""
        return np.asarray([context_hash(k) for k in self.keys], dtype=np.uint64)

    @functools.cached_property
    def item_hash(self) -> np.ndarray:
        """uint64 [N] stable hash of relation:event_id (payments) or relation:node_id."""
        return np.asarray(
            [
                stable_hash(
                    m["relation"] + ":" + (m["event_id"] if r < PAYMENT_RELATIONS else m["node_id"])
                )
                for m, r in zip(self.messages, self.relation.tolist(), strict=True)
            ],
            dtype=np.uint64,
        )

    @classmethod
    def build(cls, keys: Sequence[ContextKey], rows: Sequence[dict[str, Any]]) -> CandidateTable:
        if len(keys) != len(rows):
            raise ValueError("One candidate row per context is required")
        context, relation, time_key, messages = [], [], [], []
        for c, (key, row) in enumerate(zip(keys, rows, strict=True)):
            ordered = sorted(
                row["messages"],
                key=lambda m: (_relation(m), -int(m["event_seq"]), m["event_id"], m["node_id"]),
            )
            for m in ordered:
                r = _relation(m)
                payment = r < PAYMENT_RELATIONS
                if payment and not m["event_id"]:
                    raise ValueError("Payment candidate without an event ID")
                context.append(c)
                relation.append(r)
                time_key.append(2 * int(m["event_seq"]) if payment else 2 * key.cutoff_seq - 1)
                messages.append(m)
        table = cls(
            tuple(keys),
            np.asarray(context, dtype=np.int64),
            np.asarray(relation, dtype=np.int64),
            np.asarray(time_key, dtype=np.int64),
            np.asarray([2 * key.cutoff_seq for key in keys], dtype=np.int64),
            tuple(messages),
        )
        if len(table) and np.any(table.time_key >= table.seed_time[table.context]):
            raise ValueError("Candidate event is not strictly before its context cutoff")
        return table


def _relation(message: dict[str, Any]) -> int:
    try:
        return RELATION_INDEX[message["relation"]]
    except KeyError:
        raise ValueError(f"Unknown relation {message['relation']!r}") from None


def relation_quotas(sampler: SamplerPlan, hop: int) -> np.ndarray:
    """Maximum sampled candidates per (context, relation); hop 2 is payments-only."""
    quotas = np.zeros(NUM_RELATIONS, dtype=np.int64)
    quotas[:PAYMENT_RELATIONS] = sampler.relation_fanouts[hop - 1]
    if hop == 1:
        quotas[PAYMENT_RELATIONS:] = sampler.association_fanout
    return quotas


def selection_keys(
    table: CandidateTable, *, mode: str, step_seed: int, evaluation_seed: int, hop: int
) -> torch.Tensor:
    """Nonnegative int64 key per candidate on the CPU; smaller keys are drawn first.

    Evaluation keys are SplitMix64(hop_seed(evaluation_seed, hop), context, item):
    hop 1 keys are those of SplitMix64(evaluation_seed, context, item) and hop 2 uses
    an independent derived stream, so a root selected at both hops does not see its
    hop-1 draw again as its hop-2 draw (SamplerPlan.fingerprint versions this).
    """
    if mode == "eval":
        # The same on every device and machine.
        state = splitmix64(np.uint64(hop_seed(evaluation_seed, hop)))
        state = splitmix64(state ^ table.context_hash[table.context])
        keys = splitmix64(state ^ table.item_hash) >> np.uint64(1)
        return torch.from_numpy(keys.astype(np.int64))
    generator = torch.Generator().manual_seed(hop_seed(step_seed, hop))
    return torch.randint(0, 2**62, (len(table),), generator=generator, dtype=torch.int64)


def group_ranks(group: torch.Tensor, keys: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Rank of each row within its group by ascending key; ties keep row order."""
    order = torch.argsort(keys, stable=True)
    order = order[torch.argsort(group[order], stable=True)]
    grouped = group[order]
    counts = torch.bincount(grouped, minlength=num_groups)
    starts = torch.cumsum(counts, 0) - counts
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order), device=order.device) - starts[grouped]
    return ranks


class TorchGroupedSampler:
    """Uniform subset per (context, relation) from random keys; any torch device."""

    name = "torch"

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    def subset(self, table: CandidateTable, quotas: np.ndarray, keys: torch.Tensor) -> torch.Tensor:
        context = torch.from_numpy(table.context).to(self.device)
        relation = torch.from_numpy(table.relation).to(self.device)
        group = context * NUM_RELATIONS + relation
        ranks = group_ranks(group, keys.to(self.device), table.num_contexts * NUM_RELATIONS)
        return ranks < torch.from_numpy(quotas).to(self.device)[relation]


def merge_slots(
    table: CandidateTable,
    keep: torch.Tensor,
    keys: torch.Tensor,
    *,
    hop: int,
    fanout: int,
    association_slots: int,
) -> np.ndarray:
    """Merge kept candidates into [C, fanout] row indices (-1 is padding).

    Payments interleave by position across RELATIONS[:4] and associations across
    RELATIONS[4:], positions ordered by the selection keys. With
    `reserve = min(association_slots, n_assoc, fanout // 4)` the slots hold
    `P[:K-reserve] + A[:reserve]`, then `P[K-reserve:] + A[reserve:]` up to K.
    Hop 2 keeps payments only: `P[:K]`.
    """
    device = keep.device
    rows = torch.nonzero(keep, as_tuple=True)[0]
    num = table.num_contexts
    out = torch.full((num, fanout), -1, dtype=torch.int64, device=device)
    context = torch.from_numpy(table.context).to(device)[rows]
    relation = torch.from_numpy(table.relation).to(device)[rows]
    payment = relation < PAYMENT_RELATIONS
    if hop == 2:
        rows, context, relation, payment = (
            rows[payment],
            context[payment],
            relation[payment],
            payment[payment],
        )
    if not len(rows):
        return out.cpu().numpy()
    position = group_ranks(
        context * NUM_RELATIONS + relation, keys.to(device)[rows], num * NUM_RELATIONS
    )
    associations = NUM_RELATIONS - PAYMENT_RELATIONS
    order = torch.where(
        payment,
        position * PAYMENT_RELATIONS + relation,
        position * associations + relation - PAYMENT_RELATIONS,
    )
    rank = group_ranks(context * 2 + (~payment).long(), order, 2 * num)
    n_pay = torch.bincount(context[payment], minlength=num)
    n_assoc = torch.bincount(context[~payment], minlength=num)
    limit = min(association_slots, fanout // 4) if hop == 1 else 0
    reserve = torch.clamp(n_assoc, max=limit)
    first_association = torch.minimum(n_pay, fanout - reserve)
    slot = torch.where(payment, rank, first_association[context] + rank)
    kept = torch.where(payment, rank < (fanout - reserve)[context], slot < fanout)
    out[context[kept], slot[kept]] = rows[kept]
    return out.cpu().numpy()


def select_resampled(
    candidates: CandidateTable,
    *,
    hop: int,
    sampler: SamplerPlan,
    fanout: int,
    mode: str = "eval",
    step_seed: int = 0,
    backend: str = "torch",
    device: str | torch.device = "cpu",
    cugraph: CuGraphSampler | None = None,
) -> np.ndarray:
    """[C, fanout] candidate rows per context, prefix-filled, -1 padded.

    `backend` must already be resolved (`resolve_backend`). Evaluation always uses
    the hash-keyed torch path; `device` is where the torch sampler runs. In training,
    `cugraph` and `torch` give different (equally distributed) subsets for one
    `step_seed`; the slot order always comes from the torch keys.
    """
    if hop not in (1, 2) or fanout < 1:
        raise ValueError("Hop must be 1 or 2 and fanout positive")
    if mode not in ("train", "eval"):
        raise ValueError("Sampler mode must be train or eval")
    if backend not in ("torch", "cugraph"):
        raise ValueError("Resolve the sampler backend before selecting")
    if not len(candidates):
        return np.full((candidates.num_contexts, fanout), -1, dtype=np.int64)
    quotas = relation_quotas(sampler, hop)
    keys = selection_keys(
        candidates,
        mode=mode,
        step_seed=step_seed,
        evaluation_seed=sampler.evaluation_seed,
        hop=hop,
    )
    if mode == "train" and backend == "cugraph":
        engine = cugraph or default_cugraph_sampler()
        keep = torch.from_numpy(
            engine.subset(candidates, quotas, random_state=hop_seed(step_seed, hop), device=device)
        )
    else:
        keep = TorchGroupedSampler(device).subset(candidates, quotas, keys)
    return merge_slots(
        candidates,
        keep,
        keys.to(keep.device),
        hop=hop,
        fanout=fanout,
        association_slots=sampler.association_slots,
    )


# cuGraph -----------------------------------------------------------------------


def fanout_array(per_hop: Sequence[Sequence[int] | np.ndarray]) -> np.ndarray:
    """pylibcugraph fan-out layout: entry `hop * num_edge_types + edge_type`, int32.

    -1 would gather every edge of a type and 0 skips it.
    """
    widths = {len(q) for q in per_hop}
    if len(widths) != 1:
        raise ValueError("Every hop needs one fan-out per edge type")
    return np.ascontiguousarray(np.concatenate([np.asarray(q) for q in per_hop]), dtype=np.int32)


@dataclass(frozen=True)
class GraphArrays:
    """Batch-local graph: one vertex per context [0, C) and one per candidate [C, C+N)."""

    src: np.ndarray
    dst: np.ndarray
    edge_id: np.ndarray
    edge_type: np.ndarray  # int32
    edge_time: np.ndarray  # int64
    vertices: np.ndarray
    seeds: np.ndarray
    seed_time: np.ndarray  # int64, same dtype as edge_time
    label_offsets: np.ndarray  # int64, one label per seed

    @property
    def num_contexts(self) -> int:
        return len(self.seeds)


def graph_arrays(table: CandidateTable) -> GraphArrays:
    """Context-level vertex IDs, so cuGraph's visited set never merges two contexts."""
    num, edges = table.num_contexts, len(table)
    vertex = np.int32 if num + edges < 2**31 - 1 else np.int64
    return GraphArrays(
        src=table.context.astype(vertex),
        dst=(num + np.arange(edges)).astype(vertex),
        edge_id=np.arange(edges, dtype=vertex),
        edge_type=table.relation.astype(np.int32),
        edge_time=table.time_key.astype(np.int64),
        vertices=np.arange(num + edges, dtype=vertex),
        seeds=np.arange(num, dtype=vertex),
        seed_time=table.seed_time.astype(np.int64),
        label_offsets=np.arange(num + 1, dtype=np.int64),
    )


def sampled_rows(result: dict[str, Any], arrays: GraphArrays, device: torch.device) -> np.ndarray:
    """Candidate rows of a sampler result after on-device consistency checks."""
    for name in ("edge_id", "majors", "minors", "edge_start_time"):
        if result.get(name) is None:
            raise RuntimeError(f"cuGraph result lacks {name}")

    def tensor(name: str) -> torch.Tensor:
        return torch.from_dlpack(result[name]).to(device).long()

    edge_id, major, minor = tensor("edge_id"), tensor("majors"), tensor("minors")
    time = tensor("edge_start_time")
    num, edges = arrays.num_contexts, len(arrays.edge_id)
    if not len(edge_id):
        return np.zeros(0, dtype=np.int64)
    if not len(edge_id) == len(major) == len(minor) == len(time):
        raise RuntimeError("cuGraph result columns differ in length")
    outside = (major < 0) | (major >= num) | (edge_id < 0) | (edge_id >= edges)
    if bool(outside.any()):
        raise RuntimeError("cuGraph sampled beyond the seed contexts or candidate rows")
    seed_time = torch.from_numpy(arrays.seed_time).to(device)
    if not bool((time < seed_time[major]).all()):
        raise RuntimeError("cuGraph returned an edge at or after its seed time (temporal leakage)")
    source = torch.from_numpy(arrays.src).to(device).long()
    ok = (minor == edge_id + num) & (major == source[edge_id])
    if result.get("batch_id") is not None:
        ok &= tensor("batch_id") == major
    if result.get("edge_type") is not None:
        ok &= tensor("edge_type") == torch.from_numpy(arrays.edge_type).to(device).long()[edge_id]
    if not bool(ok.all()):
        raise RuntimeError("cuGraph result does not match the batch-local graph")
    rows = edge_id.cpu().numpy()
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("cuGraph sampled a candidate twice without replacement")
    return rows


def group_counts(table: CandidateTable, rows: np.ndarray) -> np.ndarray:
    """[C, NUM_RELATIONS] count of `rows` (indices or a boolean mask) per group."""
    groups = table.context[rows] * NUM_RELATIONS + table.relation[rows]
    size = table.num_contexts * NUM_RELATIONS
    return np.bincount(groups, minlength=size).reshape(table.num_contexts, NUM_RELATIONS)


def expected_counts(table: CandidateTable, quotas: np.ndarray) -> np.ndarray:
    """min(visible candidates, fan-out) per (context, relation), what a draw must return.

    Only rows strictly before their context's seed time count, so a table that
    deliberately holds rows at or after the cutoff (the verify script's boundary
    probe) expects none of those.
    """
    visible = table.time_key < table.seed_time[table.context]
    return np.minimum(group_counts(table, visible), quotas[None, :])


def _version(module: Any) -> tuple[int, int]:
    parts = str(getattr(module, "__version__", "0.0")).split(".")
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


def import_pylibcugraph() -> Any:
    import pylibcugraph  # pyright: ignore[reportMissingImports]

    version = _version(pylibcugraph)
    if version < MIN_PYLIBCUGRAPH:
        raise RuntimeError(
            f"pylibcugraph {pylibcugraph.__version__} predates hop-0 time filtering; "
            f"install {PYLIBCUGRAPH_PIN}"
        )
    if version >= (26, 10) and not hasattr(pylibcugraph, "neighbor_sample"):
        raise RuntimeError("pylibcugraph>=26.10 without neighbor_sample is not supported")
    return pylibcugraph


class CuGraphProbe(NamedTuple):
    """Outcome of the functional cuGraph probe on one CUDA device."""

    usable: bool
    installed: bool  # False only when cupy or pylibcugraph is not installed at all
    reason: str


def cugraph_import_error() -> tuple[bool, str | None]:
    """(installed, reason): reason is None when CUDA, cupy and pylibcugraph are usable.

    `installed` is False only for a missing module; a module that imports but fails
    (a shared-library error, an unsupported version) counts as installed.
    """
    try:
        import cupy  # noqa: F401  # pyright: ignore[reportMissingImports, reportUnusedImport]

        import_pylibcugraph()
    except ModuleNotFoundError as error:
        return False, f"{type(error).__name__}: {error}"
    except Exception as error:  # noqa: BLE001  (any import failure means cuGraph is unusable)
        return True, f"{type(error).__name__}: {error}"
    if not torch.cuda.is_available():
        return True, "CUDA is not available to torch"
    return True, None


def _probe_table() -> CandidateTable:
    """Three contexts: payments over the fan-out in two relations plus associations,
    associations only, and no candidates at all (a seed without edges)."""
    cutoff = 100
    keys = [ContextKey("Account", f"cugraph-probe-{c}", cutoff, cutoff * 1000) for c in range(3)]

    def payment(relation: str, seq: int) -> dict[str, Any]:
        event = f"{relation}:{seq}"
        return {"relation": relation, "event_seq": seq, "event_id": event, "node_id": f"p{seq}"}

    def association(relation: str, n: int) -> dict[str, Any]:
        return {"relation": relation, "event_seq": cutoff, "event_id": "", "node_id": f"n{n}"}

    rows = [
        {
            "messages": [payment("zelle_out", s) for s in range(1, 12)]
            + [payment("payment_in", s) for s in range(30, 39)]
            + [association("Account_Uses_Device", n) for n in range(2)]
        },
        {"messages": [association("Account_Owned_By_Party", n) for n in range(3)]},
        {"messages": []},
    ]
    return CandidateTable.build(keys, rows)


def probe_cugraph(device: str | torch.device, engine: CuGraphSampler | None = None) -> str | None:
    """Run `CuGraphSampler.subset` on a tiny table; None if it behaves, else the reason.

    For both hops' default resample quotas (hop 2 has association fan-out 0), each
    draw runs twice with one random_state. It must return exactly min(candidates,
    fan-out) per (context, relation), the same mask both times, and pass the leakage
    and consistency checks of `sampled_rows`. Any exception is a failure too.
    """
    try:
        engine = engine if engine is not None else default_cugraph_sampler()
        table = _probe_table()
        plan = SamplerPlan("resample")
        for hop in (1, 2):
            quotas = relation_quotas(plan, hop)
            seed = hop_seed(PROBE_SEED, hop)
            first = engine.subset(table, quotas, random_state=seed, device=device)
            again = engine.subset(table, quotas, random_state=seed, device=device)
            expected = expected_counts(table, quotas)
            if not np.array_equal(group_counts(table, first), expected):
                return f"hop {hop}: counts per (context, relation) differ from {expected.tolist()}"
            if not np.array_equal(first, again):
                return f"hop {hop}: the same random_state gave different subsets"
    except Exception as error:  # noqa: BLE001  (any failure means cuGraph is unusable here)
        return f"{type(error).__name__}: {error}"
    return None


def _probe_device(index: int) -> CuGraphProbe:
    installed, reason = cugraph_import_error()
    if reason is None and not 0 <= index < torch.cuda.device_count():
        reason = f"CUDA device {index} does not exist"
    if reason is None:
        reason = probe_cugraph(torch.device("cuda", index))
    return CuGraphProbe(reason is None, installed, reason or "ok")


_PROBE_LOCK = threading.Lock()
_PROBES: dict[int, CuGraphProbe] = {}


def cugraph_usable(device_index: int = 0) -> CuGraphProbe:
    """Whether cuGraph works on one CUDA device; probed once per process and device.

    The first call imports cupy and pylibcugraph and runs `probe_cugraph` on the
    device. Later calls (any thread) return the cached outcome.
    """
    with _PROBE_LOCK:
        if device_index not in _PROBES:
            _PROBES[device_index] = _probe_device(device_index)
        return _PROBES[device_index]


def resolve_backend(sampler: SamplerPlan, device: str | torch.device) -> str:
    """The subset backend of a run: "deterministic", "torch" or "cugraph".

    Policies other than resample are "deterministic". `torch` is always torch.
    `auto` is cugraph only on a CUDA device whose cached functional probe
    (`cugraph_usable`) passed; otherwise torch, with a warning when cuGraph is
    installed but failed the probe. Explicit `cugraph` raises with the probe's
    reason. The backends draw different (equally distributed) subsets for one step
    seed, so resolve once per run on the main thread and pass the result to every
    `make_live_batch(sampler_backend=...)` call.
    """
    if sampler.policy != "resample":
        return "deterministic"
    if sampler.backend == "torch":
        return "torch"
    device = torch.device(device)
    if device.type != "cuda":
        if sampler.backend == "cugraph":
            raise RuntimeError(
                f"Sampler backend cugraph needs a CUDA device, got {device}; "
                'set [sampler] backend = "torch" or "auto" for this host'
            )
        return "torch"
    # torch stubs type the index as int, but an unindexed CUDA device has None.
    index = device.index
    if index is None:  # pyright: ignore[reportUnnecessaryComparison]
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    probe = cugraph_usable(index)
    if probe.usable:
        return "cugraph"
    if sampler.backend == "cugraph":
        raise RuntimeError(
            f"Sampler backend cugraph cannot run on cuda:{index}: {probe.reason}. "
            f"Install {PYLIBCUGRAPH_PIN} and cupy, run "
            "scripts/temporal/verify_cugraph_sampler.py on the GPU host, "
            'or set [sampler] backend = "torch"'
        )
    if probe.installed:
        warnings.warn(
            f"cuGraph probe failed on cuda:{index} ({probe.reason}); using the torch "
            'sampler. Set [sampler] backend = "torch" to silence this, or "cugraph" '
            "to require cuGraph.",
            RuntimeWarning,
            stacklevel=2,
        )
    return "torch"


def _cupy_array(values: np.ndarray, device: torch.device) -> Any:
    # pylibcugraph reads __cuda_array_interface__ with numpy dtypes and ignores strides,
    # so hand it contiguous CuPy views of torch device tensors (zero-copy DLPack).
    import cupy  # pyright: ignore[reportMissingImports]

    tensor = torch.from_numpy(np.ascontiguousarray(values)).to(device).contiguous()
    return cupy.from_dlpack(tensor)


class CuGraphSampler:
    """Uniform per-(context, relation) subset with pylibcugraph on one CUDA GPU.

    Supports the 26.08 `heterogeneous_uniform_temporal_neighbor_sample` and the
    26.10 `neighbor_sample(starting_vertex_end_times=...)`. The 26.10 legacy
    function treats seed times as a lower bound for decreasing walks, so it is never
    used there. One ResourceHandle is kept per thread.

    Draws come from cuGraph's RNG seeded by `random_state`: equally distributed as
    the torch sampler's, but not the same subset for the same seed.
    """

    name = "cugraph"

    def __init__(self, plc: Any | None = None, *, to_device: Any | None = None) -> None:
        self.plc = plc if plc is not None else import_pylibcugraph()
        self.unified = hasattr(self.plc, "neighbor_sample")
        self.to_device = to_device or _cupy_array
        self.properties = self.plc.GraphProperties(is_symmetric=False, is_multigraph=False)
        self._local = threading.local()

    def handle(self, device: torch.device) -> Any:
        """One ResourceHandle per thread and device (cudaStreamPerThread, current RMM pool)."""
        handles = self._local.__dict__.setdefault("handles", {})
        if device.index not in handles:
            handles[device.index] = self.plc.ResourceHandle()
        return handles[device.index]

    @staticmethod
    def _scope(device: torch.device) -> contextlib.AbstractContextManager[Any]:
        if device.type != "cuda":
            return contextlib.nullcontext()
        import cupy  # pyright: ignore[reportMissingImports]

        return cupy.cuda.Device(device.index)

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        # pylibcugraph runs on the per-thread default stream and torch on its own stream.
        if device.type == "cuda":
            import cupy  # pyright: ignore[reportMissingImports]

            torch.cuda.synchronize(device)
            cupy.cuda.runtime.deviceSynchronize()

    def subset(
        self,
        table: CandidateTable,
        quotas: np.ndarray,
        *,
        random_state: int,
        device: str | torch.device = "cuda",
    ) -> np.ndarray:
        """Boolean keep mask over candidate rows (host).

        Raises unless every (context, relation) got exactly min(visible candidates,
        fan-out) rows, so a cuGraph that over- or under-samples fails loudly.
        """
        device = torch.device(device)
        # torch stubs type the index as int, but an unindexed CUDA device has None.
        if device.type == "cuda" and device.index is None:  # pyright: ignore[reportUnnecessaryComparison]
            device = torch.device("cuda", torch.cuda.current_device())
        keep = np.zeros(len(table), dtype=bool)
        if not len(table):
            return keep
        arrays = graph_arrays(table)
        plc = self.plc
        fan = fanout_array([quotas])
        with self._scope(device):
            put = functools.partial(self.to_device, device=device)
            src, dst, edge_id = put(arrays.src), put(arrays.dst), put(arrays.edge_id)
            edge_type, edge_time = put(arrays.edge_type), put(arrays.edge_time)
            vertices, seeds = put(arrays.vertices), put(arrays.seeds)
            seed_time, labels = put(arrays.seed_time), put(arrays.label_offsets)
            handle = self.handle(device)
            self._synchronize(device)
            graph = plc.SGGraph(
                handle,
                self.properties,
                src,
                dst,
                weight_array=None,
                store_transposed=False,
                renumber=False,
                do_expensive_check=False,
                edge_id_array=edge_id,
                edge_type_array=edge_type,
                edge_start_time_array=edge_time,
                vertices_array=vertices,
                drop_self_loops=False,
                drop_multi_edges=False,
                symmetrize=False,
            )
            options = dict(
                num_edge_types=NUM_RELATIONS,
                with_replacement=False,
                do_expensive_check=False,
                return_hops=False,
                renumber=False,
                compression="COO",
                random_state=int(random_state) & (2**63 - 1),
                temporal_sampling_comparison="strictly_decreasing",
            )
            if self.unified:
                result = plc.neighbor_sample(
                    handle,
                    graph,
                    seeds,
                    fan,
                    starting_vertex_end_times=seed_time,
                    starting_vertex_label_offsets=labels,
                    disjoint_sampling=True,
                    **options,
                )
            else:
                result = plc.heterogeneous_uniform_temporal_neighbor_sample(
                    handle,
                    graph,
                    None,
                    seeds,
                    seed_time,
                    labels,
                    None,
                    fan,
                    disjoint_sampling=False,
                    **options,
                )
            self._synchronize(device)
            rows = sampled_rows(result, arrays, device)
        keep[rows] = True
        counts = group_counts(table, rows)
        if np.any(counts > quotas[None, :]):
            raise RuntimeError("cuGraph exceeded a per-relation fan-out")
        if not np.array_equal(counts, expected_counts(table, quotas)):
            raise RuntimeError(
                "cuGraph under-sampled a (context, relation): got fewer than "
                "min(visible candidates, fan-out); run scripts/temporal/verify_cugraph_sampler.py"
            )
        return keep


_CUGRAPH_LOCK = threading.Lock()
_cugraph_sampler: CuGraphSampler | None = None


def default_cugraph_sampler() -> CuGraphSampler:
    global _cugraph_sampler
    with _CUGRAPH_LOCK:
        if _cugraph_sampler is None:
            _cugraph_sampler = CuGraphSampler()
        return _cugraph_sampler
