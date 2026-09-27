"""pylibcugraph's heterogeneous temporal sampler on one CUDA GPU, and its probe.

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
import threading
from typing import Any, NamedTuple

import numpy as np
import torch

from ..contract.graph_schema import ContextKey
from ..contract.sampler_plan import SamplerPlan
from .candidates import (
    NUM_RELATIONS,
    CandidateTable,
    expected_counts,
    group_counts,
    hop_seed,
    relation_quotas,
)

MIN_PYLIBCUGRAPH = (26, 4)  # hop-0 time filter (25.12) and null-label fix (26.04)
PYLIBCUGRAPH_PIN = "pylibcugraph-cu12/cu13==26.8.* (the cuda12 or cuda13 extra)"
PROBE_SEED = 20260924


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


def batch_ids_match(batch: torch.Tensor, major: torch.Tensor) -> bool:
    """Whether cuGraph's per-edge batch ids fit the seeds the edges came from.

    Label i is seed i, the major of its edges. pylibcugraph 26.08 and later do not
    return that label: they expand label offsets built over the labels that got at
    least one edge, so an id is the label's rank among those, and every seed without
    sampled edges shifts the ids after it. Either numbering is accepted.
    """
    if bool((batch == major).all()):
        return True
    rank = torch.unique(major, sorted=True, return_inverse=True)[1]
    return bool((batch == rank).all())


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
    checks = {"minors": minor == edge_id + num, "majors": major == source[edge_id]}
    if result.get("edge_type") is not None:
        edge_type = torch.from_numpy(arrays.edge_type).to(device).long()
        checks["edge_type"] = tensor("edge_type") == edge_type[edge_id]
    wrong = [name for name, ok in checks.items() if not bool(ok.all())]
    if result.get("batch_id") is not None and not batch_ids_match(tensor("batch_id"), major):
        wrong.append("batch_id")
    if wrong:
        raise RuntimeError(
            f"cuGraph result does not match the batch-local graph ({', '.join(wrong)})"
        )
    rows = edge_id.cpu().numpy()
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("cuGraph sampled a candidate twice without replacement")
    return rows


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
    no candidates at all (a seed without edges), and associations only.

    The empty seed sits between two with edges, so cuGraph's batch ids, which skip
    seeds without sampled edges, differ from the seed index at hop 1 (`sampled_rows`)."""
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
        {"messages": []},
        {"messages": [association("Account_Owned_By_Party", n) for n in range(3)]},
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
        plan = SamplerPlan()
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
                "min(visible candidates, fan-out); run "
                "`pytest -m cuda tests/integration/test_cugraph_sampler.py`"
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
