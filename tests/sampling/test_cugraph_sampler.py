"""The cuGraph sampler and its probe, on a mock library and, when present, a GPU."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from mule_pattern_learner.contract.graph_schema import (
    ASSOCIATION_RELATIONS,
    PAYMENT_RELATIONS,
    RELATIONS,
    ContextKey,
)
from mule_pattern_learner.sampling import candidates, cugraph_sampler, torch_sampler
from mule_pattern_learner.sampling.backend import select_resampled
from mule_pattern_learner.sampling.candidates import CandidateTable, selection_keys
from mule_pattern_learner.sampling.cugraph_sampler import (
    CuGraphSampler,
    fanout_array,
    graph_arrays,
    probe_cugraph,
)
from mule_pattern_learner.testing import sampler_checks
from mule_pattern_learner.testing.builders import RESAMPLE, candidate_table


class MockPLC:
    """Emulates the pylibcugraph calls CuGraphSampler makes, with their dtype rules.

    Batch ids are numbered as pylibcugraph 26.08 numbers them: by rank among the
    labels that got at least one edge. `seed_labels` returns the label itself.

    Faults: `leak` returns future edges, `drop` applies an off-by-one hop-0 time
    filter (drops same-cutoff associations), `jitter` ignores random_state and
    `broken` fails in SGGraph like a GPU without kernels for its architecture.
    """

    def __init__(
        self,
        version: str = "26.08.00",
        unified: bool = False,
        leak: bool = False,
        *,
        drop: bool = False,
        jitter: bool = False,
        broken: bool = False,
        seed_labels: bool = False,
    ) -> None:
        self.__version__, self.leak, self.drop = version, leak, drop
        self.jitter, self.broken, self.seed_labels = jitter, broken, seed_labels
        self.calls: list[dict[str, Any]] = []
        if unified:
            self.neighbor_sample = self._unified
        else:
            self.heterogeneous_uniform_temporal_neighbor_sample = self._heterogeneous

    @staticmethod
    def ResourceHandle(handle: Any = None) -> SimpleNamespace:
        return SimpleNamespace(kind="handle")

    @staticmethod
    def GraphProperties(is_symmetric: bool = False, is_multigraph: bool = False) -> SimpleNamespace:
        return SimpleNamespace(symmetric=is_symmetric, multigraph=is_multigraph)

    def SGGraph(
        self, handle: Any, properties: Any, src: Any, dst: Any, **kw: Any
    ) -> SimpleNamespace:
        if self.broken:
            raise RuntimeError("CUDA error: no kernel image is available for execution")
        src, dst = np.asarray(src), np.asarray(dst)
        assert src.dtype == dst.dtype == kw["vertices_array"].dtype == kw["edge_id_array"].dtype
        assert kw["edge_type_array"].dtype == np.int32
        assert kw["edge_start_time_array"].dtype == np.int64
        assert np.array_equal(kw["vertices_array"], np.arange(len(kw["vertices_array"])))
        assert kw["renumber"] is False and kw["weight_array"] is None
        return SimpleNamespace(src=src, dst=dst, **kw)

    def _sample(
        self,
        graph: Any,
        seeds: Any,
        times: Any,
        labels: Any,
        fan: Any,
        *,
        num_edge_types: int,
        random_state: int,
        **kw: Any,
    ) -> dict[str, np.ndarray]:
        self.calls.append(dict(kw, num_edge_types=num_edge_types, random_state=random_state))
        assert np.asarray(seeds).dtype == graph.src.dtype
        assert np.asarray(times).dtype == np.int64 and np.asarray(labels).dtype == np.int64
        assert isinstance(fan, np.ndarray) and fan.dtype == np.int32
        assert len(fan) % num_edge_types == 0 and num_edge_types > 1
        assert kw["temporal_sampling_comparison"] == "strictly_decreasing"
        assert kw["with_replacement"] is False and kw["compression"] == "COO"
        assert labels[-1] == len(seeds)
        rng = np.random.default_rng(random_state + len(self.calls) * self.jitter)
        names = ("majors", "minors", "edge_id", "edge_type", "edge_start_time", "batch_id")
        out: dict[str, list[int]] = {k: [] for k in names}
        etype, etime = graph.edge_type_array, graph.edge_start_time_array
        for label, seed in enumerate(seeds):
            for t in range(num_edge_types):
                edges = np.nonzero((graph.src == seed) & (etype == t))[0]
                if not self.leak:
                    edges = edges[etime[edges] < times[label] - int(self.drop)]
                take = rng.permutation(edges)[: fan[t]] if fan[t] >= 0 else edges
                for e in take:
                    out["majors"].append(seed)
                    out["minors"].append(graph.dst[e])
                    out["edge_id"].append(graph.edge_id_array[e])
                    out["edge_type"].append(t)
                    out["edge_start_time"].append(etime[e] + (2 if self.leak else 0))
                    out["batch_id"].append(label)
        if not self.seed_labels:
            out["batch_id"] = np.unique(out["batch_id"], return_inverse=True)[1].tolist()
        dtypes = {"edge_type": np.int32, "edge_start_time": np.int64, "batch_id": np.int32}
        return {k: np.asarray(v, dtype=dtypes.get(k, graph.src.dtype)) for k, v in out.items()}

    def _heterogeneous(
        self,
        handle: Any,
        graph: Any,
        prop: Any,
        seeds: Any,
        times: Any,
        labels: Any,
        vtypes: Any,
        fan: Any,
        **kw: Any,
    ) -> dict[str, np.ndarray]:
        assert prop is None and vtypes is None and kw.pop("disjoint_sampling") is False
        return self._sample(graph, seeds, times, labels, fan, **kw)

    def _unified(
        self,
        handle: Any,
        graph: Any,
        seeds: Any,
        fan: Any,
        *,
        starting_vertex_end_times: Any,
        starting_vertex_label_offsets: Any,
        disjoint_sampling: bool,
        **kw: Any,
    ) -> dict[str, np.ndarray]:
        assert disjoint_sampling is True
        return self._sample(
            graph, seeds, starting_vertex_end_times, starting_vertex_label_offsets, fan, **kw
        )


def host(values: np.ndarray, device: Any = None) -> np.ndarray:
    return np.ascontiguousarray(values)


def test_graph_arrays_and_fanout_layout() -> None:
    keys, rows = candidate_table(
        {"zelle_out": 3, "payment_in": 2, "Account_Uses_Device": 2}, contexts=3
    )
    table = CandidateTable.build(keys, rows)
    arrays = graph_arrays(table)
    assert arrays.src.dtype == arrays.dst.dtype == arrays.vertices.dtype == np.int32
    assert arrays.edge_type.dtype == np.int32 and arrays.edge_time.dtype == np.int64
    assert arrays.seed_time.dtype == arrays.label_offsets.dtype == np.int64
    assert arrays.dst.tolist() == list(range(3, 3 + len(table)))
    assert arrays.label_offsets.tolist() == [0, 1, 2, 3]
    assert np.all(arrays.edge_time < arrays.seed_time[arrays.src])
    quotas = [[1, 2, 3], [4, 5, 6]]
    layout = fanout_array(quotas)
    assert layout.dtype == np.int32 and layout.tolist() == [1, 2, 3, 4, 5, 6]
    assert layout[1 * 3 + 2] == quotas[1][2]  # hop * num_edge_types + edge_type
    with pytest.raises(ValueError):
        fanout_array([[1, 2], [3]])


@pytest.mark.parametrize("unified", [False, True], ids=["26.08", "26.10"])
def test_cugraph_sampler_maps_results_to_slots_with_the_torch_merge(unified: bool) -> None:
    plc = MockPLC("26.10.00" if unified else "26.08.00", unified=unified)
    engine = CuGraphSampler(plc, to_device=host)
    counts = {"zelle_out": 7, "zelle_in": 4, "payment_out": 2, "Account_Owned_By_Party": 3}
    keys, rows = candidate_table(counts | {"Account_Uses_Device": 2}, contexts=5)
    table = CandidateTable.build(keys, rows)
    sampler = replace(RESAMPLE, relation_fanouts=(3, 2), backend="cugraph")
    for seed in range(5):
        slots = select_resampled(
            table,
            hop=1,
            sampler=sampler,
            fanout=8,
            mode="train",
            step_seed=seed,
            backend="cugraph",
            device="cpu",
            cugraph=engine,
        )
        for c, row in enumerate(slots):
            chosen = [table.messages[j] for j in row if j >= 0]
            assert all(table.context[j] == c for j in row if j >= 0)
            relations = Counter(m["relation"] for m in chosen)
            assert all(relations[r] <= 3 for r in PAYMENT_RELATIONS)
            assert all(relations[r] <= 1 for r in ASSOCIATION_RELATIONS)
            assert [m["relation"] in PAYMENT_RELATIONS for m in chosen] == [True] * 6 + [False] * 2
    assert [call["random_state"] for call in plc.calls] == [0, 1, 2, 3, 4]
    assert all(call["num_edge_types"] == len(RELATIONS) for call in plc.calls)
    hop2 = select_resampled(
        table,
        hop=2,
        sampler=sampler,
        fanout=4,
        mode="train",
        step_seed=1,
        backend="cugraph",
        device="cpu",
        cugraph=engine,
    )
    assert all(table.relation[j] < 4 for j in hop2.ravel() if j >= 0)
    # Evaluation never calls cuGraph.
    before = len(plc.calls)
    table_eval = CandidateTable.build(keys, rows)
    select_resampled(
        table_eval, hop=1, sampler=sampler, fanout=8, mode="eval", backend="cugraph", cugraph=engine
    )
    assert len(plc.calls) == before


def test_cugraph_sampler_rejects_leaks_bad_versions_and_quota_overruns() -> None:
    keys, rows = candidate_table({"zelle_out": 4, "Account_Uses_Device": 1})
    table = CandidateTable.build(keys, rows)
    quotas = candidates.relation_quotas(RESAMPLE, 1)
    leaky = CuGraphSampler(MockPLC(leak=True), to_device=host)
    with pytest.raises(RuntimeError, match="temporal leakage"):
        leaky.subset(table, quotas, random_state=1, device="cpu")
    greedy = CuGraphSampler(MockPLC(), to_device=host)
    with pytest.raises(RuntimeError, match="fan-out"):
        greedy.subset(table, np.full_like(quotas, -1), random_state=1, device="cpu")
    mask = greedy.subset(table, quotas, random_state=1, device="cpu")
    assert mask.sum() == 3 + 1
    assert (
        not CuGraphSampler(MockPLC(), to_device=host)
        .subset(
            CandidateTable.build(keys, [{"messages": []}]), quotas, random_state=1, device="cpu"
        )
        .any()
    )
    fake = SimpleNamespace(__version__="25.12.00")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(__import__("sys").modules, "pylibcugraph", fake)
        with pytest.raises(RuntimeError, match="predates"):
            cugraph_sampler.import_pylibcugraph()


def test_cugraph_sampler_rejects_under_sampling() -> None:
    # An off-by-one hop-0 time filter drops the same-cutoff association silently.
    keys, rows = candidate_table({"zelle_out": 2, "Account_Uses_Device": 1})
    table = CandidateTable.build(keys, rows)
    quotas = candidates.relation_quotas(RESAMPLE, 1)
    dropping = CuGraphSampler(MockPLC(drop=True), to_device=host)
    with pytest.raises(RuntimeError, match="under-sampled"):
        dropping.subset(table, quotas, random_state=1, device="cpu")
    torch_keep = torch_sampler.TorchGroupedSampler().subset(
        table, quotas, selection_keys(table, mode="train", step_seed=1, evaluation_seed=0, hop=1)
    )
    assert bool(torch_keep.all())
    # Rows at or after the cutoff (the verify script's boundary table) are not expected.
    cutoff = 100
    key = ContextKey("Account", "boundary", cutoff, cutoff * 1000)
    boundary = CandidateTable(
        keys=(key,),
        context=np.zeros(4, dtype=np.int64),
        relation=np.asarray([0, 4, 0, 0], dtype=np.int64),
        time_key=np.asarray([198, 199, 200, 201], dtype=np.int64),
        seed_time=np.asarray([200], dtype=np.int64),
        messages=tuple({"relation": RELATIONS[r]} for r in (0, 4, 0, 0)),
    )
    engine = CuGraphSampler(MockPLC(), to_device=host)
    mask = engine.subset(boundary, np.full(len(RELATIONS), 8), random_state=3, device="cpu")
    assert mask.tolist() == [True, True, False, False]
    assert candidates.expected_counts(boundary, np.full(len(RELATIONS), 8))[0, :5].tolist() == [
        1,
        0,
        0,
        0,
        1,
    ]


@pytest.mark.parametrize("unified", [False, True], ids=["26.08", "26.10"])
def test_cugraph_probe_passes_on_a_correct_sampler(unified: bool) -> None:
    plc = MockPLC("26.10.00" if unified else "26.08.00", unified=unified)
    assert probe_cugraph("cpu", CuGraphSampler(plc, to_device=host)) is None
    # Two draws per hop, one random_state per hop, the association fan-out 0 at hop 2.
    assert len(plc.calls) == 4
    states = [call["random_state"] for call in plc.calls]
    assert states[0] == states[1] != states[2] == states[3]


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ({"broken": True}, "no kernel image"),
        ({"drop": True}, "under-sampled"),
        ({"jitter": True}, "different subsets"),
        ({"leak": True}, "temporal leakage"),
    ],
)
def test_cugraph_probe_reports_runtime_failures(fault: dict[str, Any], reason: str) -> None:
    engine = CuGraphSampler(MockPLC(**fault), to_device=host)
    failure = probe_cugraph("cpu", engine)
    assert failure is not None and reason in failure


def test_real_probe_without_cupy_reports_the_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "cupy", None)
    monkeypatch.setattr(cugraph_sampler, "_PROBES", {})
    probe = cugraph_sampler.cugraph_usable(0)
    assert not probe.usable and not probe.installed and "cupy" in probe.reason
    assert cugraph_sampler.cugraph_usable(0) is probe


def _gpu_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        cugraph_sampler.import_pylibcugraph()
        import cupy  # noqa: F401  # pyright: ignore[reportMissingImports, reportUnusedImport]
    except Exception:
        return False
    return True


@pytest.mark.cuda
@pytest.mark.skipif(not _gpu_ready(), reason="needs CUDA, cupy and pylibcugraph>=26.4")
def test_real_cugraph_matches_torch_caps_on_gpu() -> None:
    keys, rows = candidate_table(
        {"zelle_out": 9, "zelle_in": 3, "Account_Owned_By_Party": 3}, contexts=64
    )
    table = CandidateTable.build(keys, rows)
    sampler = replace(RESAMPLE, backend="cugraph")
    for seed in (1, 2):
        a = select_resampled(
            table,
            hop=1,
            sampler=sampler,
            fanout=8,
            mode="train",
            step_seed=seed,
            backend="cugraph",
            device="cuda",
        )
        b = select_resampled(
            table,
            hop=1,
            sampler=sampler,
            fanout=8,
            mode="train",
            step_seed=seed,
            backend="cugraph",
            device="cuda",
        )
        assert np.array_equal(a, b)
        torch_slots = select_resampled(
            table, hop=1, sampler=sampler, fanout=8, mode="train", step_seed=seed
        )
        # Both backends fill 3 + 3 payment slots and 1 association (the fan-out of 1).
        assert np.array_equal((a >= 0).sum(1), (torch_slots >= 0).sum(1))
        assert ((a >= 0).sum(1) == 7).all()


@pytest.mark.cuda
@pytest.mark.skipif(not _gpu_ready(), reason="needs CUDA, cupy and pylibcugraph>=26.4")
def test_real_cugraph_handles_seeds_without_edges_on_gpu() -> None:
    # The shape of a hop-2 table: empty and association-only contexts between others.
    keys, rows = candidate_table(
        {"zelle_out": 5, "payment_in": 1, "Account_Uses_Device": 2}, contexts=40
    )
    for c in range(0, 40, 3):
        rows[c] = {"messages": []}
    for c in range(1, 40, 7):
        rows[c] = {
            "messages": [m for m in rows[c]["messages"] if m["relation"] == "Account_Uses_Device"]
        }
    table = CandidateTable.build(keys, rows)
    engine = CuGraphSampler()
    for hop in (1, 2):
        quotas = candidates.relation_quotas(RESAMPLE, hop)
        keep = engine.subset(table, quotas, random_state=hop, device="cuda")
        expected = candidates.expected_counts(table, quotas)
        assert np.array_equal(candidates.group_counts(table, keep), expected)


def test_the_gpu_checks_pass_on_the_mock_and_catch_its_faults() -> None:
    # tests/integration/test_cugraph_sampler.py runs the same checks on a GPU.
    engine = CuGraphSampler(MockPLC(), to_device=host)
    sampler = replace(RESAMPLE, relation_fanouts=(8, 4))
    assert probe_cugraph("cpu", engine) is None
    sampler_checks.check_subsets(
        engine, sampler_checks.synthetic_table(24, np.random.default_rng(0)), sampler, "cpu"
    )
    sampler_checks.check_cutoff_boundary(engine, "cpu")
    sampler_checks.check_merged_slots(engine, sampler, "cpu")
    sampler_checks.check_uniform_inclusion(engine, 150, "cpu")
    leaky = CuGraphSampler(MockPLC(leak=True), to_device=host)
    with pytest.raises((AssertionError, RuntimeError)):
        sampler_checks.check_cutoff_boundary(leaky, "cpu")
    reason = probe_cugraph("cpu", CuGraphSampler(MockPLC(drop=True), to_device=host))
    assert reason is not None and "under-sampled" in reason


def test_sampled_rows_rejects_inconsistent_results() -> None:
    keys, rows = candidate_table({"zelle_out": 3})
    arrays = graph_arrays(CandidateTable.build(keys, rows))
    good = {
        "edge_id": np.asarray([0, 2], dtype=np.int32),
        "majors": np.asarray([0, 0], dtype=np.int32),
        "minors": np.asarray([1, 3], dtype=np.int32),
        "edge_start_time": arrays.edge_time[[0, 2]],
        "batch_id": np.asarray([0, 0], dtype=np.int32),
    }
    assert cugraph_sampler.sampled_rows(good, arrays, torch.device("cpu")).tolist() == [0, 2]
    for name, value in (
        ("minors", np.asarray([1, 2], dtype=np.int32)),
        ("edge_id", np.asarray([0, 9], dtype=np.int32)),
        ("batch_id", np.asarray([0, 1], dtype=np.int32)),
        ("edge_id", np.asarray([2, 2], dtype=np.int32)),
    ):
        broken = dict(good, **{name: value})
        if name == "edge_id" and value.tolist() == [2, 2]:
            broken["minors"] = np.asarray([3, 3], dtype=np.int32)
            broken["edge_start_time"] = arrays.edge_time[[2, 2]]
        with pytest.raises(RuntimeError):
            cugraph_sampler.sampled_rows(broken, arrays, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="lacks edge_start_time"):
        cugraph_sampler.sampled_rows(dict(good, edge_start_time=None), arrays, torch.device("cpu"))


def test_sampled_rows_accepts_batch_ids_ranked_over_seeds_with_edges() -> None:
    # Seed 1 has no candidates, so pylibcugraph 26.08 numbers seed 2's batch 1, not 2.
    keys, rows = candidate_table({"zelle_out": 2}, contexts=3)
    rows[1] = {"messages": []}
    arrays = graph_arrays(CandidateTable.build(keys, rows))
    result = {
        "edge_id": np.asarray([0, 1, 2], dtype=np.int32),
        "majors": np.asarray([0, 0, 2], dtype=np.int32),
        "minors": np.asarray([3, 4, 5], dtype=np.int32),
        "edge_start_time": arrays.edge_time[:3],
    }
    cpu = torch.device("cpu")
    for batch in ([0, 0, 1], [0, 0, 2]):
        ranked = dict(result, batch_id=np.asarray(batch, dtype=np.int32))
        assert cugraph_sampler.sampled_rows(ranked, arrays, cpu).tolist() == [0, 1, 2]
    for batch in ([0, 1, 1], [1, 1, 2], [0, 0, 0]):
        with pytest.raises(RuntimeError, match=r"\(batch_id\)"):
            cugraph_sampler.sampled_rows(
                dict(result, batch_id=np.asarray(batch, dtype=np.int32)), arrays, cpu
            )


@pytest.mark.parametrize("unified", [False, True], ids=["26.08", "26.10"])
@pytest.mark.parametrize("seed_labels", [False, True], ids=["ranked", "seed"])
def test_cugraph_subset_handles_seeds_without_edges(unified: bool, seed_labels: bool) -> None:
    # Hop 2 of a batch holds stubbed hubs and association-only contexts between others.
    keys, rows = candidate_table(
        {"zelle_out": 5, "payment_in": 1, "Account_Uses_Device": 2}, contexts=5
    )
    rows[0] = rows[3] = {"messages": []}
    rows[1] = {
        "messages": [m for m in rows[1]["messages"] if m["relation"] == "Account_Uses_Device"]
    }
    table = CandidateTable.build(keys, rows)
    plc = MockPLC("26.10.00" if unified else "26.08.00", unified=unified, seed_labels=seed_labels)
    engine = CuGraphSampler(plc, to_device=host)
    for hop in (1, 2):
        quotas = candidates.relation_quotas(RESAMPLE, hop)
        keep = engine.subset(table, quotas, random_state=5, device="cpu")
        expected = candidates.expected_counts(table, quotas)
        assert np.array_equal(candidates.group_counts(table, keep), expected)
