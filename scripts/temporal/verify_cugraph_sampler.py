"""Self-test of the cuGraph neighbor resampler on a CUDA host.

First it runs the functional probe that `backend = "auto"` runs once per process
before choosing cuGraph (sampler.probe_cugraph: exact counts, determinism and the
leakage checks on a tiny table). Then, on a synthetic candidate table, it compares
CuGraphSampler with the torch grouped sampler: exactly min(candidates, fan-out) per
(context, relation) (hop 2 payments-only), strict temporal validity (including
candidates exactly at the cutoff), uniform inclusion with a chi-square test over
many seeds, determinism for a fixed seed, and per-call latency. The two backends
draw different (equally distributed) subsets for one seed; only evaluation, which
is hash-keyed on the torch path, is identical across them. With --live it also
builds one real batch per backend from a prepared dataset (read-only TigerGraph
queries) and runs one deterministic CUDA training step twice.

Exit code 0 means every check that could run passed; 1 means a check failed; 2
means cuGraph could not run here (no CUDA, or cupy or a supported pylibcugraph is
missing; install the cuda12 or cuda13 extra, pylibcugraph 26.8).

Usage:
  python scripts/temporal/verify_cugraph_sampler.py [--seeds 600] [--rmm-pool 2GiB]
  python scripts/temporal/verify_cugraph_sampler.py --live [--config overrides.toml]

--live builds one batch from the live graph with the built-in run settings (a --config
file only overrides keys). It prepares the default run's cache first if needed, which
`mule-temporal train` then reuses. Roots that TigerGraph rejects are dropped and
reported, like training does.
"""

from __future__ import annotations

import os

# cuBLAS needs a fixed workspace for deterministic algorithms; set before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from collections.abc import Callable
from dataclasses import replace
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mule_pattern_learner.temporal.live.contract import RELATIONS, ContextKey, PoolPlan, SamplerPlan
from mule_pattern_learner.temporal.live.sampler import (
    NUM_RELATIONS,
    PAYMENT_RELATIONS,
    CandidateTable,
    CuGraphSampler,
    TorchGroupedSampler,
    cugraph_import_error,
    group_counts,
    import_pylibcugraph,
    probe_cugraph,
    relation_quotas,
    select_resampled,
    selection_keys,
)

CHI2_999 = {11: 31.26}  # 0.999 quantile of chi-square with 11 degrees of freedom
FAILURES: list[str] = []
Runs = list[tuple[str, Callable[[int], np.ndarray]]]  # (backend label, seed -> keep mask)


def check(condition: bool, message: str) -> None:
    print(("  ok    " if condition else "  FAIL  ") + message)
    if not condition:
        FAILURES.append(message)


def versions() -> bool:
    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"numpy {np.__version__}, torch {torch.__version__} (CUDA build {torch.version.cuda})")
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {props.name}, capability {props.major}.{props.minor}")
    else:
        print("CUDA not available to torch")
    try:
        import cupy  # pyright: ignore[reportMissingImports]

        print(f"cupy {cupy.__version__}, CUDA runtime {cupy.cuda.runtime.runtimeGetVersion()}")
    except Exception as error:  # noqa: BLE001
        print(f"cupy unavailable: {error}")
    try:
        plc = import_pylibcugraph()
        api = (
            "neighbor_sample (26.10+)"
            if hasattr(plc, "neighbor_sample")
            else ("heterogeneous_uniform_temporal_neighbor_sample (26.08)")
        )
        print(f"pylibcugraph {plc.__version__}: {api}")
    except Exception as error:  # noqa: BLE001
        print(f"pylibcugraph unavailable: {error}")
    _, reason = cugraph_import_error()
    if reason is not None:
        print(f"cuGraph unavailable: {reason}")
    return reason is None


def message(relation: str, seq: int, key: ContextKey, n: int) -> dict[str, Any]:
    payment = RELATIONS.index(relation) < PAYMENT_RELATIONS
    return {
        "relation": relation,
        "node_type": "Account" if payment else "Party",
        "node_id": f"{relation}-{n}",
        "event_id": f"{relation}:{seq}" if payment else "",
        "event_seq": seq if payment else key.cutoff_seq,
        "event_ts_ms": seq * 1000 if payment else key.cutoff_ms,
    }


def synthetic_table(contexts: int, rng: np.random.Generator, *, full: int | None = None):
    keys, rows = [], []
    for c in range(contexts):
        cutoff = int(rng.integers(50, 5000))
        key = ContextKey("Account", f"ctx{c}", cutoff, cutoff * 1000)
        messages = []
        for r, relation in enumerate(RELATIONS):
            limit = 20 if r < PAYMENT_RELATIONS else 3
            count = (
                full
                if full is not None and r < PAYMENT_RELATIONS
                else int(rng.integers(0, limit + 1))
            )
            if r < PAYMENT_RELATIONS:
                seqs = rng.choice(np.arange(1, cutoff), min(count, cutoff - 1), replace=False)
                messages += [message(relation, int(s), key, n) for n, s in enumerate(seqs)]
            elif full is None:
                messages += [message(relation, 0, key, n) for n in range(count)]
        keys.append(key)
        rows.append({"messages": messages})
    return CandidateTable.build(keys, rows)


def torch_subset(table: CandidateTable, quotas: np.ndarray, seed: int, device: str) -> np.ndarray:
    keys = selection_keys(table, mode="train", step_seed=seed, evaluation_seed=0, hop=1)
    return TorchGroupedSampler(device).subset(table, quotas, keys).cpu().numpy()


def probe(engine: CuGraphSampler, device: str = "cuda") -> None:
    reason = probe_cugraph(device, engine)
    check(reason is None, f"functional probe that backend=auto runs: {reason or 'passed'}")


def subset_checks(
    engine: CuGraphSampler, table: CandidateTable, sampler: SamplerPlan, device: str = "cuda"
) -> None:
    available = group_counts(table, np.ones(len(table), dtype=bool))
    for hop in (1, 2):
        quotas = relation_quotas(sampler, hop)
        expected = np.minimum(available, quotas[None, :])
        runs: Runs = [
            ("torch", lambda s, q=quotas: torch_subset(table, q, s, device)),
            ("cugraph", lambda s, q=quotas: engine.subset(table, q, random_state=s, device=device)),
        ]
        for name, run in runs:
            mask = run(11)
            check(
                np.array_equal(group_counts(table, mask), expected),
                f"{name} hop {hop}: exactly min(candidates, fan-out) per (context, relation)",
            )
            if hop == 2:
                check(
                    not mask[table.relation >= PAYMENT_RELATIONS].any(),
                    f"{name} hop 2 is payments-only",
                )
            valid = table.time_key[mask] < table.seed_time[table.context[mask]]
            check(bool(valid.all()), f"{name} hop {hop}: every sampled edge is before its cutoff")
            check(np.array_equal(run(11), mask), f"{name} hop {hop}: same seed, same subset")
            check(not np.array_equal(run(12), mask), f"{name} hop {hop}: new seed, new subset")


def temporal_boundary(engine: CuGraphSampler, device: str = "cuda") -> None:
    # Bypass CandidateTable.build validation to probe cuGraph's own strict filter.
    cutoff = 100
    key = ContextKey("Account", "boundary", cutoff, cutoff * 1000)
    times = np.asarray([2 * cutoff - 2, 2 * cutoff - 1, 2 * cutoff, 2 * cutoff + 1], dtype=np.int64)
    table = CandidateTable(
        keys=(key,),
        context=np.zeros(4, dtype=np.int64),
        relation=np.asarray([0, 4, 0, 0], dtype=np.int64),
        time_key=times,
        seed_time=np.asarray([2 * cutoff], dtype=np.int64),
        messages=tuple({"relation": RELATIONS[r]} for r in (0, 4, 0, 0)),
    )
    try:
        mask = engine.subset(table, np.full(NUM_RELATIONS, 8), random_state=3, device=device)
    except RuntimeError as error:  # a post-sample leakage or exact-count assertion fired
        check(False, f"cuGraph failed the cutoff boundary table: {error}")
        return
    check(
        mask.tolist() == [True, True, False, False],
        "cuGraph strictly_decreasing keeps seq < cutoff and same-cutoff associations only",
    )


def uniformity(engine: CuGraphSampler, seeds: int, device: str = "cuda") -> None:
    n, quota, contexts = 12, 3, 64
    rng = np.random.default_rng(0)
    table = synthetic_table(contexts, rng, full=n)
    first = table.relation == 0
    quotas = np.zeros(NUM_RELATIONS, dtype=np.int64)
    quotas[0] = quota
    # Candidate rank inside its (context, relation) group, in canonical order.
    starts = np.searchsorted(table.context[first], np.arange(contexts))
    rank = np.arange(first.sum()) - starts[table.context[first]]
    runs: Runs = [
        ("torch", lambda s: torch_subset(table, quotas, s, device)),
        ("cugraph", lambda s: engine.subset(table, quotas, random_state=s, device=device)),
    ]
    for name, run in runs:
        counts = np.zeros(n)
        for seed in range(seeds):
            mask = run(seed)[first]
            counts += np.bincount(rank[mask], minlength=n)
        expected = seeds * contexts * quota / n
        chi2 = float(((counts - expected) ** 2 / expected).sum())
        check(chi2 < CHI2_999[n - 1], f"{name} uniform inclusion: chi2={chi2:.1f} (df={n - 1})")


def latency(engine: CuGraphSampler, sampler: SamplerPlan, device: str = "cuda") -> None:
    rng = np.random.default_rng(1)
    for label, contexts in (("64 roots", 64), ("1024 children", 1024)):
        table = synthetic_table(contexts, rng)
        quotas = relation_quotas(sampler, 1 if contexts == 64 else 2)
        runs: Runs = [
            ("torch cpu", lambda s: torch_subset(table, quotas, s, "cpu")),
            (f"torch {device}", lambda s: torch_subset(table, quotas, s, device)),
            ("cugraph", lambda s: engine.subset(table, quotas, random_state=s, device=device)),
        ]
        for name, run in runs:
            run(0)
            start = time.perf_counter()
            for seed in range(20):
                run(seed)
            if device == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - start) / 20 * 1e3
            print(f"  {label:14s} {len(table):6d} candidates  {name:10s} {ms:7.2f} ms/call")


def merged_slots(engine: CuGraphSampler, sampler: SamplerPlan, device: str = "cuda") -> None:
    table = synthetic_table(256, np.random.default_rng(2))
    for backend in ("torch", "cugraph"):
        slots = select_resampled(
            table,
            hop=1,
            sampler=sampler,
            fanout=16,
            mode="train",
            step_seed=5,
            backend=backend,
            device=device,
            cugraph=engine,
        )
        rows = slots[slots >= 0]
        context = np.repeat(np.arange(len(slots)), (slots >= 0).sum(1))
        check(
            bool((table.context[rows] == context).all()), f"{backend} slots stay in their context"
        )
        check(len(np.unique(rows)) == len(rows), f"{backend} slots never repeat a candidate")
    evaluation = [
        select_resampled(
            table,
            hop=1,
            sampler=sampler,
            fanout=16,
            mode="eval",
            backend=b,
            device=d,
            cugraph=engine,
        )
        for b, d in (("torch", "cpu"), ("torch", device), ("cugraph", device))
    ]
    check(
        all(np.array_equal(evaluation[0], other) for other in evaluation[1:]),
        f"evaluation is hash-keyed: identical on CPU and {device} for every backend",
    )


def live(config_path: Path | None, roots: int, sampler_override: dict[str, Any]) -> None:
    from mule_pattern_learner.device import torch_runtime
    from mule_pattern_learner.temporal.live.config_schema import fanouts as configured_fanouts
    from mule_pattern_learner.temporal.live.config_schema import run_config
    from mule_pattern_learner.temporal.live.contract import FeaturePlan
    from mule_pattern_learner.temporal.live.dataset import load_prepared, sample_keys
    from mule_pattern_learner.temporal.live.hubs import load_hub_registry
    from mule_pattern_learner.temporal.live.model import build_model
    from mule_pattern_learner.temporal.live.pipeline import (
        dataset_path,
        prepare_live,
        prepared_config,
    )
    from mule_pattern_learner.temporal.live.batching import build_root_batch
    from mule_pattern_learner.temporal.live.source import open_context_source

    def make_live_batch(store: Any, keys: Any, **options: Any) -> dict[str, torch.Tensor]:
        stats = options.pop("stats", None)
        prepared = build_root_batch(store, keys, **options)
        if prepared.batch is None:
            raise ValueError("TigerGraph rejected every live root")
        if stats is not None:
            stats.update(prepared.stats)
        return prepared.batch

    config = run_config(config_path)
    plan = FeaturePlan.from_config(config)
    sampler = SamplerPlan.from_config(config)
    if sampler.policy != "resample":
        print(f"  config sampler policy is {sampler.policy}; using resample with the same pools")
        sampler = SamplerPlan(
            "resample", roots=sampler.roots, children=sampler.children, **sampler_override
        )
    # The default run's prepared cache; preparing it here is what `train` would do first.
    dataset = dataset_path(config)
    prepare_live(config, dataset)
    manifest, accounts = load_prepared(dataset)
    config = prepared_config(config, manifest)
    hubs = load_hub_registry(dataset, manifest)
    date = config["dates"]["train"][0]
    train = accounts[accounts["split"] == "train"].head(roots)
    keys = sample_keys(train, date, manifest)
    fanouts = configured_fanouts(config)
    store = open_context_source(dataset, manifest, config)
    try:
        batches = {}
        for backend in ("torch", "cugraph"):
            stats: dict[str, Any] = {}
            start = time.perf_counter()
            batches[backend] = make_live_batch(
                store,
                keys,
                fanouts=fanouts,
                device="cuda",
                plan=plan,
                sampler=replace(sampler, backend=backend),
                hubs=hubs,
                mode="train",
                step_seed=7,
                stats=stats,
            )
            print(f"  live {backend}: {time.perf_counter() - start:.2f} s, {stats}")
            check(stats["sampler_backend"] == backend, f"live batch used the {backend} backend")
        for backend, batch in batches.items():
            codes = batch["first_relation"].masked_fill(~batch["first_mask"], -1)
            caps = [int((codes == r).sum(1).max()) for r in range(PAYMENT_RELATIONS)]
            check(max(caps) <= sampler.relation_fanouts[0], f"live {backend} first-hop caps {caps}")
            second = batch["second_relation"][batch["second_mask"]]
            check(bool((second < PAYMENT_RELATIONS).all()), f"live {backend} hop 2 payments-only")
        evaluation = [
            make_live_batch(
                store,
                keys,
                fanouts=fanouts,
                device="cuda",
                plan=plan,
                sampler=replace(sampler, backend=b),
                hubs=hubs,
                mode="eval",
            )
            for b in ("torch", "cugraph")
        ]
        check(
            all(torch.equal(evaluation[0][k], evaluation[1][k]) for k in evaluation[0]),
            "live evaluation batches are identical across backends",
        )
        # One deterministic CUDA training step, twice from the same state.
        losses, grads = [], []
        with torch_runtime(torch.device("cuda"), deterministic=config["deterministic"]):
            for _ in range(2):
                torch.manual_seed(0)
                model = build_model(config, plan, dropout=0.0).cuda()
                logits = model(batches["cugraph"])
                target = torch.arange(len(logits), device="cuda") % 2
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target.float())
                loss.backward()
                losses.append(float(loss))
                grads.append(
                    torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
                )
        check(losses[0] == losses[1] and torch.equal(grads[0], grads[1]), "deterministic CUDA step")
        print(f"  live training step loss {losses[0]:.6f}")
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, default=600, help="seeds for the uniformity test")
    parser.add_argument("--rmm-pool", default=None, help="optional RMM pool size, e.g. 2GiB")
    parser.add_argument("--live", action="store_true", help="also run one real batch and step")
    parser.add_argument("--config", type=Path, help="optional overrides of the built-in run")
    parser.add_argument("--roots", type=int, default=32)
    args = parser.parse_args()
    if not versions():
        print("cuGraph cannot run on this host; nothing to verify")
        return 2
    if args.rmm_pool:
        import rmm  # pyright: ignore[reportMissingImports]

        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=args.rmm_pool,
            devices=torch.cuda.current_device(),
        )
    sampler = SamplerPlan(
        "resample",
        roots=PoolPlan(recent=8, older=4, distinct=4, associations=2),
        relation_fanouts=(8, 4),
        association_fanout=1,
    )
    engine = CuGraphSampler()
    print("probe:")
    probe(engine)
    print("synthetic candidate table:")
    table = synthetic_table(96, np.random.default_rng(42))
    print(f"  {table.num_contexts} contexts, {len(table)} candidates")
    subset_checks(engine, table, sampler)
    temporal_boundary(engine)
    merged_slots(engine, sampler)
    print(f"uniformity over {args.seeds} seeds:")
    uniformity(engine, args.seeds)
    print("latency:")
    latency(engine, sampler)
    if args.live:
        print("live batch:")
        live(args.config, args.roots, {"relation_fanouts": sampler.relation_fanouts})
    print(f"{len(FAILURES)} failed checks" if FAILURES else "all checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
