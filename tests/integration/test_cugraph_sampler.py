"""The cuGraph neighbour sampler on a CUDA host, beside the torch sampler.

The probe that `sampler.backend = "auto"` runs (which `mule check` reports too), then the
checks of mule_pattern_learner.testing.sampler_checks on the GPU. The last test also
reads the graph: one real batch per backend from the built-in run's prepared dataset
(read-only queries; it never prepares one), and one deterministic CUDA training step
run twice. These need CUDA, cupy and pylibcugraph (the cuda12 or cuda13 extra).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from mule_pattern_learner.batching.assemble import build_root_batch
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.sampler_plan import PoolPlan, SamplerPlan
from mule_pattern_learner.data.hub_registry import load_hub_registry
from mule_pattern_learner.data.manifest import load_prepared
from mule_pattern_learner.data.splits import sample_keys
from mule_pattern_learner.model.build import build_model
from mule_pattern_learner.pipeline.connect import open_context_source
from mule_pattern_learner.pipeline.prepare import find_datasets
from mule_pattern_learner.runtime.device import reserve_deterministic_cublas, torch_runtime
from mule_pattern_learner.sampling.candidates import PAYMENT_COUNT
from mule_pattern_learner.sampling.cugraph_sampler import (
    CuGraphSampler,
    cugraph_import_error,
    probe_cugraph,
)
from mule_pattern_learner.testing import sampler_checks

pytestmark = pytest.mark.cuda

# Seeds of the uniformity test.
SEEDS = 600
# Training roots of the live batch.
ROOTS = 32
# The pools and fan-outs of the synthetic checks.
SAMPLER = SamplerPlan(
    roots=PoolPlan(recent=8, older=4, distinct=4, associations=2),
    relation_fanouts=(8, 4),
    association_fanout=1,
)


@pytest.fixture(scope="module")
def engine() -> CuGraphSampler:
    _, reason = cugraph_import_error()
    if reason is not None:
        pytest.skip(f"cuGraph cannot run here: {reason}")
    # Before any CUDA work: deterministic cuBLAS GEMMs need a fixed workspace.
    reserve_deterministic_cublas()
    return CuGraphSampler()


def test_the_probe_that_auto_runs_passes(engine: CuGraphSampler) -> None:
    assert probe_cugraph("cuda", engine) is None


def test_subsets_have_exact_counts_valid_times_and_fixed_seeds(engine: CuGraphSampler) -> None:
    table = sampler_checks.synthetic_table(96, np.random.default_rng(42))
    sampler_checks.check_subsets(engine, table, SAMPLER, "cuda")


def test_the_cutoff_boundary_is_strict(engine: CuGraphSampler) -> None:
    sampler_checks.check_cutoff_boundary(engine, "cuda")


def test_merged_slots_and_evaluation_agree_across_backends(engine: CuGraphSampler) -> None:
    sampler_checks.check_merged_slots(engine, SAMPLER, "cuda")


def test_inclusion_is_uniform(engine: CuGraphSampler) -> None:
    sampler_checks.check_uniform_inclusion(engine, SEEDS, "cuda")


@pytest.mark.graph
def test_real_batches_per_backend_and_a_deterministic_cuda_step(engine: CuGraphSampler) -> None:
    config = DEFAULT_CONFIG
    plan, sampler = config.feature_plan(), config.sampler
    found = find_datasets(config)
    if len(found) != 1:
        pytest.skip(
            f"found {len(found)} datasets of the built-in run in data/; mule train prepares one"
        )
    (dataset,) = found
    manifest, accounts = load_prepared(dataset)
    hubs = load_hub_registry(dataset, manifest)
    train = accounts[accounts["split"] == "train"].head(ROOTS)
    keys = sample_keys(train, config.dataset.dates.train[0], manifest)
    contexts = open_context_source(dataset, manifest, config)

    def batch(backend: str, mode: str, stats: dict[str, Any]) -> dict[str, torch.Tensor]:
        prepared = build_root_batch(
            contexts,
            keys,
            fanouts=sampler.fanouts,
            device="cuda",
            plan=plan,
            sampler=replace(sampler, backend=backend),
            hubs=hubs,
            mode=mode,
            step_seed=7,
        )
        assert prepared.batch is not None, "TigerGraph rejected every root"
        stats.update(prepared.stats)
        return prepared.batch

    try:
        batches: dict[str, dict[str, torch.Tensor]] = {}
        for backend in ("torch", "cugraph"):
            stats: dict[str, Any] = {}
            batches[backend] = batch(backend, "train", stats)
            assert stats["sampler_backend"] == backend
        for backend, sampled in batches.items():
            codes = sampled["first_relation"].masked_fill(~sampled["first_mask"], -1)
            caps = [int((codes == r).sum(1).max()) for r in range(PAYMENT_COUNT)]
            assert max(caps) <= sampler.relation_fanouts[0], (backend, caps)
            second = sampled["second_relation"][sampled["second_mask"]]
            assert bool((second < PAYMENT_COUNT).all()), backend
        evaluation = [batch(backend, "eval", {}) for backend in ("torch", "cugraph")]
        assert all(torch.equal(evaluation[0][k], evaluation[1][k]) for k in evaluation[0])
        # One deterministic CUDA training step, twice from the same state.
        losses, grads = [], []
        with torch_runtime(torch.device("cuda"), deterministic=config.runtime.deterministic):
            for _ in range(2):
                torch.manual_seed(0)
                model = build_model(config.model, plan, sampler.fanouts[0], dropout=0.0).cuda()
                logits = model(batches["cugraph"])
                target = torch.arange(len(logits), device="cuda") % 2
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target.float())
                loss.backward()
                losses.append(float(loss))
                grads.append(
                    torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
                )
        assert losses[0] == losses[1] and torch.equal(grads[0], grads[1])
    finally:
        contexts.close()
