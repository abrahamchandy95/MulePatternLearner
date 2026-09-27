"""Choosing the subset sampler: the cuGraph probe, CPU hosts and the run's backend."""

from __future__ import annotations

from dataclasses import replace
import json
import re
from typing import Any

import pytest
import torch

from mule_pattern_learner.batching.assemble import build_batch
from mule_pattern_learner.contract.feature_groups import DEFAULT_GROUPS, FeaturePlan
from mule_pattern_learner.sampling import backend, cugraph_sampler
from mule_pattern_learner.sampling.backend import resolve_backend
from mule_pattern_learner.sampling.cugraph_sampler import CuGraphProbe
from mule_pattern_learner.testing.builders import RESAMPLE, roots
from mule_pattern_learner.testing.fake_graph import FakeStore


def _fake_probes(monkeypatch: pytest.MonkeyPatch, probe: CuGraphProbe) -> list[int]:
    """Replace the device probe; returns the device indices it was run for."""
    calls: list[int] = []

    def fake(index: int) -> CuGraphProbe:
        calls.append(index)
        return probe

    monkeypatch.setattr(cugraph_sampler, "_PROBES", {})
    monkeypatch.setattr(cugraph_sampler, "_probe_device", fake)
    return calls


def warnings_emitted(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    """The warning events printed since the last call."""
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    return [record for record in printed if record["event"] == "warning"]


def test_auto_backend_falls_back_to_torch_when_the_probe_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = CuGraphProbe(False, True, "RuntimeError: CUDA error: no kernel image")
    calls = _fake_probes(monkeypatch, broken)
    assert resolve_backend(RESAMPLE, "cuda:1") == "torch"
    (warning,) = warnings_emitted(capsys)
    assert warning["warning"] == "cugraph_probe"
    assert re.search("no kernel image.*torch sampler", warning["message"])
    with pytest.raises(RuntimeError, match="cannot run on cuda:1: RuntimeError: CUDA error"):
        resolve_backend(replace(RESAMPLE, backend="cugraph"), "cuda:1")
    assert calls == [1]  # probed once per process and device, then cached
    # cuGraph not installed at all: torch without a warning.
    _fake_probes(monkeypatch, CuGraphProbe(False, False, "ModuleNotFoundError: cupy"))
    assert resolve_backend(RESAMPLE, "cuda:0") == "torch"
    assert warnings_emitted(capsys) == []
    calls = _fake_probes(monkeypatch, CuGraphProbe(True, True, "ok"))
    assert resolve_backend(RESAMPLE, "cuda:0") == "cugraph"
    assert resolve_backend(replace(RESAMPLE, backend="cugraph"), "cuda:0") == "cugraph"
    assert resolve_backend(replace(RESAMPLE, backend="torch"), "cuda:0") == "torch"
    assert calls == [0]


def test_backend_resolution_without_cuda() -> None:
    assert resolve_backend(RESAMPLE, "cpu") == "torch"
    assert resolve_backend(replace(RESAMPLE, backend="torch"), "cuda") == "torch"
    with pytest.raises(RuntimeError, match="cugraph needs a CUDA device"):
        resolve_backend(replace(RESAMPLE, backend="cugraph"), "cpu")


def test_build_batch_uses_the_run_backend_without_resolving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: Any) -> str:
        raise AssertionError("build_batch resolved the backend again")

    monkeypatch.setattr(backend, "resolve_backend", unexpected)
    plan = FeaturePlan(DEFAULT_GROUPS, "tgat")
    store = FakeStore(RESAMPLE)
    keys = roots(4)
    options: dict[str, Any] = {"plan": plan, "sampler": RESAMPLE, "step_seed": 3}
    for mode, given, used in (
        ("train", "torch", "torch"),
        ("eval", "torch", "torch"),
        ("eval", "cugraph", "torch"),  # evaluation is hash-keyed on the torch path
    ):
        stats: dict[str, Any] = {}
        build_batch(store, keys, mode=mode, sampler_backend=given, stats=stats, **options)
        assert stats["sampler_backend"] == used
    with pytest.raises(ValueError, match="cugraph needs a CUDA batch device"):
        build_batch(store, keys, mode="train", sampler_backend="cugraph", **options)
    with pytest.raises(ValueError, match="does not fit"):
        build_batch(store, keys, mode="train", sampler_backend="numpy", **options)
    pinned: dict[str, Any] = options | {"sampler": replace(RESAMPLE, backend="torch")}
    with pytest.raises(ValueError, match="does not fit"):
        build_batch(store, keys, mode="eval", sampler_backend="cugraph", **pinned)
    # The resolved backend gives the same batch as resolving per call.
    monkeypatch.undo()
    a = build_batch(store, keys, mode="train", **options)
    b = build_batch(store, keys, mode="train", sampler_backend="torch", **options)
    assert all(torch.equal(a[n], b[n]) for n in a)
