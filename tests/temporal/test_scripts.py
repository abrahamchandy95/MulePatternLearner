"""The live temporal scripts import, show help without side effects, and run offline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from mule_pattern_learner.temporal.live.contract import SamplerPlan
from mule_pattern_learner.temporal.live.dataset import prepare
from mule_pattern_learner.temporal.live.source import StreamingContextSource, extraction_plan
from mule_pattern_learner.temporal.live.supervision import FrameObservedLabels
from temporal_fakes import (
    REPOSITORY,
    FakeExecutor,
    fixture_accounts,
    live_config,
    neighbourhood,
    supplied_labels,
)

SCRIPTS = REPOSITORY / "scripts/temporal"
# Every script that talks to the live path; each must parse --help before connecting.
LIVE_SCRIPTS = (
    "benchmark_live_batch",
    "feature_experiments",
    "render_training_queries",
    "run_live_experiments",
    "verify_cugraph_sampler",
    "verify_feature_redesign",
    "verify_live_training",
    "verify_strict_isolation",
    "verify_training_source",
)


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", LIVE_SCRIPTS)
def test_live_scripts_import_and_print_help_without_connecting(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mule_pattern_learner.temporal.live import source

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("--help must not connect to TigerGraph")

    monkeypatch.setattr(source.TigerGraphExecutor, "__init__", refuse)
    module = load(name)
    monkeypatch.setattr(sys, "argv", [name, "--help"])
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_strict_isolation_fixture_needs_explicit_write_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load("verify_strict_isolation")
    monkeypatch.setattr(module, "TigerGraphExecutor", lambda: pytest.fail("connected"))
    monkeypatch.setattr(sys, "argv", ["verify_strict_isolation"])
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 2


class PopulationExecutor(FakeExecutor):
    def run(self, name: str, params: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        if name == "temporal_training_population":
            return [{"status": "ok", "accounts": fixture_accounts().to_dict("records")}]
        return super().run(name, params, **kwargs)


def test_benchmark_builds_one_training_batch_and_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = live_config(context_storage="stream", batch_size=32, fanouts=[8, 2])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    dataset = tmp_path / "dataset"
    executor = PopulationExecutor(factory=neighbourhood, hubs=[("N3", 101)])
    prepare(
        config, dataset, executor, {"Account": 1000}, labels=FrameObservedLabels(supplied_labels())
    )
    module = load("benchmark_live_batch")

    def open_source(path: Path, manifest: dict[str, Any], training: dict[str, Any]) -> Any:
        assert path == dataset
        return StreamingContextSource(
            executor, plan=extraction_plan(training), sampler=SamplerPlan.from_config(training)
        )

    monkeypatch.setattr(module, "open_context_source", open_source)
    output = tmp_path / "report.json"
    argv = ["benchmark_live_batch", "--config", str(config_path), "--dataset", str(dataset)]
    monkeypatch.setattr(sys, "argv", [*argv, "--output", str(output), "--train-step"])
    module.main()
    report = json.loads(output.read_text())
    assert report["status"] == "passed" and report["mode"] == "train"
    assert report["roots"] == report["accepted_roots"] == 32
    assert report["batch"]["sampler_backend"] == "torch"
    assert report["batch"]["stub_children"] > 0 and report["batch"]["first_edges"] > 0
    assert report["context_requests"] > 0 and report["rest_calls"] == 0  # fakes count none
    assert report["loss"] > 0 and report["train_step_seconds"] > 0
