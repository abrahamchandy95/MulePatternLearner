"""The live temporal scripts import, show help without side effects, and run offline."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from mule_pattern_learner.contract.feature_groups import FeaturePlan, extraction_plan
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.data.contexts import StreamingContextSource, check_coverage
from mule_pattern_learner.data.preparation import prepare
from mule_pattern_learner.paths import REPOSITORY_ROOT
from mule_pattern_learner.reference import label_reveal
from mule_pattern_learner.testing.builders import (
    FrameObservedLabels,
    live_config,
    neighbourhood,
    reveal_inputs,
    scoped_accounts,
    supplied_labels,
)
from mule_pattern_learner.testing.fake_graph import FakeExecutor
from mule_pattern_learner.tigergraph import reveal
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffs
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubs
from mule_pattern_learner.tigergraph.scope import TigerGraphScope

SCRIPTS = REPOSITORY_ROOT / "scripts"
# Every script that talks to the live path; each must parse --help before connecting.
LIVE_SCRIPTS = (
    "benchmark_batch",
    "render_queries",
    "simulate_label_reveal",
    "verify_cugraph_sampler",
    "verify_feature_redesign",
    "verify_label_reveal",
    "verify_strict_isolation",
    "verify_training",
)


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The scripts that load torch; each reserves the cuBLAS workspace before anything else.
TORCH_SCRIPTS = (
    "benchmark_batch",
    "verify_cugraph_sampler",
    "verify_strict_isolation",
)


def environment_at_import(path: Path) -> list[int]:
    """Lines outside every function and class that read or change os.environ."""
    body = ast.parse(path.read_text()).body
    return [
        node.lineno
        for statement in body
        if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        for node in ast.walk(statement)
        if isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    ]


def test_no_module_or_script_reads_the_environment_when_imported() -> None:
    package = REPOSITORY_ROOT / "src/mule_pattern_learner"
    modules = [*package.rglob("*.py"), *SCRIPTS.glob("*.py")]
    assert {str(path): environment_at_import(path) for path in modules} == {
        str(path): [] for path in modules
    }


def test_entry_points_reserve_the_cublas_workspace_first() -> None:
    cli = REPOSITORY_ROOT / "src/mule_pattern_learner/cli.py"
    for path in (cli, *(SCRIPTS / f"{name}.py" for name in TORCH_SCRIPTS)):
        (main,) = (
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        first = main.body[0]
        assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call), path.name
        assert ast.unparse(first.value.func) == "reserve_deterministic_cublas", path.name
    # The other scripts never load torch, so they have no CUDA work to prepare for.
    others = [str(SCRIPTS / f"{name}.py") for name in LIVE_SCRIPTS if name not in TORCH_SCRIPTS]
    code = (
        "import importlib.util, sys\n"
        f"for path in {others!r}:\n"
        "    spec = importlib.util.spec_from_file_location('script', path)\n"
        "    spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "print('torch' in sys.modules)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("name", LIVE_SCRIPTS)
def test_live_scripts_import_and_print_help_without_connecting(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mule_pattern_learner.tigergraph.executor import TigerGraphExecutor

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("--help must not connect to TigerGraph")

    monkeypatch.setattr(TigerGraphExecutor, "__init__", refuse)
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
    monkeypatch.setattr(module, "connect", lambda config: pytest.fail("connected"))
    monkeypatch.setattr(sys, "argv", ["verify_strict_isolation"])
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 2


def test_strict_isolation_source_requests_what_the_fixture_checks_and_the_model_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load("verify_strict_isolation")
    built: list[dict[str, Any]] = []

    class Built(Exception):
        pass

    def record(executor: object, **options: Any) -> None:
        built.append(options)
        raise Built  # before the fixture writes anything

    graph = SimpleNamespace(client=SimpleNamespace(conn=None))
    monkeypatch.setattr(module, "connect", lambda config: graph)
    monkeypatch.setattr(module, "StreamingContextSource", record)
    monkeypatch.setattr(sys, "argv", ["verify_strict_isolation", "--write-fixture"])
    with pytest.raises(Built):
        module.main()
    (options,) = built
    # Without a plan the source requests its default groups.
    plan: FeaturePlan = options.get("plan", FeaturePlan())
    # The values the fixture asserts on: first-hop root features and message fields.
    checked = {
        "1h_out_count",
        "1h_out_amount",
        "1d_out_in_amount_ratio",
        "7d_out_in_amount_ratio",
        "pair_count_1h",
        "pair_count_1d",
        "pair_count_7d",
    }
    assert checked <= set(plan.node_names + plan.edge_names)
    flags = plan.query_flags(1)
    for group in ("rolling_windows", "amount_ratios", "pair_window_counts"):
        assert flags["include_" + group], group
    # The model it trains and the predictor that scores it read nothing the source skips.
    config = module.model_config()
    with StreamingContextSource(TigerGraphContextFetcher(FakeExecutor()), **options) as source:
        check_coverage(source, FeaturePlan.from_config(config), SamplerPlan.from_config(config))


def test_benchmark_builds_one_training_batch_and_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = live_config(batch_size=32, fanouts=[8, 2])
    config_path = tmp_path / "config.json"
    # The identity comes from the prepared dataset, as it does for `mule-temporal train`.
    config_path.write_text(json.dumps({k: v for k, v in config.items() if k != "dataset_id"}))
    dataset = tmp_path / "dataset"
    executor = FakeExecutor(factory=neighbourhood, hubs=[("N3", 101)], population=scoped_accounts())
    prepare(
        config,
        dataset,
        {"Account": 1000},
        FrameObservedLabels(supplied_labels()),
        scope=TigerGraphScope(executor),
        cutoffs=TigerGraphCutoffs(executor),
        hubs=TigerGraphHubs(executor),
    )
    module = load("benchmark_batch")

    def open_source(path: Path, manifest: dict[str, Any], training: dict[str, Any]) -> Any:
        assert path == dataset
        return StreamingContextSource(
            TigerGraphContextFetcher(executor),
            plan=extraction_plan(training),
            sampler=SamplerPlan.from_config(training),
        )

    monkeypatch.setattr(module, "open_context_source", open_source)
    output = tmp_path / "report.json"
    argv = ["benchmark_batch", "--config", str(config_path), "--dataset", str(dataset)]
    monkeypatch.setattr(sys, "argv", [*argv, "--output", str(output), "--train-step"])
    module.main()
    report = json.loads(output.read_text())
    assert report["status"] == "passed" and report["mode"] == "train"
    assert report["roots"] == report["accepted_roots"] == 32
    assert report["batch"]["sampler_backend"] == "torch"
    assert report["batch"]["stub_children"] > 0 and report["batch"]["first_edges"] > 0
    assert report["context_requests"] > 0 and report["rest_calls"] == 0  # fakes count none
    assert report["loss"] > 0 and report["train_step_seconds"] > 0
    assert math.isfinite(report["objective"])
    # Digests of every batch tensor (test_golden_run pins their values).
    digests = report["tensor_digests"]
    assert digests["root_positions"]["shape"] == [32] and len(digests["x"]["sha256"]) == 64


class RevealGraph:
    """The reveal's read-only inputs, and a dry run that agrees with the Python mirror."""

    def __init__(self) -> None:
        self.client = SimpleNamespace(conn=SimpleNamespace(runInterpretedQuery=self.inputs))
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def inputs(self, text: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        assert text == label_reveal.INPUTS_QUERY and set(params) == {"scope_id"}
        return reveal_inputs()

    def run(self, name: str, params: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((name, params))
        result = label_reveal.plan(reveal_inputs(), params)
        mules = result["mules"]
        rows = [
            {
                "account_id": k,
                "channel": mules[k]["channel"],
                "known_ts_ms": label_reveal.available_ms(mules[k], 10**13),
            }
            for k in result["revealed"]
        ]
        eligible = label_reveal.counts_by_split(result, "eligible")
        return [
            {
                "status": "dry_run",
                "data_end_ts_ms": 10**13,
                "eligible": {str(part): n for part, n in eligible.items() if n},
            },
            {"revealed_mules": rows},
        ]


def test_label_reveal_scripts_run_offline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    graph = RevealGraph()
    verify = load("verify_label_reveal")
    monkeypatch.setattr(verify, "connect", lambda config: graph)
    monkeypatch.setattr(sys, "argv", ["verify_label_reveal"])
    assert verify.main() == 0
    ((name, params),) = graph.calls
    # force only skips the already-revealed check; apply = FALSE writes nothing.
    assert name == reveal.REVEAL_QUERY and params["apply"] is False and params["force"] is True
    capsys.readouterr()
    simulate = load("simulate_label_reveal")
    monkeypatch.setattr(simulate, "connect", lambda config: graph)
    monkeypatch.setattr(sys, "argv", ["simulate_label_reveal", "--runs", "3"])
    simulate.main()
    report = json.loads(capsys.readouterr().out)
    assert report["salts"] == [0, 2] and len(graph.calls) == 1  # never runs the job
    assert {name: split["mules"] for name, split in report["splits"].items()} == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }
