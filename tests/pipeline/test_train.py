"""The built-in run needs no flag, file or identifier."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mule_pattern_learner.cli import build_parser
from mule_pattern_learner.config import DEFAULT_RUN
from mule_pattern_learner.paths import DEFAULT_MODEL
from mule_pattern_learner.pipeline.train import run


def test_minimal_command_and_run_defaults(tmp_path: Path) -> None:
    # `mule-temporal train` needs no flag, file or identifier.
    args = build_parser().parse_args(["train"])
    assert args.config is None and args.dataset is None and args.output == DEFAULT_MODEL
    manifest = {"source": {"dataset_id": "graph_snapshot"}}
    with (
        patch("mule_pattern_learner.pipeline.train.prepare_live", return_value=manifest) as prep,
        patch(
            "mule_pattern_learner.pipeline.train.train", return_value={"status": "complete"}
        ) as fit,
    ):
        assert run(tmp_path / "model.pt")["status"] == "complete"
        prep.assert_called_once()
        # The prepared cache lives inside the run directory.
        assert prep.call_args.args[1] == tmp_path / "model_run" / "prepared"
        fit.assert_called_once()
        config, dataset, output = fit.call_args.args
        assert output == tmp_path / "model.pt" and dataset == tmp_path / "model_run" / "prepared"
        assert config["dataset_id"] == "graph_snapshot"
        assert config["device"] == "auto"
        assert config["scope_id"] == DEFAULT_RUN["scope_id"]
