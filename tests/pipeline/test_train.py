"""The built-in run needs no flag, file or identifier."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mule_pattern_learner.cli import build_parser
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.paths import DEFAULT_MODEL
from mule_pattern_learner.pipeline.connect import open_context_source
from mule_pattern_learner.pipeline.train import run


def test_minimal_command_and_run_defaults(tmp_path: Path) -> None:
    # `mule-temporal train` needs no flag, file or identifier.
    args = build_parser().parse_args(["train"])
    assert args.dataset is None and args.output == DEFAULT_MODEL
    assert not hasattr(args, "config")
    manifest = {"source": {"source_id": "graph_snapshot"}}
    with (
        patch("mule_pattern_learner.pipeline.train.prepare_live", return_value=manifest) as prep,
        patch(
            "mule_pattern_learner.pipeline.train.train", return_value={"status": "complete"}
        ) as fit,
    ):
        assert run(tmp_path / "model.pt")["status"] == "complete"
        prep.assert_called_once()
        # The built-in run, prepared inside the run directory.
        assert prep.call_args.args == (DEFAULT_CONFIG, tmp_path / "model_run" / "prepared")
        fit.assert_called_once()
        config, dataset, output = fit.call_args.args
        assert output == tmp_path / "model.pt" and dataset == tmp_path / "model_run" / "prepared"
        assert fit.call_args.kwargs["open_contexts"] is open_context_source
        assert config is DEFAULT_CONFIG
