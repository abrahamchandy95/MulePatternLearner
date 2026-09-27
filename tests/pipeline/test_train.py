"""The built-in run needs no flag, file or identifier."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mule_pattern_learner.cli import build_parser
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.paths import DEFAULT_MODEL, DatasetPaths
from mule_pattern_learner.pipeline.connect import open_context_source
from mule_pattern_learner.pipeline.train import run


def test_minimal_command_and_run_defaults(tmp_path: Path) -> None:
    # `mule-temporal train` needs no flag, file or identifier.
    args = build_parser().parse_args(["train"])
    assert args.output == DEFAULT_MODEL
    assert not hasattr(args, "config")
    dataset = DatasetPaths.of("id", tmp_path / "data")
    with (
        patch("mule_pattern_learner.pipeline.train.prepare_live", return_value=dataset) as prep,
        patch(
            "mule_pattern_learner.pipeline.train.train", return_value={"status": "complete"}
        ) as fit,
    ):
        assert run(tmp_path / "model.pt", data=tmp_path / "data")["status"] == "complete"
        prep.assert_called_once()
        # The built-in run's dataset, in the data directory.
        assert prep.call_args.args == (DEFAULT_CONFIG, tmp_path / "data")
        fit.assert_called_once()
        config, trained_on, output = fit.call_args.args
        assert output == tmp_path / "model.pt" and trained_on == dataset
        assert fit.call_args.kwargs["open_contexts"] is open_context_source
        assert config is DEFAULT_CONFIG
