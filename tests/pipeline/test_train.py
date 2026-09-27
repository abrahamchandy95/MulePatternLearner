"""The built-in run needs no flag, file or identifier."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mule_pattern_learner.cli import build_parser
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.paths import RESULTS_DIR, DatasetPaths, RunPaths
from mule_pattern_learner.pipeline.connect import open_context_source
from mule_pattern_learner.pipeline.train import BASELINE_RUN, run


def test_minimal_command_and_run_defaults(tmp_path: Path) -> None:
    # `mule-temporal train` needs no flag, file or identifier.
    args = build_parser().parse_args(["train"])
    assert vars(args) == {"command": "train"}
    # The built-in run goes to results/baseline/seed-42/.
    assert BASELINE_RUN == RunPaths(RESULTS_DIR / "baseline" / "seed-42")
    dataset = DatasetPaths.of("id", tmp_path / "data")
    output = RunPaths(tmp_path / "run")
    with (
        patch("mule_pattern_learner.pipeline.train.prepare_live", return_value=dataset) as prep,
        patch(
            "mule_pattern_learner.pipeline.train.train", return_value={"status": "complete"}
        ) as fit,
    ):
        assert run(output, data=tmp_path / "data")["status"] == "complete"
        prep.assert_called_once()
        # The built-in run's dataset, in the data directory.
        assert prep.call_args.args == (DEFAULT_CONFIG, tmp_path / "data")
        fit.assert_called_once()
        config, trained_on, written = fit.call_args.args
        assert written == output and trained_on == dataset
        assert fit.call_args.kwargs["open_contexts"] is open_context_source
        assert config is DEFAULT_CONFIG
        # A started run is refused before anything is prepared, unless it is resumed.
        output.root.mkdir()
        output.config.write_text("{}")
        with pytest.raises(FileExistsError, match="Run already exists"):
            run(output, data=tmp_path / "data")
        assert prep.call_count == 1
        run(output, data=tmp_path / "data", resume=True)
        assert fit.call_args.kwargs["resume"] is True
