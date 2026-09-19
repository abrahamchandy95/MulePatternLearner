"""One-command preparation and training with reproducible default paths."""

from __future__ import annotations

import json
from pathlib import Path

from mule_pattern_learner.configuration import load_config
from typing import Any
from pyTigerGraph.common.exception import TigerGraphException

from .contract import fingerprint
from .dataset import ROOT, prepare
from .installation import verify_sources
from .source import TigerGraphExecutor, checked_rows
from .training import output_paths, train
from .policy import validate_protocol
from .supervision import ParquetObservedLabels

EXAMPLE_CONFIG = ROOT / "configs/temporal/live_tgat.toml"
LOCAL_CONFIG = ROOT / "configs/local/live_tgat.toml"
DEFAULT_CONFIG = LOCAL_CONFIG if LOCAL_CONFIG.exists() else EXAMPLE_CONFIG
DEFAULT_MODEL = ROOT / "models/temporal/model.pt"


def dataset_path(config: dict[str, Any]) -> Path:
    return ROOT / "artifacts/temporal" / config["dataset_id"]


def prepare_live(config: dict[str, Any], output: Path) -> dict[str, Any]:
    validate_protocol(config)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source"]["config_sha256"] != fingerprint(config):
            raise ValueError("Preparation configuration changed; use a fresh dataset directory")
        if manifest["status"] == "ready":
            # The trainer checks hashes before use. No database connection is needed.
            return manifest
    executor = TigerGraphExecutor()
    verify_sources(executor)
    if config["evaluation_protocol"] == "strict_inductive":
        scope_id = config["scope_id"]
        try:
            existing = executor.client.conn.getVerticesById("Temporal_Training_Scope", [scope_id])
        except TigerGraphException as error:
            if str(error.code) != "601":
                raise
            existing = []
        if not isinstance(existing, list):
            raise ValueError("Unexpected scope metadata response")
        if existing:
            attrs = existing[0]["attributes"]
            if (
                not attrs["ready"]
                or attrs["source_id"] != config["dataset_id"]
                or attrs["split_seed"] != int(config.get("split_seed", 42))
            ):
                raise ValueError("Scope is incomplete or belongs to a different source/partition")
        else:
            created = checked_rows(
                executor.run(
                    "temporal_create_training_scope",
                    {
                        "scope_id": scope_id,
                        "source_id": config["dataset_id"],
                        "split_seed": int(config.get("split_seed", 42)),
                    },
                )
            )
            expected = next(row["expected_members"] for row in created if "expected_members" in row)
            checked_rows(
                executor.run(
                    "temporal_finalize_training_scope",
                    {
                        "scope_id": scope_id,
                        "expected_members": expected,
                    },
                )
            )
    raw_counts = executor.client.conn.getVertexCount("*", realtime=True)
    if not isinstance(raw_counts, dict):
        raise ValueError("TigerGraph did not return counts by vertex type")
    counts = {str(name): int(count) for name, count in raw_counts.items()}
    labels_path = config.get("observed_labels")
    labels = ParquetObservedLabels(Path(labels_path)) if labels_path else None
    result = prepare(config, output, executor, counts, labels=labels)
    if executor.client.conn.getVertexCount("*", realtime=True) != counts:
        result["status"] = "source_changed"
        manifest_path.write_text(json.dumps(result, indent=2) + "\n")
        raise ValueError("Graph counts changed; freeze ingestion and prepare a fresh cache")
    return result


def run(
    output: Path = DEFAULT_MODEL,
    *,
    config_path: Path = DEFAULT_CONFIG,
    dataset: Path | None = None,
) -> dict[str, Any]:
    """Prepare if needed, train with nnPU, and save the selected model at output."""
    checkpoint, reports = output_paths(output)
    if checkpoint.exists() or reports.exists():
        raise FileExistsError(f"Experiment already exists: {output}")
    config = load_config(config_path)
    validate_protocol(config)
    if dataset is None:
        dataset = dataset_path(config)
        prepare_live(config, dataset)
    # Explicit datasets are immutable pre-existing caches, useful for experiments.
    return train(config, dataset, output)
