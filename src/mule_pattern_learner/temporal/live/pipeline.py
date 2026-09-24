"""One-command preparation and training with reproducible default paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mule_pattern_learner.configuration import load_config

from .config_schema import OPERATIONAL_DEFAULTS
from .dataset import (
    ROOT,
    check_query_hashes,
    preparation_mismatches,
    prepare,
)
from .installation import scope_header, source_counts, verify_scope, verify_sources
from .policy import validate_protocol
from .source import TigerGraphExecutor, checked_rows, transport_settings
from .supervision import label_source
from .training import output_paths, train

EXAMPLE_CONFIG = ROOT / "configs/temporal/live_tgat.toml"
LOCAL_CONFIG = ROOT / "configs/local/live_tgat.toml"
DEFAULT_CONFIG = LOCAL_CONFIG if LOCAL_CONFIG.exists() else EXAMPLE_CONFIG
DEFAULT_MODEL = ROOT / "models/temporal/model.pt"


def dataset_path(config: dict[str, Any]) -> Path:
    """artifacts/temporal/<prepared_id or dataset_id>."""
    return ROOT / "artifacts/temporal" / (config.get("prepared_id") or config["dataset_id"])


def ensure_scope(executor: TigerGraphExecutor, config: dict[str, Any]) -> None:
    """Use the frozen scope, or create it only when create_scope is explicitly set.

    Creation writes a Temporal_Training_Scope vertex and one membership edge per
    Account and Party. scope_unowned decides the accounts without an owning Party:
    - "independent": each is its own ownership group with a hashed partition.
    - "shared": unowned external accounts are visible in every phase (partition
      1, group "shared:<component>"); unowned internal accounts stay independent.
    - "linked" (the default): as "shared", and an unowned internal account whose
      only owned internal deposit counterparty is one account joins that
      account's ownership group and partition.
    An existing scope must have been created with the configured rule; it is
    inferred from the membership (temporal_scope_policy) and a mismatch raises.
    """
    scope_id = config["scope_id"]
    attrs = scope_header(executor, scope_id)
    if attrs is not None:
        counts = verify_scope(executor, config)
        print(json.dumps({"scope": scope_id, "unowned_members": counts}), flush=True)
        return
    if config.get("create_scope") is not True:
        raise ValueError(
            f"Scope {scope_id!r} does not exist on TigerGraph. Creating it writes to the "
            "graph; re-run with `mule-temporal prepare --create-scope` (or set "
            "create_scope = true), or set scope_id to an existing ready scope."
        )
    created = checked_rows(
        executor.run(
            "temporal_create_training_scope",
            {
                "scope_id": scope_id,
                "source_id": config["dataset_id"],
                "split_seed": int(config.get("split_seed", 42)),
                "unowned_policy": config.get(
                    "scope_unowned", OPERATIONAL_DEFAULTS["scope_unowned"]
                ),
            },
            timeout_s=3600.0,
            attempts=1,
        )
    )
    expected = next(row["expected_members"] for row in created if "expected_members" in row)
    checked_rows(
        executor.run(
            "temporal_finalize_training_scope",
            {"scope_id": scope_id, "expected_members": expected},
            timeout_s=3600.0,
            attempts=1,
        )
    )
    counts = verify_scope(executor, config)
    print(json.dumps({"scope": scope_id, "created": True, "unowned_members": counts}), flush=True)


def prepare_live(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Prepare (or resume) a dataset directory against the live graph.

    A ready directory is reused without connecting, but only after its GSQL
    hashes and preparation settings (PREPARATION_KEYS) match the current ones.
    """
    validate_protocol(config)
    labels = label_source(config)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        check_query_hashes(manifest, output)
        changed = preparation_mismatches(config, manifest)
        if changed:
            raise ValueError(
                f"Preparation settings changed for {output} ({', '.join(changed)}); "
                "set a new prepared_id or restore the prepared values"
            )
        if manifest["status"] == "ready":
            # The trainer re-verifies artifacts before use. No database connection is needed.
            return manifest
    transport = transport_settings(config)
    executor = TigerGraphExecutor(
        max_attempts=transport["max_query_attempts"], max_outage_s=transport["max_outage_s"]
    )
    verify_sources(executor)
    if config["evaluation_protocol"] == "strict_inductive":
        ensure_scope(executor, config)
    counts = source_counts(executor)
    result = prepare(config, output, executor, counts, labels=labels)
    if source_counts(executor) != counts:
        result["status"] = "source_changed"
        manifest_path.write_text(json.dumps(result, indent=2) + "\n")
        raise ValueError("Graph counts changed; freeze ingestion and prepare a fresh cache")
    return result


def run(
    output: Path = DEFAULT_MODEL,
    *,
    config_path: Path = DEFAULT_CONFIG,
    dataset: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Prepare if needed, train with nnPU, and save the selected model at output.

    With ``resume`` an interrupted run continues from its checkpoint_last.pt.
    """
    checkpoint, reports = output_paths(output)
    if not resume and (checkpoint.exists() or reports.exists()):
        raise FileExistsError(f"Experiment already exists: {output}; pass resume=True")
    config = load_config(config_path, live=True)
    validate_protocol(config)
    if dataset is None:
        dataset = dataset_path(config)
        prepare_live(config, dataset)
    # Explicit datasets are immutable pre-existing caches, useful for experiments.
    return train(config, dataset, output, resume=resume)
