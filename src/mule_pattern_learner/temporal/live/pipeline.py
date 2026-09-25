"""One-command preparation and training against the live graph, with built-in settings.

`mule-temporal train` needs only the TigerGraph credentials in .env. On a fresh graph
the first run installs the training queries, creates the frozen scope and reveals the
known mules through the label contract; later runs find all three in place. Settings
default to config_schema.DEFAULT_RUN, the dataset identity is read from the scope (or
derived from the graph), and the prepared cache is written inside the run directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config_schema import run_config
from .contract import fingerprint
from .dataset import (
    MANIFEST,
    ROOT,
    check_query_hashes,
    preparation_mismatches,
    prepare,
    read_manifest,
    write_manifest,
)
from .executor import TigerGraphExecutor, live_executor
from .installation import install, source_counts
from .labels import ensure_revealed_labels
from .policy import validate_protocol
from .scope import ensure_scope, scope_header
from .supervision import label_source
from .training import output_paths, train

DEFAULT_MODEL = ROOT / "models/temporal/model.pt"


def dataset_path(config: dict[str, Any], output: Path = DEFAULT_MODEL) -> Path:
    """The prepared cache of a run: <run directory>/prepared.

    An explicit prepared_id instead names a shared cache under artifacts/temporal,
    for experiments that train several models on one preparation.
    """
    if config.get("prepared_id"):
        return ROOT / "artifacts/temporal" / config["prepared_id"]
    return output_paths(output)[1] / "prepared"


def derived_dataset_id(graph: str, counts: dict[str, int]) -> str:
    """A stable name for the loaded snapshot: graph name plus a hash of its vertex counts."""
    return f"{graph}_{fingerprint(counts)[:12]}"


def resolve_identity(
    executor: TigerGraphExecutor, config: dict[str, Any], counts: dict[str, int]
) -> dict[str, Any]:
    """Fill dataset_id: an existing scope's source, else a name derived from the graph."""
    if config.get("dataset_id"):
        return config
    if config.get("evaluation_protocol") == "strict_inductive":
        attrs = scope_header(executor, config["scope_id"])
        if attrs is not None:
            return {**config, "dataset_id": str(attrs["source_id"])}
    graph = str(getattr(executor.client.conn, "graphname", "graph"))
    return {**config, "dataset_id": derived_dataset_id(graph, counts)}


def prepare_live(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Prepare (or resume) a dataset directory against the live graph.

    A ready directory is reused without connecting, but only after its GSQL
    hashes and preparation settings (PREPARATION_KEYS) match the current ones.
    Otherwise the graph is brought to a trainable state first: stale queries are
    installed, the scope is created if missing, and a strict run reveals known mules
    if the graph has none (label_policy = "graph_observed"). A shared_history run
    reads whatever labels the graph has.
    """
    validate_protocol(config)
    labels = label_source(config)
    if (output / MANIFEST).exists():
        manifest = read_manifest(output)
        if not config.get("dataset_id"):
            config = {**config, "dataset_id": manifest["source"]["dataset_id"]}
        check_query_hashes(manifest, output)
        changed = preparation_mismatches(config, manifest)
        if changed:
            raise ValueError(
                f"Preparation settings changed for {output} ({', '.join(changed)}); "
                "train into a new output, or restore the prepared values"
            )
        if manifest["status"] == "ready":
            # The trainer re-verifies artifacts before use. No database connection is needed.
            return manifest
    executor = live_executor(config)
    install(executor)
    counts = source_counts(executor)
    config = resolve_identity(executor, config, counts)
    if config["evaluation_protocol"] == "strict_inductive":
        ensure_scope(executor, config)
        # The reveal draws its splits from the scope partitions.
        if config.get("label_policy") == "graph_observed":
            ensure_revealed_labels(executor, config)
    counts = source_counts(executor)
    result = prepare(config, output, executor, counts, labels=labels)
    if source_counts(executor) != counts:
        result["status"] = "source_changed"
        write_manifest(output, result)
        raise ValueError("Graph counts changed; freeze ingestion and prepare a fresh cache")
    return result


def prepared_config(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """The run settings with the identity preparation resolved (dataset_id).

    A pinned dataset_id is kept, so preparation_mismatches reports it when it
    differs from the prepared dataset.
    """
    return {**config, "dataset_id": config.get("dataset_id") or manifest["source"]["dataset_id"]}


def run(
    output: Path = DEFAULT_MODEL,
    *,
    config_path: Path | None = None,
    dataset: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Prepare if needed, train with nnPU, and save the selected model at output.

    With ``resume`` (what `mule-temporal train` passes) an interrupted run continues
    from its checkpoint_last.pt; a finished one is an error.
    """
    checkpoint, _ = output_paths(output)
    if not resume and checkpoint.exists():
        raise FileExistsError(f"Experiment already exists: {output}; pass resume=True")
    config = run_config(config_path)
    if dataset is None:
        dataset = dataset_path(config, output)
        manifest = prepare_live(config, dataset)
    else:
        # Explicit datasets are immutable pre-existing caches, useful for experiments.
        manifest = read_manifest(dataset)
    return train(prepared_config(config, manifest), dataset, output, resume=resume)


__all__ = [
    "DEFAULT_MODEL",
    "dataset_path",
    "derived_dataset_id",
    "prepare_live",
    "prepared_config",
    "resolve_identity",
    "run",
]
