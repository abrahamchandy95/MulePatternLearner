"""Score a staged cutoff and export learned entity embeddings locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .encoding import BASIS_ID
from .model import Normalizer, TemporalModel, make_batch
from .snapshots import Snapshot
from .staging import digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", type=Path, default=Path("artifacts/temporal/stage"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new score output path")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    snapshot = Snapshot.load(args.snapshot)
    if (
        checkpoint["basis_id"] != BASIS_ID
        or checkpoint["provenance"]["stage_sha256"] != digest(args.stage / "manifest.json")
        or snapshot.metadata["stage_sha256"] != digest(args.stage / "manifest.json")
    ):
        raise ValueError("Checkpoint, stage or encoding provenance differs")
    normalizer = Normalizer(
        checkpoint["normalizer_mean"].numpy(), checkpoint["normalizer_scale"].numpy()
    )
    model = TemporalModel(
        len(normalizer.mean),
        int(config.get("hidden", 32)),
        float(config.get("dropout", 0.15)),
        str(config["variant"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    torch.set_num_threads(int(config.get("threads", 4)))
    nodes = pd.read_parquet(args.stage / "nodes.parquet")
    accounts = nodes[
        (nodes["node_type"] == "Account")
        & ~nodes["is_external"].astype(bool)
        & (nodes["subtype"] == "deposit")
        & (nodes["first_seen_ts_ms"] < snapshot.metadata["cutoff_ms"])
    ].copy()
    scores = []
    embeddings = []
    fanouts = (int(config["fanouts"][0]), int(config["fanouts"][1]))
    with torch.inference_mode():
        for offset in range(0, len(accounts), args.batch_size):
            roots = accounts.iloc[offset : offset + args.batch_size]["node_index"].to_numpy(
                np.int64
            )
            batch = make_batch(snapshot, roots, normalizer, variant=model.variant, fanouts=fanouts)
            hidden = model.encode(batch)
            scores.append(torch.sigmoid(model.head(hidden)).squeeze(-1).numpy())
            embeddings.append(hidden.numpy())
    output = accounts[["id", "node_index"]].copy()
    output["cutoff_ts_ms"] = snapshot.metadata["cutoff_ms"]
    output["score"] = np.concatenate(scores)
    output["above_validation_threshold"] = output["score"] >= checkpoint["threshold"]
    output["embedding"] = list(np.concatenate(embeddings))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    metadata = {
        "task": config["task"],
        "checkpoint_sha256": digest(args.checkpoint),
        "snapshot_sha256": digest(args.snapshot / "metadata.json"),
        "accounts": len(output),
        "learned_embedding_dimensions": int(config.get("hidden", 32)),
        "time_basis_dimensions": 64,
        "threshold": checkpoint["threshold"],
        "writes_to_graph": 0,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
