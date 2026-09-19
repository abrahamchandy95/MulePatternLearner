"""Account supervision contract, separate from all temporal feature columns."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

ACCOUNT_FACT_COLUMNS = ["id", "account_type", "is_external", "first_seen_seq", "first_seen_ts_ms"]
ACCOUNT_SUPERVISION_COLUMNS = [
    "is_mule",
    "mule_label_known",
    "is_mule_masked",
    "pu_label",
    "mule_label_effective_seq",
    "mule_label_effective_ts_ms",
    "mule_label_available_seq",
    "mule_label_available_ts_ms",
    "mule_ring_id",
    "mule_label_source",
]
ACCOUNT_LOAD_COLUMNS = ACCOUNT_FACT_COLUMNS + ACCOUNT_SUPERVISION_COLUMNS
# The integer migration appends is_mule in graph storage. CSV/PSV input order
# stays unchanged; the loading job maps named columns to storage positions.
ACCOUNT_STORAGE_COLUMNS = [name for name in ACCOUNT_LOAD_COLUMNS if name != "is_mule"] + ["is_mule"]


def validate_account_supervision(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(["id", *ACCOUNT_SUPERVISION_COLUMNS]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing account supervision columns: {sorted(missing)}")
    result = frame.copy()
    if result["id"].duplicated().any() or result["id"].astype(str).eq("").any():
        raise ValueError("Account IDs must be nonempty and unique")
    # Match GSQL INT input: literal integer 0/1, never boolean true/false.
    if not result["is_mule"].astype(str).isin(["0", "1"]).all():
        raise ValueError("is_mule must be an integer 0 or 1")
    result["is_mule"] = result["is_mule"].astype(np.int64)
    for field in ("mule_label_known", "is_mule_masked"):
        parsed = (
            result[field]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
        if parsed.isna().any():
            raise ValueError(f"Invalid boolean field: {field}")
        result[field] = parsed.astype(bool)
    numeric = [
        "pu_label",
        "mule_ring_id",
        *[
            f"mule_label_{kind}_{clock}"
            for kind in ("effective", "available")
            for clock in ("seq", "ts_ms")
        ],
    ]
    for field in numeric:
        values = pd.to_numeric(result[field], errors="raise")
        if not np.isfinite(values.to_numpy()).all() or (values != np.floor(values)).any():
            raise ValueError(f"Expected integer field: {field}")
        if (values < (-1 if field == "mule_ring_id" else 0)).any():
            raise ValueError(f"Invalid negative value: {field}")
        result[field] = values.astype(np.int64)
    known = result["mule_label_known"].to_numpy(bool)
    mule = result["is_mule"].to_numpy(bool)
    masked = result["is_mule_masked"].to_numpy(bool)
    expected = (known & mule & ~masked).astype(np.int64)
    if not np.array_equal(result["pu_label"].to_numpy(np.int64), expected):
        raise ValueError("pu_label must be 1 exactly for a known, unmasked mule; otherwise 0")
    if ((~known) & (mule | ~masked | result["mule_ring_id"].ne(-1).to_numpy())).any():
        raise ValueError("Unknown truth requires is_mule=0, is_mule_masked=true and ring=-1")
    if ((~mule) & result["mule_ring_id"].ne(-1).to_numpy()).any():
        raise ValueError("Only mule accounts may carry a mule ring ID")
    for clock in ("seq", "ts_ms"):
        effective = result[f"mule_label_effective_{clock}"].to_numpy(np.int64)
        available = result[f"mule_label_available_{clock}"].to_numpy(np.int64)
        if (known & ((effective <= 0) | (available < effective))).any():
            raise ValueError(
                "Known labels require positive effective clocks and availability at or after effectiveness"
            )
    return result


def write_account_labels(frame: pd.DataFrame, output: Path, source_manifest_sha256: str) -> int:
    """Export only known truth for evaluation; preserve graph mask independently."""
    validated = validate_account_supervision(frame)
    known = validated[validated["mule_label_known"]].copy()
    labels = known[
        [
            "id",
            "is_mule",
            "mule_label_effective_ts_ms",
            "mule_label_available_ts_ms",
            "mule_ring_id",
            "is_mule_masked",
            "pu_label",
            "mule_label_effective_seq",
            "mule_label_available_seq",
            "mule_label_source",
        ]
    ].rename(
        columns={
            "id": "account_id",
            "is_mule": "target",
            "mule_label_effective_ts_ms": "effective_ts_ms",
            "mule_label_available_ts_ms": "available_ts_ms",
            "mule_ring_id": "ring_id",
            "is_mule_masked": "graph_is_mule_masked",
            "pu_label": "graph_pu_label",
        }
    )
    labels["target"] = labels["target"].astype(np.int64)
    labels.to_parquet(output, index=False)
    metadata = {
        "target_definition": "account_mule",
        "source": "Account.is_mule in the verified loader export",
        "complete_negative_ground_truth": True,
        "negative_definition": "Only explicitly known is_mule=0 accounts are evaluation negatives; unknown accounts are omitted",
        "known_labels": len(labels),
        "positive_accounts": int(labels["target"].sum()),
        "masked_mules": int((known["is_mule"].eq(1) & known["is_mule_masked"]).sum()),
        "source_manifest_sha256": source_manifest_sha256,
        "label_source_values": sorted(known["mule_label_source"].astype(str).unique().tolist()),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return len(labels)
