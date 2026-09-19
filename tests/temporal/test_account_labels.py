from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.temporal.account_labels import (
    ACCOUNT_LOAD_COLUMNS,
    ACCOUNT_STORAGE_COLUMNS,
    validate_account_supervision,
    write_account_labels,
)
from mule_pattern_learner.temporal.supervision import (
    graph_reveal_mask,
    labels_at_cutoff,
    read_labels,
)


def accounts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                "hidden",
                "deposit",
                False,
                1,
                100,
                1,
                True,
                True,
                0,
                2,
                200,
                3,
                300,
                0,
                "simulator",
            ],
            [
                "revealed",
                "deposit",
                False,
                1,
                100,
                1,
                True,
                False,
                1,
                2,
                200,
                3,
                300,
                1,
                "simulator",
            ],
            [
                "negative",
                "deposit",
                False,
                1,
                100,
                0,
                True,
                True,
                0,
                1,
                100,
                1,
                100,
                -1,
                "simulator",
            ],
            ["unknown", "deposit", True, 1, 100, 0, False, True, 0, 0, 0, 0, 0, -1, ""],
        ],
        columns=ACCOUNT_LOAD_COLUMNS,
    )


def test_hidden_mule_truth_survives_staging_and_stored_mask(tmp_path: Path) -> None:
    source = accounts()
    label_path = tmp_path / "account_labels.parquet"
    assert write_account_labels(source, label_path, "source-checksum") == 3
    nodes = source[["id"]].assign(node_type="Account", node_index=[1, 2, 3, 4])
    labels, meta = read_labels(label_path, nodes)
    assert meta["target_definition"] == "account_mule"
    assert labels.set_index("account_id").loc["hidden", "target"] == 1
    roots = nodes.assign(split="train")
    reveal = graph_reveal_mask(roots, labels)
    assert reveal.tolist() == [False, True, False, False]
    truth, observed = labels_at_cutoff(roots, labels, 400, reveal)
    assert truth.tolist() == [1, 1, 0, -1]
    assert observed.tolist() == [False, True, False, False]
    _, too_early = labels_at_cutoff(roots, labels, 250, reveal)
    assert not too_early.any()
    assert source.loc[0, "is_mule"]  # masking never rewrites source truth


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pu_label", 1),
        ("mule_label_available_seq", 1),
        ("mule_label_available_ts_ms", 199),
        ("mule_label_effective_ts_ms", 0),
        ("mule_ring_id", -2),
        ("is_mule", "invalid"),
        ("is_mule", 2),
        ("is_mule", -1),
        ("is_mule", "true"),
        ("is_mule", True),
    ],
)
def test_reject_invalid_mask_or_clock(field: str, value: str | int) -> None:
    source = accounts().astype(object)
    source.loc[0, field] = value
    with pytest.raises(ValueError):
        validate_account_supervision(source)


def test_ring_zero_valid_and_unknown_is_not_negative(tmp_path: Path) -> None:
    validated = validate_account_supervision(accounts())
    assert validated["is_mule"].dtype == np.dtype("int64")
    assert validated.loc[0, "mule_ring_id"] == 0
    write_account_labels(validated, tmp_path / "labels.parquet", "source")
    saved = pd.read_parquet(tmp_path / "labels.parquet")
    assert "unknown" not in saved["account_id"].tolist()
    assert json.loads((tmp_path / "labels.json").read_text())["masked_mules"] == 1


def test_account_schema_contract_matches_canonical_ddl() -> None:
    import re

    root = Path(__file__).resolve().parents[2]
    ddl = (root / "gsql/schema/temporal_schema.gsql").read_text()
    block = ddl.split("ADD VERTEX Account (", 1)[1].split(") WITH", 1)[0]
    fields = re.findall(
        r"^\s*(?:PRIMARY_ID )?(\w+)\s+(?:STRING|BOOL|UINT|INT)", block, re.MULTILINE
    )
    assert fields == ACCOUNT_STORAGE_COLUMNS
    assert re.search(r"is_mule INT DEFAULT 0", block)
    loader = (root / "gsql/schema/temporal_account_loading.gsql").read_text()
    columns = re.findall(r'\$"(\w+)"', loader)
    assert columns == ACCOUNT_STORAGE_COLUMNS
    header = loader.split("DEFINE HEADER account_header =", 1)[1].split(";", 1)[0]
    assert re.findall(r'"(\w+)"', header) == ACCOUNT_LOAD_COLUMNS
    assert np.array_equal(
        validate_account_supervision(accounts())["pu_label"].to_numpy(), [0, 1, 0, 0]
    )


def test_full_staging_keeps_account_labels_out_of_model_features(tmp_path: Path) -> None:
    import csv
    import hashlib
    from mule_pattern_learner.temporal.staging import (
        ASSOCIATIONS,
        ENTITY_COLUMNS,
        EVENT_COLUMNS,
        stage,
    )

    rows: dict[str, list[list[object]]] = {
        "Account": accounts().values.tolist(),
        "Party": [["party", "person", 1, 100]],
        "Token": [["token", 1, 100, "phone", "zelle"]],
        "Device": [["device", "mobile", 1, 100]],
        "IP": [["ip", 1, 100]],
        "Address": [["address", 1, 100, "US"]],
        "Zelle_Transfer": [
            ["T1", "1970-01-01 00:00:00", 400, 4, 10, True, "USD", "mobile", 1, True, 5, 500]
        ],
        "Transfer_From_Account": [["T1", "hidden", 400, 4]],
        "Transfer_To_Account": [["T1", "revealed", 400, 4]],
        "Party_Owns_Account": [["party", "hidden", 1, 0, 1.0, "simulator"]],
    }
    dataset_names = [*ENTITY_COLUMNS, *EVENT_COLUMNS, *ASSOCIATIONS]
    dataset_names += [
        prefix + "_" + role
        for prefix in ("Transfer", "Transaction")
        for role in (
            "From_Account",
            "To_Account",
            "From_Token",
            "To_Token",
            "Used_Device",
            "Used_IP",
        )
    ]
    datasets = {}
    for name in dataset_names:
        path = tmp_path / (name + ".psv")
        with path.open("w", newline="") as stream:
            csv.writer(stream, delimiter="|", lineterminator="\n").writerows(rows.get(name, []))
        count = len(rows.get(name, []))
        datasets[name] = {
            "rows": count,
            "shards": [
                {
                    "path": str(path),
                    "rows": count,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ],
        }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_database": "unit_fixture",
                "graphname": "Mule_Pattern_Learner",
                "datasets": datasets,
            }
        )
    )
    (tmp_path / "tigergraph_verification.json").write_text(
        json.dumps(
            {"passed": True, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
        )
    )
    report = stage(manifest, tmp_path / "staged")
    assert report["known_account_labels"] == 3
    nodes = pd.read_parquet(tmp_path / "staged/nodes.parquet")
    events = pd.read_parquet(tmp_path / "staged/events.parquet")
    forbidden = {
        "is_mule",
        "mule_label_known",
        "is_mule_masked",
        "pu_label",
        "mule_ring_id",
        "fraud_label",
        "label_known",
    }
    assert not forbidden & set(nodes.columns)
    assert not forbidden & set(events.columns)
    labels = pd.read_parquet(tmp_path / "staged/account_labels.parquet").set_index("account_id")
    assert labels.loc["hidden", "target"] == 1
    assert labels.loc["hidden", "graph_pu_label"] == 0
