"""Convert the earlier Account.is_mule BOOL deployment to INT, preserving labels.

Run once with loading paused. TigerGraph requires dropping/readding the attribute
to change its type. A local backup precedes all writes; account vertices, edges,
other attributes and CSV input column positions remain intact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, cast

from pyTigerGraph import TigerGraphConnection

from mule_pattern_learner.temporal.live.labels import ACCOUNT_STORAGE_COLUMNS
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings

from install_account_supervision import GRAPH, QUERIES
from install_time_encoding import gsql

ROOT = Path(__file__).resolve().parents[2]


def export_labels(conn: TigerGraphConnection) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    cursor = ""
    while True:
        blocks = conn.runInstalledQuery(
            QUERIES[0], {"after_id": cursor, "batch_size": 10000}, usePost=True
        )
        rows = [row for block in blocks for row in block.get("accounts", [])]
        if not rows:
            return labels
        for row in rows:
            key = str(row["v_id"])
            if key <= cursor or key in labels:
                raise RuntimeError("Supervision export did not advance")
            labels[key] = cast(dict[str, Any], row["attributes"])
        cursor = max(str(row["v_id"]) for row in rows)


def main(output: Path) -> None:
    settings = Settings()
    if settings.graphname != GRAPH:
        raise ValueError("Unexpected graph")
    conn = Client(settings).conn
    prefix = f"USE GRAPH {GRAPH}\n"
    schema = conn.getSchema(force=True)
    account = next(v for v in schema["VertexTypes"] if v["Name"] == "Account")
    label = next(a for a in account["Attributes"] if a["AttributeName"] == "is_mule")
    if label["AttributeType"]["Name"] != "BOOL":
        raise ValueError("This one-time conversion requires the earlier BOOL schema")
    old_columns = ["id", *[a["AttributeName"] for a in account["Attributes"]]]
    new_columns = [name for name in old_columns if name != "is_mule"] + ["is_mule"]
    if new_columns != ACCOUNT_STORAGE_COLUMNS:
        raise ValueError("Unexpected Account column contract; no change applied")
    status = gsql(conn, prefix + "SHOW LOADING STATUS ALL")
    if "There is no running loading jobs" not in status:
        raise RuntimeError("Loading is active; no schema change applied")
    before = cast(dict[str, int], conn.getVertexCount("*", realtime=True))
    catalog = str(conn.gsql(prefix + "LS"))
    installed = re.findall(r"- (\w+)\(.*\) \(installed v2\)", catalog)
    labels = export_labels(conn)
    if len(labels) != before["Account"]:
        raise RuntimeError("Incomplete label backup; no change applied")
    if any(type(row["is_mule"]) is not bool for row in labels.values()):
        raise RuntimeError("Unexpected label values; no change applied")
    repairs = []
    for match in re.finditer(
        r"CREATE LOADING JOB (\w+) FOR GRAPH \w+ \{.*?\n    \}", catalog, re.DOTALL
    ):
        definition = match.group(0)
        if not re.search(r"TO VERTEX Account\b", definition):
            continue
        values = re.search(r"(TO VERTEX Account VALUES\()([^)]*)(\))", definition)
        if values is None:
            raise RuntimeError("Unsupported Account loader; no change applied")
        tokens = [token.strip() for token in values[2].split(",")]
        if len(tokens) != len(old_columns):
            raise RuntimeError("Unexpected Account loader width; no change applied")
        mapping = dict(zip(old_columns, tokens, strict=True))
        updated = (
            definition[: values.start(2)]
            + ", ".join(mapping[name] for name in new_columns)
            + definition[values.end(2) :]
        )
        repairs.append((match.group(1), updated))
    if {name for name, _ in repairs} != {"mt_load_account", "load_temporal_accounts"}:
        raise RuntimeError("Unexpected loading jobs; review before converting")
    backup = ROOT / "artifacts/temporal/schema/is_mule_integer_backup.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite the recovery copy from an interrupted migration.
    with backup.open("x") as stream:
        json.dump(
            {
                "schema": schema,
                "counts": before,
                "catalog": catalog,
                "labels": labels,
                "repaired_loading_definitions": repairs,
            },
            stream,
        )
    print(f"Backed up {len(labels)} account supervision records", flush=True)
    for name in QUERIES:
        gsql(conn, prefix + f"DROP QUERY {name}")
    for job, operation in (
        ("drop_boolean_mule_label", "DROP ATTRIBUTE (is_mule)"),
        ("add_integer_mule_label", "ADD ATTRIBUTE (is_mule INT DEFAULT 0)"),
    ):
        gsql(
            conn,
            prefix
            + f"CREATE SCHEMA_CHANGE JOB {job} FOR GRAPH {GRAPH} {{\n"
            + f"  ALTER VERTEX Account {operation};\n}}",
        )
        gsql(conn, prefix + f"RUN SCHEMA_CHANGE JOB {job} -N -warn")
        gsql(conn, prefix + f"DROP JOB {job}")
    positives = [(key, {"is_mule": 1}) for key, row in labels.items() if row["is_mule"]]
    for start in range(0, len(positives), 1000):
        batch = positives[start : start + 1000]
        if conn.upsertVertices("Account", batch) != len(batch):
            raise RuntimeError(f"Incomplete positive restoration; recovery backup: {backup}")
    for name, definition in repairs:
        result = conn._req(  # pyright: ignore[reportPrivateUsage]
            "PUT",
            conn.gsUrl + "/gsql/v1/loading-jobs",
            headers={"Content-Type": "text/plain"},
            params={"graph": GRAPH},
            data=definition,
            resKey="",
        )
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(f"Loader repair failed: {name}")
        print(f"Updated Account column mapping: {name}", flush=True)
    gsql(conn, (ROOT / "gsql/temporal/account_supervision.gsql").read_text())
    gsql(conn, prefix + "INSTALL QUERY " + ", ".join(installed))
    verify_conversion(conn, schema, before, labels, repairs, installed, backup, output)


def verify_conversion(
    conn: TigerGraphConnection,
    schema: dict[str, Any],
    before: dict[str, int],
    labels: dict[str, dict[str, Any]],
    repairs: list[tuple[str, str]],
    installed: list[str],
    backup: Path,
    output: Path,
) -> None:
    """Check the restored graph against the pre-conversion backup; write the record."""
    prefix = f"USE GRAPH {GRAPH}\n"
    account = next(v for v in schema["VertexTypes"] if v["Name"] == "Account")
    new_columns = ACCOUNT_STORAGE_COLUMNS
    final_schema = conn.getSchema(force=True)
    final_account = next(v for v in final_schema["VertexTypes"] if v["Name"] == "Account")
    final_label = next(a for a in final_account["Attributes"] if a["AttributeName"] == "is_mule")
    assert final_label["AttributeType"]["Name"] == "INT"
    assert ["id", *[a["AttributeName"] for a in final_account["Attributes"]]] == new_columns
    original_other = [a for a in account["Attributes"] if a["AttributeName"] != "is_mule"]
    assert original_other == [
        a for a in final_account["Attributes"] if a["AttributeName"] != "is_mule"
    ]
    final_labels = export_labels(conn)
    expected = {key: {**row, "is_mule": int(row["is_mule"])} for key, row in labels.items()}
    if final_labels != expected or any(
        type(row["is_mule"]) is not int for row in final_labels.values()
    ):
        raise RuntimeError(f"Label preservation check failed; recovery backup: {backup}")
    counts = conn.getVertexCount("*", realtime=True)
    if counts != before:
        raise RuntimeError("Vertex counts changed during conversion")
    final_catalog = str(conn.gsql(prefix + "LS"))
    if re.search(r"(?i)disabled", final_catalog):
        raise RuntimeError("A loading job or query is disabled")
    for name in installed:
        if not re.search(rf"- {name}\(.*\) \(installed v2\)", final_catalog):
            raise RuntimeError(f"Query is no longer installed: {name}")
    validation = conn.runInstalledQuery(QUERIES[1], usePost=True)
    if any(value for row in validation for key, value in row.items() if key.startswith("invalid_")):
        raise RuntimeError("Account supervision validation failed")
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "graphname": GRAPH,
        "account_attributes": final_account["Attributes"],
        "vertex_counts": counts,
        "existing_account_loaders_repaired": [name for name, _ in repairs],
        "new_account_loader": "load_temporal_accounts",
        "installed_supervision_queries": list(QUERIES),
        "validation": validation,
        "data_deleted": False,
        "labels_inferred_or_populated": False,
        "label_contract": "Ground truth is integer is_mule (0/1); hidden positives retain is_mule=1 and have is_mule_masked=true, pu_label=0. Unknown truth has mule_label_known=false.",
        "integer_conversion": {
            "previous_type": "BOOL",
            "current_type": "INT",
            "labels_verified": len(final_labels),
            "positives_preserved": sum(bool(row["is_mule"]) for row in labels.values()),
            "csv_column_order_preserved": True,
            "installed_queries_preserved": installed,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    # Parse before main() so --help never connects to TigerGraph.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/temporal/reports/temporal_account_supervision_deployment.json",
        help="Where the deployment record is written",
    )
    args = parser.parse_args()
    try:
        main(args.output)
    except BaseException as exc:
        import traceback

        print(f"Integer conversion error: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
