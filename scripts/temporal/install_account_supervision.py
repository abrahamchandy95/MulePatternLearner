"""Add Account mule truth and mask fields to the configured live temporal graph.

Updates the canonical loading contract and preserves existing account loaders.
Does not clear data, manufacture labels, or change any account's ground truth.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings

from install_time_encoding import gsql

ROOT = Path(__file__).resolve().parents[2]
GRAPH = "Mule_Pattern_Learner"
FIELDS = (
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
)
QUERIES = ("temporal_get_account_supervision", "temporal_validate_account_supervision")


def main() -> None:
    settings = Settings()
    if settings.graphname != GRAPH:
        raise ValueError("Unexpected graph")
    conn = Client(settings).conn
    prefix = f"USE GRAPH {GRAPH}\n"
    before = conn.getVertexCount("*", realtime=True)
    schema = conn.getSchema(force=True)
    account = next(v for v in schema["VertexTypes"] if v["Name"] == "Account")
    names = {a["AttributeName"] for a in account["Attributes"]}
    present = names & set(FIELDS)
    if present and present != set(FIELDS):
        raise ValueError("Partial Account migration; inspect before continuing")
    if present:
        label = next(a for a in account["Attributes"] if a["AttributeName"] == "is_mule")
        if label["AttributeType"]["Name"] != "INT":
            raise ValueError(
                "Run convert_mule_label_to_integer.py before installing integer queries"
            )
    catalog = str(conn.gsql(prefix + "LS"))
    backup = ROOT / "artifacts/temporal/schema/account_supervision_preflight.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_text(
            json.dumps({"schema": schema, "counts": before, "catalog": catalog}, indent=2) + "\n"
        )
    restored = []
    if not present:
        status = gsql(conn, prefix + "SHOW LOADING STATUS ALL")
        if "There is no running loading jobs" not in status:
            raise RuntimeError("Loading is active; no schema change applied")
        repairs = []
        for match in re.finditer(
            r"CREATE LOADING JOB (\w+) FOR GRAPH \w+ \{.*?\n    \}", catalog, re.DOTALL
        ):
            definition = match.group(0)
            if not re.search(r"TO VERTEX Account\b", definition):
                continue
            updated, count = re.subn(
                r"(TO VERTEX Account VALUES\([^)]*)(\))",
                lambda m: m[1] + ", _" * len(FIELDS) + m[2],
                definition,
            )
            if count != 1:
                raise RuntimeError("Unsupported account loading definition")
            repairs.append((match.group(1), updated))
        migration = (ROOT / "gsql/schema/migrations/account_mule_supervision.gsql").read_text()
        gsql(conn, migration.split("RUN SCHEMA_CHANGE JOB", 1)[0])
        gsql(conn, prefix + "RUN SCHEMA_CHANGE JOB add_account_mule_supervision -warn")
        after = conn.getSchema(force=True)
        account = next(v for v in after["VertexTypes"] if v["Name"] == "Account")
        if not set(FIELDS) <= {a["AttributeName"] for a in account["Attributes"]}:
            raise RuntimeError("Account attributes were not added")
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
                raise RuntimeError(f"Account loader repair failed: {name}")
            restored.append(name)
        gsql(conn, prefix + "DROP JOB add_account_mule_supervision")
    loader = (ROOT / "gsql/schema/temporal_account_loading.gsql").read_text()
    if "CREATE LOADING JOB load_temporal_accounts " not in catalog:
        gsql(conn, loader)
    gsql(conn, (ROOT / "gsql/temporal/account_supervision.gsql").read_text())
    gsql(conn, prefix + "INSTALL QUERY " + ", ".join(QUERIES))
    final_catalog = str(conn.gsql(prefix + "LS"))
    for name in QUERIES:
        if not re.search(rf"- {name}\(.*\) \(installed v2\)", final_catalog):
            raise RuntimeError(f"Missing installed query: {name}")
    if re.search(r"(?i)disabled", final_catalog):
        raise RuntimeError("A loading job or query is disabled; inspect the catalog")
    counts = conn.getVertexCount("*", realtime=True)
    if counts != before:
        raise RuntimeError("Graph vertex counts changed during the schema update")
    validation = conn.runInstalledQuery("temporal_validate_account_supervision", usePost=True)
    schema = conn.getSchema(force=True)
    account = next(v for v in schema["VertexTypes"] if v["Name"] == "Account")
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "graphname": GRAPH,
        "account_attributes": account["Attributes"],
        "vertex_counts": counts,
        "existing_account_loaders_repaired": restored,
        "new_account_loader": "load_temporal_accounts",
        "installed_supervision_queries": list(QUERIES),
        "validation": validation,
        "data_deleted": False,
        "labels_inferred_or_populated": False,
        "label_contract": "Ground truth is integer is_mule (0/1); hidden positives retain is_mule=1 and have is_mule_masked=true, pu_label=0. Unknown truth has mule_label_known=false.",
    }
    destination = ROOT / "docs/temporal_account_supervision_deployment.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
