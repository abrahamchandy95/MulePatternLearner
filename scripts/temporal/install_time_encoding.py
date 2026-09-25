"""Install the additive temporal features and GSQL queries using the project's .env.

Preserves and repairs the two existing positional payment loading definitions
if the schema change disables them. Never recreates the graph or loads data.
"""

import argparse
import json
from pathlib import Path
import re

from pyTigerGraph import TigerGraphConnection

from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
GRAPH = "Mule_Pattern_Learner"
FIELDS = {
    "pair_delta_t_ms",
    "pair_delta_t_present",
    "pair_time_encoding",
    "time_encoding_basis_id",
    "pair_previous_event_id",
    "pair_sender_id",
    "pair_recipient_type",
    "pair_recipient_id",
}
QUERY_NAMES = (
    "temporal_fourier64_values",
    "temporal_fourier64",
    "zelle_pair_time64",
    "payment_pair_time64",
)


def gsql(conn: TigerGraphConnection, source: str) -> str:
    result = conn.gsql(source)
    text = result if isinstance(result, str) else json.dumps(result)
    print(text, flush=True)
    if re.search(r"(?im)^(?:.*(?:syntax|semantic) error|error[: ]|failed\b)", text):
        raise RuntimeError("GSQL rejected the operation; see output above")
    return text


def main() -> None:
    settings = Settings()
    if settings.graphname != GRAPH:
        raise RuntimeError(f"These queries require graph {GRAPH}")
    conn = Client(settings).conn
    prefix = f"USE GRAPH {GRAPH}\n"
    schema = conn.getSchema(force=True)
    counts = conn.getVertexCount("*")
    catalog = str(conn.gsql(prefix + "LS"))
    backup = ROOT / "docs" / "temporal_encoding_preflight.json"
    if not backup.exists():
        backup.write_text(
            json.dumps({"schema": schema, "counts": counts, "catalog": catalog}, indent=2) + "\n"
        )
    vertex_types = {v["Name"]: v for v in schema["VertexTypes"]}
    present = [
        FIELDS & {a["AttributeName"] for a in vertex_types[name]["Attributes"]}
        for name in ("Zelle_Transfer", "Payment_Transaction")
    ]
    if any(present) and not all(fields == FIELDS for fields in present):
        raise RuntimeError("Partial temporal migration: inspect before continuing")

    if not all(present):
        status = gsql(conn, prefix + "SHOW LOADING STATUS ALL")
        if "There is no running loading jobs" not in status:
            raise RuntimeError("Loading activity detected; no schema change attempted")
        repairs = []
        for match in re.finditer(
            r"CREATE LOADING JOB (\w+) FOR GRAPH \w+ \{.*?\n    \}", catalog, re.DOTALL
        ):
            definition = match.group(0)
            if not re.search(r"TO VERTEX (?:Zelle_Transfer|Payment_Transaction)\b", definition):
                continue
            updated, count = re.subn(
                r"(TO VERTEX (?:Zelle_Transfer|Payment_Transaction) VALUES\([^)]*)(\))",
                lambda m: m[1] + ", _, _, _, _, _, _, _, _" + m[2],
                definition,
            )
            if count != 1:
                raise RuntimeError("Unsupported existing loader; no schema change attempted")
            repairs.append((match.group(1), updated))
        migration = (ROOT / "gsql/schema/migrations/temporal_encoding_attributes.gsql").read_text()
        # Run each phase separately: never drop the migration job after a failed run.
        creation = migration.split("RUN SCHEMA_CHANGE JOB", 1)[0]
        gsql(conn, creation)
        gsql(conn, prefix + "RUN SCHEMA_CHANGE JOB add_temporal_encoding_attributes -warn")
        after = conn.getSchema(force=True)
        for vertex in after["VertexTypes"]:
            if vertex["Name"] in ("Zelle_Transfer", "Payment_Transaction"):
                if not FIELDS <= {a["AttributeName"] for a in vertex["Attributes"]}:
                    raise RuntimeError("Migration did not add the expected fields")
        for name, definition in repairs:
            # pyTigerGraph's updateLoadingJob omits the required text/plain header.
            result = conn._req(  # pyright: ignore[reportPrivateUsage]
                "PUT",
                conn.gsUrl + "/gsql/v1/loading-jobs",
                headers={"Content-Type": "text/plain"},
                params={"graph": GRAPH},
                data=definition,
                resKey="",
            )
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(f"Restore loader {name}: {result}")
            print(f"Updated loading definition: {name}", flush=True)
        gsql(conn, prefix + "DROP JOB add_temporal_encoding_attributes")

    for filename in (
        "temporal_fourier64.gsql",
        "zelle_pair_time64.gsql",
        "payment_pair_time64.gsql",
    ):
        gsql(conn, (ROOT / "gsql/features" / filename).read_text())
    gsql(conn, prefix + "INSTALL QUERY " + ", ".join(QUERY_NAMES))
    listing = str(conn.gsql(prefix + "LS"))
    for name in QUERY_NAMES:
        if not re.search(rf"- {name}\(.*\) \(installed v2\)", listing):
            raise RuntimeError(f"Query not installed: {name}")
    print(
        json.dumps({"installed": QUERY_NAMES, "vertex_counts": conn.getVertexCount("*")}),
        flush=True,
    )


if __name__ == "__main__":
    # Parse before main() so --help never connects to TigerGraph.
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
