"""Verify live account loader/masking with isolated, cleaned-up synthetic records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import csv
import json
from pathlib import Path
from typing import Any, cast
import uuid

from pyTigerGraph import TigerGraphException

from mule_pattern_learner.temporal.live.labels import ACCOUNT_LOAD_COLUMNS
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings

ROOT = Path(__file__).resolve().parents[2]


def main(output: Path) -> None:
    print("Connecting for account supervision verification", flush=True)
    conn = Client(Settings()).conn
    print("Connected", flush=True)
    if conn.graphname != "Mule_Pattern_Learner":
        raise ValueError("Unexpected graph")
    before = conn.getVertexCount("*", realtime=True)
    baseline = conn.runInstalledQuery("temporal_validate_account_supervision", usePost=True)[0]
    prefix = "zz__account_supervision_test_" + uuid.uuid4().hex + "_"
    identifiers = [prefix + name for name in ("hidden", "revealed", "negative", "unknown")]
    base = 1_800_000_000_000

    def attributes(identifier: str) -> dict[str, Any]:
        result = cast(list[dict[str, Any]], conn.getVerticesById("Account", identifier))
        return cast(dict[str, Any], result[0]["attributes"])

    for identifier in identifiers:
        try:
            if cast(list[dict[str, Any]], conn.getVerticesById("Account", identifier)):
                raise RuntimeError("Fixture ID collision")
        except TigerGraphException as exc:
            if str(exc.code) != "601":
                raise
    rows = [
        [
            identifiers[0],
            "deposit",
            "false",
            1,
            base,
            1,
            "true",
            "true",
            0,
            2,
            base + 1000,
            3,
            base + 2000,
            0,
            "schema_fixture",
        ],
        [
            identifiers[1],
            "deposit",
            "false",
            1,
            base,
            1,
            "true",
            "false",
            1,
            2,
            base + 1000,
            3,
            base + 2000,
            1,
            "schema_fixture",
        ],
        [
            identifiers[2],
            "deposit",
            "false",
            1,
            base,
            0,
            "true",
            "true",
            0,
            1,
            base,
            1,
            base,
            -1,
            "schema_fixture",
        ],
        [
            identifiers[3],
            "deposit",
            "true",
            1,
            base,
            0,
            "false",
            "true",
            0,
            0,
            0,
            0,
            0,
            -1,
            "",
        ],
    ]
    checks = []
    try:
        print("Loading four isolated test accounts", flush=True)
        stream = io.StringIO()
        csv.writer(stream, lineterminator="\n").writerows(rows)
        # REST++ streaming loads require data rows without a CSV header, even
        # though server-file loading through this job uses HEADER=true.
        loaded = conn.runLoadingJobWithData(
            stream.getvalue(), "accounts", "load_temporal_accounts", sep=","
        )
        print("Loader response: " + json.dumps(loaded), flush=True)
        if isinstance(loaded, dict) and loaded.get("error"):
            raise RuntimeError("Fixture loading failed")
        for identifier, row in zip(identifiers, rows, strict=True):
            attrs = attributes(identifier)
            expected = dict(zip(ACCOUNT_LOAD_COLUMNS, row, strict=True))
            for name in ("is_external", "mule_label_known", "is_mule_masked"):
                expected[name] = expected[name] == "true"
            for name, value in expected.items():
                assert attrs[name] == value, f"Loader mapping mismatch: {name}"
            assert type(attrs["is_mule"]) is int
            assert type(attrs["is_mule_masked"]) is bool
        checks.append("is_mule stores integer 0/1; is_mule_masked remains boolean")
        hidden = attributes(identifiers[0])
        assert hidden["is_mule"] and hidden["is_mule_masked"] and hidden["pu_label"] == 0
        assert hidden["mule_ring_id"] == 0
        revealed = attributes(identifiers[1])
        assert revealed["is_mule"] and not revealed["is_mule_masked"] and revealed["pu_label"] == 1
        assert attributes(identifiers[2])["mule_label_known"]
        assert not attributes(identifiers[3])["mule_label_known"]
        check = conn.runInstalledQuery("temporal_validate_account_supervision", usePost=True)[0]
        for key in (
            "invalid_mule",
            "invalid_pu",
            "invalid_unknown",
            "invalid_clocks",
            "invalid_ring",
        ):
            assert check[key] == baseline[key]
        assert check["true_mules"] == baseline["true_mules"] + 2
        assert check["masked_mules"] == baseline["masked_mules"] + 1
        assert check["revealed_positives"] == baseline["revealed_positives"] + 1
        checks.append(
            "All fifteen Account columns load correctly; hidden truth, known negatives, unknowns and ring zero remain distinct"
        )
        conn.upsertVertex("Account", identifiers[1], {"is_mule_masked": True, "pu_label": 0})
        after_mask = attributes(identifiers[1])
        assert after_mask["is_mule"] and after_mask["mule_label_known"]
        assert after_mask["is_mule_masked"] and after_mask["pu_label"] == 0
        checks.append("Masking a revealed mule preserves its true label")
        conn.upsertVertex("Account", identifiers[0], {"pu_label": 1})
        invalid = conn.runInstalledQuery("temporal_validate_account_supervision", usePost=True)[0]
        assert invalid["invalid_pu"] == baseline["invalid_pu"] + 1
        conn.upsertVertex("Account", identifiers[0], {"pu_label": 0})
        checks.append("Inconsistent mask/PU state is detected")
        conn.upsertVertex("Account", identifiers[0], {"is_mule": 2})
        invalid = conn.runInstalledQuery("temporal_validate_account_supervision", usePost=True)[0]
        assert invalid["invalid_mule"] == baseline["invalid_mule"] + 1
        conn.upsertVertex("Account", identifiers[0], {"is_mule": 1})
        checks.append("Invalid integer mule labels outside 0/1 are detected")
        page = conn.runInstalledQuery(
            "temporal_get_account_supervision",
            {"after_id": prefix, "batch_size": 100},
            usePost=True,
        )
        exported = [
            item
            for block in page
            for item in block.get("accounts", [])
            if item["v_id"] in identifiers
        ]
        assert len(exported) == 4
        checks.append("Paginated supervision export includes truth and mask fields")
    except BaseException as exc:
        print(f"Verification failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        print("Removing isolated test accounts", flush=True)
        for identifier in identifiers:
            conn.delVerticesById("Account", identifier)
    # Default statistics can lag deletes by 30 seconds; verify the actual graph.
    after_counts = conn.getVertexCount("*", realtime=True)
    if after_counts != before:
        raise RuntimeError(f"Vertex counts differ: before={before}, after={after_counts}")
    final = conn.runInstalledQuery("temporal_validate_account_supervision", usePost=True)[0]
    if final != baseline:
        raise RuntimeError(f"Validation differs: before={baseline}, after={final}")
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "checks": checks,
        "vertex_counts_preserved": before,
        "temporary_accounts_removed": 4,
        "final_validation": final,
        "production_account_labels_modified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    # Parse before main() so --help never connects to TigerGraph.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/temporal/reports/temporal_account_supervision_tests.json",
        help="Where the test record is written",
    )
    args = parser.parse_args()
    try:
        main(args.output)
    except BaseException as exc:
        import traceback

        print(f"Account verification error: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
