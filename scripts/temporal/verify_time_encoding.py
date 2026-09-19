"""Live integration checks with isolated, temporary synthetic graph records.

Run explicitly after installing the queries. Existing records are never edited.
Only randomly named vertices created here are deleted, including their edges.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import uuid
from typing import Any, cast

from pyTigerGraph import TigerGraphException

from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
BASIS = "log1p_s_400d_32x_sincos_v1"


def reference(delta_ms: int) -> list[float]:
    u = math.log1p(delta_ms / 1000) / math.log1p(34_560_000)
    return [
        value
        for i in range(32)
        for value in (
            math.sin(2 * math.pi * 0.125 * 16 ** (i / 31) * u),
            math.cos(2 * math.pi * 0.125 * 16 ** (i / 31) * u),
        )
    ]


def close(actual: list[float], delta_ms: int) -> None:
    assert len(actual) == 64, len(actual)
    expected = reference(delta_ms)
    assert all(math.isfinite(x) for x in actual)
    assert max(abs(x - y) for x, y in zip(actual, expected, strict=True)) < 1e-5


def main() -> None:
    conn = Client(Settings()).conn
    if conn.graphname != "Mule_Pattern_Learner":
        raise RuntimeError("Unexpected graph")
    before = conn.getVertexCount("*", realtime=True)
    created: list[tuple[str, str]] = []
    prefix = "__time64_test_" + uuid.uuid4().hex + "_"
    base = 1_800_000_000_000
    checks: list[str] = []

    def vertices(kind: str, identifier: str) -> list[dict[str, Any]]:
        try:
            return cast(list[dict[str, Any]], conn.getVerticesById(kind, identifier))
        except TigerGraphException as exc:
            if str(exc.code) == "601":  # REST++ missing primary ID
                return []
            raise

    def create(kind: str, suffix: str, attrs: dict[str, Any]) -> str:
        identifier = prefix + suffix
        assert not vertices(kind, identifier)
        created.append((kind, identifier))
        assert conn.upsertVertex(kind, identifier, attrs) == 1
        return identifier

    def run(name: str, **params: Any) -> list[dict[str, Any]]:
        if "sender" in params:
            # pyTigerGraph uses a one-tuple to serialize VERTEX<Account> in JSON.
            params["sender"] = (params["sender"],)
        return conn.runInstalledQuery(name, params, usePost=True, timeout=120000)

    def summary(result: list[dict[str, Any]]) -> dict[str, Any]:
        return next(row for row in result if "status" in row)

    def event_rows(result: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [row for row in result if "event_id" in row]

    def attrs(kind: str, identifier: str) -> dict[str, Any]:
        return vertices(kind, identifier)[0]["attributes"]

    try:
        for delta in (0, 1, 1000, 180000, 1209600000, 34560000000, 10**13):
            result = summary(run("temporal_fourier64", delta_t_ms=delta))
            assert result["status"] == "ok" and result["basis_id"] == BASIS
            close(result["time_encoding"], delta)
        assert summary(run("temporal_fourier64", delta_t_ms=-1))["status"] == "invalid_delta_t"
        checks.append(
            "64 dimensions, coordinate order, numerical reference, zero/large/negative gaps"
        )

        entity = {"first_seen_seq": 1, "first_seen_ts_ms": base - 1000}
        sender = create("Account", "sender", entity)
        recipient = create("Account", "recipient", entity)
        another = create("Account", "another", entity)
        token = create("Token", "token", {**entity, "token_kind": "handle"})

        def payment(
            suffix: str,
            seq: int,
            timestamp: int,
            *,
            rail: str = "zelle",
            target_type: str = "Account",
            target: str | None = None,
            token_role: bool = False,
        ) -> str:
            kind = "Zelle_Transfer" if rail == "zelle" else "Payment_Transaction"
            role = "Transfer" if rail == "zelle" else "Transaction"
            fields: dict[str, Any] = {
                "event_seq": seq,
                "event_ts_ms": timestamp,
                "amount": 10.0,
                "amount_present": True,
            }
            if rail != "zelle":
                fields["payment_rail"] = rail
            event = create(kind, suffix, fields)
            clocks = {"event_seq": seq, "event_ts_ms": timestamp}
            assert (
                conn.upsertEdge(
                    kind,
                    event,
                    role + "_From_Account",
                    "Account",
                    sender,
                    clocks,
                    vertexMustExist=True,
                )
                == 1
            )
            assert (
                conn.upsertEdge(
                    kind,
                    event,
                    role + "_To_" + target_type,
                    target_type,
                    target or recipient,
                    clocks,
                    vertexMustExist=True,
                )
                == 1
            )
            if token_role:
                assert (
                    conn.upsertEdge(
                        kind,
                        event,
                        role + "_To_Token",
                        "Token",
                        token,
                        clocks,
                        vertexMustExist=True,
                    )
                    == 1
                )
            return event

        first = payment("first", 10, base + 1000, token_role=True)
        second = payment("second", 20, base + 1100)
        payment("same_time", 30, base + 1100)
        payment("future", 40, base + 5000)
        params = dict(
            sender=sender,
            recipient_type="Account",
            recipient_id=recipient,
            seed_seq=40,
            seed_ts_ms=base + 5000,
        )
        result = run("zelle_pair_time64", **params)
        assert summary(result)["event_count"] == 3
        rows = event_rows(result)
        assert [r["pair_delta_t_ms"] for r in rows] == [0, 100, 0]
        assert [r["pair_delta_t_present"] for r in rows] == [False, True, True]
        assert rows[0]["pair_time_encoding"] == []
        for row in rows:
            close(row["age_time_encoding"], base + 5000 - row["event_ts_ms"])
            if row["pair_delta_t_present"]:
                close(row["pair_time_encoding"], row["pair_delta_t_ms"])
        assert attrs("Zelle_Transfer", second)["pair_time_encoding"] == []
        checks.append(
            "strict sequence cutoff, integer 100ms gap at epoch scale, zero-gap mask, read-only default"
        )

        capped = run("zelle_pair_time64", **params, max_events=1, persist=True)
        assert summary(capped)["status"] == "history_limit_exceeded"
        assert attrs("Zelle_Transfer", second)["pair_time_encoding"] == []
        for _ in range(2):
            assert summary(run("zelle_pair_time64", **params, persist=True))["status"] == "ok"
        persisted = attrs("Zelle_Transfer", second)
        close(persisted["pair_time_encoding"], 100)
        assert persisted["pair_previous_event_id"] == first
        assert persisted["pair_sender_id"] == sender
        assert persisted["pair_recipient_id"] == recipient
        assert persisted["time_encoding_basis_id"] == BASIS
        assert attrs("Zelle_Transfer", first)["pair_time_encoding"] == []
        checks.append(
            "history cap prevents writes; persistence round-trip and idempotent list replacement"
        )

        earlier = run("zelle_pair_time64", **{**params, "seed_seq": 20})
        assert summary(earlier)["event_count"] == 1
        later = run("zelle_pair_time64", **{**params, "seed_ts_ms": base + 8000})
        assert event_rows(later)[1]["pair_time_encoding"] == rows[1]["pair_time_encoding"]
        assert event_rows(later)[1]["age_time_encoding"] != rows[1]["age_time_encoding"]
        assert (
            summary(
                run("zelle_pair_time64", **{**params, "seed_seq": 100, "seed_ts_ms": base + 1200})
            )["event_count"]
            == 3
        )
        checks.append(
            "cutoff-dependent age changes; pair-gap encoding stays fixed; timestamp cutoff"
        )

        payment("token_first", 50, base + 10000, target_type="Token", target=token)
        payment("token_second", 60, base + 13000, target_type="Token", target=token)
        token_result = run(
            "zelle_pair_time64",
            sender=sender,
            recipient_type="Token",
            recipient_id=token,
            seed_seq=100,
            seed_ts_ms=base + 20000,
            persist=True,
        )
        assert summary(token_result)["event_count"] == 2
        close(event_rows(token_result)[1]["pair_time_encoding"], 3000)
        checks.append(
            "unresolved token recipient; observed account takes precedence over routing token"
        )

        payment("ach_first", 70, base + 14000, rail="ach")
        payment("wire", 75, base + 14500, rail="wire")
        last_ach = payment("ach_second", 80, base + 16000, rail="ach")
        ach = run(
            "payment_pair_time64",
            sender=sender,
            recipient_type="Account",
            recipient_id=recipient,
            seed_seq=100,
            seed_ts_ms=base + 20000,
            payment_rail="ach",
            persist=True,
        )
        assert summary(ach)["event_count"] == 2
        close(attrs("Payment_Transaction", last_ach)["pair_time_encoding"], 2000)
        checks.append("non-Zelle encoding and separate payment rails")

        # Windows are inclusive at their lower timestamp boundary, while the
        # seed event itself is excluded by its authoritative sequence number.
        window_cutoff = base + 8 * 86400000
        ages = (604800001, 604800000, 86400001, 86400000, 3600001, 3600000, 0)
        for rail in ("zelle", "ach"):
            for index, age in enumerate(ages):
                payment(
                    f"window_{rail}_{index}",
                    100 + index,
                    window_cutoff - age,
                    rail=rail,
                    target=another,
                )
            window_params: dict[str, Any] = dict(
                sender=sender,
                recipient_type="Account",
                recipient_id=another,
                seed_seq=1000,
                seed_ts_ms=window_cutoff,
            )
            query = "zelle_pair_time64" if rail == "zelle" else "payment_pair_time64"
            if rail != "zelle":
                window_params["payment_rail"] = rail
            window = summary(run(query, **window_params))
            assert window["status"] == "ok" and window["event_count"] == 7
            assert [window[key] for key in ("count_1h", "count_24h", "count_7d")] == [2, 4, 6]
        checks.append("exact 1h/24h/7d count boundaries for Zelle and non-Zelle payments")

        backfill = payment("backfill", 15, base + 1050)
        result = run("zelle_pair_time64", **params, persist=True)
        assert summary(result)["event_count"] == 4
        assert attrs("Zelle_Transfer", second)["pair_previous_event_id"] == backfill
        close(attrs("Zelle_Transfer", second)["pair_time_encoding"], 50)
        checks.append("late event recomputes predecessor and gap")

        invalid = payment("out_of_order", 25, base + 900)
        assert (
            summary(run("zelle_pair_time64", **params, persist=True))["status"]
            == "invalid_event_order"
        )
        close(attrs("Zelle_Transfer", second)["pair_time_encoding"], 50)
        conn.delVerticesById("Zelle_Transfer", invalid)
        created.remove(("Zelle_Transfer", invalid))
        dup = payment("duplicate_seq", 20, base + 1100)
        assert (
            summary(run("zelle_pair_time64", **params, persist=True))["status"]
            == "invalid_event_order"
        )
        conn.delVerticesById("Zelle_Transfer", dup)
        created.remove(("Zelle_Transfer", dup))
        conn.upsertEdge(
            "Zelle_Transfer",
            second,
            "Transfer_To_Account",
            "Account",
            another,
            {"event_seq": 20, "event_ts_ms": base + 1100},
            vertexMustExist=True,
        )
        assert (
            summary(run("zelle_pair_time64", **params, persist=True))["status"]
            == "invalid_event_roles_or_clocks"
        )
        checks.append(
            "nonmonotonic timestamps, duplicate sequence and ambiguous recipients reject before writes"
        )
        print(json.dumps({"checks_passed": checks}), flush=True)
    finally:
        # Delete only this run's new random IDs; vertex deletion removes fixture edges.
        for kind, identifier in reversed(created):
            conn.delVerticesById(kind, identifier)
        for kind, identifier in created:
            assert not vertices(kind, identifier), "Fixture cleanup incomplete"

    after = conn.getVertexCount("*", realtime=True)
    assert after == before, "Counts changed during verification; inspect concurrent activity"
    files = list((ROOT / "gsql/features").glob("*time64.gsql"))
    files += [
        ROOT / "gsql/features/temporal_fourier64.gsql",
        ROOT / "gsql/schema/migrations/temporal_encoding_attributes.gsql",
    ]
    evidence = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "graph": conn.graphname,
        "basis_id": BASIS,
        "dimensions": 64,
        "checks_passed": checks,
        "temporary_fixture_cleanup": "verified",
        "vertex_counts_before": before,
        "vertex_counts_after": after,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    (ROOT / "docs/temporal_encoding_deployment.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    print("Verified: all fixture records removed; original vertex counts unchanged.", flush=True)


if __name__ == "__main__":
    main()
