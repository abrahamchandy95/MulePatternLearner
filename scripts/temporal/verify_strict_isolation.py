"""Live integration test using isolated temporary vertices, cleaned up afterwards.

This WRITES to TigerGraph: it upserts UUID-prefixed fixture vertices and edges,
then deletes exactly those. Existing business data and labels are never
modified, but vertex counts change while it runs, so never run it during a
streamed training run (verify_frozen_source would see the drift). It requires
--write-fixture. Run after installing the scoped training queries.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from uuid import uuid4
from typing import Any
from pyTigerGraph.common.exception import TigerGraphException

import torch

from mule_pattern_learner.temporal.encoding import BASIS_ID
from mule_pattern_learner.temporal.live.batching import make_live_batch
from mule_pattern_learner.temporal.live.contract import ContextKey, contract_fingerprint
from mule_pattern_learner.temporal.live.model import LiveTGAT
from mule_pattern_learner.temporal.live.predictor import TemporalPredictor
from mule_pattern_learner.temporal.live.source import TigerGraphExecutor, StreamingContextSource
from mule_pattern_learner.device import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--write-fixture",
        action="store_true",
        help="Confirm that temporary fixture vertices may be written to and deleted from TigerGraph",
    )
    if not parser.parse_args().write_fixture:
        parser.error("this live test writes temporary vertices; pass --write-fixture to run it")
    executor = TigerGraphExecutor()
    conn = executor.client.conn
    prefix = "temporal_fixture_" + uuid4().hex + "_"
    scope = prefix + "scope"
    base = 1710000000000
    created: dict[str, list[str]] = defaultdict(list)

    def exists(kind: str, key: str) -> bool:
        try:
            rows = conn.getVerticesById(kind, [key])
        except TigerGraphException as error:
            if str(error.code) == "601":
                return False
            raise
        if not isinstance(rows, list):
            raise ValueError("Unexpected vertex response")
        return bool(rows)

    def put(kind: str, name: str, attrs: dict[str, Any]) -> str:
        key = prefix + name
        assert key.startswith(prefix)
        if exists(kind, key):
            raise ValueError("Fixture ID collision")
        created[kind].append(key)
        conn.upsertVertex(kind, key, attrs)
        return key

    def link(a_type: str, a: str, edge: str, b_type: str, b: str, attrs: dict[str, Any]) -> None:
        assert a.startswith(prefix) and b.startswith(prefix)
        if "valid_from_seq" in attrs:
            query = (
                "USE GRAPH Mule_Pattern_Learner\nINTERPRET QUERY () FOR GRAPH Mule_Pattern_Learner { "
                f"INSERT INTO {edge} VALUES ({json.dumps(a)} {a_type}, "
                f"{json.dumps(b)} {b_type}, {attrs['valid_from_seq']}, "
                f'{attrs["valid_to_seq"]}, 1.0, ""); PRINT "ok" AS status; }}'
            )
            result = str(conn.gsql(query))
            if '"ok"' not in result or "Error" in result:
                raise RuntimeError(result)
        else:
            conn.upsertEdge(a_type, a, edge, b_type, b, attrs)

    def payment(
        name: str,
        seq: int,
        sender: str,
        recipient: str,
        kind: str = "Zelle_Transfer",
        target: str = "Account",
    ) -> str:
        attrs = {
            "event_seq": seq,
            "event_ts_ms": base + seq * 100,
            "amount": 10.0,
            "amount_present": True,
            "currency": "USD",
        }
        if kind == "Payment_Transaction":
            attrs["payment_rail"] = "ach"
        event = put(kind, name, attrs)
        stem = "Transfer" if kind == "Zelle_Transfer" else "Transaction"
        clock = {"event_seq": seq, "event_ts_ms": attrs["event_ts_ms"]}
        link(kind, event, stem + "_From_Account", "Account", sender, clock)
        link(kind, event, stem + "_To_" + target, target, recipient, clock)
        return event

    source = StreamingContextSource(executor, capacity=0)

    def fetch_all(keys: list[ContextKey]) -> list[dict[str, Any]]:
        rows = source.fetch(keys)
        missing = [key for key, row in zip(keys, rows, strict=True) if row is None]
        if missing:
            raise AssertionError(f"Fixture contexts rejected {dict(source.rejections)}: {missing}")
        return [row for row in rows if row is not None]

    association = {"valid_from_seq": 1, "valid_to_seq": 0}
    try:
        put(
            "Temporal_Training_Scope",
            "scope",
            {"source_id": prefix, "split_seed": 42, "ready": True},
        )
        entities = {}
        for kind, name in [
            ("Account", "a"),
            ("Account", "b"),
            ("Account", "c"),
            ("Party", "p"),
            ("Party", "q"),
            ("Token", "token"),
            ("Device", "device"),
        ]:
            attrs: dict[str, Any] = {"first_seen_seq": 1, "first_seen_ts_ms": base}
            if kind == "Account":
                attrs.update(account_type="deposit", is_external=False)
            entities[name] = put(kind, name, attrs)
            if kind in ("Account", "Party"):
                link(
                    kind,
                    entities[name],
                    "Entity_In_Training_Scope",
                    "Temporal_Training_Scope",
                    scope,
                    {"partition": 3 if name in ("b", "q") else 1, "group_id": name},
                )
        a, b, c = (entities[n] for n in ("a", "b", "c"))
        link("Party", entities["p"], "Party_Owns_Account", "Account", a, association)
        link("Party", entities["q"], "Party_Owns_Account", "Account", b, association)
        link("Party", entities["p"], "Party_Uses_Device", "Device", entities["device"], association)
        link("Account", a, "Account_Uses_Device", "Device", entities["device"], association)
        payment("z1", 10, a, c)
        payment("z2", 20, a, c)
        payment("p1", 30, a, c, "Payment_Transaction")
        payment("p2", 40, a, c, "Payment_Transaction")
        payment("t1", 50, a, entities["token"], target="Token")
        keys = [
            ContextKey(kind, entities[name], 1000, base + 100000, scope, 1)
            for kind, name in [("Account", "a"), ("Token", "token"), ("Device", "device")]
        ]
        baseline = fetch_all(keys)
        assert baseline[0]["features"]["1h_out_count"] == 5
        assert baseline[0]["features"]["1h_out_amount"] == 50
        for window in ("1d", "7d"):
            assert baseline[0]["features"][window + "_out_in_amount_ratio"] == 50
            assert baseline[2]["features"][window + "_out_in_amount_ratio"] == 0
        for name in ("z2", "p2"):
            event = next(m for m in baseline[0]["messages"] if m["event_id"] == prefix + name)
            assert event["gap_present"] and event["gap_ms"] == 1000
            assert all(
                event[field] == 1 for field in ("pair_count_1h", "pair_count_1d", "pair_count_7d")
            )
        hidden = payment("heldout_in", 400, b, a)
        payment("heldout_out", 410, a, b, "Payment_Transaction")
        payment("heldout_token", 420, b, entities["token"], target="Token")
        link("Party", entities["q"], "Party_Uses_Device", "Device", entities["device"], association)
        link("Party", entities["q"], "Party_Uses_Token", "Token", entities["token"], association)
        link("Token", entities["token"], "Token_Bound_To_Account", "Account", b, association)
        assert fetch_all(keys) == baseline, "Held-out events/associations changed training inputs"
        conn.upsertVertex("Zelle_Transfer", hidden, {"amount": 999999.0})
        payment("future", 1100, a, c)
        link(
            "Party",
            entities["p"],
            "Party_Uses_Token",
            "Token",
            entities["token"],
            {"valid_from_seq": 1100, "valid_to_seq": 0},
        )
        link(
            "Party",
            entities["p"],
            "Party_Uses_Device",
            "Device",
            entities["device"],
            {"valid_from_seq": 1, "valid_to_seq": 1200},
        )
        assert fetch_all(keys) == baseline, (
            "Hidden mutation/future event or association changed training inputs"
        )
        shared = fetch_all([replace(key, scope_id="", visibility_phase=3) for key in keys])
        assert shared[0]["features"] != baseline[0]["features"], (
            "Fixture did not exercise exclusion"
        )
        # A held-out root is a per-request rejection: None, counted by status.
        rejected = source.fetch([ContextKey("Account", b, 1000, base + 100000, scope, 1)])
        if rejected != [None] or not source.rejections["invisible_entity"]:
            raise AssertionError("Held-out account accepted as a training root")
        # Real accelerator update, then inductive prediction for B with unchanged weights.
        device = choose_device()
        model = LiveTGAT(16, 4, 0).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        batch = make_live_batch(source, keys[:1], fanouts=(2, 2), device=device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(batch), torch.ones(1, device=device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory(prefix="temporal_model_check_") as temp:
            checkpoint = Path(temp) / "model.pt"
            torch.save(
                {
                    "state_dict": {n: v.cpu() for n, v in model.state_dict().items()},
                    "config": {
                        "hidden": 16,
                        "heads": 4,
                        "dropout": 0.0,
                        "fanouts": [2, 2],
                        "batch_size": 16,
                    },
                    "contract": contract_fingerprint(),
                    "basis_id": BASIS_ID,
                    "threshold": 0.5,
                },
                checkpoint,
            )
            predictor = TemporalPredictor(checkpoint, source)
            result = predictor.predict([ContextKey("Account", b, 1000, base + 100000)])
            assert len(result) == 1 and 0 <= result.score.iloc[0] <= 1
            arrival = put(
                "Account",
                "arrived_after_training",
                {
                    "first_seen_seq": 1500,
                    "first_seen_ts_ms": base + 150000,
                    "account_type": "deposit",
                    "is_external": False,
                },
            )
            fresh = predictor.predict([ContextKey("Account", arrival, 1600, base + 160000)])
            assert len(fresh) == 1 and 0 <= fresh.score.iloc[0] <= 1
        print(
            json.dumps(
                {
                    "strict_heldout_invariance": "passed",
                    "future_event_invariance": "passed",
                    "future_valid_time_association_invariance": "passed",
                    "scoped_pair_history_parity": "passed",
                    "new_account_prediction": "passed",
                    "accelerator": str(device),
                    "feature_contexts_checked": len(keys),
                    "loss": float(loss.detach().cpu()),
                }
            ),
            flush=True,
        )
    finally:
        source.close()
        for kind, ids in reversed(list(created.items())):
            assert all(key.startswith(prefix) for key in ids)
            conn.delVerticesById(kind, ids)
        for kind, ids in created.items():
            assert all(not exists(kind, key) for key in ids), "Fixture cleanup incomplete"
        print("Temporary fixture removed; existing business vertices were untouched.", flush=True)


if __name__ == "__main__":
    main()
