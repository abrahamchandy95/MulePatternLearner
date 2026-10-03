"""Held-out and future data never change training inputs, on the graph itself.

This WRITES to TigerGraph, so it runs only with --allow-graph-writes: it upserts
UUID-prefixed fixture vertices and edges, then deletes exactly those. Existing business
data and labels are never modified, but vertex counts change while it runs, so never
run it during a training run (verify_frozen_source would see the drift). Run it after
the training queries are installed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import replace
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pyTigerGraph.common.exception import TigerGraphException
import torch

from mule_pattern_learner.batching.assemble import build_batch
from mule_pattern_learner.config import DEFAULT_CONFIG, RunConfig
from mule_pattern_learner.contract.feature_groups import (
    FeaturePlan,
    contract_fingerprint,
    extraction_plan,
)
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.contract.server import GRAPH_NAME, SCOPE_VERTEX
from mule_pattern_learner.contract.time_basis import BASIS_ID
from mule_pattern_learner.data.contexts import ContextSource
from mule_pattern_learner.inference.predictor import Predictor
from mule_pattern_learner.inference.saved_model import SavedModel
from mule_pattern_learner.model.build import build_model
from mule_pattern_learner.runtime.device import choose_device
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher
from mule_pattern_learner.tigergraph.executor import TigerGraphExecutor

pytestmark = pytest.mark.graph_write

# The fixture's clock: event n happens at BASE + 100 n milliseconds.
BASE = 1710000000000


def model_config() -> RunConfig:
    """A small built-in model with the default pools, which the source uses too."""
    small = {"model": {"hidden": 16, "heads": 4, "dropout": 0.0}, "training": {"batch_size": 16}}
    return replace(DEFAULT_CONFIG.with_changes(small), sampler=SamplerPlan(fanouts=(2, 2)))


def source_plan(model: FeaturePlan) -> FeaturePlan:
    """What the test's source requests: the model's inputs, which its checks read."""
    return extraction_plan(model)


class TemporaryVertices:
    """Vertices and edges whose ids share a fresh UUID prefix; remove() deletes only them."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn
        self.prefix = "fixture_" + uuid4().hex + "_"
        self.created: dict[str, list[str]] = defaultdict(list)

    def exists(self, kind: str, key: str) -> bool:
        try:
            rows = self.conn.getVerticesById(kind, [key])
        except TigerGraphException as error:
            if str(error.code) == "601":
                return False
            raise
        if not isinstance(rows, list):
            raise ValueError("Unexpected vertex response")
        return bool(rows)

    def put(self, kind: str, name: str, attrs: dict[str, Any]) -> str:
        key = self.prefix + name
        if self.exists(kind, key):
            raise ValueError("Fixture ID collision")
        self.created[kind].append(key)
        self.conn.upsertVertex(kind, key, attrs)
        return key

    def link(
        self, a_type: str, a: str, edge: str, b_type: str, b: str, attrs: dict[str, Any]
    ) -> None:
        assert a.startswith(self.prefix) and b.startswith(self.prefix)
        if "valid_from_seq" in attrs:
            query = (
                f"USE GRAPH {GRAPH_NAME}\nINTERPRET QUERY () FOR GRAPH {GRAPH_NAME} {{ "
                f"INSERT INTO {edge} VALUES ({json.dumps(a)} {a_type}, "
                f"{json.dumps(b)} {b_type}, {attrs['valid_from_seq']}, "
                f'{attrs["valid_to_seq"]}, 1.0, ""); PRINT "ok" AS status; }}'
            )
            result = str(self.conn.gsql(query))
            if '"ok"' not in result or "Error" in result:
                raise RuntimeError(result)
        else:
            self.conn.upsertEdge(a_type, a, edge, b_type, b, attrs)

    def payment(
        self,
        name: str,
        seq: int,
        sender: str,
        recipient: str,
        kind: str = "Zelle_Transfer",
        target: str = "Account",
    ) -> str:
        attrs: dict[str, Any] = {
            "event_seq": seq,
            "event_ts_ms": BASE + seq * 100,
            "amount": 10.0,
            "amount_present": True,
            "currency": "USD",
        }
        if kind == "Payment_Transaction":
            attrs["payment_rail"] = "ach"
        event = self.put(kind, name, attrs)
        stem = "Transfer" if kind == "Zelle_Transfer" else "Transaction"
        clock = {"event_seq": seq, "event_ts_ms": attrs["event_ts_ms"]}
        self.link(kind, event, stem + "_From_Account", "Account", sender, clock)
        self.link(kind, event, stem + "_To_" + target, target, recipient, clock)
        return event

    def remove(self) -> None:
        for kind, ids in reversed(list(self.created.items())):
            assert all(key.startswith(self.prefix) for key in ids)
            self.conn.delVerticesById(kind, ids)
        for kind, ids in self.created.items():
            assert not any(self.exists(kind, key) for key in ids), "Fixture cleanup incomplete"


@pytest.fixture
def vertices(graph: TigerGraphExecutor) -> Iterator[TemporaryVertices]:
    fixture = TemporaryVertices(graph.client.conn)
    try:
        yield fixture
    finally:
        fixture.remove()


def test_held_out_and_future_data_never_change_training_inputs(
    graph: TigerGraphExecutor, vertices: TemporaryVertices, tmp_path: Path
) -> None:
    config = model_config()
    plan, sampler = config.feature_plan(), config.sampler
    put, link, payment = vertices.put, vertices.link, vertices.payment
    scope = vertices.prefix + "scope"
    association = {"valid_from_seq": 1, "valid_to_seq": 0}
    # The predictor below reads from this source, so it requests the model's inputs too.
    with ContextSource(
        TigerGraphContextFetcher(graph), plan=source_plan(plan), sampler=sampler, capacity=0
    ) as source:

        def fetch_all(keys: list[ContextKey]) -> list[dict[str, Any]]:
            rows = source.fetch(keys)
            missing = [key for key, row in zip(keys, rows, strict=True) if row is None]
            assert not missing, f"Fixture contexts rejected {dict(source.rejections)}: {missing}"
            return [row for row in rows if row is not None]

        put(SCOPE_VERTEX, "scope", {"source_id": vertices.prefix, "split_seed": 42, "ready": True})
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
            attrs: dict[str, Any] = {"first_seen_seq": 1, "first_seen_ts_ms": BASE}
            if kind == "Account":
                attrs.update(account_type="deposit", is_external=False)
            entities[name] = put(kind, name, attrs)
            if kind in ("Account", "Party"):
                link(
                    kind,
                    entities[name],
                    "Entity_In_Training_Scope",
                    SCOPE_VERTEX,
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
            ContextKey(kind, entities[name], 1000, BASE + 100000, scope, 1)
            for kind, name in [("Account", "a"), ("Token", "token"), ("Device", "device")]
        ]
        baseline = fetch_all(keys)
        # A's five payments, of 10.0 each, and its deposit flag; the device has neither.
        events = [m for m in baseline[0]["messages"] if m["event_id"]]
        assert len(events) == 5 and sum(m["amount"] for m in events) == 50
        assert baseline[0]["features"] == {"is_deposit": 1}
        assert baseline[2]["features"] == {}
        for name in ("z2", "p2"):
            event = next(m for m in events if m["event_id"] == vertices.prefix + name)
            assert event["gap_present"] and event["gap_ms"] == 1000
            assert event["pair_prior_count"] == 1 and event["pair_first_present"]
        # Held-out events and associations leave the training inputs as they were.
        hidden = payment("heldout_in", 400, b, a)
        payment("heldout_out", 410, a, b, "Payment_Transaction")
        payment("heldout_token", 420, b, entities["token"], target="Token")
        link("Party", entities["q"], "Party_Uses_Device", "Device", entities["device"], association)
        link("Party", entities["q"], "Party_Uses_Token", "Token", entities["token"], association)
        link("Token", entities["token"], "Token_Bound_To_Account", "Account", b, association)
        assert fetch_all(keys) == baseline, "Held-out events/associations changed training inputs"
        # So do a hidden mutation, a future event and future association changes.
        graph.client.conn.upsertVertex("Zelle_Transfer", hidden, {"amount": 999999.0})
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
        assert shared[0]["messages"] != baseline[0]["messages"], (
            "Fixture did not exercise exclusion"
        )
        # A held-out root is a per-request rejection: None, counted by status.
        rejected = source.fetch([ContextKey("Account", b, 1000, BASE + 100000, scope, 1)])
        assert rejected == [None] and source.rejections["invisible_entity"], (
            "Held-out account accepted as a training root"
        )
        # A real accelerator update, then inductive prediction for B with unchanged weights.
        device = choose_device()
        model = build_model(config.model, plan, sampler.fanouts[0]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        batch = build_batch(source, keys[:1], device=device, plan=plan, sampler=sampler)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(batch), torch.ones(1, device=device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        saved = tmp_path / "model.pt"
        torch.save(
            {
                "format": SavedModel.FORMAT,
                "state_dict": {n: v.cpu() for n, v in model.state_dict().items()},
                "config": config.to_dict(),
                "input_fingerprint": plan.fingerprint(),
                "contract": contract_fingerprint(),
                "basis_id": BASIS_ID,
                "threshold": 0.5,
            },
            saved,
        )
        predictor = Predictor(saved, source)
        (result,), _ = predictor.score_keys([[ContextKey("Account", b, 1000, BASE + 100000)]])
        assert len(result) == 1 and 0 <= result.score.iloc[0] <= 1
        arrival = put(
            "Account",
            "arrived_after_training",
            {
                "first_seen_seq": 1500,
                "first_seen_ts_ms": BASE + 150000,
                "account_type": "deposit",
                "is_external": False,
            },
        )
        (fresh,), _ = predictor.score_keys([[ContextKey("Account", arrival, 1600, BASE + 160000)]])
        assert len(fresh) == 1 and 0 <= fresh.score.iloc[0] <= 1
