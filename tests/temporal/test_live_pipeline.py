from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import re

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.temporal.encoding import BASIS_ID, fourier64
from mule_pattern_learner.temporal.live.batching import child_key, make_live_batch, node_features
from mule_pattern_learner.temporal.live.contract import ContextKey, FEATURE_NAMES
from mule_pattern_learner.temporal.live.dataset import (
    assign_groups,
    validate_dates,
)
from mule_pattern_learner.temporal.live.model import LiveTGAT
from mule_pattern_learner.temporal.live.queries import render_context_query
from mule_pattern_learner.temporal.live.source import ContextStore, validate_context


def context(key: ContextKey, messages: list[dict] | None = None) -> dict:
    messages = messages or []
    row = {
        **asdict(key),
        "status": "ok",
        "basis_id": BASIS_ID,
        "features": {
            "is_deposit": 1,
            "age_days": 1,
            "1d_out_count": 2,
            "1d_out_in_amount_ratio": 0,
            "7d_out_in_amount_ratio": 0,
        },
        "messages": messages,
        "age_encoding": {},
        "gap_encoding": {},
    }
    for message in messages:
        if message["event_id"]:
            name = message["relation"] + ":" + message["event_id"]
            row["age_encoding"][name] = fourier64(np.array([message["age_ms"]]))[0].tolist()
            if message["gap_present"]:
                row["gap_encoding"][name] = fourier64(np.array([message["gap_ms"]]))[0].tolist()
    return row


def message(seq: int, ts: int, parent: ContextKey) -> dict:
    return {
        "node_type": "Account",
        "node_id": "neighbor",
        "relation": "zelle_out",
        "rail": "zelle",
        "event_id": "E" + str(seq),
        "event_seq": seq,
        "event_ts_ms": ts,
        "amount": 20,
        "amount_present": True,
        "age_ms": parent.cutoff_ms - ts,
        "gap_ms": 0,
        "gap_present": True,
        "pair_count_1h": 1,
        "pair_count_1d": 1,
        "pair_count_7d": 1,
        "peer_first_ms": 1,
        "peer_external": False,
        "peer_deposit": True,
    }


class FakeExecutor:
    def __init__(self, rows: dict[ContextKey, dict]) -> None:
        self.rows = rows
        self.requested: list[ContextKey] = []

    def run(self, name: str, params: dict) -> list[dict]:
        assert name == "temporal_training_context"
        assert len(params["node_ids"]) <= 16
        keys = [
            ContextKey(*args, params.get("scope_id", ""), params.get("visibility_phase", 3))
            for args in zip(
                params["node_types"],
                params["node_ids"],
                params["cutoff_seqs"],
                params["cutoff_times"],
                strict=True,
            )
        ]
        self.requested.extend(keys)
        return [
            {**deepcopy(self.rows.get(key, context(key))), "request_index": i}
            for i, key in enumerate(keys)
        ]


def test_recursive_context_keeps_same_neighbor_at_two_different_event_times(tmp_path: Path) -> None:
    root = ContextKey("Account", "root", 100, 1000)
    messages = [message(90, 900, root), message(80, 800, root)]
    source = FakeExecutor({root: context(root, messages)})
    store = ContextStore(tmp_path / "cache.sqlite", {"dataset": "one"}, source)
    batch = make_live_batch(store, [root], fanouts=(2, 2))
    assert child_key(messages[0]) in source.requested
    assert child_key(messages[1]) in source.requested
    assert len(set(batch["neighbor_positions"][0].tolist())) == 2
    model = LiveTGAT(hidden=16, heads=4, dropout=0)
    logits = model(batch)
    logits.sum().backward()
    assert torch.isfinite(logits).all()
    assert model.edge.weight.grad is not None
    assert torch.count_nonzero(model.edge.weight.grad[:, 7:]) > 0
    # A real zero pair gap has cosine coordinates; it is not missing time.
    assert torch.all(batch["first_edge"][0, 0, 72::2] == 1)
    store.close()
    offline = ContextStore(tmp_path / "cache.sqlite", {"dataset": "one"})
    again = make_live_batch(offline, [root], fanouts=(2, 2))
    for key in batch:
        torch.testing.assert_close(batch[key], again[key])
    assert offline.query_calls == 0
    offline.close()


def test_future_and_same_event_are_rejected() -> None:
    key = ContextKey("Account", "root", 100, 1000)
    for seq, ts in ((100, 1000), (101, 900), (90, 1001)):
        row = context(key, [message(seq, min(ts, 1000), key)])
        row["messages"][0]["event_ts_ms"] = ts
        with pytest.raises(ValueError):
            validate_context(key, row)


def test_basis_and_clock_corruption_are_rejected() -> None:
    key = ContextKey("Account", "root", 100, 1000)
    row = context(key, [message(90, 900, key)])
    bad = deepcopy(row)
    bad["age_encoding"]["zelle_out:E90"][0] += 0.1
    with pytest.raises(ValueError, match="encoding"):
        validate_context(key, bad)
    bad = deepcopy(row)
    bad["cutoff_seq"] = 101
    with pytest.raises(ValueError, match="differs"):
        validate_context(key, bad)
    bad = deepcopy(row)
    bad["messages"][0]["gap_present"] = False
    with pytest.raises(ValueError, match="Missing predecessor"):
        validate_context(key, bad)


def test_labels_cannot_enter_node_features() -> None:
    row = context(ContextKey("Account", "root", 100, 1000))
    assert node_features(row).shape == (len(FEATURE_NAMES),)
    row["features"]["is_mule"] = 1
    with pytest.raises(ValueError, match="Unrecognized"):
        node_features(row)


def test_amount_ratios_are_required_from_gsql_and_preserved_by_tensor_conversion() -> None:
    key = ContextKey("Account", "root", 100, 1000)
    row = context(key)
    row["features"].update({"1d_out_in_amount_ratio": 2.5, "7d_out_in_amount_ratio": 100.0})
    validate_context(key, row)
    features = node_features(row)
    assert features[FEATURE_NAMES.index("1d_out_in_amount_ratio")] == pytest.approx(np.log1p(2.5))
    assert features[FEATURE_NAMES.index("7d_out_in_amount_ratio")] == pytest.approx(np.log1p(100.0))
    del row["features"]["1d_out_in_amount_ratio"]
    with pytest.raises(ValueError, match="missing amount ratios"):
        validate_context(key, row)


def test_cache_provenance_offline_miss_and_batch_bound(tmp_path: Path) -> None:
    keys = [ContextKey("Account", str(i), 100, 1000) for i in range(35)]
    source = FakeExecutor({})
    store = ContextStore(tmp_path / "cache.sqlite", {"dataset": "one"}, source)
    store.fetch(keys)
    assert store.query_calls == 3
    store.fetch(keys)
    assert store.query_calls == 3
    store.close()
    with pytest.raises(ValueError, match="provenance"):
        ContextStore(tmp_path / "cache.sqlite", {"dataset": "two"})
    offline = ContextStore(tmp_path / "cache.sqlite", {"dataset": "one"})
    with pytest.raises(ValueError, match="Offline"):
        offline.fetch([ContextKey("Account", "absent", 100, 1000)])
    offline.close()


def test_isolated_entities_and_model_ablations(tmp_path: Path) -> None:
    key = ContextKey("Token", "alone", 100, 1000)
    store = ContextStore(tmp_path / "cache.sqlite", {}, FakeExecutor({}))
    batch = make_live_batch(store, [key])
    assert not batch["first_mask"].any()
    for variant in ("temporal", "no_fourier", "tabular"):
        assert torch.isfinite(LiveTGAT(16, 4, 0, variant)(batch)).all()
    store.close()


def test_coowners_share_split_even_through_another_account() -> None:
    accounts = pd.DataFrame(
        {"account_id": ["a", "b", "c", "d"], "owner_ids": [["x"], ["x", "y"], ["y"], ["z"]]}
    )
    assigned = assign_groups(accounts, 42)
    assert assigned.iloc[:3]["group_id"].nunique() == 1
    assert assigned.iloc[:3]["split"].nunique() == 1


def test_query_renderer_matches_reviewed_source_and_uses_no_labels() -> None:
    root = Path(__file__).resolve().parents[2]
    text = render_context_query()
    assert re.sub(r"\s+", "", text) == re.sub(
        r"\s+", "", (root / "gsql/temporal/training_context.gsql").read_text()
    )
    for field in ("is_mule", "fraud_label", "pu_label", "ring_id", "pair_time_encoding"):
        assert field not in text
    assert "temporal_fourier64_values" in text
    assert "e.valid_from_seq <= state_seq" in text
    assert "state_seq < e.valid_to_seq" in text


def test_dates_must_have_forward_chronological_splits() -> None:
    config = {
        "dates": {"train": ["2024-07-01"], "validation": ["2024-06-01"], "test": ["2025-01-01"]}
    }
    with pytest.raises(ValueError, match="overlap"):
        validate_dates(config)


def test_query_comparison_preserves_string_literal_case_and_spacing() -> None:
    from mule_pattern_learner.temporal.live.installation import normalized

    assert normalized('PRINT "USD";') != normalized('PRINT "usd";')
    assert normalized('PRINT "a b";') != normalized('PRINT "ab";')
    assert normalized('PRINT  "USD";') == normalized('print "USD";')
