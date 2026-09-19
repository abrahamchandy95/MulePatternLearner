from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.temporal.encoding import fourier64
from mule_pattern_learner.temporal.model import Normalizer, TemporalModel, make_batch, nnpu_loss
from mule_pattern_learner.temporal.snapshots import Snapshot, build_snapshot, visible
from mule_pattern_learner.temporal.supervision import (
    account_groups,
    labels_at_cutoff,
    reveal_mask,
    split_accounts,
)


def fixture_stage(path: Path, *, future_amount: float = 10, future_end: int = 50) -> None:
    path.mkdir()
    pd.DataFrame(
        [
            [1, "p1", "Party", "person", False, 1, 100],
            [2, "a1", "Account", "deposit", False, 1, 100],
            [3, "a2", "Account", "deposit", False, 1, 100],
            [4, "t1", "Token", "phone", False, 1, 100],
            [5, "d1", "Device", "mobile", False, 1, 100],
            [6, "ip1", "IP", "unknown", False, 1, 100],
            [7, "addr1", "Address", "unknown", False, 1, 100],
            [8, "future", "Account", "deposit", False, 40, 4000],
        ],
        columns=[
            "node_index",
            "id",
            "node_type",
            "subtype",
            "is_external",
            "first_seen_seq",
            "first_seen_ts_ms",
        ],
    ).to_parquet(path / "nodes.parquet", index=False)
    pd.DataFrame(
        [
            ["T1", 10.0, 1000, 10, True, 1, 2, 3, 4, 0, 5, 6, 3, False, 0],
            ["T2", 20.0, 2000, 20, True, 1, 2, 3, 4, 0, 5, 6, 3, True, 1000],
            ["T3", 30.0, 2000, 30, True, 1, 2, 0, 4, 4, 5, 6, 4, False, 0],
            ["T4", future_amount, 4000, 40, True, 1, 2, 8, 4, 0, 5, 6, 8, False, 0],
        ],
        columns=[
            "event_id",
            "amount",
            "event_ts_ms",
            "event_seq",
            "amount_present",
            "rail",
            "sender",
            "recipient",
            "sender_token",
            "recipient_token",
            "device",
            "ip",
            "counterparty",
            "gap_present",
            "pair_gap_ms",
        ],
    ).to_parquet(path / "events.parquet", index=False)
    pd.DataFrame(
        [
            [1, 2, "Party_Owns_Account", 1, 0],
            [4, 2, "Token_Bound_To_Account", 5, 20],
            [4, 2, "Token_Bound_To_Account", 25, future_end],
            [4, 3, "Token_Bound_To_Account", 40, 0],
            [1, 8, "Party_Owns_Account", 40, 0],
        ],
        columns=["src", "dst", "relation", "valid_from_seq", "valid_to_seq"],
    ).to_parquet(path / "associations.parquet", index=False)
    (path / "manifest.json").write_text(json.dumps({"max_ts_ms": 4000}))


def test_fourier_matches_gsql_basis_and_rejects_negative() -> None:
    actual = fourier64(np.array([0, 1000, 34_560_000_000], dtype=np.int64))
    np.testing.assert_array_equal(actual[0, ::2], 0)
    np.testing.assert_array_equal(actual[0, 1::2], 1)
    frequencies = 0.125 * 16 ** (np.arange(32) / 31)
    np.testing.assert_allclose(actual[2, ::2], np.sin(2 * np.pi * frequencies), atol=1e-6)
    np.testing.assert_allclose(actual[2, 1::2], np.cos(2 * np.pi * frequencies), atol=1e-6)
    with pytest.raises(ValueError):
        fourier64(np.array([-1], dtype=np.int64))


def test_future_payments_and_future_association_ends_do_not_change_snapshot(tmp_path: Path) -> None:
    fixture_stage(tmp_path / "a")
    fixture_stage(tmp_path / "b", future_amount=1e12, future_end=999999)
    for name in ("a", "b"):
        build_snapshot(tmp_path / name, tmp_path / (name + "_snap"), 3000)
    a = Snapshot.load(tmp_path / "a_snap")
    b = Snapshot.load(tmp_path / "b_snap")
    for name in ("x", "neighbors", "relation", "rail", "edge", "age_ms", "gap_ms"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.metadata["seed_seq"] == 30
    assert not a.x[8].any()
    assert 8 not in a.neighbors
    # An unresolved recipient remains a Token, and a repeated directed pair
    # produces two separate sampled messages (not a collapsed edge).
    assert list(a.neighbors[2]).count(3) == 2
    assert 4 in a.neighbors[2]
    feature = a.metadata["feature_names"].index("Token_Bound_To_Account_active")
    assert a.x[2, feature] == pytest.approx(np.log1p(1))
    assert a.x[3, feature] == 0


def test_tied_timestamp_group_excluded_together(tmp_path: Path) -> None:
    fixture_stage(tmp_path / "stage")
    meta = build_snapshot(tmp_path / "stage", tmp_path / "snap", 2000)
    assert meta["history_events"] == 1
    assert meta["seed_seq"] == 10


def test_valid_time_is_half_open_and_supports_reenrollment() -> None:
    start = np.array([5, 25], dtype=np.int64)
    end = np.array([20, 0], dtype=np.int64)
    assert visible(start, end, 20).tolist() == [False, False]
    assert visible(start, end, 25).tolist() == [False, True]


def test_masks_nested_account_stable_and_dark_ring_zero_group_isolates(tmp_path: Path) -> None:
    fixture_stage(tmp_path / "stage")
    nodes = pd.read_parquet(tmp_path / "stage/nodes.parquet")
    associations = pd.read_parquet(tmp_path / "stage/associations.parquet")
    group = account_groups(nodes, associations)[2]
    accounts = split_accounts(nodes, associations, test_groups={group})
    assert accounts.loc[accounts["node_index"].isin([2, 8]), "split"].eq("test").all()
    for seed in (1, 42, 123):
        low = reveal_mask(accounts, 0.1, seed)
        high = reveal_mask(accounts, 0.5, seed)
        assert np.all(~low | high)
        assert not reveal_mask(accounts, 0, seed).any()
        assert reveal_mask(accounts, 1, seed).all()


def test_label_availability_and_hidden_identity() -> None:
    accounts = pd.DataFrame(
        {"node_index": [1, 2, 3, 4], "split": ["train", "train", "test", "train"]}
    )
    labels = pd.DataFrame(
        {
            "node_index": [1, 2, 3, 4],
            "target": [1, 1, 1, 0],
            "effective_ts_ms": [100, 100, 100, 0],
            "available_ts_ms": [200, 400, 200, 0],
        }
    )
    truth, known = labels_at_cutoff(accounts, labels, 300, np.array([True, True, True, False]))
    assert truth.tolist() == [1, 1, 1, 0]
    assert known.tolist() == [True, False, False, False]
    _, hidden = labels_at_cutoff(accounts, labels, 999, np.zeros(4, dtype=bool))
    assert not hidden.any()


@pytest.mark.parametrize("variant", ["tabular", "no_fourier", "temporal"])
def test_model_bounded_batch_finite_gradient_and_padding(tmp_path: Path, variant: str) -> None:
    fixture_stage(tmp_path / "stage")
    build_snapshot(tmp_path / "stage", tmp_path / "snapshot", 3000)
    snapshot = Snapshot.load(tmp_path / "snapshot")
    normalizer = Normalizer.fit(snapshot)
    batch = make_batch(
        snapshot, np.array([2, 3, 7], dtype=np.int64), normalizer, variant=variant, fanouts=(4, 2)
    )
    model = TemporalModel(snapshot.x.shape[1], hidden=8, dropout=0, variant=variant)
    logits = model(batch)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
    nnpu_loss(logits[:1], logits, prior=0.1).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if variant != "tabular":
        assert batch["second_x"].shape[0] <= 3 * (1 + 4)
        assert batch["second_x"].shape[1] == 2
        assert not batch["first_mask"][2].any()


def test_nnpu_requires_nonempty_samples() -> None:
    with pytest.raises(ValueError):
        nnpu_loss(torch.empty(0), torch.ones(2), 0.01)
    with pytest.raises(ValueError):
        nnpu_loss(torch.ones(2), torch.ones(2), 0)


def test_nnpu_training_checkpoint_and_zero_label_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import Any
    from mule_pattern_learner.temporal import training

    fixture_stage(tmp_path / "stage")
    nodes = pd.read_parquet(tmp_path / "stage/nodes.parquet")
    extra = pd.DataFrame(
        [[i, f"a{i}", "Account", "deposit", False, 1, 100] for i in (9, 10, 11)],
        columns=nodes.columns,
    )
    nodes = pd.concat([nodes, extra], ignore_index=True)
    nodes.to_parquet(tmp_path / "stage/nodes.parquet", index=False)
    dates = ["1970-01-01T00:00:05", "1970-01-01T00:00:06", "1970-01-01T00:00:07"]
    for date in dates:
        build_snapshot(tmp_path / "stage", tmp_path / "snapshots" / date, training.timestamp(date))
    accounts = nodes[nodes["node_index"].isin([2, 3, 8, 9, 10, 11])][
        ["node_index", "id", "first_seen_ts_ms"]
    ].copy()
    accounts["group_id"] = accounts["id"]
    accounts["split"] = ["train", "train", "validation", "validation", "test", "test"]

    def fixed_split(*args: Any, **kwargs: Any) -> pd.DataFrame:
        return accounts.reset_index(drop=True)

    monkeypatch.setattr(training, "split_accounts", fixed_split)
    labels = accounts[["id"]].rename(columns={"id": "account_id"})
    labels["target"] = [1, 0, 1, 0, 1, 0]
    labels["effective_ts_ms"] = 100
    labels["available_ts_ms"] = 200
    labels["ring_id"] = [0, -1, 0, -1, 0, -1]
    label_path = tmp_path / "labels.parquet"
    labels.to_parquet(label_path, index=False)
    label_path.with_suffix(".json").write_text(
        json.dumps({"target_definition": "confirmed_mule", "complete_negative_ground_truth": True})
    )
    config = {
        "stage": str(tmp_path / "stage"),
        "snapshots": str(tmp_path / "snapshots"),
        "task": "mule_pu",
        "labels": str(label_path),
        "dates": {"train": [dates[0]], "validation": [dates[1]], "test": [dates[2]]},
        "variant": "temporal",
        "seed": 42,
        "reveal_fraction": 1.0,
        "class_prior": 0.1,
        "epochs": 2,
        "steps_per_epoch": 1,
        "batch_size": 4,
        "fanouts": [4, 2],
        "hidden": 8,
        "threads": 1,
    }
    result = training.run(config, tmp_path / "run")
    assert result["status"] == "complete"
    assert result["counts"]["train"][0]["revealed_positive_accounts"] == 1
    assert result["metrics"]["test"]["n"] == 2
    saved = torch.load(tmp_path / "run/model.pt", weights_only=True)
    assert saved["basis_id"] == "log1p_s_400d_32x_sincos_v1"
    assert saved["provenance"]["code_sha256"]
    skipped = training.run({**config, "reveal_fraction": 0.0}, tmp_path / "skip")
    assert skipped["status"] == "skipped"
    assert not (tmp_path / "skip/model.pt").exists()
