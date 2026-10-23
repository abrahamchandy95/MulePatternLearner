"""The contract lock and the model files (docs/design.md, "Training and selection": the model
file holds every query version, every layout with its count transform and Fourier basis, the
community, degree and ablation flags, and the settings; mule score refuses a model built for
other query versions or settings and names the fields that differ)."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.model.time_encoding import FREQUENCIES, SCALE
from mule_pattern_learner.subgraph.layouts import ACCOUNT_LAYOUT, RELATIONS
from mule_pattern_learner.tigergraph.queries import VERSIONS
from mule_pattern_learner.training import contract
from mule_pattern_learner.training.contract import differences, load, lock, save

SUBGRAPH = DEFAULT_CONFIG.subgraph
TRAINING = DEFAULT_CONFIG.training


def _lock(**changes: object) -> contract.Lock:
    subgraph = replace(SUBGRAPH, **changes)
    return lock("gnn", community=True, subgraph=subgraph, training=TRAINING)


def test_the_lock_holds_every_field_the_design_names() -> None:
    found = _lock()
    assert found["format"] == contract.FORMAT and found["ranker"] == "gnn"
    assert found["community"] is True
    for name, version in VERSIONS.items():
        assert found[f"versions.{name}"] == version
    assert found["layout.node.account"][:2] == [["is_external", "flag"], ["hub", "flag"]]
    assert len(found["layout.node.account"]) == len(ACCOUNT_LAYOUT.columns)
    assert len(found["layout.relations"]) == len(RELATIONS) == 16
    assert found["layout.edge.account.pays.account"][0] == ["amount", "amount"]
    assert found["layout.scored"] == "account"
    assert found["encoding.count"] == "log(1 + n)"
    assert found["fourier.frequencies"] == FREQUENCIES.tolist()
    assert found["fourier.scale"] == SCALE
    assert found["degrees.customer"] == [
        "owned_by",
        "rev_enrolled_on_device",
        "rev_enrolled_on_ip",
    ]
    assert found["seeds"] == 5
    # The settings of the design: lookback 90; caps 32, 16, 8, 8 devices and IPs, 48;
    # time-ordered lists 256; community 64; hub 200; crowd 8.
    assert {name: found[f"settings.{name}"] for name in asdict(SUBGRAPH)} == {
        "lookback_days": 90,
        "recent_payments": 256,
        "scored_payments": 32,
        "counterparty_payments": 16,
        "own_accounts": 8,
        "devices_ips": 8,
        "co_users": 48,
        "community": 64,
        "hub_threshold": 200,
        "crowd_threshold": 8,
        "new_windows": [7, 30],
        "sharing_days": 30,
    }


def test_a_model_file_reads_back_under_its_lock(tmp_path: Path) -> None:
    path = tmp_path / "gnn_seed0.pt"
    content = {"seed": 0, "state": {"w": torch.arange(3.0)}, "name": "x", "bytes": b"\x00\x01"}
    save(path, _lock(), content)
    found = load(path, _lock())
    assert found.keys() == content.keys()
    assert torch.equal(found["state"]["w"], torch.arange(3.0))
    assert found["bytes"] == b"\x00\x01"
    assert [item.name for item in tmp_path.iterdir()] == ["gnn_seed0.pt"]


def test_a_changed_cap_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    save(path, _lock(), {"seed": 0})
    with pytest.raises(ValueError, match=r"settings\.co_users \(the model's 48, now 47\)"):
        load(path, _lock(co_users=47))


def test_every_differing_field_is_named() -> None:
    stored = _lock()
    current = _lock(hub_threshold=150, crowd_threshold=9)
    current["versions.read_money_flow"] = "changed"
    del current["seeds"]
    found = differences(stored, current)
    assert [line.split(" (")[0] for line in found] == [
        "seeds",
        "settings.crowd_threshold",
        "settings.hub_threshold",
        "versions.read_money_flow",
    ]
    assert found[0] == 'seeds (the model\'s 5, now "missing")'
    assert differences(stored, _lock()) == []


def test_the_ablation_and_community_flags_and_a_rankers_own_fields_are_locked() -> None:
    base = lock("gnn", community=True, subgraph=SUBGRAPH, training=TRAINING)
    ablation = lock("gnn_without_community", community=False, subgraph=SUBGRAPH, training=TRAINING)
    assert [line.split(" (")[0] for line in differences(base, ablation)] == ["community", "ranker"]
    trees = lock(
        "xgboost_own",
        community=False,
        subgraph=SUBGRAPH,
        training=TRAINING,
        extra={"columns": ("a", "b")},
    )
    assert trees["xgboost_own.columns"] == ["a", "b"]
    fewer = replace(TRAINING, seeds=(0, 1, 2))
    assert lock("gnn", community=True, subgraph=SUBGRAPH, training=fewer)["seeds"] == 3


def test_a_file_that_is_no_model_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.pt"
    torch.save({"state": {}}, path)
    with pytest.raises(ValueError, match="not a model file"):
        load(path, _lock())
    with pytest.raises(ValueError, match="names neither"):
        save(tmp_path / "x.pt", _lock(), {"lock": {}})
