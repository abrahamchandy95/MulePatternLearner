"""The contract lock and the model files (docs/design.md, "Training and selection": the model
file holds every query version, every layout with its count transform and Fourier basis, the
community, degree and ablation flags, and the settings; mule score refuses a model built for
other query versions or settings and names the fields that differ)."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mule_pattern_learner.config import DEFAULT_CONFIG, FULL_DESIGN_CONFIG
from mule_pattern_learner.model import encode as encode_module
from mule_pattern_learner.model.encode import Kind
from mule_pattern_learner.model.time_encoding import FREQUENCIES, SCALE, encode_reference
from mule_pattern_learner.subgraph.layouts import ACCOUNT_LAYOUT, RELATIONS
from mule_pattern_learner.tigergraph.queries import VERSIONS
from mule_pattern_learner.training import contract
from mule_pattern_learner.training.contract import differences, load, lock, save

SUBGRAPH = DEFAULT_CONFIG.subgraph
# The five seeds of the full design, which the lock counts.
TRAINING = FULL_DESIGN_CONFIG.training


def _lock(**changes: object) -> contract.Lock:
    subgraph = replace(SUBGRAPH, **changes)
    return lock("gnn", community=True, subgraph=subgraph, seeds=len(TRAINING.seeds))


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
    assert {name for name in found if name.startswith("encoding.")} == {
        f"encoding.{kind.value}" for kind in Kind
    }
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
    seeds = len(TRAINING.seeds)
    base = lock("gnn", community=True, subgraph=SUBGRAPH, seeds=seeds)
    ablation = lock("gnn_without_community", community=False, subgraph=SUBGRAPH, seeds=seeds)
    assert [line.split(" (")[0] for line in differences(base, ablation)] == ["community", "ranker"]
    trees = lock("xgboost_own", community=False, subgraph=SUBGRAPH, extra={"columns": ("a", "b")})
    assert trees["xgboost_own.columns"] == ["a", "b"]
    assert lock("gnn", community=True, subgraph=SUBGRAPH, seeds=3)["seeds"] == 3


def test_a_ranker_with_no_seed_has_no_seed_count_in_its_lock() -> None:
    """The seed count is the GNN's and its ablations': a lock made with none leaves the field
    out, so a ranker trained once, like each XGBoost comparison, reads as a model with no seed
    and the number of GNN seeds never reaches its file."""
    alone = lock("xgboost_own", community=False, subgraph=SUBGRAPH)
    assert "seeds" not in alone
    assert differences(alone, lock("xgboost_own", community=False, subgraph=SUBGRAPH)) == []
    counted = lock("xgboost_own", community=False, subgraph=SUBGRAPH, seeds=5)
    assert differences(counted, alone) == ['seeds (the model\'s 5, now "missing")']


def test_a_file_that_is_no_model_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.pt"
    torch.save({"state": {}}, path)
    with pytest.raises(ValueError, match="not a model file"):
        load(path, _lock())
    with pytest.raises(ValueError, match="names neither"):
        save(tmp_path / "x.pt", _lock(), {"lock": {}})


def _probe(kind: Kind, raw: tuple[float, ...]) -> list[float]:
    """The encoded numbers the lock holds for the probe row raw of a kind."""
    for held, encoded in _lock()[f"encoding.{kind.value}"]:
        if held == list(raw):
            return encoded
    raise AssertionError(f"No probe row {raw} of {kind}")


def test_every_kind_has_a_probe_and_the_lock_holds_what_the_encoder_makes_of_it() -> None:
    """Worked out by hand: a flag and a share as they are; a count of 7 is log(1 + 7) = 2.0794,
    12,345 cents are 123.45 dollars and log(1 + 123.45) = 4.8241, both to two decimals; an
    absent duration is its presence bit 0 and 64 zeros, a present one its bit 1 and the 32 sines
    then the 32 cosines of 2 pi f u, u = log1p(seconds) / log1p(10 years)."""
    assert set(contract.PROBES) == set(Kind)
    assert _probe(Kind.FLAG, (1.0,)) == [1.0]
    assert _probe(Kind.SHARE, (0.25,)) == [0.25]
    assert _probe(Kind.COUNT, (0.0,)) == [0.0]
    assert _probe(Kind.COUNT, (7.0,)) == [2.08] and round(math.log(1 + 7), 2) == 2.08
    assert _probe(Kind.AMOUNT, (12_345.0,)) == [4.82] and round(math.log(1 + 123.45), 2) == 4.82
    assert _probe(Kind.DURATION, (0.0, 0.0)) == [0.0] * 65
    for milliseconds in (3_600_000, 7_776_000_000):
        expected = encode_reference(np.array([milliseconds]))[0]
        found = _probe(Kind.DURATION, (1.0, float(milliseconds)))
        assert found[0] == 1.0 and len(found) == 65
        np.testing.assert_allclose(found[1:], expected, atol=0.006)


def test_changing_the_amount_or_count_transform_changes_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """log(1 + cents) instead of log(1 + cents / 100), and another count transform, each
    leave every other field equal and are named by their own."""
    before = _lock()
    monkeypatch.setattr(encode_module, "CENTS", 1.0)
    cents = _lock()
    assert [line.split(" (")[0] for line in differences(before, cents)] == ["encoding.amount"]
    monkeypatch.undo()

    def shifted(x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + 2.0)

    monkeypatch.setattr(torch, "log1p", shifted)
    counts = _lock()
    changed = [line.split(" (")[0] for line in differences(before, counts)]
    assert "encoding.count" in changed and "encoding.flag" not in changed
    monkeypatch.undo()
    assert differences(before, _lock()) == []


def test_a_model_built_under_another_encoding_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.pt"
    save(path, _lock(), {"seed": 0})
    monkeypatch.setattr(encode_module, "CENTS", 1.0)
    with pytest.raises(ValueError, match=r"encoding\.amount"):
        load(path, _lock())


def test_the_probe_is_encoded_on_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lock computed on a training machine with a GPU is the lock a laptop computes."""
    devices: list[object] = []
    real = contract.encode

    def spy(raw: torch.Tensor, layout: object, device: object = None) -> torch.Tensor:
        devices.append(device)
        return real(raw, layout, device)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(contract, "encode", spy)
    fields = contract.encoding_fields()
    assert devices == ["cpu"] * len(Kind)
    assert {name: len(value) for name, value in fields.items()} == {
        f"encoding.{kind.value}": len(contract.PROBES[kind]) for kind in Kind
    }
