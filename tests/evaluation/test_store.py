"""The evaluation store: the answer key, read_labels' rows, the fallback list and the pass record
round-trip through their files, read back through the queries' parsers, and a pass record holds
pass or fail per check and nothing else."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mule_pattern_learner.data.labels import read_identified
from mule_pattern_learner.evaluation.answer_key import read_answer_key
from mule_pattern_learner.evaluation.store import (
    RECORD_FILE,
    AuditRecord,
    Store,
    parse_record,
    record_text,
)
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.tigergraph.queries import VERSIONS

IDENTITY = "0123456789abcdef0123456789abcdef"


def test_the_answer_key_and_labels_round_trip(data_dir: Path, graph: ReferenceGraph) -> None:
    store = Store(IDENTITY)
    key = read_answer_key(Spy(graph))
    labels = read_identified(Spy(graph))
    store.write_answer_key(key)
    store.write_labels(labels)
    assert store.folder == data_dir / "evaluation" / IDENTITY
    assert store.answer_key() == key
    assert store.labels() == labels and labels.accounts == {"a_mule", "a_drop"}
    # Nothing is left behind by the writes but the files themselves.
    assert sorted(path.name for path in store.folder.iterdir()) == [
        "answer_key.parquet",
        "labels.parquet",
    ]


def test_a_stored_answer_key_is_reparsed_by_its_parser(
    data_dir: Path, graph: ReferenceGraph
) -> None:
    store = Store(IDENTITY)
    store.write_answer_key(read_answer_key(Spy(graph)))
    frame = pd.read_parquet(store.folder / "answer_key.parquet")
    frame.loc[0, "is_mule"] = 2
    frame.to_parquet(store.folder / "answer_key.parquet", index=False)
    with pytest.raises(ValueError, match="is_mule 2"):
        store.answer_key()


def test_the_fallback_list_round_trips(data_dir: Path) -> None:
    store = Store(IDENTITY)
    keys = pd.DataFrame({"t_ms": [1, 2], "account_id": ["a", "b"], "weight": [1.0, 10.0]})
    assert not store.has_fallback()
    store.write_fallback(keys)
    assert store.has_fallback()
    pd.testing.assert_frame_equal(store.fallback(), keys)


def _record(**checks: bool) -> AuditRecord:
    return AuditRecord("D1", "development", IDENTITY, "c" * 32, "s" * 32, dict(VERSIONS), checks)


def test_a_pass_record_round_trips_and_names_its_failures(data_dir: Path) -> None:
    store = Store(IDENTITY)
    assert store.record() is None
    record = _record(**{"gate_record": True, "label.truth_join": False, "clock_probe": False})
    store.write_record(record)
    assert store.record() == record
    assert record.failed == ("clock_probe", "label.truth_join")
    text = (store.folder / RECORD_FILE).read_text()
    assert '"label.truth_join" = false' in text and 'world = "D1"' in text


def test_a_pass_record_holds_nothing_but_its_fields(tmp_path: Path) -> None:
    text = record_text(_record(clock_probe=True))
    path = tmp_path / RECORD_FILE
    assert parse_record(text, path) == _record(clock_probe=True)
    for planted in (
        text + 'roc_auc = "0.51"\n',
        text.replace('"clock_probe" = true', '"clock_probe" = 0.51'),
        text.replace('world = "D1"', "world = 1"),
    ):
        with pytest.raises(ValueError):
            parse_record(planted, path)


def test_a_record_goes_to_its_own_worlds_store(data_dir: Path) -> None:
    with pytest.raises(ValueError, match="its own store"):
        Store("f" * 32).write_record(_record())
    store = Store(IDENTITY)
    (store.folder).mkdir(parents=True)
    (store.folder / RECORD_FILE).write_text(record_text(_record()).replace(IDENTITY, "f" * 32))
    with pytest.raises(ValueError, match="pass record of the world"):
        store.record()
