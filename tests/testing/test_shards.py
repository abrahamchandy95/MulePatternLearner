"""The loader's shard reader, on export folders written here in the loader's format
(tf_gnn_loader_v2, format 9): export_manifest.json with each dataset's columns and types in
file order and its shards in order, each with its rows and SHA-256; headerless PSV shards as
PostgreSQL's COPY TEXT writes them, booleans in lower case."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mule_pattern_learner.contract import TABLES
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.reference.infrastructure import world
from mule_pattern_learner.testing.shards import MANIFEST, read_export, write_export
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, Mule, T, WorldBuilder

D = DAY_MS


def _world() -> WorldBuilder:
    w = WorldBuilder()
    w.household(["ann", "ben"], T - 400 * D, "d_home", "i_home")
    w.customer("mia", T - 300 * D, mule=Mule(T - 60 * D, T - 10 * D, 2, identified=True))
    w.external("x_out", T - 500 * D)
    w.token("t_ann", "p_ann", "a_ann", T - 200 * D)
    w.token("t_mia", "p_mia", "a_mia", T - 200 * D)
    w.pay("x_1", "a_ann", "a_ben", T - 2 * D, 12.5, "ach", device="d_home", ip="i_home")
    w.pay("x_2", "a_ben", "x_out", T - D, 1.005, "card")
    w.pay("x_3", "a_mia", "x_out", T - 3 * D, 1e20, "cash")
    w.pay(
        "z_1", "a_ann", "a_mia", T - HOUR_MS, 900.0, "zelle", ip="i_home", tokens=("t_ann", "t_mia")
    )
    return w


def test_the_reader_gives_back_every_table_of_a_two_shard_export(tmp_path: Path) -> None:
    tables = _world().tables()
    manifest = write_export(tmp_path, tables)
    assert len(manifest["datasets"]["Account"]["shards"]) == 2
    assert len(manifest["datasets"]["Zelle_Transfer"]["shards"]) == 1
    assert manifest["datasets"]["Transfer_Used_Device"]["shards"] == []
    found = read_export(tmp_path)
    assert list(found) == [table.name for table in TABLES]
    for name, expected in tables.items():
        pd.testing.assert_frame_equal(found[name], expected, check_exact=True)
    assert world(ReferenceGraph(found)) == world(ReferenceGraph(tables))


def test_a_shard_line_is_read_as_copy_text_writes_it(tmp_path: Path) -> None:
    manifest = write_export(tmp_path, WorldBuilder().tables())
    lines = {
        "Account": "a_1|deposit|false|1704067200000|0|1|0|7|1704153600000|1704240000000\n"
        "x_1|unknown|true|1600000000000|0|-1|-1|-1|0|0\n",
        "Payment_Transaction": "x_9|100|ach|1719792000000\n",
        "Zelle_Transfer": "z_9|1e+20|1719791999999\n",
    }
    for name, text in lines.items():
        _replace_shards(tmp_path, manifest, name, text)
    found = read_export(tmp_path)
    assert list(found["Account"].itertuples(index=False, name=None)) == [
        (
            "a_1",
            "deposit",
            False,
            1_704_067_200_000,
            0,
            1,
            0,
            7,
            1_704_153_600_000,
            1_704_240_000_000,
        ),
        ("x_1", "unknown", True, 1_600_000_000_000, 0, -1, -1, -1, 0, 0),
    ]
    # PostgreSQL writes a whole double without a point and a large one with an exponent.
    assert list(found["Payment_Transaction"].itertuples(index=False, name=None)) == [
        ("x_9", 100.0, "ach", 1_719_792_000_000)
    ]
    assert found["Zelle_Transfer"]["amount"].tolist() == [1e20]


def _replace_shards(directory: Path, manifest: dict[str, Any], name: str, text: str) -> None:
    """The dataset name's rows replaced by one shard holding text, with the manifest's bytes,
    rows and SHA-256 to match, so that only what text holds is read."""
    path = directory / "shards" / f"{name.lower()}_replaced.psv"
    data = text.encode()
    path.write_bytes(data)
    rows = text.count("\n")
    # A relative path is taken from the export folder.
    shard = {
        "path": str(path.relative_to(directory)),
        "bytes": len(data),
        "rows": rows,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    manifest["datasets"][name].update(rows=rows, bytes=len(data), shards=[shard])
    (directory / MANIFEST).write_text(json.dumps(manifest))


def _manifest(change: Callable[[dict[str, Any]], None]) -> Callable[[Path, dict[str, Any]], None]:
    def plant(directory: Path, manifest: dict[str, Any]) -> None:
        change(manifest)
        (directory / MANIFEST).write_text(json.dumps(manifest))

    return plant


def _line(name: str, text: str) -> Callable[[Path, dict[str, Any]], None]:
    def plant(directory: Path, manifest: dict[str, Any]) -> None:
        _replace_shards(directory, manifest, name, text)

    return plant


def _tamper(directory: Path, manifest: dict[str, Any]) -> None:
    path = Path(manifest["datasets"]["Party"]["shards"][0]["path"])
    path.write_bytes(path.read_bytes().replace(b"p_ann", b"p_amy"))


def _columns(name: str, columns: list[list[str]]) -> Callable[[dict[str, Any]], None]:
    return lambda manifest: manifest["datasets"][name].update(columns=columns)


PARTY = [["id", "STRING"], ["joined_ms", "UINT"]]
ACCOUNT = "a_1|deposit|{}|1|0|0|0|-1|0|0\n"
REFUSED: list[tuple[str, Callable[[Path, dict[str, Any]], None], str]] = [
    ("another separator", _manifest(lambda m: m.update(separator=",")), "headerless PSV"),
    ("a header", _manifest(lambda m: m.update(header=True)), "headerless PSV"),
    ("a missing dataset", _manifest(lambda m: m["datasets"].pop("Token")), "not the contract's"),
    (
        "an extra dataset",
        _manifest(lambda m: m["datasets"].update(Address={})),
        "not the contract's",
    ),
    (
        "a renamed column",
        _manifest(_columns("Party", [["id", "STRING"], ["join_ms", "UINT"]])),
        "Party has the columns",
    ),
    (
        "a column of another type",
        _manifest(_columns("Party", [["id", "STRING"], ["joined_ms", "INT"]])),
        "Party has the columns",
    ),
    ("reordered columns", _manifest(_columns("Party", PARTY[::-1])), "Party has the columns"),
    ("a dropped column", _manifest(_columns("Party", PARTY[:1])), "Party has the columns"),
    ("a changed shard", _tamper, "SHA-256 differs"),
    (
        "a shard of other rows",
        _manifest(lambda m: m["datasets"]["Party"]["shards"][0].update(rows=9)),
        "not the manifest's 9",
    ),
    (
        "a dataset of other rows",
        _manifest(lambda m: m["datasets"]["Party"].update(rows=9)),
        "not the manifest's 9",
    ),
    ("an upper-case boolean", _line("Account", ACCOUNT.format("True")), "'True', not a BOOL"),
    ("a boolean as a digit", _line("Account", ACCOUNT.format("0")), "'0', not a BOOL"),
    ("a NULL", _line("Party", "p_1|\n"), "'', not a UINT"),
    ("an empty id", _line("Party", "|5\n"), "'', not a STRING"),
    ("a negative UINT", _line("Party", "p_1|-5\n"), "'-5', not a UINT"),
    ("a fractional UINT", _line("Party", "p_1|5.0\n"), "'5.0', not a UINT"),
    ("a NaN amount", _line("Zelle_Transfer", "z_1|NaN|5\n"), "'NaN', not a DOUBLE"),
    ("an infinite amount", _line("Zelle_Transfer", "z_1|Infinity|5\n"), "not a DOUBLE"),
    ("an amount with an underscore", _line("Zelle_Transfer", "z_1|1_0|5\n"), "not a DOUBLE"),
    ("an overflowing amount", _line("Zelle_Transfer", "z_1|1e999|5\n"), "beyond a finite double"),
    ("a time beyond 64 bits", _line("Party", "p_1|99999999999999999999\n"), "UINT beyond 64 bits"),
    ("too few values", _line("Party", "p_1\n"), "1 values, not Party's 2"),
    ("a separator inside a value", _line("Party", "p|1|5\n"), "3 values, not Party's 2"),
    ("no final line feed", _line("Party", "p_1|5"), "does not end with a line feed"),
]


@pytest.mark.parametrize(
    ("plant", "message"), [(p, m) for _, p, m in REFUSED], ids=[n for n, _, _ in REFUSED]
)
def test_the_reader_refuses_an_export_unlike_the_contract_or_its_manifest(
    tmp_path: Path, plant: Callable[[Path, dict[str, Any]], None], message: str
) -> None:
    manifest = write_export(tmp_path, _world().tables())
    read_export(tmp_path)
    plant(tmp_path, manifest)
    with pytest.raises(ValueError, match=message):
        read_export(tmp_path)
