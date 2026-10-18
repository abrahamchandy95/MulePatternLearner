"""The loader's shard reader: one export folder of tf_gnn_loader_v2, read into the contract
frames the reference graph takes, so that mule parity compares the live graph with the
reference graph on the exact bytes TigerGraph loaded (docs/design.md, "Commands").

An export folder holds export_manifest.json and the shards it lists. Per contract table, its
dataset, the manifest names the columns with their graph types in file order, and the shards
in order, each with its rows and SHA-256. A shard is headerless PSV, as PostgreSQL's COPY TEXT
writes it: one line per row ending in a line feed, values separated by "|", booleans as true
or false in lower case, and an empty value for NULL. Values are read as TigerGraph loads them,
with nothing unescaped.

The reader refuses a manifest whose datasets, columns or types are unlike contract.py's, a
shard whose bytes or rows are not the manifest's, and a value that is not of its column's
type, a NULL included. An export holds label columns, so only testing/ and evaluation/ open
one (the Files that carry labels guard); finding the export of the loaded world under the
export root is parity's.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from ..contract import TABLES, Table
from .reference.graph import CONTRACT_DTYPES, DTYPES, frame

MANIFEST = "export_manifest.json"
SEPARATOR = "|"
LINE_END = "\n"
# The text each graph type's value must be, whole: a string of at least one character (an
# empty value is NULL), digits, a sign and digits, true or false in lower case, and a double
# as PostgreSQL writes one, with no infinity, NaN, space or underscore.
PATTERNS = {
    "STRING": r".+",
    "UINT": r"[0-9]+",
    "INT": r"-?[0-9]+",
    "BOOL": r"true|false",
    "DOUBLE": r"-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?",
}


def read_export(directory: Path) -> dict[str, pd.DataFrame]:
    """The contract frames of the export in directory, one per table, rows in shard order."""
    manifest: object = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{directory / MANIFEST} holds no manifest")
    found: Mapping[str, Any] = manifest
    if found.get("separator") != SEPARATOR or found.get("header") is not False:
        raise ValueError(
            f"{directory} is not headerless PSV separated by {SEPARATOR!r}: separator "
            f"{found.get('separator')!r}, header {found.get('header')!r}"
        )
    datasets = found.get("datasets")
    names = {table.name for table in TABLES}
    if not isinstance(datasets, Mapping) or set(datasets) != names:
        given = sorted(datasets) if isinstance(datasets, Mapping) else datasets
        raise ValueError(
            f"{directory} holds the datasets {given}, not the contract's {sorted(names)}"
        )
    return {table.name: _dataset(directory, table, datasets[table.name]) for table in TABLES}


def _dataset(directory: Path, table: Table, dataset: object) -> pd.DataFrame:
    """A dataset's rows, from its shards in the manifest's order."""
    if not isinstance(dataset, Mapping):
        raise ValueError(f"The manifest's {table.name} is {dataset!r}, not a dataset")
    entry: Mapping[str, Any] = dataset
    columns = entry.get("columns")
    expected = [[column.name, column.graph_type] for column in table.columns]
    if columns != expected:
        raise ValueError(f"{table.name} has the columns {columns}, not the contract's {expected}")
    shards = entry.get("shards")
    if not isinstance(shards, list):
        raise ValueError(f"The manifest's {table.name} lists no shards")
    frames = [_shard(directory, table, shard) for shard in shards]
    found = pd.concat(frames, ignore_index=True) if frames else frame(table.name, [])
    if entry.get("rows") != len(found):
        raise ValueError(
            f"{table.name} has {len(found)} rows, not the manifest's {entry.get('rows')}"
        )
    return found


def _shard(directory: Path, table: Table, shard: object) -> pd.DataFrame:
    """A shard's rows, once its bytes have the manifest's SHA-256 and rows. Its path is taken
    from the export folder when it is relative; the loader writes it absolute."""
    if not isinstance(shard, Mapping) or not isinstance(shard.get("path"), str):
        raise ValueError(f"The manifest lists the {table.name} shard {shard!r}")
    entry: Mapping[str, Any] = shard
    path = directory / str(entry["path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
        raise ValueError(f"{path} is not the shard the manifest lists: its SHA-256 differs")
    text = data.decode("utf-8")
    if text and not text.endswith(LINE_END):
        raise ValueError(f"{path} does not end with a line feed")
    lines = pd.Series(text.split(LINE_END)[:-1], dtype=DTYPES["STRING"])
    if entry.get("rows") != len(lines):
        raise ValueError(f"{path} has {len(lines)} rows, not the manifest's {entry.get('rows')}")
    if lines.empty:
        return frame(table.name, [])
    width = len(table.columns)
    counts = lines.str.count(re.escape(SEPARATOR)) + 1
    wrong = (counts != width).to_numpy()
    if wrong.any():
        line = int(wrong.argmax())
        raise ValueError(
            f"{path}:{line + 1} has {counts.iloc[line]} values, not {table.name}'s {width}"
        )
    parts = lines.str.split(SEPARATOR, expand=True, regex=False)
    return pd.DataFrame(
        {
            column.name: _column(
                parts[index], column.graph_type, f"{path} {table.name}.{column.name}"
            )
            for index, column in enumerate(table.columns)
        }
    )


def _column(texts: pd.Series, graph_type: str, where: str) -> pd.Series:
    """A column of a graph type, from its values' text as COPY TEXT writes it."""
    valid = texts.str.fullmatch(PATTERNS[graph_type]).to_numpy(dtype=bool)
    if not valid.all():
        row = int((~valid).argmax())
        raise ValueError(f"{where} row {row + 1} is {texts.iloc[row]!r}, not a {graph_type}")
    if graph_type == "STRING":
        return texts
    if graph_type == "BOOL":
        return (texts == "true").astype(CONTRACT_DTYPES[graph_type])
    try:
        values = texts.astype(CONTRACT_DTYPES[graph_type])
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{where} holds a {graph_type} beyond 64 bits") from error
    finite = np.isfinite(values.to_numpy())
    if not finite.all():
        row = int((~finite).argmax())
        raise ValueError(f"{where} row {row + 1} is {texts.iloc[row]!r}, beyond a finite double")
    return values
