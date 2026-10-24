"""The evaluation store of a world (docs/design.md, "Cache"): the folder beside the cache, keyed
by the world's identity, that holds what mule audit read and found, and that only evaluation/
and testing/ open (the Files that carry labels guard):

    data/evaluation/<world identity>/answer_key.parquet   read_answer_key's rows
    data/evaluation/<world identity>/labels.parquet       read_labels' rows
    data/evaluation/<world identity>/fallback.parquet     the test-role fallback list
    data/evaluation/<world identity>/tables/<name>.parquet the audit's report tables
    data/evaluation/<world identity>/audit.toml           the pass record

Every file is written through a temporary file and a rename, so it appears whole or not at
all. Stored rows of a query are read back through that query's parser, so they pass the live
checks.

The pass record holds what mule evaluate and the mule train command read and nothing else:
pass or fail per check, the query versions, the key of the cache the audit read (its query
versions and subgraph settings), a hash of the audit's own settings, and the world's name,
kind and identity. No value, bound or count of a check is in it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import tomllib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..data.labels import Labels, label_identity
from ..paths import evaluation_store_dir
from ..tigergraph.queries import READ_LABELS, VERSIONS, read_labels
from .answer_key import COLUMNS, AnswerKey, parse_rows
from .atomic import write_bytes, write_parquet

ANSWER_KEY_FILE = "answer_key.parquet"
LABELS_FILE = "labels.parquet"
FALLBACK_FILE = "fallback.parquet"
RECORD_FILE = "audit.toml"
TABLES = "tables"
ANSWER_KEY_SCHEMA = pa.schema(
    [
        ("account_id", pa.string()),
        ("is_mule", pa.int64()),
        ("mule_ring_id", pa.int64()),
        ("mule_onset_ms", pa.int64()),
        ("mule_last_ms", pa.int64()),
        ("opened_ms", pa.int64()),
        ("account_type", pa.string()),
        ("owned", pa.bool_()),
    ]
)
LABELS_SCHEMA = pa.schema(
    [("account_id", pa.string()), ("mule_onset_ms", pa.int64()), ("mule_last_ms", pa.int64())]
)
FALLBACK_SCHEMA = pa.schema(
    [("t_ms", pa.int64()), ("account_id", pa.string()), ("weight", pa.float64())]
)


@dataclass(frozen=True)
class AuditRecord:
    """mule audit's pass record of a world: its name, kind and identity, the cache key and the
    hash of the audit's settings it was made under, every query version, and pass or fail per
    check."""

    world: str
    kind: str
    identity: str
    cache_key: str
    settings: str
    versions: Mapping[str, str] = field(default_factory=lambda: dict(VERSIONS))
    checks: Mapping[str, bool] = field(default_factory=lambda: dict[str, bool]())

    @property
    def failed(self) -> tuple[str, ...]:
        """The checks that failed, by name, sorted."""
        return tuple(sorted(name for name, passed in self.checks.items() if not passed))


def _quoted(text: str) -> str:
    """A TOML basic string: JSON's escapes are TOML's."""
    return json.dumps(text)


def _value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _quoted(value)
    raise ValueError(f"A pass record holds strings and booleans, not {value!r}")


def record_text(record: AuditRecord) -> str:
    """A pass record as TOML: its scalar fields, then a table of versions and one of checks,
    every key quoted."""
    lines = [
        f"{name} = {_value(getattr(record, name))}\n"
        for name in ("world", "kind", "identity", "cache_key", "settings")
    ]
    for table in ("versions", "checks"):
        entries: Mapping[str, object] = getattr(record, table)
        lines.append(f"\n[{table}]\n")
        lines += [f"{_quoted(name)} = {_value(entries[name])}\n" for name in sorted(entries)]
    return "".join(lines)


def parse_record(text: str, path: Path) -> AuditRecord:
    """A pass record read back, once it holds exactly the fields of AuditRecord with their
    types."""
    found = tomllib.loads(text)
    scalars = ("world", "kind", "identity", "cache_key", "settings")
    if set(found) != {*scalars, "versions", "checks"}:
        raise ValueError(f"{path} holds the keys {sorted(found)}, not a pass record's")
    for name in scalars:
        if not isinstance(found[name], str):
            raise ValueError(f"{path} holds {name} = {found[name]!r}, not a string")
    versions = found["versions"]
    checks = found["checks"]
    if not isinstance(versions, dict) or not all(
        isinstance(value, str) for value in versions.values()
    ):
        raise ValueError(f"{path} holds versions that are not names of versions")
    if not isinstance(checks, dict) or not all(
        isinstance(value, bool) for value in checks.values()
    ):
        raise ValueError(f"{path} holds checks that are not pass or fail")
    return AuditRecord(
        found["world"],
        found["kind"],
        found["identity"],
        found["cache_key"],
        found["settings"],
        {str(name): str(value) for name, value in versions.items()},
        {str(name): bool(value) for name, value in checks.items()},
    )


class Store:
    """The evaluation store of the world of an identity."""

    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.folder = evaluation_store_dir(identity)

    def write_answer_key(self, key: AnswerKey) -> None:
        frame = key.frame()
        write_parquet(
            self.folder / ANSWER_KEY_FILE,
            pa.Table.from_pandas(frame, schema=ANSWER_KEY_SCHEMA, preserve_index=False),
        )

    def answer_key(self) -> AnswerKey:
        """The stored answer key, reparsed by read_answer_key's parser."""
        rows = pq.read_table(self.folder / ANSWER_KEY_FILE, schema=ANSWER_KEY_SCHEMA).to_pylist()
        if rows and tuple(rows[0]) != COLUMNS:
            raise ValueError(f"The stored answer key has the columns {list(rows[0])}")
        return parse_rows(rows)

    def write_labels(self, labels: Labels) -> None:
        rows = [(mule.account_id, mule.mule_onset_ms, mule.mule_last_ms) for mule in labels.mules]
        frame = pd.DataFrame(rows, columns=LABELS_SCHEMA.names)
        write_parquet(
            self.folder / LABELS_FILE,
            pa.Table.from_pandas(frame, schema=LABELS_SCHEMA, preserve_index=False),
        )

    def labels(self) -> Labels:
        """The stored identified mules, reparsed by read_labels' parser; their identity must be
        the label identity of their rows."""
        rows = pq.read_table(self.folder / LABELS_FILE, schema=LABELS_SCHEMA).to_pylist()
        response: list[dict[str, object]] = [
            {"status": "ok", "version": VERSIONS[READ_LABELS]},
            {"accounts": [{"attributes": row} for row in rows]},
        ]
        mules = read_labels.parse(response, "", max(1, len(rows)))
        return Labels(mules, label_identity(mules))

    def write_fallback(self, keys: pd.DataFrame) -> None:
        write_parquet(
            self.folder / FALLBACK_FILE,
            pa.Table.from_pandas(
                keys.loc[:, list(FALLBACK_SCHEMA.names)],
                schema=FALLBACK_SCHEMA,
                preserve_index=False,
            ),
        )

    def fallback(self) -> pd.DataFrame:
        """The stored test-role fallback list: (t_ms, account_id, weight)."""
        return pq.read_table(self.folder / FALLBACK_FILE, schema=FALLBACK_SCHEMA).to_pandas()

    def write_table(self, name: str, frame: pd.DataFrame) -> None:
        """Write one of the audit's report tables."""
        write_parquet(
            self.folder / TABLES / f"{name}.parquet",
            pa.Table.from_pandas(frame, preserve_index=False),
        )

    def table(self, name: str) -> pd.DataFrame:
        return pq.read_table(self.folder / TABLES / f"{name}.parquet").to_pandas()

    def write_record(self, record: AuditRecord) -> None:
        if record.identity != self.identity:
            raise ValueError(
                f"A pass record of the world {record.identity} goes to its own store, not to "
                f"{self.identity}'s"
            )
        text = record_text(record)
        write_bytes(self.folder / RECORD_FILE, lambda temporary: temporary.write_text(text))

    def record(self) -> AuditRecord | None:
        """The stored pass record, or None when mule audit has not written one; one of another
        world's identity is refused."""
        path = self.folder / RECORD_FILE
        if not path.is_file():
            return None
        found = parse_record(path.read_text(), path)
        if found.identity != self.identity:
            raise ValueError(f"{path} is the pass record of the world {found.identity}")
        return found
