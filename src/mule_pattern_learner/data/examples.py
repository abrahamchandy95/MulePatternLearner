"""The examples and their weights (docs/design.md, "Worlds, examples and weights"). An example is
an (account, Monday t); every one is a scorable account of t.

Training world:
- a positive is an identified mule active at t and scorable at t, weighing 1 over its number
  of positive Mondays, so every mule counts once however long it ran;
- an identified mule's other Mondays, when it is scorable, are dropped, never unlabelled;
- the unlabelled examples of t are the training draw (draw.lowest) of n = min(pool, max(15 x
  positives, 500)) accounts from the pool, every scorable account of t that is no identified
  mule; hidden mules can be drawn. Each weighs the pool over n, so each Monday's draw stands
  for that Monday's scorable population;
- each label's weights are then divided by their mean over the training examples, once.

Selection world, by pu_label alone: every scorable account on every Monday, positive when it is
an identified mule active then, and left out of the ranking when it is an identified mule on
one of its other Mondays. Under the fallback scoring mode: the positives, plus the fallback
sample (draw.sampled) of every other scorable account that is no identified mule, hidden mules
at the same rate, each weighing the sample's weight.

Test world, under whole scoring: every scorable account on every Monday. Its fallback list
depends on the truth, so mule audit builds it, never this module.

The tables are written to the examples folder of the world, keyed by the query versions and
settings they read (examples_key) and by the label identity, each through a temporary file and
a rename.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import CalendarConfig, DataConfig
from ..paths import IDENTITY, examples_dir
from ..tigergraph.queries import LIST_ACCOUNTS, READ_LABELS, VERSIONS
from ..tigergraph.queries.read_labels import IdentifiedMule
from ..tigergraph.world import canonical_hash
from .draw import lowest, sampled
from .labels import active

# The labels of the training examples.
POSITIVE = 1
UNLABELLED = 0
# The tables of the examples folder.
TRAINING_FILE = "training.parquet"
SELECTION_FILE = "selection.parquet"
TEST_FILE = "test.parquet"
TRAINING_SCHEMA = pa.schema(
    [
        ("t_ms", pa.int64()),
        ("account_id", pa.string()),
        ("label", pa.int64()),
        ("weight", pa.float64()),
    ]
)
SELECTION_SCHEMA = pa.schema(
    [
        ("t_ms", pa.int64()),
        ("account_id", pa.string()),
        ("positive", pa.bool_()),
        ("ranked", pa.bool_()),
        ("weight", pa.float64()),
    ]
)
TEST_SCHEMA = pa.schema([("t_ms", pa.int64()), ("account_id", pa.string())])
# The queries whose rows the examples are made of.
EXAMPLE_QUERIES = (LIST_ACCOUNTS, READ_LABELS)


@dataclass(frozen=True)
class Tally:
    """What the training weights and the report need of a Monday: how many accounts are
    scorable then, its positives, by id, and how many identified mules are dropped."""

    t_ms: int
    scorable: int
    positives: tuple[str, ...]
    dropped: int


@dataclass(frozen=True)
class Monday:
    """A Monday's scorable accounts, by id; its positives, the identified mules active then
    and scorable then, by id; and how many are dropped, the identified mules scorable then and
    not active: examples on their other Mondays, so no mule that is not an example that Monday
    (not yet open, or closed) is counted."""

    t_ms: int
    accounts: tuple[str, ...]
    positives: tuple[str, ...]
    dropped: int

    def tally(self) -> Tally:
        return Tally(self.t_ms, len(self.accounts), self.positives, self.dropped)


def monday(t_ms: int, accounts: Collection[str], mules: Sequence[IdentifiedMule]) -> Monday:
    """The Monday t's scorable accounts, positives and dropped identified mules."""
    scorable = set(accounts)
    held = [mule for mule in mules if mule.account_id in scorable]
    positives = sorted(mule.account_id for mule in held if active(mule, t_ms))
    dropped = sum(1 for mule in held if not active(mule, t_ms))
    return Monday(t_ms, tuple(sorted(scorable)), tuple(positives), dropped)


@dataclass(frozen=True)
class Draw:
    """A training Monday's unlabelled draw: the size of its pool and the accounts drawn, in
    the draw's order. Each weighs pool / len(drawn) before the weights are divided by their
    mean."""

    t_ms: int
    pool: int
    drawn: tuple[str, ...]

    @property
    def weight(self) -> float:
        return self.pool / len(self.drawn) if self.drawn else 0.0


def unlabelled(day: Monday, identified: Collection[str], data: DataConfig) -> Draw:
    """The Monday's training draw: n = min(pool, max(per_positive x positives,
    min_unlabelled)) of the scorable accounts that are no identified mule."""
    pool = [account for account in day.accounts if account not in identified]
    n = min(len(pool), max(data.per_positive * len(day.positives), data.min_unlabelled))
    return Draw(day.t_ms, len(pool), tuple(lowest(data.draw_key, pool, day.t_ms, n)))


def training_frame(days: Sequence[Tally], draws: Sequence[Draw]) -> pd.DataFrame:
    """The training examples, by Monday, label (positives first) and account: each positive
    weighing 1 over its mule's positive Mondays and each unlabelled the pool over the draw,
    then each label's weights divided by their mean."""
    counts = Counter(account for day in days for account in day.positives)
    rows: list[tuple[int, str, int, float]] = [
        (day.t_ms, account, POSITIVE, 1 / counts[account])
        for day in days
        for account in day.positives
    ]
    rows += [
        (draw.t_ms, account, UNLABELLED, draw.weight) for draw in draws for account in draw.drawn
    ]
    found = pd.DataFrame(rows, columns=["t_ms", "account_id", "label", "weight"])
    found = found.astype({"t_ms": "int64", "account_id": "str", "label": "int64"})
    found["weight"] = found["weight"].astype("float64")
    for label in (POSITIVE, UNLABELLED):
        chosen = found["label"] == label
        if chosen.any():
            found.loc[chosen, "weight"] /= found.loc[chosen, "weight"].mean()
    found = found.sort_values(["t_ms", "label", "account_id"], ascending=[True, False, True])
    return found.reset_index(drop=True)


def selection_frame(day: Monday, identified: Collection[str], data: DataConfig) -> pd.DataFrame:
    """The Monday's selection keys. Whole: every scorable account, each weighing 1, positive
    when it is a positive, and out of the ranking when it is an identified mule on another of
    its Mondays. Fallback: the positives, weighing 1, and the fallback sample of the scorable
    accounts that are no identified mule, each weighing the sample's weight."""
    positives = set(day.positives)
    if data.scoring == "whole":
        accounts = list(day.accounts)
        positive = [account in positives for account in accounts]
        ranked = [account in positives or account not in identified for account in accounts]
        weights = [1.0] * len(accounts)
    else:
        pool = [account for account in day.accounts if account not in identified]
        drawn = sampled(data.draw_key, pool, day.t_ms, data.fallback_weight)
        accounts = [*day.positives, *drawn]
        positive = [True] * len(day.positives) + [False] * len(drawn)
        ranked = [True] * len(accounts)
        weights = [1.0] * len(day.positives) + [float(data.fallback_weight)] * len(drawn)
    found = pd.DataFrame(
        {
            "t_ms": pd.Series([day.t_ms] * len(accounts), dtype="int64"),
            "account_id": pd.Series(accounts, dtype="str"),
            "positive": pd.Series(positive, dtype="bool"),
            "ranked": pd.Series(ranked, dtype="bool"),
            "weight": pd.Series(weights, dtype="float64"),
        }
    )
    return found.sort_values("account_id").reset_index(drop=True)


def scored_frame(day: Monday) -> pd.DataFrame:
    """The Monday's test keys under whole scoring: every scorable account."""
    return pd.DataFrame(
        {
            "t_ms": pd.Series([day.t_ms] * len(day.accounts), dtype="int64"),
            "account_id": pd.Series(day.accounts, dtype="str"),
        }
    )


def examples_key(calendar: CalendarConfig, data: DataConfig) -> str:
    """The folder under a world's examples that holds the examples of these settings: the hash
    of the versions of list_accounts and read_labels, the calendar and the data settings. The
    transport, subgraph, model and training settings change no example."""
    settings = {
        "versions": {name: VERSIONS[name] for name in EXAMPLE_QUERIES},
        "calendar": {**asdict(calendar), "export_start": calendar.export_start.isoformat()},
        "data": {**asdict(data), "draw_key": data.draw_key.hex()},
    }
    return canonical_hash(settings)


def examples_folder(identity: str, key: str, label_identity: str) -> Path:
    """The examples of a world, for the settings of key and the labels of label_identity."""
    for what, value in (("examples key", key), ("label identity", label_identity)):
        if not IDENTITY.fullmatch(value):
            raise ValueError(f"An {what} is 32 hex digits, got {value!r}")
    return examples_dir(identity) / key / label_identity


class TableWriter:
    """A Parquet table written a frame at a time into a temporary file, which commit renames
    into place and abort deletes, so the table appears whole or not at all."""

    def __init__(self, path: Path, schema: pa.Schema) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.schema = schema
        self.temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        self._writer = pq.ParquetWriter(self.temporary, schema)

    def write(self, frame: pd.DataFrame) -> None:
        self._writer.write_table(
            pa.Table.from_pandas(frame, schema=self.schema, preserve_index=False)
        )

    def commit(self) -> None:
        self._writer.close()
        self.temporary.replace(self.path)

    def abort(self) -> None:
        self._writer.close()
        self.temporary.unlink(missing_ok=True)


def read_table(path: Path, schema: pa.Schema) -> pd.DataFrame:
    """A table of an examples folder."""
    return pq.read_table(path, schema=schema).to_pandas()
