"""The diagnostic feature table: one row per sampled account of each split, at its cutoff.

Each split's sample is its audit sample (evaluation.sample.audit_sample): every mule of
the split's frozen population and AUDIT_NEGATIVES uniform non-mules drawn with the
dataset's split seed, each with its inclusion probability. The validation and test rows
are therefore the accounts every audit of the dataset's runs scores, so a baseline fitted
here and a run's audit rank the same accounts; train is sampled the same way.

Each account is read at its split's cutoff, with its split's visibility, through the
training context query and the dataset's context source, as the model reads the root:
its own inputs as batching.features.node_matrix builds them (the `model` family: entity
flags, the hub flag and the pool counts, log1p where the model applies it) and its hop-1
candidate pool as batching.features.edge_block encodes it (the `messages` family: the
mean and maximum of each edge input over the pool's payments, and counts by relation
and rail). An account TigerGraph rejects keeps its row, marked rejected, without
features.

Ground truth chooses the sample and labels the rows, for analysis only; nothing here
feeds a model. The table reads the graph through ports (data.contexts.ContextReader and
data.ports.ScopeReader) and truth as the table evaluation.truth.checked_truth checks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import FEATURE_TABLE_COLUMNS
from ..batching.features import edge_block, node_matrix
from ..config import RunConfig
from ..contract.bounds import BATCH_CONTEXTS
from ..contract.feature_groups import FeaturePlan
from ..contract.graph_schema import PAYMENT_RELATIONS, RAILS, ContextKey
from ..contract.server import CONTEXT_CONTRACT
from ..data.contexts import ContextReader
from ..data.hub_registry import HubRegistry
from ..data.ports import ScopeReader
from ..data.splits import sample_keys
from ..evaluation.audit import audit_population
from ..evaluation.sample import audit_sample
from ..evaluation.truth import checked_truth
from ..runtime.progress import emit

# The splits the table samples, each at its own cutoff.
FEATURE_SPLITS = ("train", "validation", "test")
# The feature families, in column order.
FAMILIES = ("model", "messages")
DAY_MS = 86_400_000


def column(family: str, name: str) -> str:
    """A feature's column in the table: <family>__<name>."""
    return f"{family}__{name}"


def family_of(name: str) -> str:
    """The family of a feature column."""
    return name.split("__", 1)[0]


def feature_columns(frame: pd.DataFrame, families: Iterable[str] = FAMILIES) -> list[str]:
    """The table's feature columns of these families, in table order."""
    chosen = set(families)
    return [name for name in frame.columns if "__" in str(name) and family_of(str(name)) in chosen]


def split_sample(
    scope: ScopeReader, config: RunConfig, split: str, truth: pd.DataFrame
) -> tuple[str, pd.DataFrame]:
    """A split's cutoff date and its audit sample: account ids, split, revealed and truth."""
    dates = config.dataset.dates[split]
    if len(dates) != 1:
        raise ValueError(f"The feature table needs one {split} cutoff, not {len(dates)}")
    (date,) = dates
    population = audit_population(scope, split, date)
    if not len(population):
        raise ValueError(f"No eligible accounts in the {split} population")
    return date, audit_sample(population, truth, seed=config.dataset.split_seed)


def model_features(rows: Sequence[dict[str, Any]], plan: FeaturePlan) -> dict[str, np.ndarray]:
    """The root's own model inputs of each row (batching.features.node_matrix), by column."""
    values = node_matrix(rows, plan, pooled=len(rows)).astype(np.float64)
    return {column("model", name): values[:, j] for j, name in enumerate(plan.node_names)}


# The edge inputs the messages family summarises: every one of the plan but the Fourier
# columns, which encode the ages and gaps summarised as days instead, and is_event.
def summarised_edges(plan: FeaturePlan) -> list[str]:
    return [n for n in plan.edge_names if "_fourier_" not in n and n != "is_event"]


def message_features(
    row: dict[str, Any], key: ContextKey, plan: FeaturePlan, hubs: HubRegistry
) -> dict[str, float]:
    """The root's hop-1 candidate pool as its edge inputs encode it, summarised.

    The mean and maximum of each edge input (batching.features.edge_block) over the
    pool's payments, the youngest and median payment age in days, the counts of payments
    by relation and rail, and the pool's distinct peers, how
    many of them the hub registry withholds at the root's cutoff, and the shares of
    payments with an external or a deposit peer.
    """
    messages = row["messages"]
    events = [m for m in messages if m["event_id"]]
    names = plan.edge_names
    edges = edge_block(events, plan)["edge"] if events else np.zeros((0, len(names)))
    values: dict[str, float] = {"events": len(events), "associations": len(messages) - len(events)}
    for name in summarised_edges(plan):
        found = edges[:, names.index(name)].astype(np.float64)
        values[f"mean_{name}"] = float(found.mean()) if len(found) else 0.0
        values[f"max_{name}"] = float(found.max()) if len(found) else 0.0
    ages = np.asarray([m["age_ms"] for m in events], dtype=np.float64) / DAY_MS
    values["min_age_days"] = float(ages.min()) if len(ages) else 0.0
    values["median_age_days"] = float(np.median(ages)) if len(ages) else 0.0
    for relation in PAYMENT_RELATIONS:
        values[f"relation_{relation}"] = sum(m["relation"] == relation for m in events)
    for rail in RAILS:
        values[f"rail_{rail}"] = sum(m["rail"] == rail for m in events)
    peers = {(m["node_type"], m["node_id"]) for m in messages}
    values["peers"] = len(peers)
    values["hub_peers"] = sum(
        hubs.is_stub(typ, node, key.cutoff_seq, key.visibility_phase) for typ, node in peers
    )
    for flag in ("peer_external", "peer_deposit"):
        values[f"{flag}_share"] = float(np.mean([bool(m[flag]) for m in events])) if events else 0.0
    return {column("messages", name): value for name, value in values.items()}


def training_rows(contexts: ContextReader, keys: list[ContextKey]) -> list[dict[str, Any] | None]:
    """The training query's row of each key, None where TigerGraph rejected it."""
    rows: list[dict[str, Any] | None] = []
    for start in range(0, len(keys), BATCH_CONTEXTS):
        rows += contexts.fetch(keys[start : start + BATCH_CONTEXTS], hop=1)
    return rows


def feature_names(plan: FeaturePlan) -> list[str]:
    """The table's feature columns, family by family, in the order they are written."""
    edges = summarised_edges(plan)
    messages = [
        "events",
        "associations",
        *(f"{stat}_{name}" for name in edges for stat in ("mean", "max")),
        "min_age_days",
        "median_age_days",
        *(f"relation_{relation}" for relation in PAYMENT_RELATIONS),
        *(f"rail_{rail}" for rail in RAILS),
        "peers",
        "hub_peers",
        "peer_external_share",
        "peer_deposit_share",
    ]
    return [
        *(column("model", name) for name in plan.node_names),
        *(column("messages", name) for name in messages),
    ]


def split_features(
    keys: list[ContextKey],
    trained: list[dict[str, Any] | None],
    plan: FeaturePlan,
    hubs: HubRegistry,
) -> tuple[pd.DataFrame, np.ndarray]:
    """The feature columns of a split's accounts and which of them were rejected.

    A rejected account's features are missing.
    """
    names = feature_names(plan)
    values = np.full((len(keys), len(names)), np.nan)
    kept = [(index, row) for index, row in enumerate(trained) if row is not None]
    if kept:
        model = model_features([row for _, row in kept], plan)
        for place, (index, row) in enumerate(kept):
            record = {name: float(found[place]) for name, found in model.items()}
            record |= message_features(row, keys[index], plan, hubs)
            values[index] = [record[name] for name in names]
    rejected = np.ones(len(keys), dtype=bool)
    rejected[[index for index, _ in kept]] = False
    return pd.DataFrame(values, columns=names), rejected


def build_feature_table(
    config: RunConfig,
    manifest: dict[str, Any],
    hubs: HubRegistry,
    *,
    scope: ScopeReader,
    truth: pd.DataFrame,
    contexts: ContextReader,
) -> pd.DataFrame:
    """The feature table of a prepared dataset (see the module docstring).

    ``manifest`` and ``hubs`` are the dataset's, so the keys and the withheld peers are
    those training uses; ``contexts`` is a context source of the configuration on the
    dataset's frozen source, which the caller opens and closes.
    """
    answer = checked_truth(truth)
    plan = config.feature_plan()
    frames: list[pd.DataFrame] = []
    for split in FEATURE_SPLITS:
        date, sample = split_sample(scope, config, split, answer)
        keys = sample_keys(sample, date, manifest)
        trained = training_rows(contexts, keys)
        features, rejected = split_features(keys, trained, plan, hubs)
        meta = pd.DataFrame(
            {
                "account_id": sample.account_id.astype(str),
                "split": split,
                "date": date,
                "is_mule": sample.is_mule.astype("int64"),
                "revealed": sample.revealed.astype(bool),
                "ring_id": sample.ring_id.astype("int64"),
                "label_source": sample.label_source.astype(str),
                "inclusion_probability": sample.inclusion_probability.astype(np.float64),
                "weight": 1 / sample.inclusion_probability.astype(np.float64),
                "rejected": rejected,
                "context_contract": CONTEXT_CONTRACT,
            }
        )
        frames.append(pd.concat([meta[list(FEATURE_TABLE_COLUMNS)], features], axis=1))
        emit(
            {
                "event": "feature_table",
                "split": split,
                "date": date,
                "accounts": len(sample),
                "mules": int(sample.is_mule.sum()),
                "rejected": int(rejected.sum()),
            }
        )
    return pd.concat(frames, ignore_index=True)


def current(frame: pd.DataFrame, plan: FeaturePlan) -> bool:
    """Whether a feature table is the one this code would read for a feature plan.

    It must have been read with this code's query text and hold exactly the columns
    this code writes for the plan; on the frozen source such a table is read again
    the same, so `mule diagnose` keeps it.
    """
    return bool(
        len(frame)
        and frame.context_contract.eq(CONTEXT_CONTRACT).all()
        and feature_columns(frame) == feature_names(plan)
    )


def usable(frame: pd.DataFrame) -> pd.DataFrame:
    """The accounts with features: those TigerGraph did not reject."""
    return frame[~frame.rejected.astype(bool)].reset_index(drop=True)
