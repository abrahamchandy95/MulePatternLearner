"""The diagnostic feature table on the fake graph, against the batching code.

The feature table's own check: for the same keys, its families equal what
batching.features computes from the training query's rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.batching.features import edge_block, node_matrix
from mule_pattern_learner.config import RunConfig
from mule_pattern_learner.contract.feature_groups import extraction_plan
from mule_pattern_learner.contract.graph_schema import SPLIT_PHASE
from mule_pattern_learner.contract.server import CONTEXT_CONTRACT
from mule_pattern_learner.data.hub_registry import load_hub_registry
from mule_pattern_learner.data.manifest import read_manifest
from mule_pattern_learner.data.splits import sample_keys
from mule_pattern_learner.diagnostics import feature_table
from mule_pattern_learner.diagnostics.feature_table import (
    FEATURE_SPLITS,
    build_feature_table,
    current,
    feature_columns,
    feature_names,
    summarised_edges,
    usable,
)
from mule_pattern_learner.pipeline.connect import context_source
from mule_pattern_learner.testing.builders import (
    ground_truth_rows,
    scope_population,
    unit_config,
)
from mule_pattern_learner.testing.fake_graph import (
    PREPARED_ACCOUNTS,
    FakeTigerGraph,
    prepared_graph,
)
from mule_pattern_learner.tigergraph.context_query import query_context_rows
from mule_pattern_learner.tigergraph.oracle import TigerGraphTruthReader
from mule_pattern_learner.tigergraph.scope import TigerGraphScopeReader

# The account the context query rejects.
REJECTED = "S0003"


@pytest.fixture(scope="module")
def built(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[RunConfig, FakeTigerGraph, dict[str, Any], pd.DataFrame]:
    config = unit_config()
    data = tmp_path_factory.mktemp("data")
    graph, dataset = prepared_graph(data, config, statuses={REJECTED: "history_capacity_exceeded"})
    manifest = read_manifest(dataset)
    contexts = context_source(graph, config)
    try:
        table = build_feature_table(
            config,
            manifest,
            load_hub_registry(dataset, manifest),
            scope=TigerGraphScopeReader(graph),
            truth=TigerGraphTruthReader(graph).read(),
            contexts=contexts,
        )
    finally:
        contexts.close()
    return config, graph, manifest, table


def test_each_split_holds_its_audit_sample_with_truth_and_weights(
    built: tuple[RunConfig, FakeTigerGraph, dict[str, Any], pd.DataFrame],
) -> None:
    config, _, _, table = built
    population = pd.DataFrame(scope_population(PREPARED_ACCOUNTS))
    truth = pd.DataFrame(ground_truth_rows(scope_population(PREPARED_ACCOUNTS)))
    truth = truth.set_index("account_id")
    assert table.split.unique().tolist() == list(FEATURE_SPLITS)
    for split in FEATURE_SPLITS:
        rows = table[table.split == split]
        members = population[population.partition == SPLIT_PHASE[split]]
        # The population is small, so the sample holds every account of the partition.
        assert rows.account_id.tolist() == sorted(members.account_id)
        assert (rows.date == config.dataset.dates[split][0]).all()
        assert rows.is_mule.tolist() == truth.loc[rows.account_id].is_mule.tolist()
        assert rows.ring_id.tolist() == truth.loc[rows.account_id].mule_ring_id.tolist()
        assert np.allclose(rows.weight, 1 / rows.inclusion_probability)
        observed = members.set_index("account_id").observed_positive
        assert rows.revealed.tolist() == observed.loc[rows.account_id].tolist()
    plan = config.feature_plan()
    assert current(table, plan)
    assert set(table.context_contract) == {CONTEXT_CONTRACT}
    stale = table.assign(context_contract="context_old")
    assert not current(stale, plan) and not current(table.iloc[:0], plan)
    # A table of other columns, from older code or another plan, is not current either.
    assert not current(table.drop(columns="messages__peers"), plan)


def test_the_training_families_equal_the_batching_features_of_the_same_keys(
    built: tuple[RunConfig, FakeTigerGraph, dict[str, Any], pd.DataFrame],
) -> None:
    config, graph, manifest, table = built
    plan = config.feature_plan()
    assert feature_columns(table) == feature_names(plan)
    for split in FEATURE_SPLITS:
        rows = usable(table[table.split == split])
        (date,) = config.dataset.dates[split]
        keys = sample_keys(rows, date, manifest)
        assert all(key.visibility_phase == SPLIT_PHASE[split] for key in keys)
        contexts = [
            row
            for start in range(0, len(keys), 8)
            for row in query_context_rows(
                graph,
                keys[start : start + 8],
                plan=extraction_plan(plan),
                sampler=config.sampler,
            )
        ]
        model = node_matrix(contexts, plan, pooled=len(contexts))
        expected = pd.DataFrame(model, columns=[f"model__{n}" for n in plan.node_names])
        assert np.array_equal(rows[expected.columns].to_numpy(), expected.to_numpy(np.float64))
        names = plan.edge_names
        for place, context in enumerate(contexts):
            events = [m for m in context["messages"] if m["event_id"]]
            edges = edge_block(events, plan)["edge"]
            for name in summarised_edges(plan):
                found = edges[:, names.index(name)]
                assert rows[f"messages__mean_{name}"].iloc[place] == pytest.approx(found.mean())
                assert rows[f"messages__max_{name}"].iloc[place] == pytest.approx(found.max())
            assert rows["messages__events"].iloc[place] == len(events)
    # The pools differ between accounts, so the comparison is not of constants.
    assert usable(table)["messages__mean_amount"].nunique() > 1


def test_rejected_accounts_have_no_features(
    built: tuple[RunConfig, FakeTigerGraph, dict[str, Any], pd.DataFrame],
) -> None:
    _, _, _, table = built
    (rejected,) = table.index[table.account_id == REJECTED]
    assert table.rejected.tolist().count(True) == 1 and table.rejected[rejected]
    assert table.loc[rejected, feature_columns(table)].isna().all()
    assert REJECTED not in set(usable(table).account_id)


def test_a_split_with_several_cutoffs_is_refused(tmp_path: Path) -> None:
    config = unit_config(dataset={"dates": {"train": ["2024-05-01", "2024-07-01"]}})
    with pytest.raises(ValueError, match="one train cutoff, not 2"):
        feature_table.split_sample(
            TigerGraphScopeReader(FakeTigerGraph()), config, "train", pd.DataFrame()
        )
