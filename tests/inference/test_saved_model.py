"""Models saved before the layered restructure load and score as they did.

The fixtures were written by commit 9925f69, whose training and checkpoint code is the
code the saved models on the CUDA host come from. They are the built-in run, the tabular
variant and the no_fourier variant, each trained for two steps with 8 hidden units on the
fake graph. Their configurations hold keys that the restructure retires, such as
context_storage, evaluation_protocol, label_policy and variant. The literals are the
scores that commit's Predictor gave eight test accounts; this code must give the
same scores. Floating point rounding differs between machines, hence the tolerance.

`dataset/` is the dataset that commit prepared for the built-in model: its manifest
records preparation keys and a query file the restructure retires. The model still
scores it as it did, and today's code prepares the same accounts and labels from the
same settings.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
import shutil
from typing import Any

import pandas as pd
import pytest
import torch

from mule_pattern_learner.config import BUILT_IN_SAMPLER, DEFAULT_CONFIG, RunConfig
from mule_pattern_learner.contract.clock import cutoff_ms
from mule_pattern_learner.contract.feature_groups import (
    CORE_GROUPS,
    FeaturePlan,
    contract_fingerprint,
    extraction_plan,
)
from mule_pattern_learner.contract.fingerprints import fingerprint
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import PoolPlan
from mule_pattern_learner.data.contexts import ContextSource, build_context_source
from mule_pattern_learner.data.hub_registry import load_hub_registry
from mule_pattern_learner.data.manifest import dataset_id, load_prepared, read_manifest
from mule_pattern_learner.data.preparation import prepare
from mule_pattern_learner.data.splits import eligible_mask, sample_keys
from mule_pattern_learner.inference.predictor import Predictor
from mule_pattern_learner.inference.saved_model import (
    REQUIRED_SETTINGS,
    SavedModel,
    saved_run_config,
)
from mule_pattern_learner.paths import DATA_DIR, DatasetPaths, RunPaths
from mule_pattern_learner.testing.builders import neighbourhood, scope_population
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffs
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubs
from mule_pattern_learner.tigergraph.labels import TigerGraphObservedLabels
from mule_pattern_learner.tigergraph.scope import TigerGraphScope
from mule_pattern_learner.training.trainer import train

FIXTURES = Path(__file__).parent / "fixtures" / "saved_models"
RELATIVE = 1e-5
# Test-partition accounts of the fake scope, at the fake graph's test cutoff sequence.
ACCOUNTS = ("S0004", "S0009", "S0014", "S0019", "S0024", "S0029", "S0034", "S0039")
SCORES = {
    "built_in": [
        0.4991507417307398,
        0.5007370076392473,
        0.5002217446308491,
        0.4998289155840587,
        0.500896578159351,
        0.5003809564941073,
        0.4991540060069826,
        0.49990310473488864,
    ],
    "tabular": [
        0.5340324079266534,
        0.5216267707009064,
        0.5216267707009064,
        0.5340324079266534,
        0.5216267707009064,
        0.5340324079266534,
        0.5340324079266534,
        0.5216267707009064,
    ],
    "no_fourier": [
        0.568559441682482,
        0.5614372077809549,
        0.5589734916980292,
        0.5617949737483596,
        0.5556401659101458,
        0.5661534213136434,
        0.5634052456284869,
        0.5619818620262174,
    ],
}

# The settings the fixtures were trained and prepared with, over the built-in run, and
# the source id of the fake graph's data they name.
CHANGES: dict[str, Any] = {
    "dataset": {"seed_limits": {"train": 40, "validation": 16, "test": 16}},
    "model": {"hidden": 8, "heads": 2, "dropout": 0.0},
    "training": {"epochs": 1, "steps_per_epoch": 2, "batch_size": 8},
    "runtime": {"device": "cpu", "threads": 1},
}
SOURCE = "load_fixture"
# What that commit's inference.score gave the dataset's test accounts with built_in.pt: every
# eligible test account at the test cutoff, scored with the dataset's hub registry.
DATASET_SCORES = {
    "S0009": 0.5007370076392473,
    "S0014": 0.5002217446308491,
    "S0019": 0.4998289155840587,
    "S0029": 0.5003809564941073,
    "S0034": 0.4991540060069826,
    "S0044": 0.500269264422404,
    "S0049": 0.501590554359506,
    "S0059": 0.4995948039882288,
    "S0079": 0.5013632801106169,
    "S0084": 0.5003109690332304,
    "S0109": 0.5015115724461501,
    "S0119": 0.4992631679146766,
    "S0129": 0.49867206472206704,
    "S0154": 0.5018713894459883,
    "S0159": 0.5004211802902023,
    "S0174": 0.4990827842659545,
    "S0179": 0.5004852529022983,
    "S0189": 0.5004320097012165,
    "S0194": 0.4994300410757545,
}


@pytest.mark.parametrize("name", sorted(SCORES))
def test_models_saved_before_the_restructure_score_as_they_did(name: str) -> None:
    saved = SavedModel.load(FIXTURES / f"{name}.pt")
    keys = [
        ContextKey("Account", account, 103, cutoff_ms("2025-01-01"), saved.config.scope.id, 3)
        for account in ACCOUNTS
    ]
    contexts = fixture_source(saved.config)
    try:
        (frame,), rejected = Predictor(saved, contexts).score_keys([keys])
    finally:
        contexts.close()
    assert rejected == []
    assert frame.account_id.tolist() == list(ACCOUNTS)
    for have, want in zip(frame.score.tolist(), SCORES[name], strict=True):
        assert math.isclose(have, want, rel_tol=RELATIVE), (name, have, want)


def test_the_built_in_inputs_keep_their_recorded_fingerprints() -> None:
    assert contract_fingerprint() == (
        "530e46c91b07b254d38722e57117b917761d2b68176e7eb1cb5e2e9de32307da"
    )
    assert FeaturePlan().fingerprint() == (
        "9cf8c7606ce403da4e74757799aa77cfa6fdcd0736686b23c38bce592fd08238"
    )
    assert FeaturePlan(architecture="summary").fingerprint() == (
        "77c38ffb11af48f5485d4e1efc9cd92e826990d3650738864b4a18cfc478be58"
    )


def test_models_whose_columns_moved_are_refused(tmp_path: Path) -> None:
    # Columns follow registry order now; before, the window groups' columns sat elsewhere.
    plan = FeaturePlan((*CORE_GROUPS, "rolling_windows"))
    recorded = fingerprint(
        {"contract": contract_fingerprint(), "groups": sorted(plan.groups), "architecture": "split"}
    )
    saved = SavedModel(tmp_path / "model.pt", {"input_fingerprint": recorded})
    with pytest.raises(ValueError, match="input groups"):
        saved.check_inputs(plan)


def fixture_source(config: RunConfig) -> ContextSource:
    executor = FakeTigerGraph(factory=neighbourhood)
    return build_context_source(
        TigerGraphContextFetcher(executor),
        extraction_plan(config.feature_plan()),
        config.sampler,
        config.transport,
    )


def test_the_dataset_prepared_before_the_restructure_scores_as_it_did(tmp_path: Path) -> None:
    dataset = DatasetPaths(tmp_path / "prepared")
    shutil.copytree(FIXTURES / "dataset", dataset.root)
    saved = SavedModel.load(FIXTURES / "built_in.pt")
    assert saved.config == DEFAULT_CONFIG.with_changes(CHANGES)
    # Every eligible test account at the test cutoff, with the dataset's hub registry.
    manifest, accounts = load_prepared(dataset)
    saved.check_dataset(dataset)
    date = "2025-01-01"
    accounts = accounts[eligible_mask(accounts, "test", date)]
    contexts = fixture_source(saved.config)
    try:
        predictor = Predictor(saved, contexts, hubs=load_hub_registry(dataset, manifest))
        size = predictor.batch_size
        frames, rejected = predictor.score_keys(
            sample_keys(accounts.iloc[start : start + size], date, manifest)
            for start in range(0, len(accounts), size)
        )
    finally:
        contexts.close()
    frame = pd.concat(frames, ignore_index=True)
    assert rejected == [] and frame.account_id.tolist() == list(DATASET_SCORES)
    for account, have in zip(frame.account_id, frame.score, strict=True):
        assert math.isclose(have, DATASET_SCORES[account], rel_tol=RELATIVE), account


def test_the_same_settings_prepare_the_dataset_of_the_old_code(tmp_path: Path) -> None:
    config = DEFAULT_CONFIG.with_changes(CHANGES)
    executor = FakeTigerGraph(factory=neighbourhood, population=scope_population(200))
    dataset = DatasetPaths(tmp_path / "dataset")
    prepare(
        config,
        SOURCE,
        dataset,
        {"Account": 200},
        TigerGraphObservedLabels(),
        scope=TigerGraphScope(executor),
        cutoffs=TigerGraphCutoffs(executor),
        hub_reader=TigerGraphHubs(executor),
    )
    fixture = DatasetPaths(FIXTURES / "dataset")
    for have, want in (
        (dataset.accounts, fixture.accounts),
        (dataset.observed_labels, fixture.observed_labels),
    ):
        pd.testing.assert_frame_equal(pd.read_parquet(have), pd.read_parquet(want))
    # And a new model trains on it, which names the dataset by its dataset id.
    run = RunPaths(tmp_path / "run")
    result = train(config, dataset, run, contexts=fixture_source(config))
    assert result["status"] == "complete"
    saved = SavedModel.load(run.model)
    assert saved.payload["format"] == SavedModel.FORMAT == 1
    recorded = read_manifest(dataset)["source"]["dataset_id"]
    assert saved.dataset_id == recorded == dataset_id(SOURCE, config) == result["dataset_id"]
    assert saved.dataset(tmp_path) == DatasetPaths.of(recorded, tmp_path)
    assert saved.dataset() == DatasetPaths.of(recorded, DATA_DIR)
    assert "dataset" not in saved.payload and "training_protocol" not in saved.payload


def test_a_model_of_another_format_is_refused_and_old_ones_keep_their_directory(
    tmp_path: Path,
) -> None:
    old = SavedModel.load(FIXTURES / "built_in.pt")
    assert "format" not in old.payload and old.dataset_id is None
    # A model saved before FORMAT 1 recorded its dataset's directory.
    recorded = old.payload["dataset"]
    assert old.dataset(tmp_path) == DatasetPaths(Path(recorded))
    newer = tmp_path / "newer.pt"
    torch.save({**old.payload, "format": SavedModel.FORMAT + 1}, newer)
    with pytest.raises(ValueError, match="format 2; this code reads 1"):
        SavedModel.load(newer)


# The fan-outs and sampler table every configuration the old code saved holds, here the
# built-in run's.
OLD_SAMPLING: dict[str, Any] = {
    "fanouts": [16, 4],
    "sampler": {**asdict(BUILT_IN_SAMPLER.roots), "children": asdict(BUILT_IN_SAMPLER.children)},
}


def old(**settings: Any) -> dict[str, Any]:
    """An old flat configuration: OLD_SAMPLING with settings."""
    return {**OLD_SAMPLING, **settings}


def test_old_configurations_convert_through_the_key_table() -> None:
    built_in = saved_run_config(old())
    assert built_in == DEFAULT_CONFIG
    # Settings of removed paths were saved with the one value that remains.
    retired = {
        "context_storage": "stream",
        "evaluation_protocol": "strict_inductive",
        "label_policy": "graph_observed",
        "observed_labels": None,
        "extraction_groups": ["entity_meta", "rolling_windows"],
        "fanouts": [16, 4],
        "sampler": {**asdict(BUILT_IN_SAMPLER.roots), "policy": "resample"},
        "dataset_id": "load_fixture",
        "prepared_id": None,
        "prepare_batch_size": 16,
    }
    converted = saved_run_config(retired)
    assert converted.sampler.roots == BUILT_IN_SAMPLER.roots
    assert converted.sampler.children == replace(BUILT_IN_SAMPLER.roots, associations=0)
    assert replace(converted, sampler=DEFAULT_CONFIG.sampler) == DEFAULT_CONFIG
    for key, value, message in (
        ("context_storage", "sqlite", "context_storage = 'sqlite' is no longer supported"),
        ("evaluation_protocol", "shared_history", "evaluation_protocol = 'shared_history'"),
        ("label_policy", "observed", "label_policy = 'observed' is no longer supported"),
        ("observed_labels", "x.parquet", "observed_labels = 'x.parquet' is no longer supported$"),
        ("variant", "wide", "variant = 'wide' is no longer supported"),
        ("stage", "offline", "Unknown saved configuration key"),
    ):
        with pytest.raises(ValueError, match=message):
            saved_run_config(old(**{key: value}))
    with pytest.raises(ValueError, match="sampler.policy = 'recent' is no longer supported"):
        saved_run_config(old(sampler={"policy": "recent"}))
    # The old code gave absent fan-outs and sampler pools values of its own, which
    # nothing else would catch, so a table without them is refused.
    for missing in REQUIRED_SETTINGS:
        partial = {k: v for k, v in old().items() if k != missing}
        with pytest.raises(ValueError, match=f"names no {missing}; the code that saved it"):
            saved_run_config(partial)
    with pytest.raises(ValueError, match="names no fanouts or sampler"):
        saved_run_config({"fanouts": None})
    # The model variants became settings.
    assert saved_run_config(old(variant="temporal")) == built_in
    tabular = saved_run_config(old(variant="tabular"))
    assert tabular == replace(built_in, model=replace(built_in.model, architecture="summary"))
    no_fourier = saved_run_config(old(variant="no_fourier")).features
    assert no_fourier == tuple(g for g in built_in.features if g != "time_encoding")
    # A [sampler] table's absent pool keys were those of PoolPlan(), and its children
    # pool the roots pool without associations, changed by [sampler.children].
    partial = saved_run_config(old(sampler={"recent": 4, "children": {"older": 1}})).sampler
    assert partial.roots == PoolPlan(recent=4)
    assert partial.children == PoolPlan(recent=4, older=1, associations=0)
    assert partial.relation_fanouts == BUILT_IN_SAMPLER.relation_fanouts
    # The reservoir seed and the reveal salt defaulted to the training seed.
    seeded = saved_run_config(old(seed=7, reveal_salt=None, reveal_per_split=None))
    assert (seeded.training.seed, seeded.dataset.seed, seeded.scope.reveal_salt) == (7, 7, 7)
    assert seeded.scope.reveal_per_split == built_in.scope.reveal_per_split
    pinned = saved_run_config(old(seed=7, cohort_seed=3, reveal_salt=5))
    assert (pinned.dataset.seed, pinned.scope.reveal_salt) == (3, 5)
    # A null limit saved on purpose stays null.
    unlimited = saved_run_config(old(steps_per_epoch=None, evaluation_unlabeled_limit=None))
    assert unlimited.training.steps_per_epoch is None
    assert unlimited.training.proxy_unlabeled_limit is None
    # A model saved since the typed configuration holds RunConfig.to_dict().
    assert saved_run_config(tabular.to_dict()) == tabular
