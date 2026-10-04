"""Models this code saves load and score as they did when they were saved.

The fixtures are three models and the dataset they were trained on, which write_fixtures
wrote with this code: the baseline, no_attention and drop_time_encoding variants of the
built-in run, each trained for two steps with 8 hidden units on the fake graph. The
literals are the scores the fixtures gave when they were written, and loading them must
give the same scores, so a change to what model.pt holds or to how a model is built from
it shows here. Floating point rounding differs between machines, hence the tolerance.

A change that refuses the fixtures or changes their scores on purpose (a new
SavedModel.FORMAT, other model inputs or modules, or another text of a query file the
dataset records the hash of) writes them again with
`python tests/inference/test_saved_model.py`, which prints the new literals, and says so.
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd
import pytest
import torch

from mule_pattern_learner.config import DEFAULT_CONFIG, RunConfig
from mule_pattern_learner.contract.clock import cutoff_ms
from mule_pattern_learner.contract.feature_groups import (
    FeaturePlan,
    contract_fingerprint,
    extraction_plan,
)
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.data.contexts import ContextSource, build_context_source
from mule_pattern_learner.data.hub_registry import load_hub_registry
from mule_pattern_learner.data.manifest import dataset_id, load_prepared, read_manifest
from mule_pattern_learner.data.preparation import prepare
from mule_pattern_learner.data.splits import eligible_mask, sample_keys
from mule_pattern_learner.experiments.variants import VARIANTS
from mule_pattern_learner.inference.predictor import Predictor
from mule_pattern_learner.inference.saved_model import SavedModel
from mule_pattern_learner.paths import DATA_DIR, DatasetPaths, RunPaths
from mule_pattern_learner.testing.builders import neighbourhood, scope_population
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffReader
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubReader
from mule_pattern_learner.tigergraph.labels import TigerGraphObservedLabelReader
from mule_pattern_learner.tigergraph.scope import TigerGraphScopeReader
from mule_pattern_learner.training.trainer import train

FIXTURES = Path(__file__).parent / "fixtures" / "saved_models"
RELATIVE = 1e-5
# Test-partition accounts of the fake scope, at the fake graph's test cutoff sequence.
ACCOUNTS = ("S0004", "S0009", "S0014", "S0019", "S0024", "S0029", "S0034", "S0039")
SCORES = {
    "baseline": [
        0.4999965010210873,
        0.5017275023221025,
        0.50141239267892,
        0.500041031744239,
        0.5019356160122372,
        0.5000090003013601,
        0.4989037610460633,
        0.49968605399532245,
    ],
    "no_attention": [
        0.5340324079266534,
        0.5216267892925102,
        0.5216267892925102,
        0.5340324079266534,
        0.5216267892925102,
        0.5340324079266534,
        0.5340324079266534,
        0.5216267892925102,
    ],
    "drop_time_encoding": [
        0.5679751354264553,
        0.5621443056969948,
        0.5610226289089402,
        0.5622329530671222,
        0.5598875356025597,
        0.5657153681341759,
        0.5625862853374058,
        0.5623272956173293,
    ],
}

# The settings the fixtures were prepared and trained with, over the built-in run and
# then each variant's change, and the source id of the fake graph's data they name.
CHANGES: dict[str, Any] = {
    "dataset": {"seed_limits": {"train": 40, "validation": 16, "test": 16}},
    "model": {"hidden": 8, "heads": 2, "dropout": 0.0},
    "training": {"epochs": 1, "steps_per_epoch": 2, "batch_size": 8},
    "runtime": {"device": "cpu", "threads": 1},
}
BASE = DEFAULT_CONFIG.with_changes(CHANGES)
SEED = 42
SOURCE = "load_fixture"
# What the baseline fixture gave the dataset's test accounts: every eligible test
# account at the test cutoff, scored with the dataset's hub registry.
DATASET_SCORES = {
    "S0009": 0.5017275023221025,
    "S0014": 0.50141239267892,
    "S0019": 0.500041031744239,
    "S0029": 0.500009000999852,
    "S0034": 0.4989037610460633,
    "S0044": 0.49925061045448016,
    "S0049": 0.5007689310928054,
    "S0059": 0.49912746804860086,
    "S0079": 0.5016716895036526,
    "S0084": 0.5006339610649492,
    "S0109": 0.5007848684283,
    "S0119": 0.5004635377390388,
    "S0129": 0.4995606361156572,
    "S0154": 0.5016560597556299,
    "S0159": 0.49913571395380213,
    "S0174": 0.4999670535326481,
    "S0179": 0.5018592582069735,
    "S0189": 0.5006520045950844,
    "S0194": 0.49986877315979006,
}


def fixture_config(name: str) -> RunConfig:
    return VARIANTS[name].config(BASE, SEED)


def fixture_source(config: RunConfig) -> ContextSource:
    executor = FakeTigerGraph(factory=neighbourhood)
    return build_context_source(
        TigerGraphContextFetcher(executor),
        extraction_plan(config.feature_plan()),
        config.sampler,
        config.transport,
    )


def prepare_fixture_dataset(dataset: DatasetPaths) -> None:
    """Prepare the fixtures' dataset from the fake scope of 200 accounts."""
    executor = FakeTigerGraph(factory=neighbourhood, population=scope_population(200))
    prepare(
        BASE,
        SOURCE,
        dataset,
        {"Account": 200},
        TigerGraphObservedLabelReader(),
        scope=TigerGraphScopeReader(executor),
        cutoffs=TigerGraphCutoffReader(executor),
        hub_reader=TigerGraphHubReader(executor),
    )


def write_fixtures(root: Path) -> None:
    """Prepare the fixtures' dataset in root/dataset and save each model as root/<name>.pt."""
    dataset = DatasetPaths(root / "dataset")
    prepare_fixture_dataset(dataset)
    with tempfile.TemporaryDirectory() as runs:
        for name in SCORES:
            config = fixture_config(name)
            run = RunPaths(Path(runs) / name)
            result = train(config, dataset, run, contexts=fixture_source(config))
            assert result["status"] == "complete", result
            shutil.copyfile(run.model, root / f"{name}.pt")


def fixture_scores(saved: SavedModel) -> list[float]:
    """The model's scores of ACCOUNTS at the test cutoff."""
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
    return frame.score.tolist()


def dataset_scores(saved: SavedModel, dataset: DatasetPaths) -> dict[str, float]:
    """The model's scores of every eligible test account, with the dataset's hub registry."""
    manifest, accounts = read_manifest(dataset), pd.read_parquet(dataset.accounts)
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
    assert rejected == []
    frame = pd.concat(frames, ignore_index=True)
    return dict(zip(frame.account_id.tolist(), frame.score.tolist(), strict=True))


@pytest.mark.parametrize("name", list(SCORES))
def test_saved_models_score_as_they_did(name: str) -> None:
    saved = SavedModel.load(FIXTURES / f"{name}.pt")
    assert saved.payload["format"] == SavedModel.FORMAT == 1
    assert saved.config == fixture_config(name)
    for have, want in zip(fixture_scores(saved), SCORES[name], strict=True):
        assert math.isclose(have, want, rel_tol=RELATIVE), (name, have, want)


def test_the_built_in_inputs_keep_their_recorded_fingerprints() -> None:
    # What a model trained now records.
    assert contract_fingerprint() == (
        "51bec43d1cf620b40dac7bee957f129d35f371eef4508f549b473d4768a29b28"
    )
    assert FeaturePlan().fingerprint() == (
        "59c1f4e2696fe2665eee40507c7d73e8a4d231643a1e05f63d1e653a7363205c"
    )
    assert FeaturePlan(architecture="summary").fingerprint() == (
        "a768f5b406749101d592e5d5599261ef77a960e713e09395c0d86c6913861d1f"
    )


def test_a_model_of_another_contract_is_refused() -> None:
    saved = SavedModel.load(FIXTURES / "baseline.pt")
    assert saved.payload["contract"] == contract_fingerprint()
    assert saved.payload["input_fingerprint"] == FeaturePlan().fingerprint()
    saved.check_contract()
    saved.check_inputs(saved.config.feature_plan())
    foreign = SavedModel(Path("model.pt"), {**saved.payload, "contract": "another"})
    with pytest.raises(ValueError, match="contract differs"):
        foreign.check_contract()


def test_a_model_that_names_a_group_training_does_not_read_is_refused() -> None:
    # An analytics group is unknown to training, so such a model is refused, not misread.
    payload = SavedModel.load(FIXTURES / "baseline.pt").payload
    config = payload["config"]
    saved = SavedModel(
        Path("model.pt"),
        {**payload, "config": {**config, "features": [*config["features"], "rolling_windows"]}},
    )
    with pytest.raises(ValueError, match="unknown feature groups"):
        _ = saved.config


def test_the_fixture_dataset_scores_as_it_did(tmp_path: Path) -> None:
    dataset = DatasetPaths(tmp_path / "prepared")
    shutil.copytree(FIXTURES / "dataset", dataset.root)
    # The fixture passes the gate every run's dataset passes, its query hashes included.
    load_prepared(dataset)
    saved = SavedModel.load(FIXTURES / "baseline.pt")
    saved.check_dataset(dataset)
    assert saved.dataset_id == read_manifest(dataset)["source"]["dataset_id"]
    scores = dataset_scores(saved, dataset)
    assert list(scores) == list(DATASET_SCORES)
    for account, have in scores.items():
        assert math.isclose(have, DATASET_SCORES[account], rel_tol=RELATIVE), account


def test_this_code_writes_the_fixtures_again(tmp_path: Path) -> None:
    # The same settings prepare the same accounts, labels and hubs, and train models of
    # the fixtures' layout, which name the dataset by its dataset id.
    write_fixtures(tmp_path)
    dataset, fixture = DatasetPaths(tmp_path / "dataset"), DatasetPaths(FIXTURES / "dataset")
    for have, want in (
        (dataset.accounts, fixture.accounts),
        (dataset.observed_labels, fixture.observed_labels),
        (dataset.hubs, fixture.hubs),
    ):
        pd.testing.assert_frame_equal(pd.read_parquet(have), pd.read_parquet(want))
    recorded = read_manifest(dataset)["source"]["dataset_id"]
    assert recorded == dataset_id(SOURCE, BASE) == read_manifest(fixture)["source"]["dataset_id"]
    for name in SCORES:
        saved = SavedModel.load(tmp_path / f"{name}.pt")
        kept = SavedModel.load(FIXTURES / f"{name}.pt")
        assert saved.payload.keys() == kept.payload.keys(), name
        assert saved.config == kept.config and saved.dataset_id == recorded, name
        assert saved.dataset(tmp_path) == DatasetPaths.of(recorded, tmp_path)
        assert saved.dataset() == DatasetPaths.of(recorded, DATA_DIR)


def test_a_model_of_another_format_or_of_none_is_refused(tmp_path: Path) -> None:
    payload = SavedModel.load(FIXTURES / "baseline.pt").payload
    newer, unmarked = tmp_path / "newer.pt", tmp_path / "unmarked.pt"
    torch.save({**payload, "format": SavedModel.FORMAT + 1}, newer)
    with pytest.raises(ValueError, match="format 2; this code reads 1"):
        SavedModel.load(newer)
    torch.save({k: v for k, v in payload.items() if k != "format"}, unmarked)
    with pytest.raises(ValueError, match="records no format; this code reads format 1"):
        SavedModel.load(unmarked)


def test_a_model_reads_back_its_run_config(tmp_path: Path) -> None:
    config = DEFAULT_CONFIG.with_changes({"model": {"architecture": "summary"}})
    saved = SavedModel(tmp_path / "model.pt", {"config": config.to_dict()})
    assert saved.config == config


if __name__ == "__main__":
    # Write the fixtures again and print the literals they now give.
    for path in FIXTURES.glob("*.pt"):
        path.unlink()
    shutil.rmtree(FIXTURES / "dataset", ignore_errors=True)
    write_fixtures(FIXTURES)
    print("SCORES = {")
    for model in SCORES:
        print(f"    {model!r}: {fixture_scores(SavedModel.load(FIXTURES / f'{model}.pt'))!r},")
    print("}")
    baseline = SavedModel.load(FIXTURES / "baseline.pt")
    print(f"DATASET_SCORES = {dataset_scores(baseline, DatasetPaths(FIXTURES / 'dataset'))!r}")
    print(f"FeaturePlan().fingerprint() = {FeaturePlan().fingerprint()!r}")
