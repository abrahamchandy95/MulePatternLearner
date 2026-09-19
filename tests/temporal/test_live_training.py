from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.device import choose_device, select_device
from mule_pattern_learner.temporal.live.cli import build_parser
from mule_pattern_learner.temporal.live.contract import fingerprint
from mule_pattern_learner.temporal.live.dataset import prepare
from mule_pattern_learner.temporal.live.pipeline import DEFAULT_CONFIG, run
from mule_pattern_learner.temporal.live.sampling import pu_batches
from mule_pattern_learner.temporal.live.supervision import (
    FrameObservedLabels,
    label_summary,
    align_observed_labels,
)
from mule_pattern_learner.temporal.live.evaluation import (
    ParquetEvaluationTruth,
    evaluate_predictions,
)
from mule_pattern_learner.configuration import load_config
from mule_pattern_learner.temporal.live.policy import validate_protocol
from mule_pattern_learner.temporal.live.source import StreamingContextSource
from mule_pattern_learner.temporal.live.training import train
from mule_pattern_learner.temporal.training import timestamp
from test_live_pipeline import FakeExecutor


def config() -> dict:
    value = load_config(DEFAULT_CONFIG)
    return {
        **value,
        "dataset_id": "unit_fixture",
        "epochs": 2,
        "hidden": 16,
        "dropout": 0,
        "batch_size": 16,
        "steps_per_epoch": 2,
        "device": "cpu",
        "evaluation_protocol": "shared_history",
        "context_storage": "sqlite",
    }


def accounts() -> pd.DataFrame:
    c = config()
    # Use ownership grouping's actual splitter, with enough groups per partition.
    return pd.DataFrame(
        {
            "account_id": [f"A{i:04}" for i in range(1000)],
            "first_seen_ts_ms": timestamp(c["dates"]["train"][0]) - 1000,
            "owner_ids": [[f"P{i:04}"] for i in range(1000)],
            "observed_positive": False,
            "known_from_ms": 0,
        }
    )


def assigned_accounts() -> pd.DataFrame:
    from mule_pattern_learner.temporal.live.dataset import assign_groups

    return assign_groups(accounts(), 42)


@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_choose_device_prefers_available_accelerator(cuda: bool, mps: bool, expected: str) -> None:
    with (
        patch("torch.cuda.is_available", return_value=cuda),
        patch("torch.backends.mps.is_available", return_value=mps),
    ):
        assert choose_device().type == expected
        assert select_device().type == expected
        assert choose_device("auto").type == expected
        assert choose_device("cpu").type == "cpu"
        if not mps:
            with pytest.raises(ValueError, match="MPS"):
                choose_device("mps")


def supplied_labels() -> pd.DataFrame:
    a = assigned_accounts()
    chosen = a.groupby("split").head(20)
    return pd.DataFrame(
        {
            "account_id": chosen.account_id,
            "known_positive": True,
            "known_from_ms": timestamp("2024-01-01"),
        }
    )


def test_observed_labels_have_no_oracle_and_preserve_split_isolation() -> None:
    a = assigned_accounts()
    labels = align_observed_labels(a, supplied_labels())
    assert label_summary(labels) == {"train": 20, "validation": 20, "test": 20}
    assert labels.pu_label.sum() == 20
    assert not labels.loc[labels.split != "train", "pu_label"].any()
    bad = supplied_labels().assign(is_mule=1)
    with pytest.raises(ValueError, match="Oracle"):
        align_observed_labels(a, bad)
    with pytest.raises(ValueError, match="discovery"):
        align_observed_labels(a, supplied_labels().assign(known_from_ms=0))


def test_strict_claim_fails_before_preparation_or_training() -> None:
    with pytest.raises(ValueError, match="frozen TigerGraph scope_id"):
        validate_protocol({"evaluation_protocol": "strict_inductive"})
    with pytest.raises(ValueError, match="explicitly"):
        validate_protocol({})


def test_nnpu_draws_only_known_positives_and_covers_the_label_blind_marginal() -> None:
    indices = np.arange(100)
    observed = indices < 20
    draws = list(pu_batches(indices, observed, np.random.default_rng(42), 16))
    assert all(observed[p].all() for p, _ in draws)
    assert sorted(np.concatenate([u for _, u in draws]).tolist()) == indices.tolist()
    assert all(len(p) == 4 for p, _ in draws)
    assert len(list(pu_batches(indices, observed, np.random.default_rng(42), 16, max_steps=2))) == 2


class PreparedExecutor(FakeExecutor):
    def __init__(self, directory: Path):
        super().__init__({})
        self.directory = directory

    def run(self, name: str, params: dict) -> list[dict]:
        if name == "temporal_training_population":
            assert params["include_observed"] is False
            return [{"status": "ok", "accounts": accounts().to_dict("records")}]
        if name == "temporal_training_cutoffs":
            return [
                {
                    "status": "ok",
                    "last_visible_seqs": {
                        str(ms): 100 + i for i, ms in enumerate(params["cutoff_times"])
                    },
                }
            ]
        # A label mask must already be frozen when the first feature query starts.
        frozen = pd.read_parquet(self.directory / "observed_labels.parquet")
        assert label_summary(frozen) == {"train": 20, "validation": 20, "test": 20}
        return super().run(name, params)


def test_hidden_truth_cannot_change_updates_or_checkpoint_selection(tmp_path: Path) -> None:
    c = config()
    dataset = tmp_path / "dataset"
    prepare(
        c,
        dataset,
        PreparedExecutor(dataset),
        {"Account": 1000},
        labels=FrameObservedLabels(supplied_labels()),
    )
    first = train(c, dataset, tmp_path / "first.pt")
    saved_first = torch.load(tmp_path / "first.pt", weights_only=True)
    # The oracle is a separate file that is never opened by training.
    a = pd.read_parquet(dataset / "accounts.parquet")
    assert "is_mule" not in a.columns
    truth = a[["account_id"]].assign(is_mule=np.arange(len(a)) % 2)
    truth_path = tmp_path / "evaluation_truth.parquet"
    truth.to_parquet(truth_path, index=False)
    before = evaluate_predictions(
        tmp_path / "first_run/test_predictions.parquet",
        tmp_path / "first.pt",
        ParquetEvaluationTruth(truth_path),
    )
    truth["is_mule"] = 1 - truth.is_mule
    truth.to_parquet(truth_path, index=False)
    after = evaluate_predictions(
        tmp_path / "first_run/test_predictions.parquet",
        tmp_path / "first.pt",
        ParquetEvaluationTruth(truth_path),
    )
    assert before != after
    second = train(c, dataset, tmp_path / "second.pt")
    saved_second = torch.load(tmp_path / "second.pt", weights_only=True)
    assert first["history"] == second["history"]
    assert first["best_epoch"] == second["best_epoch"]
    assert first["validation_proxy"] == second["validation_proxy"]
    assert saved_first["threshold"] == saved_second["threshold"]
    assert first["observed_label_proxy"] == second["observed_label_proxy"]
    for name, value in saved_first["state_dict"].items():
        torch.testing.assert_close(value, saved_second["state_dict"][name], rtol=0, atol=0)
    assert first["database_calls_during_training"] == 0
    assert first["known_mules"] == {"train": 20, "validation": 20, "test": 20}


def test_minimal_command_and_run_defaults(tmp_path: Path) -> None:
    args = build_parser().parse_args(["train", "--output", "model.pt"])
    assert args.config == DEFAULT_CONFIG and args.dataset is None
    with (
        patch("mule_pattern_learner.temporal.live.pipeline.load_config", return_value=config()),
        patch("mule_pattern_learner.temporal.live.pipeline.prepare_live") as prep,
        patch(
            "mule_pattern_learner.temporal.live.pipeline.train", return_value={"status": "complete"}
        ) as fit,
    ):
        assert run(tmp_path / "model.pt")["status"] == "complete"
        prep.assert_called_once()
        fit.assert_called_once()
        assert fit.call_args.args[-1] == tmp_path / "model.pt"


def test_ready_pipeline_reuses_cache_without_connecting(tmp_path: Path) -> None:
    from mule_pattern_learner.temporal.live.pipeline import prepare_live

    c = config()
    manifest = {"status": "ready", "source": {"config_sha256": fingerprint(c)}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with patch("mule_pattern_learner.temporal.live.pipeline.TigerGraphExecutor") as client:
        assert prepare_live(c, tmp_path) == manifest
        client.assert_not_called()


def test_streaming_preparation_and_training_never_create_disk_context_cache(tmp_path: Path) -> None:
    c = {**config(), "context_storage": "stream"}
    dataset = tmp_path / "stream"
    executor = PreparedExecutor(dataset)
    manifest = prepare(
        c, dataset, executor, {"Account": 1000}, labels=FrameObservedLabels(supplied_labels())
    )
    assert manifest["cached_contexts"] == 0
    assert not executor.requested
    assert not (dataset / "contexts.sqlite").exists()
    source = StreamingContextSource(executor, capacity=4)
    result = train(c, dataset, tmp_path / "model.pt", contexts=source)
    assert result["database_calls_during_training"] > 0
    assert not (dataset / "contexts.sqlite").exists()
    assert len(source.memory) <= 4


def test_streaming_and_sqlite_return_identical_features_with_bounded_retention(
    tmp_path: Path,
) -> None:
    from mule_pattern_learner.temporal.live.source import ContextStore
    from mule_pattern_learner.temporal.live.contract import ContextKey

    keys = [ContextKey("Account", str(i), 100, 1000) for i in range(80)]
    source = FakeExecutor({})
    memory = StreamingContextSource(source, capacity=3)
    disk = ContextStore(tmp_path / "contexts.sqlite", {}, FakeExecutor({}))
    assert memory.fetch(keys) == disk.fetch(keys)
    assert len(memory.memory) == 3
    calls = memory.query_calls
    assert memory.fetch(keys[-3:]) == disk.fetch(keys[-3:])
    assert memory.query_calls == calls
    disk.close()
