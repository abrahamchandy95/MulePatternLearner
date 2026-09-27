"""model.pt: the selected model, as scoring and audits read it.

`training.summary.model_payload` builds its payload: the selected weights, the
configuration (RunConfig.to_dict()), the fingerprint of the feature plan, the
threshold, the id and manifest digest of the dataset and SavedModel.FORMAT. Readers
load it once and pass the SavedModel on; each checks only what it relies on.

A model saved before FORMAT 1 records no format, and names its dataset by directory
instead of by dataset id. One saved before the typed configuration also holds a flat
table of the old setting names: SavedModel.config converts it through SAVED_SETTINGS,
the one table from old names to RunConfig fields, so such models load and score as
they did.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

import torch

from ..artifacts import atomic_write
from ..config import DEFAULT_CONFIG, RunConfig
from ..contract.feature_groups import BUILT_IN_GROUPS, FeaturePlan, contract_fingerprint
from ..contract.sampler_plan import PoolPlan
from ..contract.time_basis import BASIS_ID
from ..data.manifest import manifest_digest
from ..paths import DATA_DIR, DatasetPaths

# Where each setting of a configuration saved before the typed configuration lives in
# RunConfig: the dotted name of its field, or None for a setting that names nothing
# now. Its [sampler] table held the roots pool beside the sampler's own settings, and
# a [sampler.children] table of changes to the roots pool without associations.
SAVED_SETTINGS: dict[str, str | None] = {
    # The graph identity is the source id the prepared dataset records, and a dataset
    # is found by its run, so neither is a setting.
    "dataset_id": None,
    "prepared_id": None,
    "scope_id": "scope.id",
    "create_scope": "scope.create",
    "scope_unowned": "scope.unowned",
    "reveal_per_split": "scope.reveal_per_split",
    "reveal_salt": "scope.reveal_salt",
    "dates": "dataset.dates",
    "seed_limits": "dataset.seed_limits",
    "cohort_seed": "dataset.seed",
    "split_seed": "dataset.split_seed",
    "fanouts": "sampler.fanouts",
    "sampler.recent": "sampler.roots.recent",
    "sampler.older": "sampler.roots.older",
    "sampler.distinct": "sampler.roots.distinct",
    "sampler.associations": "sampler.roots.associations",
    "sampler.max_history": "sampler.roots.max_history",
    "sampler.children.recent": "sampler.children.recent",
    "sampler.children.older": "sampler.children.older",
    "sampler.children.distinct": "sampler.children.distinct",
    "sampler.children.associations": "sampler.children.associations",
    "sampler.children.max_history": "sampler.children.max_history",
    "sampler.relation_fanouts": "sampler.relation_fanouts",
    "sampler.association_fanout": "sampler.association_fanout",
    "sampler.association_slots": "sampler.association_slots",
    "sampler.backend": "sampler.backend",
    "sampler.evaluation_seed": "sampler.evaluation_seed",
    "feature_groups": "features",
    "architecture": "model.architecture",
    "hidden": "model.hidden",
    "heads": "model.heads",
    "dropout": "model.dropout",
    "slot_sum": "model.slot_sum",
    "class_prior": "loss.class_prior",
    "positive_weight": "loss.positive_weight",
    "seed": "training.seed",
    "epochs": "training.epochs",
    "steps_per_epoch": "training.steps_per_epoch",
    "batch_size": "training.batch_size",
    "patience": "training.patience",
    "learning_rate": "training.learning_rate",
    "weight_decay": "training.weight_decay",
    "weight_average_decay": "training.weight_average_decay",
    "evaluation_unlabeled_limit": "training.proxy_unlabeled_limit",
    "request_batch_size": "transport.request_batch_size",
    "query_concurrency": "transport.query_concurrency",
    "context_lru_capacity": "transport.context_lru_capacity",
    "encoding_check_every": "transport.encoding_check_every",
    "max_query_attempts": "transport.max_query_attempts",
    "max_outage_s": "transport.max_outage_s",
    # The batch size of the removed SQLite context storage.
    "prepare_batch_size": None,
    "device": "runtime.device",
    "threads": "runtime.threads",
    "deterministic": "runtime.deterministic",
    "prefetch_batches": "runtime.prefetch_batches",
    "checkpoint_every_steps": "runtime.checkpoint_every_steps",
    "log_every_steps": "runtime.log_every_steps",
    "max_rejected_root_fraction": "runtime.max_rejected_root_fraction",
    # Settings of removed paths, saved with the one value this code implements
    # (RETIRED_VALUES), and the extraction groups, which only widened the request.
    "context_storage": None,
    "evaluation_protocol": None,
    "label_policy": None,
    "observed_labels": None,
    "sampler.policy": None,
    "extraction_groups": None,
    # The model variants, which became settings (saved_run_config).
    "variant": None,
}
# Saved settings whose null value took a default: the reveal's built-in budget, and the
# training seed for the reservoir seed and the reveal salt.
SEEDED_DEFAULTS = frozenset({"reveal_per_split", "reveal_salt", "cohort_seed"})
RETIRED_VALUES: dict[str, object] = {
    "context_storage": "stream",
    "evaluation_protocol": "strict_inductive",
    "label_policy": "graph_observed",
    "observed_labels": None,
    "sampler.policy": "resample",
}


def _flat(table: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """A saved table's settings by dotted name; only the sampler tables nest."""
    flat: dict[str, Any] = {}
    for key, value in table.items():
        name = prefix + key
        if name in ("sampler", "sampler.children") and isinstance(value, dict):
            flat.update(_flat(value, name + "."))
        else:
            flat[name] = value
    return flat


def saved_run_config(saved: dict[str, Any]) -> RunConfig:
    """The RunConfig of a saved configuration: RunConfig.to_dict(), or the old flat table.

    An old table goes through SAVED_SETTINGS, and what it leaves out takes the value
    the old code gave it: a setting it leaves absent is the built-in run's; a
    [sampler] table's absent pool keys are those of PoolPlan() and its children pool
    is the roots pool without associations, changed by [sampler.children]; an absent
    or null reservoir seed and reveal salt are the training seed, and a null reveal
    budget is the built-in one. The variant "tabular" is the summary architecture and
    "no_fourier" the feature groups without time_encoding.
    """
    if set(saved) == {field.name for field in fields(RunConfig)}:
        return RunConfig.from_dict(saved)
    flat = _flat(saved)
    unknown = sorted(set(flat) - SAVED_SETTINGS.keys())
    if unknown:
        raise ValueError(f"Unknown saved configuration key(s): {', '.join(unknown)}")
    for name, value in RETIRED_VALUES.items():
        if name in flat and flat[name] != value:
            remains = "" if value is None else f": only {value!r} remains"
            raise ValueError(f"{name} = {flat[name]!r} is no longer supported{remains}")
    table: dict[str, Any] = {}
    for name, value in flat.items():
        target = SAVED_SETTINGS[name]
        if target is None or (value is None and name in SEEDED_DEFAULTS):
            continue
        *sections, key = target.split(".")
        node = table
        for section in sections:
            node = node.setdefault(section, {})
        node[key] = value
    if isinstance(saved.get("sampler"), dict):
        sampler = table.setdefault("sampler", {})
        roots = sampler["roots"] = {**asdict(PoolPlan()), **sampler.get("roots", {})}
        sampler["children"] = {**roots, "associations": 0, **sampler.get("children", {})}
    seed = flat.get("seed", DEFAULT_CONFIG.training.seed)
    table.setdefault("dataset", {}).setdefault("seed", seed)
    table.setdefault("scope", {}).setdefault("reveal_salt", seed)
    variant = flat.get("variant", "temporal")
    if variant == "tabular":
        table.setdefault("model", {})["architecture"] = "summary"
    elif variant == "no_fourier":
        groups = table.get("features") or BUILT_IN_GROUPS
        table["features"] = [group for group in groups if group != "time_encoding"]
    elif variant != "temporal":
        raise ValueError(f"variant = {variant!r} is no longer supported")
    return RunConfig.from_dict(table)


@dataclass(frozen=True)
class SavedModel:
    """A model.pt payload and where it was read from."""

    # The payload layout this code writes. A payload without a format is older, and
    # loads through the conversions above.
    FORMAT: ClassVar[int] = 1

    path: Path
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> SavedModel:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        recorded = payload.get("format")
        if recorded is not None and recorded != cls.FORMAT:
            raise ValueError(
                f"{path} is a model of format {recorded}; this code reads {cls.FORMAT}"
            )
        return cls(path, payload)

    @classmethod
    def of(cls, model: Path | SavedModel) -> SavedModel:
        """An already loaded model, or the one at a path."""
        return model if isinstance(model, SavedModel) else cls.load(model)

    def save(self) -> None:
        """Write the payload to path atomically, so a crash never leaves a truncated model."""
        with atomic_write(self.path) as pending:
            torch.save(self.payload, pending)

    @property
    def config(self) -> RunConfig:
        """The training configuration, converted when the model predates RunConfig."""
        return saved_run_config(self.payload["config"])

    @property
    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.payload["state_dict"]

    @property
    def threshold(self) -> float:
        return float(self.payload["threshold"])

    @property
    def selected_on(self) -> str | None:
        return self.payload.get("selected_on")

    @property
    def dataset_id(self) -> str | None:
        """The id of the dataset the model was trained on; None before FORMAT 1."""
        return self.payload.get("dataset_id")

    def dataset(self, data: Path = DATA_DIR) -> DatasetPaths | None:
        """The prepared dataset the model was trained on: its dataset id's directory in data.

        A model saved before FORMAT 1 recorded the directory itself, if anything.
        """
        if self.dataset_id is not None:
            return DatasetPaths.of(self.dataset_id, data)
        value = self.payload.get("dataset")
        return DatasetPaths(Path(value)) if value else None

    def check_contract(self) -> None:
        """Refuse a model saved under another feature or time-basis contract."""
        if (
            self.payload["contract"] != contract_fingerprint()
            or self.payload["basis_id"] != BASIS_ID
        ):
            raise ValueError("The model's feature/time contract differs from this sampler")

    def check_inputs(self, plan: FeaturePlan) -> None:
        """Refuse a model whose inputs differ from those of its configuration.

        The input fingerprint also covers the pool groups' definitions (amount bands,
        pass-through thresholds), which the contract fingerprint leaves out.
        """
        if self.payload.get("input_fingerprint") != plan.fingerprint():
            raise ValueError(
                "The model's input groups or pool definitions differ from its configuration"
            )

    def check_dataset(
        self,
        dataset: DatasetPaths,
        message: str = "The model belongs to a different prepared dataset",
    ) -> None:
        """Refuse a prepared dataset other than the one this model was trained on."""
        if self.payload.get("dataset_manifest_sha256") != manifest_digest(dataset):
            raise ValueError(message)
