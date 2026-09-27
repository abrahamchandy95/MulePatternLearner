"""model.pt: the selected model, as training writes it and scoring and audits read it.

SavedModel.selected builds its payload: the selected weights, the configuration
(RunConfig.to_dict()), the fingerprint of the feature plan, the threshold, the id and
manifest digest of the dataset and SavedModel.FORMAT, with what the run was trained on
for the record. Readers load it once and pass the SavedModel on; each checks only what
it relies on.

A model saved before FORMAT 1 records no format, and names its dataset by directory
instead of by dataset id. One saved before the typed configuration also holds a flat
table of the old setting names, which SavedModel.config converts
(inference.saved_settings), so such models load and score as they did.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

import torch

from ..artifacts import atomic_write
from ..config import RunConfig
from ..contract.feature_groups import FeaturePlan, contract_fingerprint
from ..contract.graph_schema import EVALUATION_PROTOCOL
from ..contract.time_basis import BASIS_ID
from ..data.manifest import manifest_digest
from ..paths import DATA_DIR, DatasetPaths
from .saved_settings import converted_run_config


@dataclass(frozen=True)
class SavedModel:
    """A model.pt payload and where it was read from."""

    # The payload layout this code writes. A payload without a format is older, and
    # loads through the conversions of config and dataset.
    FORMAT: ClassVar[int] = 1

    path: Path
    payload: dict[str, Any]

    @classmethod
    def selected(
        cls,
        path: Path,
        *,
        state: dict[str, torch.Tensor],
        config: RunConfig,
        dataset: DatasetPaths,
        dataset_id: str,
        threshold: float,
        known_mules: dict[str, int],
        device: torch.device,
        backend: str,
    ) -> SavedModel:
        """The model a run selected, to be saved at path.

        ``state`` holds the selected weights and ``threshold`` the validation threshold;
        the feature plan and sampler are config's. The known mules per split, the
        training device and the sampler backend are recorded, not read.
        """
        plan, sampler = config.feature_plan(), config.sampler
        payload = {
            "format": cls.FORMAT,
            "state_dict": state,
            "config": config.to_dict(),
            "basis_id": BASIS_ID,
            "contract": contract_fingerprint(),
            "dataset_id": dataset_id,
            "dataset_manifest_sha256": manifest_digest(dataset),
            "threshold": threshold,
            "feature_dim": len(plan.node_names),
            "input_fingerprint": plan.fingerprint(),
            "sampler": sampler.query_params(),
            "sampler_fingerprint": sampler.fingerprint(),
            "selected_on": "validation_observed_label_proxy_ap",
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "known_mules": known_mules,
            "training_device": str(device),
            "sampler_backend": backend,
        }
        return cls(path, payload)

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
        saved = self.payload["config"]
        if set(saved) == {field.name for field in fields(RunConfig)}:
            return RunConfig.from_dict(saved)
        return converted_run_config(saved)

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
