"""model.pt: the selected model, as scoring and audits read it.

`training.summary.model_payload` builds its payload. Readers load it once and pass the
ModelCheckpoint on; each checks only what it relies on (the configuration is
validated where TemporalPredictor, score_new_accounts and evaluate_final_population
use it, never on load).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..config import validate_config
from ..contract.feature_groups import FeaturePlan, contract_fingerprint
from ..contract.time_basis import BASIS_ID
from ..data.manifest import manifest_digest


@dataclass(frozen=True)
class ModelCheckpoint:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> ModelCheckpoint:
        return cls(path, torch.load(path, map_location="cpu", weights_only=True))

    @classmethod
    def of(cls, checkpoint: Path | ModelCheckpoint) -> ModelCheckpoint:
        """An already loaded checkpoint, or the one at a path."""
        return checkpoint if isinstance(checkpoint, ModelCheckpoint) else cls.load(checkpoint)

    @property
    def config(self) -> dict[str, Any]:
        """The training configuration as saved; see validated_config."""
        return self.payload["config"]

    def validated_config(self) -> dict[str, Any]:
        return validate_config(self.config)

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
    def training_protocol(self) -> str | None:
        return self.payload.get("training_protocol")

    @property
    def dataset(self) -> Path | None:
        """The prepared dataset recorded at training time, if any."""
        value = self.payload.get("dataset")
        return Path(value) if value else None

    def check_contract(self) -> None:
        """Refuse a model saved under another feature or time-basis contract."""
        if (
            self.payload["contract"] != contract_fingerprint()
            or self.payload["basis_id"] != BASIS_ID
        ):
            raise ValueError("Checkpoint feature/time contract differs from this sampler")

    def check_inputs(self, plan: FeaturePlan) -> None:
        """Refuse a model whose inputs differ from those of its configuration.

        The input fingerprint also covers the pool groups' definitions (amount bands,
        pass-through thresholds), which the contract fingerprint leaves out.
        """
        if self.payload.get("input_fingerprint") != plan.fingerprint():
            raise ValueError(
                "Checkpoint input groups or pool definitions differ from its configuration"
            )

    def check_dataset(
        self,
        dataset: Path,
        message: str = "Checkpoint belongs to a different prepared dataset",
    ) -> None:
        """Refuse a prepared dataset other than the one this model was trained on."""
        if self.payload.get("dataset_manifest_sha256") != manifest_digest(dataset):
            raise ValueError(message)
