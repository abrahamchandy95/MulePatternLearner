"""The composition root's connection: only the pipeline builds TigerGraph adapters.

connect reads the connection settings from the repository .env when it is called and
builds the executor with a configuration's retry budgets. open_context_source opens the
context source of a prepared dataset on the frozen graph; the pipeline hands it to the
use cases that open one (a data.contexts.ContextOpener).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import transport_settings
from ..contract.feature_groups import extraction_plan
from ..contract.sampler_plan import SamplerPlan, sampler_pools
from ..data.contexts import StreamingContextSource, streaming_source
from ..tigergraph.connection import Settings
from ..tigergraph.executor import TigerGraphExecutor
from ..tigergraph.provenance import verify_frozen_source


def connect(config: dict[str, Any]) -> TigerGraphExecutor:
    """A connected executor with the retry budgets of a training or preparation config."""
    transport = transport_settings(config)
    return TigerGraphExecutor(
        settings=Settings(),
        max_attempts=transport["max_query_attempts"],
        max_outage_s=transport["max_outage_s"],
    )


def open_context_source(
    dataset: Path, manifest: dict[str, Any], config: dict[str, Any] | None = None
) -> StreamingContextSource:
    """Open the live source of a prepared dataset; `config` is the training config.

    It requests the training model's groups and hop-2 flags (extraction_plan) with the
    prepared candidate pools, on a connection with the config's retry budgets whose
    source is checked to be the frozen one. `config` defaults to the prepared
    configuration.
    """
    prepared: dict[str, Any] = manifest["config"]
    training = prepared if config is None else config
    plan = extraction_plan(training)
    sampler = SamplerPlan.from_config(training)
    if sampler_pools(sampler) != sampler_pools(SamplerPlan.from_config(prepared)):
        raise ValueError(
            f"Sampler candidate pools differ from the preparation in {dataset}; prepare a "
            "new dataset (set prepared_id) or restore the prepared [sampler] pools"
        )
    executor = connect(training)
    verify_frozen_source(executor, manifest)
    return streaming_source(executor, plan, sampler, training)
