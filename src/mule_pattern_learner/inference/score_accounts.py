"""Score arbitrary accounts at a date with a run's model."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..artifacts import atomic_write, pending_path
from ..contract.graph_schema import ContextKey
from ..contract.sampler_plan import SamplerPlan
from ..data.contexts import ContextReader, close_source
from ..data.hub_registry import HubRegistry, hub_threshold, warn_hub_stubs
from ..data.ports import CutoffReader, HubReader
from ..data.splits import resolve_cutoff
from ..runtime.progress import emit
from .predictor import Predictor
from .rejections import rejection_summary
from .saved_model import SavedModel

SCORE_SCHEMA = pa.schema(
    [
        ("account_id", pa.string()),
        # Float64 probabilities (see model.probabilities_from_logits).
        ("score", pa.float64()),
        ("embedding", pa.list_(pa.float64())),
        ("predicted_mule", pa.bool_()),
        ("date", pa.string()),
        ("cutoff_utc", pa.string()),
    ]
)


def query_hubs(reader: HubReader, cutoff_seqs: list[int], sampler: SamplerPlan) -> HubRegistry:
    """Unscoped hub registry for arbitrary cutoffs, with the saved model's threshold.

    Scoring arbitrary accounts runs unscoped, so its rows carry visibility phase 3.
    """
    return reader.hub_registry(cutoff_seqs, threshold=hub_threshold(sampler))


def id_batches(ids: Iterable[str], size: int) -> Iterator[list[str]]:
    """Consume input IDs lazily; never allocate a database-wide ID map."""
    iterator = iter(ids)
    while batch := list(islice(iterator, size)):
        yield batch


def read_account_ids(path: Path) -> Iterator[str]:
    with path.open() as stream:
        for line in stream:
            value = line.strip()
            if value:
                yield value


def check_new_outputs(output: Path, rejected_output: Path) -> None:
    """Refuse scores of new accounts that exist, or that another run is writing."""
    # A pending file is another scoring run's output in the making.
    for path in (output, rejected_output, pending_path(output), pending_path(rejected_output)):
        if path.exists():
            raise FileExistsError(path)


def score_new_accounts(
    model: Path | SavedModel,
    account_ids: Iterable[str],
    date: str,
    output: Path,
    *,
    rejected_output: Path,
    contexts: ContextReader,
    cutoffs: CutoffReader,
    hub_reader: HubReader,
    hubs: HubRegistry | None = None,
) -> dict[str, Any]:
    """Score arbitrary existing-in-TigerGraph account IDs without a training manifest.

    Inference can use all history available at its cutoff. Strict experiment
    scoring uses scoped ContextKeys through Predictor instead. IDs that
    TigerGraph rejects (missing, not yet visible, over capacity) are not scored:
    they are listed in ``rejected_output``. The result reports rejected roots and
    masked child contexts separately (see ``rejection_summary``). A date before the
    first visible event is refused, since no account could be scored at it.
    The date's cutoff comes from ``cutoffs``, the unscoped hub registry from
    ``hub_reader`` (``hubs`` replaces it) and the contexts from ``contexts``, which
    scoring closes; the pipeline opens them on a connection whose installed queries it
    has verified.
    """
    count = rejected = supplied = 0
    examples: list[str] = []
    failed = True
    try:
        check_new_outputs(output, rejected_output)
        saved = SavedModel.of(model)
        seq, ms = resolve_cutoff(cutoffs, date)
        predictor = Predictor(saved, contexts, hubs=hubs)
        output.parent.mkdir(parents=True, exist_ok=True)
        if hubs is None:
            predictor.hubs = query_hubs(hub_reader, [seq], predictor.sampler)
            warn_hub_stubs(predictor.hubs, predictor.plan)
        # The scores replace output first, then the rejected IDs (if any) their file.
        with predictor.runtime(), atomic_write(rejected_output) as rejected_pending:
            with (
                atomic_write(output) as pending,
                pq.ParquetWriter(pending, SCORE_SCHEMA) as writer,
                rejected_pending.open("w") as rejected_stream,
            ):
                batches = (
                    [ContextKey("Account", value, seq, ms) for value in ids]
                    for ids in id_batches(account_ids, predictor.batch_size)
                )
                for frame, bad in predictor.stream(batches):
                    supplied += len(frame) + len(bad)
                    for key in bad:
                        rejected_stream.write(key.node_id + "\n")
                        if len(examples) < 20:
                            examples.append(key.node_id)
                    rejected += len(bad)
                    if len(frame):
                        frame["date"] = date
                        frame["cutoff_utc"] = date
                        writer.write_table(
                            pa.Table.from_pandas(frame, schema=SCORE_SCHEMA, preserve_index=False)
                        )
                        count += len(frame)
                if not supplied:
                    raise ValueError("No account IDs supplied")
            if not rejected:
                rejected_pending.unlink()
        failed = False
    finally:
        close_source(contexts, failed=failed)
    result = {
        "accounts": count,
        **rejection_summary(contexts, rejected, predictor.totals),
        "rejected_examples": examples,
        "rejected_output": str(rejected_output) if rejected else None,
        "output": str(output),
        "device": str(predictor.device),
        "database_calls": contexts.database_calls,
        "scope": "available_history_at_prediction",
    }
    # The whole result, so the run's events.jsonl keeps every scoring's rejections.
    emit({"event": "score", "date": date, **result})
    return result
