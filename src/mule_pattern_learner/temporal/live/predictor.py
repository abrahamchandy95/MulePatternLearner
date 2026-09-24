"""Inductive prediction from bounded contexts, independent of training account IDs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
import inspect
from itertools import islice
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mule_pattern_learner.device import choose_device

from ..common import timestamp
from ..encoding import BASIS_ID
from .batching import make_live_batch
from .config_schema import validate_config
from .contract import ContextKey, FeaturePlan, SamplerPlan, contract_fingerprint
from .hubs import HubRegistry, hub_threshold, query_hub_registry
from .memory import BatchLimits
from .model import LiveTGAT
from .sampling import BatchPrefetcher
from .source import (
    ContextSource,
    QueryExecutor,
    StreamingContextSource,
    checked_rows,
    live_executor,
    run_query,
    transport_settings,
)

SCORE_SCHEMA = pa.schema(
    [
        ("account_id", pa.string()),
        ("score", pa.float32()),
        ("embedding", pa.list_(pa.float64())),
        ("predicted_mule", pa.bool_()),
        ("date", pa.string()),
        ("cutoff_utc", pa.string()),
    ]
)


def _setting(config: dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


def query_hubs(
    executor: QueryExecutor, cutoff_seqs: list[int], sampler: SamplerPlan
) -> HubRegistry:
    """Unscoped hub registry for arbitrary cutoffs, with the checkpoint's threshold.

    Score-new runs unscoped, so its rows carry visibility phase 3.
    """
    return query_hub_registry(executor, cutoff_seqs, threshold=hub_threshold(sampler))


def warn_hub_stubs(hubs: HubRegistry, plan: FeaturePlan) -> None:
    """Warn once when hub children become stubs the model cannot recognise as hubs."""
    if len(hubs) and "hub_indicator" not in plan.groups:
        warnings.warn(
            f"The hub registry lists {len(hubs)} hub rows but the feature plan has no "
            "hub_indicator group: hub children are replaced by stubs without history, which "
            "this model cannot tell apart from dormant accounts. Add hub_indicator to "
            "feature_groups to give stubs their history_withheld flag.",
            UserWarning,
            stacklevel=2,
        )


def close_source(store: ContextSource, *, failed: bool) -> None:
    """Close a context source; after a failure, do not wait for its in-flight requests.

    Sources whose ``close`` accepts ``wait`` are closed with ``wait=False`` after an
    error or KeyboardInterrupt, so the error surfaces without waiting for REST retries.
    """
    close = store.close
    if failed and "wait" in inspect.signature(close).parameters:
        close(wait=False)  # pyright: ignore[reportCallIssue]
    else:
        close()


def rejection_summary(
    source: ContextSource, rejected_roots: int, totals: Counter[str]
) -> dict[str, Any]:
    """Root and child rejections reported separately.

    ``rejected`` counts the roots that were not scored. ``rejected_children`` counts
    the child contexts masked out of scored batches. ``rejection_events_by_status``
    is the source's raw counter: every rejected row served by a fetch at either hop,
    cache replays included, so it is not a count of accounts. Sources that count per
    hop (``rejections_by_hop``) also give the root and child statuses.
    """
    by_hop = getattr(source, "rejections_by_hop", None)
    return {
        "rejected": rejected_roots,
        "rejected_roots_by_status": None if by_hop is None else dict(by_hop.get(1, {})),
        "rejected_children": int(totals["rejected_children"]),
        "rejected_children_by_status": None if by_hop is None else dict(by_hop.get(2, {})),
        "stub_children": int(totals["stub_children"]),
        "rejection_events_by_status": dict(getattr(source, "rejections", {}) or {}),
    }


def streaming_source(
    executor: QueryExecutor, plan: FeaturePlan, sampler: SamplerPlan, config: dict[str, Any]
) -> ContextSource:
    """Live source with the transport settings of a training configuration."""
    transport = transport_settings(config)
    return StreamingContextSource(
        executor,
        plan=plan,
        sampler=sampler,
        capacity=transport["context_lru_capacity"],
        request_batch_size=transport["request_batch_size"],
        concurrency=transport["query_concurrency"],
        encoding_check_every=transport["encoding_check_every"],
    )


def check_coverage(store: ContextSource, plan: FeaturePlan, sampler: SamplerPlan) -> None:
    """The source must request every input the model reads, with the model's pools."""
    source_plan = getattr(store, "plan", None)
    source_sampler = getattr(store, "sampler", None)
    if not isinstance(source_plan, FeaturePlan) or not isinstance(source_sampler, SamplerPlan):
        raise ValueError("Context source must expose its FeaturePlan and SamplerPlan")
    # A summary model never fetches children, so only its first hop matters.
    for hop in (1,) if plan.architecture == "summary" else (1, 2):
        have = source_plan.query_flags(hop)
        missing = sorted(k for k, v in plan.query_flags(hop).items() if v and not have.get(k))
        if missing:
            raise ValueError(f"Context source does not request {missing} at hop {hop}")
    if (source_sampler.roots, source_sampler.children) != (sampler.roots, sampler.children):
        raise ValueError("Context source candidate pools differ from the model sampler")


class PinnedRoots:
    """Serve already fetched root rows to make_live_batch without a second request.

    Anything else (children, other keys) goes to the wrapped source, so concurrent
    batch builders cannot evict a batch's roots between filtering and assembly.
    """

    def __init__(
        self, store: ContextSource, keys: list[ContextKey], rows: list[dict[str, Any]]
    ) -> None:
        self.store = store
        self.rows = dict(zip(keys, rows, strict=True))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)

    def fetch(self, keys: list[ContextKey], *, hop: int = 1) -> list[dict[str, Any] | None]:
        if hop == 1 and all(key in self.rows for key in keys):
            return [self.rows[key] for key in keys]
        return self.store.fetch(keys, hop=hop)


@dataclass
class RootBatch:
    """One assembled batch for the accepted roots; rejected roots are reported, not scored."""

    requested: list[ContextKey]
    accepted: np.ndarray
    batch: dict[str, torch.Tensor] | None
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def keys(self) -> list[ContextKey]:
        return [k for k, ok in zip(self.requested, self.accepted, strict=True) if ok]

    @property
    def rejected(self) -> list[ContextKey]:
        return [k for k, ok in zip(self.requested, self.accepted, strict=True) if not ok]


def build_root_batch(
    store: ContextSource,
    keys: list[ContextKey],
    *,
    fanouts: tuple[int, int],
    device: str | torch.device,
    plan: FeaturePlan,
    sampler: SamplerPlan,
    hubs: HubRegistry,
    mode: str,
    step_seed: int = 0,
    sampler_backend: str | None = None,
) -> RootBatch:
    """Drop roots TigerGraph rejected (per-request status), then assemble the rest.

    Rejections are counted by status on ``store.rejections``; the batch statistics
    carry ``rejected_roots``. A batch with no accepted root has ``batch=None``.
    ``sampler_backend`` is the backend resolved once per run (``resolve_backend``);
    None lets make_live_batch resolve it.
    """
    rows = store.fetch(keys, hop=1)
    accepted = np.fromiter((row is not None for row in rows), dtype=bool, count=len(keys))
    stats: dict[str, Any] = {"rejected_roots": int(len(keys) - accepted.sum())}
    kept = [k for k, ok in zip(keys, accepted, strict=True) if ok]
    if not kept:
        return RootBatch(keys, accepted, None, stats)
    pinned = PinnedRoots(store, kept, [row for row in rows if row is not None])
    batch = make_live_batch(
        pinned,  # type: ignore[arg-type]
        kept,
        fanouts=fanouts,
        device=device,
        plan=plan,
        sampler=sampler,
        hubs=hubs,
        mode=mode,
        step_seed=step_seed,
        stats=stats,
        sampler_backend=sampler_backend,
    )
    return RootBatch(keys, accepted, batch, stats)


class TemporalPredictor:
    """The same feature/weight contract for old and newly arriving accounts.

    Scoring uses the deterministic evaluation sampler. ``hubs`` must be the registry
    for the scored cutoff (training's dataset registry or ``query_hubs``); without
    it no child is stubbed and hub children are masked out when TigerGraph rejects them.
    """

    def __init__(
        self,
        checkpoint: Path,
        contexts: ContextSource | None = None,
        device: str = "auto",
        *,
        executor: QueryExecutor | None = None,
        hubs: HubRegistry | None = None,
    ) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["contract"] != contract_fingerprint() or payload["basis_id"] != BASIS_ID:
            raise ValueError("Checkpoint feature/time contract differs from this sampler")
        self.config: dict[str, Any] = validate_config(payload["config"])
        self.plan = FeaturePlan.from_config(self.config)
        self.sampler = SamplerPlan.from_config(self.config)
        if payload.get("input_fingerprint", self.plan.fingerprint()) != self.plan.fingerprint():
            raise ValueError("Checkpoint input groups differ from its configuration")
        self.threshold = float(payload["threshold"])
        created = contexts is None
        if contexts is None:
            if executor is None:
                raise ValueError("Provide a context source or query executor")
            contexts = streaming_source(executor, self.plan, self.sampler, self.config)
        try:
            check_coverage(contexts, self.plan, self.sampler)
            self.contexts = contexts
            self.hubs = hubs if hubs is not None else HubRegistry.empty()
            warn_hub_stubs(self.hubs, self.plan)
            # Batch statistics of everything streamed (stub and rejected children).
            self.totals: Counter[str] = Counter()
            self.device = choose_device(device)
            # CUDA batches are assembled on the device by the prefetch workers.
            self.batch_device = self.device if self.device.type == "cuda" else torch.device("cpu")
            self.fanouts: tuple[int, int] = tuple(self.config.get("fanouts", [8, 4]))  # type: ignore[assignment]
            self.batch_size = min(int(_setting(self.config, "batch_size", 64)), 128)
            BatchLimits().validate_model(
                self.batch_size,
                self.fanouts,
                int(_setting(self.config, "hidden", 64)),
                self.plan,
                self.sampler,
            )
            self.prefetch = int(_setting(self.config, "prefetch_batches", 2))
            self.model = LiveTGAT(
                int(_setting(self.config, "hidden", 64)),
                int(_setting(self.config, "heads", 4)),
                float(_setting(self.config, "dropout", 0.15)),
                _setting(self.config, "variant", "temporal"),
                plan=self.plan,
            ).to(self.device)
            self.model.load_state_dict(payload["state_dict"])
            self.model.eval()
            torch.set_num_threads(int(_setting(self.config, "threads", 4)))
        except BaseException:
            if created:
                contexts.close()
            raise

    def prepare(self, keys: list[ContextKey]) -> RootBatch:
        """Fetch and assemble one batch; safe to call from prefetch worker threads."""
        BatchLimits().validate_model(
            len(keys),
            self.fanouts,
            int(_setting(self.config, "hidden", 64)),
            self.plan,
            self.sampler,
        )
        return build_root_batch(
            self.contexts,
            keys,
            fanouts=self.fanouts,
            device=self.batch_device,
            plan=self.plan,
            sampler=self.sampler,
            hubs=self.hubs,
            mode="eval",
        )

    def infer(self, prepared: RootBatch) -> pd.DataFrame:
        """Scores and embeddings for the accepted roots only; CPU outputs."""
        keys = prepared.keys
        if prepared.batch is None:
            return pd.DataFrame(
                {
                    "account_id": pd.Series([], dtype=object),
                    "score": np.zeros(0, dtype=np.float32),
                    "embedding": pd.Series([], dtype=object),
                    "predicted_mule": np.zeros(0, dtype=bool),
                }
            )
        with torch.inference_mode():
            batch = {k: v.to(self.device) for k, v in prepared.batch.items()}
            hidden = self.model.encode(batch)
            probabilities = torch.sigmoid(self.model.head(hidden).squeeze(-1)).cpu().numpy()
            vectors = hidden.cpu().tolist()
        return pd.DataFrame(
            {
                "account_id": [key.node_id for key in keys],
                "score": probabilities,
                "embedding": vectors,
                "predicted_mule": probabilities >= self.threshold,
            }
        )

    def predict(self, keys: list[ContextKey]) -> pd.DataFrame:
        """Rows for accepted keys only; rejected keys are counted on the source."""
        return self.infer(self.prepare(keys))

    def stream(
        self, batches: Iterable[list[ContextKey]]
    ) -> Iterator[tuple[pd.DataFrame, list[ContextKey]]]:
        """Score key batches in order, prefetching the next ones on worker threads.

        Integer batch statistics are summed into ``self.totals``.
        """
        with BatchPrefetcher(self.prepare, batches, depth=self.prefetch) as prepared:
            for item in prepared:
                self.totals.update(
                    {
                        k: int(v)
                        for k, v in item.stats.items()
                        if isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                    }
                )
                yield self.infer(item), item.rejected


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


def rejected_path(output: Path) -> Path:
    return output.with_name(output.name + ".rejected.txt")


def score_new_accounts(
    checkpoint: Path,
    account_ids: Iterable[str],
    date: str,
    output: Path,
    *,
    executor: QueryExecutor | None = None,
    contexts: ContextSource | None = None,
    hubs: HubRegistry | None = None,
) -> dict[str, Any]:
    """Score arbitrary existing-in-TigerGraph account IDs without a training manifest.

    Inference can use all history available at its cutoff. Strict experiment
    scoring uses scoped ContextKeys through TemporalPredictor instead. IDs that
    TigerGraph rejects (missing, not yet visible, over capacity) are not scored:
    they are listed in ``<output>.rejected.txt``. The result reports rejected roots and
    masked child contexts separately (see ``rejection_summary``).
    """
    rejected_output = rejected_path(output)
    pending = output.with_name(output.name + ".pending")
    rejected_pending = rejected_output.with_name(rejected_output.name + ".pending")
    for path in (output, rejected_output, pending, rejected_pending):
        if path.exists():
            raise FileExistsError(path)
    if executor is None:
        from .installation import verify_sources

        # The checkpoint's retry budgets (max_query_attempts, max_outage_s).
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        live = live_executor(validate_config(payload["config"]))
        verify_sources(live)
        executor = live
    ms = timestamp(date) - 1
    result = checked_rows(
        run_query(executor, "temporal_training_cutoffs", {"cutoff_times": [ms]}, timeout_s=900.0)
    )
    clocks = next(row["last_visible_seqs"] for row in result if "last_visible_seqs" in row)
    seq = int(clocks[str(ms)]) + 1
    predictor = TemporalPredictor(checkpoint, contexts, executor=executor, hubs=hubs)
    source = predictor.contexts
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    count = rejected = supplied = 0
    examples: list[str] = []
    failed = True
    try:
        if hubs is None:
            predictor.hubs = query_hubs(executor, [seq], predictor.sampler)
            warn_hub_stubs(predictor.hubs, predictor.plan)
        writer = pq.ParquetWriter(pending, SCORE_SCHEMA)
        with rejected_pending.open("w") as rejected_stream:
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
        writer.close()
        writer = None
        pending.replace(output)
        if rejected:
            rejected_pending.replace(rejected_output)
        failed = False
    finally:
        if writer is not None:
            writer.close()
        for path in (pending, rejected_pending):
            if path.exists():
                path.unlink()
        close_source(source, failed=failed)
    return {
        "accounts": count,
        **rejection_summary(source, rejected, predictor.totals),
        "rejected_examples": examples,
        "rejected_output": str(rejected_output) if rejected else None,
        "output": str(output),
        "device": str(predictor.device),
        "database_calls": source.query_calls,
        "scope": "available_history_at_prediction",
    }
