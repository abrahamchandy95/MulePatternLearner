"""`mule diagnose`: the diagnostic study of the built-in run's dataset, and its graph reads.

Only the pipeline builds adapters, so diagnose_built_in composes the study: it prepares
the dataset and hands diagnostics.study.diagnose a TigerGraphStudyReader, which builds
the adapters on one connection. It is the only reader that installs the analytics
queries (installer.install with analytics=True, when their text differs); `mule install`
and `mule train` never do.
"""

from __future__ import annotations

from typing import Any

from ..config import DEFAULT_CONFIG, RunConfig
from ..data.context_cache import ContextCache
from ..data.contexts import ContextSource
from ..data.manifest import read_manifest
from ..diagnostics.study import analyses, diagnose
from ..evaluation.truth import TruthReader
from ..paths import DatasetPaths
from ..tigergraph.analytics_query import TigerGraphAnalyticsFetcher
from ..tigergraph.executor import ConnectionExecutor
from ..tigergraph.installer import install
from ..tigergraph.oracle import TigerGraphRevealInputReader
from ..tigergraph.provenance import verify_frozen_source
from ..tigergraph.reveal import reveal_parameters
from ..tigergraph.scope import TigerGraphScopeReader
from .connect import Session, context_source
from .evaluate import SharedTruth
from .prepare import prepare_dataset
from .train import BASELINE_RUN


def diagnose_built_in(analysis: str | None = None) -> dict[str, Any]:
    """`mule diagnose [ANALYSIS]`: the study of the built-in run's dataset and run.

    The dataset is prepared as `mule train` prepares it, so a ready one needs no
    connection, and the study's graph reads share that session's one connection.
    """
    session = Session(DEFAULT_CONFIG.transport)
    dataset = prepare_dataset(DEFAULT_CONFIG, session=session)
    return diagnose(
        analyses(analysis),
        config=DEFAULT_CONFIG,
        dataset=dataset,
        run=BASELINE_RUN,
        graph=TigerGraphStudyReader(DEFAULT_CONFIG, dataset, session),
    )


class TigerGraphStudyReader:
    """What the diagnostic study of a prepared dataset reads, on one connection.

    The first read connects (the session's connection), and checks that the graph is
    still the frozen source the dataset was prepared from; every read after uses that
    connection. The oracle reads the truth once (pipeline.evaluate.SharedTruth). The
    context source is the configuration's, with the dataset's disk tier, which the
    training runs and audits of the dataset share. The analytics fetcher comes after the
    analytics queries are installed where their text differs. The reveal's parameters
    need no graph.
    """

    def __init__(self, config: RunConfig, dataset: DatasetPaths, session: Session) -> None:
        self.config = config
        self.dataset = dataset
        self.session = session
        self._verified: ConnectionExecutor | None = None
        self._truth = SharedTruth(session)
        self._installed: dict[str, Any] | None = None

    def executor(self) -> ConnectionExecutor:
        """The session's connection, once its source is checked to be the frozen one."""
        if self._verified is None:
            executor = self.session.executor()
            verify_frozen_source(executor, read_manifest(self.dataset))
            self._verified = executor
        return self._verified

    def oracle(self) -> TruthReader:
        """The graph's oracle truth on the checked connection, read once however often."""
        self.executor()
        return self._truth

    def scope(self) -> TigerGraphScopeReader:
        return TigerGraphScopeReader(self.executor())

    def contexts(self) -> ContextSource:
        executor = self.executor()
        cache = ContextCache.of(self.dataset, read_manifest(self.dataset))
        return context_source(executor, self.config, cache)

    def analytics(self) -> TigerGraphAnalyticsFetcher:
        executor = self.executor()
        if self._installed is None:
            self._installed = install(executor, analytics=True)
        return TigerGraphAnalyticsFetcher(executor)

    def reveal_inputs(self) -> list[dict[str, Any]]:
        return TigerGraphRevealInputReader(self.executor()).read(self.config.scope.id)

    def reveal_parameters(self) -> dict[str, Any]:
        return reveal_parameters(self.config.scope, self.config.dataset.dates, apply=False)
