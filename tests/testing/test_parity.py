"""A live graph against the reference graph, on the same export: every query must agree.

Needs the TigerGraph of the repository .env, loaded from a PhantomLedger export by
gsql/schema/loading.gsql, with its splits written and the queries installed (`mule
install`), and MULE_PARITY_EXPORT naming that export's directory (one <table>.csv per
table). The reference graph reads the whole export into memory, so use a small world.
Nothing here writes to the live graph. Marked graph: run it with `pytest -m graph`.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.evaluation.answer_key import AnswerKeyRow, parse_answer_key
from mule_pattern_learner.testing.reference_graph import DAY_MS, ReferenceGraph, canonical
from mule_pattern_learner.tigergraph.connection import ConnectionSettings
from mule_pattern_learner.tigergraph.executor import (
    QueryExecutor,
    TigerGraphExecutor,
    account_pages,
)
from mule_pattern_learner.tigergraph.installer import verify
from mule_pattern_learner.tigergraph.queries import (
    FETCH_SUBGRAPHS,
    LIST_POPULATION,
    NODES,
    READ_ANSWER_KEY,
    READ_SUPERVISION,
    RELATIONS,
    WRITE_SPLIT,
    Phase,
    PopulationAccount,
    SubgraphRequest,
    Supervision,
    parse_population,
    parse_subgraphs,
    parse_supervision,
    population_params,
    split_params,
    subgraph_params,
    supervision_params,
)

pytestmark = pytest.mark.graph

EXPORT = "MULE_PARITY_EXPORT"
# Seeds checked per date and phase, at most MULES of them true mules, BATCH per request.
SEEDS = 64
MULES = 16
BATCH = 16


@pytest.fixture(scope="module")
def live() -> TigerGraphExecutor:
    executor = TigerGraphExecutor(settings=ConnectionSettings())
    verify(executor)
    return executor


def _splits(live: TigerGraphExecutor, vertex_type: str) -> dict[str, int]:
    """Every vertex's split on the live graph, through its built-in vertex endpoint."""

    def vertices(conn: Any) -> Any:
        return conn.getVertices(vertex_type, select="split")

    found = live.call(vertices, what=f"{vertex_type} splits")
    return {str(item["v_id"]): int(item["attributes"]["split"]) for item in found}


@pytest.fixture(scope="module")
def reference(live: TigerGraphExecutor) -> ReferenceGraph:
    """The export, with the live graph's party splits written as write_split writes them."""
    directory = os.environ.get(EXPORT)
    if not directory:
        pytest.skip(f"{EXPORT} must name the directory of the export the live graph holds")
    graph = ReferenceGraph.from_csv(Path(directory))
    written = {party: split for party, split in _splits(live, "Party").items() if split in (0, 1)}
    if written:
        graph.run(WRITE_SPLIT, split_params(written))
    return graph


def _monday(ms: int) -> int:
    """The first Monday 00:00 UTC at or after ms (1970-01-01 was a Thursday)."""
    day = -(-ms // DAY_MS)
    return (day + (-(day + 3)) % 7) * DAY_MS


def _dates(graph: ReferenceGraph) -> list[int]:
    """Three Mondays among the export's payments: after a quarter, a half and three
    quarters of them."""
    times = sorted(payment.ts for payment in graph.payments.values())
    if not times:
        pytest.skip("The export has no payments")
    return [_monday(times[len(times) * quarter // 4]) for quarter in (1, 2, 3)]


def _population(executor: QueryExecutor, t: int, phase: Phase) -> list[PopulationAccount]:
    pages = account_pages(executor, LIST_POPULATION, population_params(t, phase))
    return [account for page in pages for account in parse_population(page)]


def test_the_account_splits_agree(live: TigerGraphExecutor, reference: ReferenceGraph) -> None:
    assert _splits(live, "Account") == reference.account_split


def test_the_population_agrees(live: TigerGraphExecutor, reference: ReferenceGraph) -> None:
    for t in _dates(reference):
        for phase in Phase:
            assert _population(live, t, phase) == _population(reference, t, phase), (t, phase)


def test_the_supervision_agrees(live: TigerGraphExecutor, reference: ReferenceGraph) -> None:
    def supervision(executor: QueryExecutor, cutoff: int, phase: Phase) -> list[Supervision]:
        rows = executor.run(READ_SUPERVISION, supervision_params(cutoff, phase))
        return sorted(parse_supervision(rows, cutoff), key=lambda row: row.account_id)

    for cutoff in _dates(reference):
        for phase in Phase:
            expected = supervision(reference, cutoff, phase)
            assert supervision(live, cutoff, phase) == expected, (cutoff, phase)


def test_the_answer_key_agrees(live: TigerGraphExecutor, reference: ReferenceGraph) -> None:
    def answer_key(executor: QueryExecutor) -> list[AnswerKeyRow]:
        pages = account_pages(executor, READ_ANSWER_KEY, {})
        return [row for page in pages for row in parse_answer_key(page)]

    assert answer_key(live) == answer_key(reference)


def _seeds(reference: ReferenceGraph, t: int, phase: Phase) -> list[str]:
    """Up to SEEDS accounts of the population at t, MULES of them true mules if there are
    that many, chosen the same way on every run."""
    population = [account.account_id for account in _population(reference, t, phase)]
    mules = [account for account in population if reference.accounts[account].is_mule == 1]
    rng = random.Random(t)
    chosen = set(rng.sample(mules, min(MULES, len(mules))))
    others = [account for account in population if account not in chosen]
    chosen |= set(rng.sample(others, min(SEEDS - len(chosen), len(others))))
    return sorted(chosen)


def test_the_subgraphs_agree(live: TigerGraphExecutor, reference: ReferenceGraph) -> None:
    """Every node and edge with its features, list by list, for each seed."""
    for t in _dates(reference):
        for phase in Phase:
            seeds = _seeds(reference, t, phase)
            for start in range(0, len(seeds), BATCH):
                request = SubgraphRequest(tuple(seeds[start : start + BATCH]), t, phase)
                params = subgraph_params(request, DEFAULT_CONFIG.subgraph)
                expected = canonical(
                    parse_subgraphs(reference.run(FETCH_SUBGRAPHS, params), request)
                )
                actual = canonical(parse_subgraphs(live.run(FETCH_SUBGRAPHS, params), request))
                for got, want in zip(actual.subgraphs, expected.subgraphs, strict=True):
                    for name in (*NODES, *RELATIONS):
                        where = f"{want.seed} at {t} in {phase}: {name}"
                        assert getattr(got, name) == getattr(want, name), where


def test_the_refusals_agree(live: TigerGraphExecutor, reference: ReferenceGraph) -> None:
    t = _dates(reference)[0]
    [seed] = _seeds(reference, t, Phase.TRAINING)[:1]
    params = subgraph_params(SubgraphRequest((seed,), t, Phase.TRAINING), DEFAULT_CONFIG.subgraph)
    later = sorted(a.id for a in reference.accounts.values() if a.opened_ms > t)[:1]
    held_out = sorted(a for a, split in reference.account_split.items() if split == 1)[:1]
    changes: list[dict[str, Any]] = [
        {"lookback_days": 10},
        {"seed_ids": [seed, seed]},
        {"seed_ids": [seed, "no such account"]},
    ]
    if later or held_out:
        changes.append({"seed_ids": [seed, *later, *held_out]})
    for change in changes:
        changed = {**params, **change}
        [want] = reference.run(FETCH_SUBGRAPHS, changed)
        [got] = live.run(FETCH_SUBGRAPHS, changed)
        assert got["status"] == want["status"], change
        assert sorted(got.get("seeds", [])) == want.get("seeds", []), change
