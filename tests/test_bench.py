"""mule bench: the table it writes, and the projections that price the owner's questions
(docs/design.md, "Scaling and the benchmark").

The loaded world is testing.fixtures.batch_world behind a spy executor, with the two payments
that span the export window (the first on day 0, the last on the window's last day) that mule
prepare's calendar check asks for. Its two scored Mondays are T, 2024-07-01, and T + 1 week.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mule_pattern_learner import bench
from mule_pattern_learner.config import (
    DEFAULT_CONFIG,
    FULL_DESIGN_CONFIG,
    BenchConfig,
    CalendarConfig,
    RunConfig,
)
from mule_pattern_learner.data.labels import read_identified
from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.calendar import DAY_0, Spy, identity_text, world_of
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, T
from mule_pattern_learner.tigergraph.executor import BenchExecutor
from mule_pattern_learner.tigergraph.queries import (
    PROFILE_HUBS,
    QUERY_FILES,
    READ_ANSWER_KEY,
    READ_HUBS,
    read_hubs,
)

W = 7 * DAY_MS
CALENDAR = CalendarConfig(export_start=date(2024, 1, 1), history_days=182, scored_mondays=2)
# Two Mondays, 8 unlabelled accounts and 4 identified mules on each, at 1 and 2 concurrent
# requests; a small synthetic ranking and matrix; projected to 1,000 and 100,000 people.
CONFIG = replace(
    DEFAULT_CONFIG,
    calendar=CALENDAR,
    # The full design's way of scoring: every pass builds the selection examples anew.
    data=FULL_DESIGN_CONFIG.data,
    training=FULL_DESIGN_CONFIG.training,
    audit=FULL_DESIGN_CONFIG.audit,
    bench=BenchConfig(
        mondays=2,
        unlabelled=8,
        positives=4,
        concurrency=(1, 2),
        bootstrap_rows=2_000,
        bootstrap_replicates=2,
        handoff_rows=500,
        projected_people=(1_000, 100_000),
    ),
)


class FakeClient:
    """The client's one operation the bench uses: a longer read timeout."""

    @contextmanager
    def request_timeout(self, read_s: float) -> Generator[None, None, None]:
        yield


class Live(Spy):
    """The loaded world as a connection executor: a spy over a reference graph, and a
    connection that reinstalls queries and tells its version, and has no operation that reads
    TigerGraph's metrics, which the bench never asks for."""

    graph_name = "Mule_Pattern_Learner"

    def __init__(self, graph: ReferenceGraph, *, fail: str | None = None) -> None:
        super().__init__(graph)
        self.client: Any = FakeClient()
        self.installed: list[tuple[list[str], str | None]] = []
        self.fail = fail

    def call(
        self, operation: Callable[[Any], Any], *, what: str, attempts: int | None = None
    ) -> Any:
        installed, fail = self.installed, self.fail

        class Connection:
            def getVer(self) -> str:  # noqa: N802
                return "4.2.5"

            def installQueries(self, names: list[str], flag: str | None = None) -> str:  # noqa: N802
                installed.append((names, flag))
                if names == [fail]:
                    raise RuntimeError("compilation failed")
                return "installed"

        return operation(Connection())

    def gsql(self, text: str, *, what: str = "gsql", attempts: int | None = None) -> str:
        raise AssertionError("the bench runs no GSQL statement")


@pytest.fixture(scope="module")
def graph() -> ReferenceGraph:
    w = fixtures.batch_world()
    last = DAY_0 + (CALENDAR.export_days - 1) * DAY_MS
    w.external("x_far", fixtures.OPENED)
    w.pay("x_first", "a_h1", "x_far", DAY_0 + HOUR_MS, 25.0, "ach")
    w.pay("x_last", "x_far", "a_h2", last + 23 * HOUR_MS, 40.0, "ach")
    return w.graph()


@pytest.fixture
def records(graph: ReferenceGraph, tmp_path: Path) -> list[Path]:
    """The identity file the loader writes for the world, as the world of seed 1001, D1."""
    path = tmp_path / "world_1001.identity.toml"
    path.write_text(identity_text(world_of(graph), 1001))
    return [path]


@pytest.fixture(scope="module")
def measured(
    graph: ReferenceGraph, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Live, pd.DataFrame, Path]:
    """One bench run of the world, which the tests of its table read."""
    folder = tmp_path_factory.mktemp("bench")
    records = folder / "world_1001.identity.toml"
    records.write_text(identity_text(world_of(graph), 1001))
    live = Live(graph)
    loaded, table, path = bench.run(live, CONFIG, records=[records], folder=folder / "out")
    assert loaded.config.name == "D1"
    return live, table, path


def _row(table: pd.DataFrame, section: str, name: str, variant: str = "") -> float:
    found = table[
        (table["section"] == section) & (table["name"] == name) & (table["variant"] == variant)
    ]
    assert len(found) == 1, f"{len(found)} rows of {section}.{name} {variant!r}"
    return float(found["value"].iloc[0])


def test_the_bench_writes_a_table_named_for_the_world(
    measured: tuple[Live, pd.DataFrame, Path],
) -> None:
    _, table, path = measured
    assert path.name == "D1.parquet" and path.parent.name == "out"
    assert tuple(table.columns) == bench.COLUMNS
    assert pd.read_parquet(path).equals(table)
    assert table["value"].dtype == "float64"
    # Every part of the design's list is measured.
    assert set(table["section"]) >= {
        "world",
        "fetch",
        "subgraph",
        "cache",
        "build",
        "gpu",
        "handoff",
        "profile",
        "hubs",
        "install",
        "co_users",
        "selection",
        "audit",
        "bootstrap",
        "scoring",
    }
    # No number is missing, and TigerGraph's memory, which this workspace's secret may not
    # read, is no row of the table: it is read in the Cloud console.
    assert table["value"].notna().all()
    assert "tigergraph_memory" not in set(table["name"])


def test_the_bench_runs_no_query_it_may_not(measured: tuple[Live, pd.DataFrame, Path]) -> None:
    live, _, _ = measured
    names = set(live.names())
    assert PROFILE_HUBS in names and READ_HUBS in names
    assert READ_ANSWER_KEY not in names
    assert names <= set(QUERY_FILES)


def test_each_query_is_timed_at_each_concurrency_with_its_bytes(
    measured: tuple[Live, pd.DataFrame, Path],
) -> None:
    live, table, _ = measured
    fetch = table[table["section"] == "fetch"]
    for query in ("read_money_flow", "read_payment_events", "read_shared_identity"):
        for concurrency in (1, 2):
            variant = f"unlabelled,concurrency={concurrency}"
            # 2 Mondays of 8 accounts, 32 to a request: a request a Monday.
            assert _row(table, "fetch", f"{query}.requests", variant) == 2
            assert _row(table, "fetch", f"{query}.accounts", variant) == 16
            assert _row(table, "fetch", f"{query}.wall_seconds", variant) > 0
            assert _row(table, "fetch", f"{query}.request_bytes_max", variant) > 100
            assert _row(table, "fetch", f"{query}.response_bytes_max", variant) > 100
    # 2 concurrencies, a bench fill and the other uses: B1 ran for every group at each.
    assert fetch["name"].str.endswith("response_bytes_per_account_p90").any()
    # The account-hub parameter is in the request: a Monday's request names its hubs.
    sent = [params for name, params in live.calls if name == "read_counterparties"]
    assert any(params["hub_accounts"] for params in sent)
    # B1's work counter, and its worst account.
    worst = fetch[fetch["name"] == "read_money_flow.worst_scan_steps"]
    assert set(worst["variant"]) <= {"unlabelled", "positives"} and (worst["value"] >= 0).all()
    assert all(note.startswith("a_") for note in worst["note"])


def test_selection_is_priced_by_its_passes(measured: tuple[Live, pd.DataFrame, Path]) -> None:
    _, table, _ = measured
    # 5 seeds x (3 priors + 4 ablations) x 30 epochs, and with the prior fixed 5 x (1 + 4) x 30.
    assert _row(table, "selection", "passes_per_rotation", "grid") == 1_050
    assert _row(table, "selection", "passes_per_rotation", "fixed") == 750
    build_rate = _row(table, "selection", "build_seconds_per_example")
    forward_rate = _row(table, "selection", "forward_seconds_per_example")
    assert build_rate > 0 and forward_rate > 0
    per_person = _row(table, "selection", "scorable_accounts_per_person_monday")
    # The world has 224 scorable accounts on each Monday and D1 is configured at 30,000 people.
    assert per_person == pytest.approx(224 / 30_000)
    for people in (1_000, 30_000, 100_000):
        variant = f"people={people}"
        # Two scored Mondays of the test calendar.
        examples = _row(table, "selection", "examples_per_pass", variant)
        assert examples == pytest.approx(per_person * people * 2)
        for priors, passes in (("grid", 1_050), ("fixed", 750)):
            for cost, rate in (("build", build_rate), ("build+forward", build_rate + forward_rate)):
                label = f"{variant},priors={priors},cost={cost}"
                seconds = _row(table, "selection", "projected_seconds_per_rotation", label)
                assert seconds == pytest.approx(rate * examples * passes)
                days = _row(table, "selection", "projected_days_per_rotation", label)
                assert days == pytest.approx(seconds / 86_400)


def test_passes_are_priced_by_one_build_when_the_examples_are_kept() -> None:
    """Proof of concept (docs/design.md, "Proof-of-concept settings (25 October 2026)"): with the
    examples kept, the first pass builds them and every pass runs the forward pass, so the build
    is priced once, for the selection world's passes and the test world's runs alike."""
    built, scored = bench.pass_seconds(0.02, 0.001, 1_000, 750, reuse=True)
    assert built == pytest.approx(20.0) and scored == pytest.approx(20.0 + 750.0)
    # Built anew on every pass, the full design's way: the build is priced 750 times.
    built, scored = bench.pass_seconds(0.02, 0.001, 1_000, 750, reuse=False)
    assert built == pytest.approx(15_000.0) and scored == pytest.approx(15_750.0)


def test_the_selection_rows_say_whether_the_examples_are_kept(
    measured: tuple[Live, pd.DataFrame, Path],
) -> None:
    _, table, _ = measured
    held = table[
        (table["section"] == "selection") & (table["name"] == "projected_seconds_per_rotation")
    ]
    assert len(held) and set(held["note"]) == {"built anew on every pass"}


def test_the_audit_and_the_bootstrap_are_priced_by_their_counts(
    measured: tuple[Live, pd.DataFrame, Path],
) -> None:
    _, table, _ = measured
    per_person = _row(table, "selection", "scorable_accounts_per_person_monday")
    # The sparse fetch rate: the seconds of both groups over the accounts of both.
    fetch = table[table["section"] == "fetch"]
    fill = (
        fetch[fetch["name"] == "fill_seconds"]["value"].sum()
        / fetch[fetch["name"] == "fill_accounts"]["value"].sum()
    )
    positives = table[
        (table["section"] == "hubs") & (table["name"] == "read_hubs_seconds")
    ]  # one per sampled Monday
    assert len(positives) == 2
    runs = _row(table, "bootstrap", "runs")
    # 5 seeds of the full model and of each of 4 ablations, and each XGBoost comparison once.
    assert runs == 5 * 5 + 2 == 27
    assert _row(table, "bootstrap", "replicates") == 1_000
    rate = _row(table, "bootstrap", "seconds_per_row_replicate", "rows=2000,replicates=2")
    for people in (1_000, 100_000):
        variant = f"people={people}"
        examples = per_person * people * 2
        # 1,000 replicates x 27 runs x the ranked account-Mondays.
        assert _row(table, "bootstrap", "projected_seconds", variant) == pytest.approx(
            1_000 * 27 * examples * rate
        )
        # The audit fetches 200,000 non-mule account-Mondays, and at least the identified
        # mules' active ones.
        fetched = _row(table, "audit", "account_mondays_fetched_at_least", variant)
        assert fetched >= 200_000
        assert _row(table, "audit", "projected_fetch_seconds_at_least", variant) == pytest.approx(
            fill * fetched
        )
    # The whole world against its fallback: a tenth of the other accounts, and the mules.
    whole = _row(table, "scoring", "account_mondays", "people=100000,scoring=whole")
    fallback = _row(table, "scoring", "account_mondays", "people=100000,scoring=fallback")
    assert whole == pytest.approx(per_person * 100_000 * 2)
    assert whole / 10 <= fallback < whole
    runs_pass = _row(table, "scoring", "scoring_pass_seconds", "people=100000,scoring=whole")
    assert _row(
        table, "scoring", "test_scoring_seconds_all_runs", "people=100000,scoring=whole"
    ) == pytest.approx(runs_pass * 27)


def test_what_is_measured_of_the_sample_is_consistent(
    measured: tuple[Live, pd.DataFrame, Path],
) -> None:
    _, table, _ = measured
    # Examples built: the unlabelled and the identified accounts of both Mondays.
    built = _row(table, "build", "examples", "sample")
    assert 16 <= built <= 24
    assert _row(table, "cache", "bytes_total", "sample") > 0
    assert _row(table, "cache", "bytes_per_scored_account", "sample") == pytest.approx(
        _row(table, "cache", "bytes_total", "sample") / built
    )
    assert _row(table, "handoff", "bytes_per_row", "rows=500") == pytest.approx(102 * 4, abs=1)
    assert _row(table, "handoff", "bytes_written", "rows=500") > 500 * 102 * 4
    assert _row(table, "gpu", "step_seconds", "examples=" + str(int(built))) > 0
    # The identified mules are no account hubs, which must be 0.
    hubs = table[table["name"] == "identified_active_that_are_hubs"]
    assert len(hubs) == 2 and (hubs["value"] == 0).all() and (hubs["note"] == "must be 0").all()
    # The shop that 201 customers pay is an account hub, and the histogram shows it.
    above = table[table["name"] == "accounts_above_hub"]
    assert above["value"].max() == 1
    assert set(table[table["section"] == "install"]["variant"]) == set(QUERY_FILES)


def test_a_sample_keeps_every_identified_active_mule_and_fetches_the_first_few(
    graph: ReferenceGraph,
) -> None:
    executor = BenchExecutor(Live(graph))
    config = replace(CONFIG, bench=replace(CONFIG.bench, positives=1))
    [sample] = bench.take_samples(
        executor, [T], read_identified(executor), config, bench.Measures()
    )
    # a_mule and a_r1 are identified and active on T, and the fetched sample holds the first.
    assert sample.identified == ("a_mule", "a_r1") and sample.identified_active == 2
    assert sample.positives == ("a_mule",)


def test_an_identified_mule_that_is_a_hub_is_counted_beyond_the_fetched_sample(
    graph: ReferenceGraph,
) -> None:
    executor = BenchExecutor(Live(graph))
    hubs = read_hubs.parse(executor.run(READ_HUBS, read_hubs.params(T)))
    assert len(hubs.accounts) == 1
    hub = hubs.accounts[0].account_id
    # Three identified active mules, the hub last; only the first is fetched.
    sample = bench.Sample(T, 224, ("a_mule", "a_r1", hub), ("a_mule",), (), hubs)
    measures = bench.Measures()
    bench.measure_profile(executor, [sample], CONFIG, measures)
    table = measures.frame()
    variant = f"t={T}"
    assert _row(table, "hubs", "identified_active_that_are_hubs", variant) == 1
    assert _row(table, "hubs", "identified_active_hub_share", variant) == pytest.approx(1 / 3)


def test_the_bench_imports_nothing_of_the_answer_key() -> None:
    """The benchmark never reads the truth (docs/design.md, "Scaling and the benchmark"), and no
    import contract holds it to that: its imports name no module or name of the answer key."""
    tree = ast.parse(Path(bench.__file__).read_text())
    named = [
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom | ast.Import)
        for name in [
            *(getattr(node, "module", None) or "").split("."),
            *(alias.name for alias in node.names),
        ]
    ]
    assert not [name for name in named if "answer_key" in name.lower()]


def test_every_query_is_installed_again_with_force_and_one_that_fails_is_named(
    graph: ReferenceGraph, records: list[Path], tmp_path: Path
) -> None:
    failing = "read_community"
    live = Live(graph, fail=failing)
    _, table, _ = bench.run(live, CONFIG, records=records, folder=tmp_path)
    assert live.installed == [([name], "-force") for name in QUERY_FILES]
    seconds = table[table["section"] == "install"].set_index("variant")
    assert (
        seconds["value"].drop(failing).notna().all() and (seconds["value"] > 0).drop(failing).all()
    )
    assert pd.isna(seconds.loc[failing, "value"])
    assert "compilation failed" in str(seconds.loc[failing, "note"])
    # The installs are left out when asked.
    quiet = Live(graph)
    _, other, _ = bench.run(
        quiet, CONFIG, records=records, folder=tmp_path / "quiet", installs=False
    )
    assert quiet.installed == [] and "install" not in set(other["section"])


def test_a_request_tigergraph_refuses_is_a_row_of_the_table_not_the_end_of_the_bench(
    graph: ReferenceGraph, records: list[Path], tmp_path: Path
) -> None:
    live = Live(graph)
    # B6 answers more than the response limit for a request of more than one account.
    live.too_large = ("read_community", 1)
    _, table, _ = bench.run(live, CONFIG, records=records, folder=tmp_path, installs=False)
    failed = table[table["name"] == "read_community.requests_failed"]
    assert set(failed["variant"]) == {
        "unlabelled,concurrency=1",
        "unlabelled,concurrency=2",
        "positives,concurrency=1",
        "positives,concurrency=2",
    }
    assert (failed["value"] > 0).all() and failed["note"].str.contains("ResponseTooLarge").all()
    # The other queries had none, and the fetch into the cache halved the refused requests.
    ok = table[table["name"] == "read_money_flow.requests_failed"]
    assert (ok["value"] == 0).all()
    assert not table["name"].str.startswith("read_community.response_bytes").any()
    assert _row(table, "build", "examples", "sample") > 0


def test_the_mondays_measured_are_the_first_the_last_and_those_between() -> None:
    mondays = list(range(52))
    assert bench.chosen_mondays(mondays, 3) == (0, 26, 51)
    assert bench.chosen_mondays(mondays, 2) == (0, 51)
    assert bench.chosen_mondays(mondays, 1) == (0,)
    assert bench.chosen_mondays(mondays, 5) == (0, 13, 26, 38, 51)
    assert bench.chosen_mondays(mondays[:2], 3) == (0, 1)


def test_the_projections_are_products_of_what_was_measured() -> None:
    # 1.5 scorable accounts per person on a Monday, 30,000 people, 52 Mondays: 2,340,000.
    examples = bench.examples_at(1.5, 30_000, 52)
    assert examples == 2_340_000
    # A millisecond an example over the 1,050 passes of a rotation: 2,457,000 s, 28.4 days.
    seconds = bench.seconds_projected(0.001, examples, 1_050)
    assert seconds == pytest.approx(2_457_000)
    assert seconds / bench.SECONDS_PER_DAY == pytest.approx(28.44, abs=0.01)
    assert bench.seconds_projected(2.0, 10) == 20
    assert bench.runs_of(5, 4, 2) == 27
    # The estimate of 35 gave each XGBoost comparison five seeds, as the GNN has: 5 x (1 + 4 + 2).
    assert bench.runs_of(5, 4, 0) + 2 * 5 == 35


def test_the_headline_numbers_read_back(measured: tuple[Live, pd.DataFrame, Path]) -> None:
    _, table, _ = measured
    found = bench.lines(table)
    assert any(line.startswith("selection, days per rotation, people=100000") for line in found)
    assert any(line.startswith("audit fetch, seconds at least, people=1000") for line in found)
    assert any(line.startswith("bootstrap, seconds, people=30000") for line in found)


def test_a_bench_config_is_part_of_the_run_config() -> None:
    assert isinstance(CONFIG, RunConfig) and CONFIG.bench.concurrency == (1, 2)
    assert CONFIG.calendar.mondays()[:2] == (T, T + W)
