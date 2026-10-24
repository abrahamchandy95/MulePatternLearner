"""mule parity (docs/design.md, "Commands" and "The pipeline"): every installed query, run on the
loaded world, and the reference graph on the same export, compared.

The export is found under the loader's export root (paths.export_root): the folder whose manifest
counts equal the live read_world's counts and whose reference read_world, computed from its
shards, has the live identity (find_export). Then, on a seeded sample of scored Mondays, and on
each a seeded sample of accounts with true mules oversampled to a share (the truth comes from the
reference graph, which holds every column):

- the infrastructure queries, whole: read_world, read_labels and read_answer_key once, and per
  Monday every page of list_accounts, read_hubs and profile_hubs;
- B1 to B6, in requests of parity.together accounts, each answer parsed by the query's own
  parser and compared in the form the cache stores it (subgraph.cache's tables), the rows of
  each kind sorted, doubles compared to parity.tolerance and everything else exactly: TigerGraph
  prints a double to 16 digits, so a share of two integers differs from the reference's in the
  last digit at times (the cents sums are integers, and equal exactly);
- an account fetched alone against the same account among the others of its request, on the
  live graph (per-request state is kept by the scored account, never in a vertex flag two
  accounts of a request could share);
- B5's leaf flags against read_hubs' leaves, both read from the live graph;
- the refusals, each sent raw, as the builders refuse them: a t that is no Monday, a page out of
  range, no account, a repeated account, an unknown account, an account that does not exist at t
  and a hub that does not exist at t (the status rows are compared exactly).

B6 prints no per-node leaf flag, so its hubs and crowds are checked by the parity of its whole
community block, which meets B6's inline crowd rule only where a sampled community reaches a
crowd. The sample's accounts are therefore counted, from the reference graph, by how many of
their communities touch a leaf (a node of the first two rounds' members that is a crowd) and how
many hold a crowd between two members, and the counts are reported: a parity run that meets no
crowd proves nothing of the rule, and says so (the review of 24 October 2026, pending the
owner's confirmation).

A query passes when every one of its checks matched and it was checked at all. mule parity
writes the versions that passed, the TigerGraph version and the world's identity to
gsql/parity.toml (installer.parity_versions reads it), which fetch needs before it runs a query.
The record holds this run's queries alone, so a version that failed leaves no record, and a
record of an earlier run never vouches for a query this run did not pass.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
import json
import math
from pathlib import Path
import random
from typing import Any

from ..config import DEFAULT_CONFIG, ParityConfig, RunConfig
from ..paths import export_root, parity_record
from ..subgraph.cache import HUBS, Table
from ..subgraph.fetch import PARAMS, PER_ACCOUNT
from ..tigergraph.executor import ConnectionExecutor, ParityExecutor
from ..tigergraph.queries import (
    LIST_ACCOUNTS,
    PROFILE_HUBS,
    QUERY_FILES,
    READ_ANSWER_KEY,
    READ_COMMUNITY,
    READ_COUNTERPARTIES,
    READ_HUBS,
    READ_LABELS,
    READ_PAYMENT_EVENTS,
    READ_SHARED_IDENTITY,
    READ_WORLD,
    VERSIONS,
    list_accounts,
    profile_hubs,
    read_answer_key,
    read_hubs,
    read_labels,
    read_world,
)
from ..tigergraph.queries.read_hubs import Hubs
from ..tigergraph.queries.read_shared_identity import IdentityGraph
from ..tigergraph.queries.read_world import EDGE_TYPES, VERTEX_TYPES, World
from ..tigergraph.queries.rows import PAGE_LIMIT
from ..tigergraph.world import identity
from .reference import ReferenceGraph
from .reference.community import members
from .reference.graph import NODE_KINDS, Account
from .reference.infrastructure import scorable, world
from .reference.payment_events import Lookback
from .reference.shared_identity import Sharing, seen_node
from .shards import MANIFEST, read_export

# The queries mule parity runs, which is every registered one, in the registry's order.
QUERIES = tuple(QUERY_FILES)
# Those of the per-account queries (subgraph.fetch.PER_ACCOUNT) that take the account hubs.
TAKES_HUBS = frozenset(
    {READ_COUNTERPARTIES, READ_PAYMENT_EVENTS, READ_SHARED_IDENTITY, READ_COMMUNITY}
)
# How many mismatches of a query are kept to be shown, and how many differences of one answer.
MOST_MISMATCHES = 20
MOST_DIFFERENCES = 4
# The id no account has, and the days a true mule stays active after its last payment.
NO_ACCOUNT = "~no_such_account~"
WEEK_MS = 7 * 86_400_000

type Rows = dict[str, list[dict[str, Any]]]
type Node = tuple[str, str]


class ParityError(RuntimeError):
    """A parity run that cannot start or finish: no export of the loaded world, or a live
    request that raised."""


@dataclass
class Mismatch:
    """Where one check of a query differed, and how."""

    where: str
    what: str


@dataclass
class QueryParity:
    """What parity found of one query: its version, how many checks it made, how many differed
    (mismatched) and the first of them."""

    query: str
    version: str
    checks: int = 0
    mismatched: int = 0
    mismatches: list[Mismatch] = field(default_factory=lambda: list[Mismatch]())

    @property
    def passed(self) -> bool:
        return self.checks > 0 and self.mismatched == 0

    def record(self, where: str, differences: Sequence[str]) -> None:
        """Count a check, a mismatch when differences is not empty."""
        self.checks += 1
        if differences:
            self.mismatched += 1
            if len(self.mismatches) < MOST_MISMATCHES:
                self.mismatches.append(Mismatch(where, "; ".join(differences)))


@dataclass
class Communities:
    """What the sampled B6 requests met: how many accounts' communities were sampled, how many
    reached a leaf (a crowd) in their first two rounds, and how many hold a crowd between two
    members."""

    sampled: int = 0
    touch_a_leaf: int = 0
    crowd_between_members: int = 0


@dataclass(frozen=True)
class Parity:
    """A parity run: the world's identity and TigerGraph's version, the export it compared with,
    the Mondays sampled, each query's result and what B6's sample met."""

    identity: str
    tigergraph: str
    export: Path
    mondays: tuple[int, ...]
    queries: Mapping[str, QueryParity]
    communities: Communities

    @property
    def passed(self) -> tuple[str, ...]:
        """The queries that passed, in the registry's order."""
        return tuple(name for name in QUERIES if self.queries[name].passed)

    @property
    def failed(self) -> tuple[str, ...]:
        """The queries that did not pass, in the registry's order."""
        return tuple(name for name in QUERIES if not self.queries[name].passed)

    def lines(self) -> list[str]:
        """The run as a person reads it: a line per query, its first mismatches under it, and
        the communities that met a leaf."""
        found = [f"world {self.identity}, TigerGraph {self.tigergraph}, export {self.export}"]
        for name in QUERIES:
            result = self.queries[name]
            state = "passed" if result.passed else "FAILED"
            found.append(f"{name}: {state}, {result.checks} checks, {result.mismatched} differ")
            found += [f"    {item.where}: {item.what}" for item in result.mismatches[:5]]
        meet = self.communities
        found.append(
            f"B6: {meet.sampled} sampled communities, {meet.touch_a_leaf} touch a leaf, "
            f"{meet.crowd_between_members} hold a crowd between two members"
        )
        if not meet.touch_a_leaf:
            found.append(
                "B6's inline crowd rule met no crowd on this sample, so parity did not test it"
            )
        return found


# The comparison of two answers.


def _stable(value: object) -> object:
    """A value with its doubles rounded to 9 significant digits, so that rows sort alike on
    both sides of a comparison."""
    if isinstance(value, float):
        return float(f"{value:.9g}")
    if isinstance(value, Mapping):
        mapping: Mapping[Any, Any] = value
        return {str(key): _stable(item) for key, item in mapping.items()}
    if isinstance(value, list | tuple):
        sequence: Sequence[Any] = value
        return [_stable(item) for item in sequence]
    return value


def _sorted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: json.dumps(_stable(row), sort_keys=True))


def differences(live: object, reference: object, tolerance: float, path: str = "") -> list[str]:
    """How two canonical answers differ, at most MOST_DIFFERENCES places: doubles within the
    relative tolerance are one, everything else is equal only if it is."""
    found: list[str] = []
    named = path or "answer"
    if isinstance(live, Mapping) and isinstance(reference, Mapping):
        left: Mapping[Any, Any] = live
        right: Mapping[Any, Any] = reference
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                side = "live" if key in left else "reference"
                found.append(f"{named}.{key} only in {side}")
            else:
                found += differences(left[key], right[key], tolerance, f"{named}.{key}")
    elif isinstance(live, list | tuple) and isinstance(reference, list | tuple):
        first: Sequence[Any] = live
        second: Sequence[Any] = reference
        if len(first) != len(second):
            found.append(f"{named} has {len(first)} rows live, {len(second)} in the reference")
        for index, (one, other) in enumerate(zip(first, second, strict=False)):
            found += differences(one, other, tolerance, f"{named}[{index}]")
            if len(found) >= MOST_DIFFERENCES:
                break
    elif (
        isinstance(live, float | int)
        and isinstance(reference, float | int)
        and not isinstance(live, bool)
        and not isinstance(reference, bool)
        and (isinstance(live, float) or isinstance(reference, float))
    ):
        if not math.isclose(live, reference, rel_tol=tolerance, abs_tol=0.0):
            found.append(f"{named} is {live!r} live, {reference!r} in the reference")
    elif live != reference or type(live) is not type(reference):
        found.append(f"{named} is {live!r} live, {reference!r} in the reference")
    return found[:MOST_DIFFERENCES]


@dataclass(frozen=True)
class Failure:
    """An answer that could not be read: the parser refused it, a refusal included."""

    why: str


def _canonical[R](table: Table[R], answer: R) -> Rows:
    """An answer in the form the cache keeps it, each kind's rows sorted."""
    return {name: _sorted(rows) for name, rows in table.encode(answer).items()}


def _of_account(table: Table[Any], rows: Rows, account: str) -> Rows:
    """The rows of one account of a canonical answer, of each kind, by the column that names
    the account each row belongs to."""
    return {
        kind.name: [row for row in rows[kind.name] if row[kind.key] == account]
        for kind in table.kinds
    }


# Finding the export of the loaded world.


def _manifest_rows(folder: Path) -> dict[str, int] | None:
    """The rows of each dataset of the export in folder, or None when it holds no manifest."""
    path = folder / MANIFEST
    if not path.is_file():
        return None
    datasets = json.loads(path.read_text(encoding="utf-8")).get("datasets")
    if not isinstance(datasets, dict):
        return None
    found: dict[str, int] = {}
    for name, dataset in datasets.items():
        rows = dataset.get("rows") if isinstance(dataset, dict) else None
        if isinstance(rows, int) and not isinstance(rows, bool):
            found[str(name)] = rows
    return found


def counts_match(rows: Mapping[str, int], live: World) -> bool:
    """Whether an export's rows per table equal the live world's counts of each vertex type and
    of each of the 15 forward edge types."""
    expected = {**live.vertices, **live.edges}
    return all(rows.get(name) == expected[name] for name in (*VERTEX_TYPES, *EDGE_TYPES))


def world_differences(live: World, other: World) -> list[str]:
    """The fields of read_world in which two worlds differ, by name."""
    found: list[str] = []
    for name, value in asdict(live).items():
        held: Any = asdict(other)[name]
        if isinstance(value, dict):
            found += [f"{name}.{key}" for key in sorted(value) if value[key] != held.get(key)]
        elif value != held:
            found.append(name)
    return found


def find_export(root: Path, live: World) -> tuple[Path, ReferenceGraph]:
    """The loader's export under root whose manifest counts equal the live world's and whose
    reference read_world has the live identity, with the reference graph of its shards. root
    itself and each folder in it are looked at, by name; the first match is taken, since two
    exports of one identity are one world. Refused when none matches, naming how the nearest
    differs."""
    if not root.is_dir():
        raise ParityError(f"The export root {root} is no folder")
    folders = [root, *sorted(path for path in root.iterdir() if path.is_dir())]
    wanted = identity(live)
    nearest: list[str] | None = None
    candidates = 0
    for folder in folders:
        rows = _manifest_rows(folder)
        if rows is None:
            continue
        candidates += 1
        if not counts_match(rows, live):
            continue
        graph = ReferenceGraph(read_export(folder))
        found = world(graph)
        if identity(found) == wanted:
            return folder, graph
        differing = world_differences(live, found)
        nearest = differing if nearest is None or len(differing) < len(nearest) else nearest
    if not candidates:
        raise ParityError(f"No export folder with a {MANIFEST} is in {root}")
    if nearest is None:
        raise ParityError(
            f"None of the {candidates} exports in {root} has the loaded world's counts per table"
        )
    raise ParityError(
        f"No export in {root} has the loaded world's identity {wanted}; the nearest differs in "
        f"{', '.join(nearest[:8])}"
    )


# The sample.


def sample_mondays(mondays: Sequence[int], config: ParityConfig) -> tuple[int, ...]:
    """The seeded sample of scored Mondays, in increasing order; all of them when the calendar
    has no more than parity.mondays."""
    if len(mondays) <= config.mondays:
        return tuple(sorted(mondays))
    return tuple(sorted(random.Random(config.seed).sample(list(mondays), config.mondays)))


def _active(account: Account, t_ms: int) -> bool:
    """Whether the account is a true mule active on the Monday t."""
    return account.is_mule == 1 and account.mule_onset_ms < t_ms <= account.mule_last_ms + WEEK_MS


def sample_accounts(
    graph: ReferenceGraph, t_ms: int, config: ParityConfig
) -> tuple[list[str], list[str]]:
    """The seeded sample of the Monday's scorable accounts: its true mules, up to the share
    parity.mule_share of parity.accounts, those active that Monday first, and then other
    accounts. The mules, and the others, each by increasing id."""
    view = graph.as_of(t_ms)
    scored = [account for account in graph.accounts.values() if scorable(view, account)]
    rng = random.Random(f"{config.seed}|{t_ms}")
    mules = sorted(
        (account for account in scored if account.is_mule == 1), key=lambda account: account.id
    )
    wanted = min(len(mules), round(config.accounts * config.mule_share))
    active = [account for account in mules if _active(account, t_ms)]
    rest = [account for account in mules if not _active(account, t_ms)]
    chosen = rng.sample(active, min(len(active), wanted))
    chosen += rng.sample(rest, min(len(rest), wanted - len(chosen)))
    others = sorted((account for account in scored if account.is_mule != 1), key=lambda a: a.id)
    ordinary = rng.sample(others, min(len(others), config.accounts - len(chosen)))
    return sorted(account.id for account in chosen), sorted(account.id for account in ordinary)


# B6's leaves.


def _held(sharing: Sharing, customer: str) -> set[Node]:
    """Every device and IP a customer holds before the leaf test: its links current in the last
    30 days and the nodes its own payments of those days were seen on."""
    nodes: set[Node] = {
        (kind, node) for kind in NODE_KINDS for node in sharing.current(customer, kind)
    }
    for payment in sharing.own_sent(customer):
        if payment.ts >= sharing.recent:
            for kind in NODE_KINDS:
                node = seen_node(payment, kind)
                if node is not None:
                    nodes.add((kind, node))
    return nodes


def community_leaves(sharing: Sharing, scored: str, hubs: frozenset[str]) -> tuple[bool, bool]:
    """Whether the scored account's community, as the reference builds it, touches a leaf (a
    crowd held by a member of its first two rounds, which B6 must not expand) and holds a crowd
    between two members (a leaf two members of any round hold)."""
    owner = sharing.view.owner(scored)
    if owner is None:
        return False, False
    found = members(sharing, owner, hubs)
    expanded: set[Node] = set()
    holders: dict[Node, int] = {}
    for customer, number in found.round.items():
        leaves = {node for node in _held(sharing, customer) if sharing.leaf(node)}
        if number <= 1:
            expanded |= leaves
        for node in leaves:
            holders[node] = holders.get(node, 0) + 1
    return bool(expanded), any(count >= 2 for count in holders.values())


# The run.


class _Comparison:
    """One parity run's two sides and what it found: the live graph through the parity
    executor, the reference graph, and the result of each query."""

    def __init__(self, live: ParityExecutor, graph: ReferenceGraph, config: ParityConfig) -> None:
        self.live = live
        self.graph = graph
        self.config = config
        self.found = {name: QueryParity(name, VERSIONS[name]) for name in QUERIES}
        self.communities = Communities()

    def check(self, query: str, where: str, live: object, reference: object) -> None:
        """Compare two canonical answers of a request that should be answered: a side that could
        not be read is a difference, whichever side and even when both could not, since two
        requests that fail alike prove nothing."""
        if isinstance(live, Failure) or isinstance(reference, Failure):
            self.found[query].record(where, [f"live: {_why(live)}; reference: {_why(reference)}"])
            return
        self.found[query].record(where, differences(live, reference, self.config.tolerance))

    def ask(
        self, graph: bool, query: str, params: dict[str, Any]
    ) -> list[dict[str, Any]] | Failure:
        """A query's rows from the live graph, or from the reference one when graph is set; a
        live request that raised is a Failure naming the error."""
        if graph:
            return self.graph.run(query, params)
        try:
            return self.live.run(query, params)
        except Exception as error:
            return Failure(f"{type(error).__name__}: {str(error)[:300]}")

    def read[R](
        self,
        graph: bool,
        query: str,
        params: dict[str, Any],
        read: Callable[[Sequence[object]], R],
    ) -> R | Failure:
        """A query's answer from one side, read by the query's own parser: a Failure when the
        request raised or the parser refused the answer."""
        rows = self.ask(graph, query, params)
        if isinstance(rows, Failure):
            return rows
        try:
            return read(rows)
        except ValueError as error:
            return Failure(f"{type(error).__name__}: {str(error)[:300]}")

    def both[R](
        self, query: str, params: dict[str, Any], read: Callable[[Sequence[object]], R]
    ) -> tuple[R | Failure, R | Failure]:
        """A query's answer from each side, each read by the query's own parser."""
        return self.read(False, query, params, read), self.read(True, query, params, read)

    def paged[R](
        self,
        query: str,
        make: Callable[[str], dict[str, Any]],
        parse: Callable[[Sequence[object], str], tuple[R, ...]],
    ) -> tuple[list[R] | Failure, list[R] | Failure]:
        """Every page of a paged query from each side, concatenated, until one comes back short
        (a side that failed gives its Failure)."""
        sides: list[list[R] | Failure] = []
        for graph in (False, True):
            found: list[R] = []
            after_id = ""
            outcome: list[R] | Failure = found
            while True:
                rows = self.ask(graph, query, make(after_id))
                if isinstance(rows, Failure):
                    outcome = rows
                    break
                try:
                    page = parse(rows, after_id)
                except ValueError as error:
                    outcome = Failure(f"{type(error).__name__}: {str(error)[:300]}")
                    break
                found += page
                if len(page) < PAGE_LIMIT:
                    break
                after_id = str(getattr(page[-1], "account_id"))
            sides.append(outcome)
        return sides[0], sides[1]


def _why(value: object) -> str:
    return value.why if isinstance(value, Failure) else "answered"


def _plain(rows: Sequence[object]) -> list[dict[str, Any]]:
    """Dataclass rows as plain dicts, sorted."""
    return _sorted([asdict(row) for row in rows])  # pyright: ignore[reportArgumentType]


def _infrastructure(run: _Comparison) -> None:
    """The queries with no Monday: the world, the identified mules and the answer key, whole."""
    live, reference = run.both(READ_WORLD, read_world.params(), read_world.parse)
    run.check(READ_WORLD, "world", _world(live), _world(reference))
    labels = run.paged(
        READ_LABELS,
        lambda after_id: read_labels.params(after_id, PAGE_LIMIT),
        lambda rows, after_id: read_labels.parse(rows, after_id, PAGE_LIMIT),
    )
    run.check(READ_LABELS, "all pages", _listed(labels[0]), _listed(labels[1]))
    key = run.paged(
        READ_ANSWER_KEY,
        lambda after_id: read_answer_key.params(after_id, PAGE_LIMIT),
        lambda rows, after_id: read_answer_key.parse(rows, after_id, PAGE_LIMIT),
    )
    run.check(READ_ANSWER_KEY, "all pages", _listed(key[0]), _listed(key[1]))
    for query, make in (
        (READ_LABELS, read_labels.params),
        (READ_ANSWER_KEY, read_answer_key.params),
    ):
        for label, params in {
            "a page of 0": make("", 1) | {"page": 0},
            "a page above the limit": make("", 1) | {"page": PAGE_LIMIT + 1},
        }.items():
            _refusal(run, query, label, params)


def _world(value: World | Failure) -> object:
    return value if isinstance(value, Failure) else {"world": [asdict(value)]}


def _listed(value: Sequence[object] | Failure) -> object:
    return value if isinstance(value, Failure) else {"accounts": _plain(value)}


def _monday_infrastructure(run: _Comparison, t_ms: int, where: str) -> tuple[Hubs | Failure, Hubs]:
    """The queries of one Monday that name no account: the scorable accounts, the hubs and the
    hub profile. The live hubs, which B5's leaf flags are checked against, and the reference's,
    which every request of the Monday names its account hubs from, so that a difference in
    read_hubs shows in read_hubs alone."""
    live_pages, reference_pages = run.paged(
        LIST_ACCOUNTS,
        lambda after_id: list_accounts.params(t_ms, after_id, PAGE_LIMIT),
        lambda rows, after_id: list_accounts.parse(rows, after_id, PAGE_LIMIT),
    )
    run.check(LIST_ACCOUNTS, where, _listed(live_pages), _listed(reference_pages))
    live, reference = run.both(READ_HUBS, read_hubs.params(t_ms), read_hubs.parse)
    run.check(READ_HUBS, where, _hubs(live), _hubs(reference))
    live_profile, reference_profile = run.both(
        PROFILE_HUBS, profile_hubs.params(t_ms), profile_hubs.parse
    )
    run.check(
        PROFILE_HUBS,
        where,
        live_profile if isinstance(live_profile, Failure) else {"profile": [asdict(live_profile)]},
        reference_profile
        if isinstance(reference_profile, Failure)
        else {"profile": [asdict(reference_profile)]},
    )
    if isinstance(reference, Failure):
        raise ParityError(f"The reference graph's read_hubs failed on {where}: {reference.why}")
    return live, reference


def _hubs(value: Hubs | Failure) -> object:
    return value if isinstance(value, Failure) else _canonical(HUBS, value)


def _per_account(
    run: _Comparison, t_ms: int, where: str, accounts: Sequence[str], hubs: Hubs
) -> list[IdentityGraph]:
    """B1 to B6 on the Monday's sample, in requests of parity.together accounts, each against
    the reference; an account alone against itself among the others of its request. The live B5
    answers, which the leaf flags are read from."""
    config = run.config
    hub_ids = sorted(hub.account_id for hub in hubs.accounts)
    chunks = [
        accounts[start : start + config.together]
        for start in range(0, len(accounts), config.together)
    ]
    identity_answers: list[IdentityGraph] = []
    for query, table in PER_ACCOUNT.items():
        among: dict[str, Rows] = {}
        for chunk in chunks:
            params = PARAMS[query](t_ms, tuple(chunk), hub_ids)
            live, reference = run.both(
                query, params, lambda rows, chunk=chunk, table=table: table.parse(rows, chunk)
            )
            run.check(
                query,
                f"{where}, {len(chunk)} accounts from {chunk[0]}",
                live if isinstance(live, Failure) else _canonical(table, live),
                reference if isinstance(reference, Failure) else _canonical(table, reference),
            )
            if isinstance(live, Failure):
                continue
            canonical = _canonical(table, live)
            for account in chunk:
                among[account] = _of_account(table, canonical, account)
            if query == READ_SHARED_IDENTITY:
                identity_answers += list(live)
        full = [chunk for chunk in chunks if len(chunk) == config.together] or chunks[:1]
        for account in _alone(run, accounts, full):
            params = PARAMS[query](t_ms, (account,), hub_ids)
            lone = run.read(
                False,
                query,
                params,
                lambda rows, account=account, table=table: table.parse(rows, [account]),
            )
            if isinstance(lone, Failure):
                run.found[query].record(
                    f"{where}, {account} alone", [f"the live answer alone failed: {lone.why}"]
                )
            elif account not in among:
                run.found[query].record(
                    f"{where}, {account} alone",
                    [
                        "the live answer among the others could not be read, so it has no rows to equal"
                    ],
                )
            else:
                run.check(
                    query,
                    f"{where}, {account} alone against {len(full[0]) - 1} others",
                    _of_account(table, _canonical(table, lone), account),
                    among[account],
                )
    return identity_answers


def _alone(run: _Comparison, accounts: Sequence[str], full: Sequence[Sequence[str]]) -> list[str]:
    """The accounts fetched alone: of the accounts in full requests, a true mule first, then an
    ordinary account, up to parity.alone."""
    pool = [account for chunk in full for account in chunk]
    mules = [a for a in pool if run.graph.accounts[a].is_mule == 1]
    others = [a for a in pool if run.graph.accounts[a].is_mule != 1]
    chosen: list[str] = []
    for kind in (mules, others):
        if kind and len(chosen) < run.config.alone:
            chosen.append(kind[0])
    for account in pool:
        if len(chosen) >= run.config.alone:
            break
        if account not in chosen:
            chosen.append(account)
    return chosen[: run.config.alone]


def _leaf_flags(
    run: _Comparison, where: str, answers: Sequence[IdentityGraph], hubs: Hubs | Failure
) -> None:
    """B5's inline leaf flags against the leaves of the live read_hubs of the same Monday, both
    from the live graph."""
    if isinstance(hubs, Failure):
        run.found[READ_SHARED_IDENTITY].record(where, [f"live read_hubs failed: {hubs.why}"])
        return
    leaves = {
        "device": {leaf.node_id for leaf in hubs.devices},
        "ip": {leaf.node_id for leaf in hubs.ips},
    }
    wrong = [
        f"{kind} {row.node_id} of {graph.scored_id} is flagged "
        f"{'a leaf' if row.leaf else 'no leaf'}, read_hubs says the opposite"
        for graph in answers
        for kind, nodes in (("device", graph.devices), ("ip", graph.ips))
        for row in nodes
        if row.leaf != (row.node_id in leaves[kind])
    ]
    run.found[READ_SHARED_IDENTITY].record(
        f"{where}, leaf flags against read_hubs", wrong[:MOST_DIFFERENCES]
    )


def _refusals(run: _Comparison, t_ms: int, where: str, accounts: Sequence[str], hubs: Hubs) -> None:
    """The refusals of every query, each request sent raw, compared as status rows."""
    late = sorted(
        account.id
        for account in run.graph.accounts.values()
        if not account.is_external and account.opened_ms >= t_ms
    )
    hub_ids = sorted(hub.account_id for hub in hubs.accounts)
    mistaken = t_ms + 1
    for query, probes in (
        (
            LIST_ACCOUNTS,
            {
                "t that is no Monday": list_accounts.params(t_ms, "", 1) | {"t_ms": mistaken},
                "a page of 0": list_accounts.params(t_ms, "", 1) | {"page": 0},
                "a page above the limit": list_accounts.params(t_ms, "", 1)
                | {"page": PAGE_LIMIT + 1},
            },
        ),
        (
            READ_HUBS,
            {"t that is no Monday": read_hubs.params(t_ms) | {"t_ms": mistaken}},
        ),
        (
            PROFILE_HUBS,
            {"t that is no Monday": profile_hubs.params(t_ms) | {"t_ms": mistaken}},
        ),
    ):
        for label, params in probes.items():
            _refusal(run, query, f"{where}, {label}", params)
    sample = tuple(accounts[:2])
    for query in PER_ACCOUNT:
        valid = PARAMS[query](t_ms, sample, hub_ids)
        probes = {
            "t that is no Monday": valid | {"t_ms": mistaken},
            "no account": valid | {"account_ids": []},
            "a repeated account": valid | {"account_ids": [sample[0], sample[0]]},
            "an unknown account": valid | {"account_ids": [sample[0], NO_ACCOUNT]},
        }
        if late:
            probes["an account that does not exist at t"] = valid | {"account_ids": [late[0]]}
            if query in TAKES_HUBS:
                probes["a hub that does not exist at t"] = valid | {
                    "hub_accounts": [{"id": late[0]}]
                }
        for label, params in probes.items():
            _refusal(run, query, f"{where}, {label}", params)


def _refusal(run: _Comparison, query: str, where: str, params: dict[str, Any]) -> None:
    live = run.ask(False, query, params)
    reference = run.ask(True, query, params)
    if isinstance(live, Failure):
        run.found[query].record(where, [f"live request raised {live.why}"])
        return
    run.found[query].record(
        where,
        [] if live == reference else [f"refused as {live!r} live, {reference!r} in the reference"],
    )


def run_on(
    live: ParityExecutor,
    graph: ReferenceGraph,
    mondays: Sequence[int],
    config: ParityConfig,
) -> tuple[dict[str, QueryParity], Communities]:
    """The comparison of the live graph and the reference graph on the sampled Mondays, as the
    module says."""
    comparison = _Comparison(live, graph, config)
    _infrastructure(comparison)
    for t_ms in mondays:
        where = f"Monday {t_ms}"
        live_hubs, hubs = _monday_infrastructure(comparison, t_ms, where)
        mules, others = sample_accounts(graph, t_ms, config)
        accounts = [*mules, *others]
        if not accounts:
            continue
        answers = _per_account(comparison, t_ms, where, accounts, hubs)
        _leaf_flags(comparison, where, answers, live_hubs)
        _refusals(comparison, t_ms, where, accounts, hubs)
        sharing = Sharing(Lookback(graph.as_of(t_ms)))
        hub_ids = frozenset(hub.account_id for hub in hubs.accounts)
        for account in accounts:
            touches, between = community_leaves(sharing, account, hub_ids)
            comparison.communities.sampled += 1
            comparison.communities.touch_a_leaf += int(touches)
            comparison.communities.crowd_between_members += int(between)
    return comparison.found, comparison.communities


def record_text(parity: Parity, today: date) -> str:
    """The parity record of a run: a [[passed]] table per query that passed, with its version,
    the TigerGraph version, the world's identity and the date (installer.PARITY_FIELDS)."""
    lines = [
        "# Written by `mule parity`: the query versions that matched the reference graph on the\n",
        "# world named, committed with the code. A version with no entry here is never fetched.\n",
    ]
    for name in parity.passed:
        lines.append(
            "\n[[passed]]\n"
            f'query = "{name}"\n'
            f'version = "{VERSIONS[name]}"\n'
            f'tigergraph = "{parity.tigergraph}"\n'
            f'world = "{parity.identity}"\n'
            f"date = {today.isoformat()}\n"
        )
    return "".join(lines)


def run(
    executor: ConnectionExecutor,
    config: RunConfig = DEFAULT_CONFIG,
    *,
    exports: Path | None = None,
    record: Path | None = None,
    today: date | None = None,
) -> Parity:
    """Run mule parity on the loaded world: find its export (under exports, by default the
    loader's export root in .env), compare, and write the record (record, by default
    gsql/parity.toml) of the queries that passed. The run's Parity; the record is written even
    when some query failed, holding the others."""
    live = ParityExecutor(executor)
    found = read_world.parse(live.run(READ_WORLD, read_world.params()))
    folder, graph = find_export(export_root() if exports is None else exports, found)
    version = str(executor.call(lambda conn: conn.getVer(), what="getVer"))
    mondays = sample_mondays(config.calendar.mondays(), config.parity)
    queries, communities = run_on(live, graph, mondays, config.parity)
    parity = Parity(identity(found), version, folder, mondays, queries, communities)
    path = parity_record() if record is None else record
    path.write_text(record_text(parity, date.today() if today is None else today))
    return parity
