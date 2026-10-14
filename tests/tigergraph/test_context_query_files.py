"""Offline checks of the context query file (no server contact).

The file is the source: nothing generates it, so these tests keep it in step with the
Python contracts it shares (the feature groups, the request and pool bounds, the
contract it prints) and with the rules a context query keeps. Text is compared whatever
its layout (testing.gsql_blocks).
"""

from __future__ import annotations

import re

from mule_pattern_learner.contract.bounds import POOL, REQUEST_KEYS
from mule_pattern_learner.contract.graph_schema import (
    ASSOCIATION_TARGETS,
    ASSOCIATIONS,
    NO_SPLIT,
    PHASE_SPLIT,
)
from mule_pattern_learner.contract.server import CONTEXT_CONTRACT, CONTEXT_QUERY
from mule_pattern_learner.testing.gsql_blocks import FOURIER_QUERY, as_interpreted, blocks, flat
from mule_pattern_learner.tigergraph.gsql_text import (
    definitions,
    parameter_names,
    query_contract,
)

# The context query's parameters: the requests, their phase and pools, and the spot check.
PARAMETERS = {
    "node_types",
    "node_ids",
    "cutoff_seqs",
    "cutoff_times",
    "per_relation",
    "visibility_phase",
    "k_old",
    "k_div",
    "k_assoc",
    "max_history",
    "emit_encodings",
}


def signature(query: str) -> str:
    return query.split("CREATE OR REPLACE QUERY", 1)[1].split("SYNTAX V2", 1)[0]


def test_the_context_query_uses_no_labels(text: str) -> None:
    for field in ("is_mule", "fraud_label", "pu_label", "ring_id"):
        assert field not in text
    assert FOURIER_QUERY in text
    assert "e.valid_from_seq <= state_seq" in text
    assert "state_seq < e.valid_to_seq" in text


def test_the_query_reads_only_usd_payments_and_uses_no_global_reference_date(text: str) -> None:
    # Every payment read is in USD: the candidates of the four relations and those of the
    # prior-pair scan.
    query = flat(text)
    candidates = re.findall(
        r"\w*Candidates_\w+ = SELECT t FROM .*? WHERE (.*?)(?: ACCUM |;)", query
    )
    assert len(candidates) == 6
    assert all('t.currency == "USD"' in where for where in candidates)
    assert '"history_capacity_exceeded"' in text
    assert "now()" not in text.lower()


def test_the_query_prints_the_contract_of_its_text(text: str) -> None:
    # A changed query needs a new contract; the failure prints the value to set.
    derived = query_contract("context_", text, CONTEXT_CONTRACT)
    assert derived == CONTEXT_CONTRACT, derived
    assert text.count(f'"{CONTEXT_CONTRACT}" AS contract_version') == 1
    # Comments, whitespace and the case outside string literals do not count; any other
    # change names a new contract.
    reformatted = "/* a comment */\n" + "".join(f"    {line}\n" for line in text.splitlines())
    assert query_contract("context_", reformatted, CONTEXT_CONTRACT) == CONTEXT_CONTRACT
    renamed = text.replace(f"QUERY {CONTEXT_QUERY}(", "QUERY fetch_other_context(")
    assert query_contract("context_", renamed, CONTEXT_CONTRACT) != CONTEXT_CONTRACT


def test_the_query_computes_every_group_whatever_the_plan(text: str) -> None:
    # No group flag: the query computes every group TigerGraph supplies, and a source checks
    # the groups of its plan. Only the spot check's Fourier vectors are on request.
    assert parameter_names(definitions(text)[CONTEXT_QUERY]) == PARAMETERS
    assert "include_" not in text


def test_signature_order_and_the_bounds_of_contract_bounds(text: str) -> None:
    params = flat(signature(text))
    assert "INT max_history = 2048, BOOL emit_encodings = FALSE )" in params
    # The request and pool checks refuse what contract.bounds excludes.
    pool = {name: (bound.low, bound.high) for name, bound in POOL.items()}
    for bound in (
        f"@@request_ids.size() > {REQUEST_KEYS.high}",
        "per_relation < {} OR per_relation > {}".format(*pool["recent"]),
        "k_old < {} OR k_old > {}".format(*pool["older"]),
        "k_div < {} OR k_div > {}".format(*pool["distinct"]),
        "k_assoc < {} OR k_assoc > {}".format(*pool["associations"]),
        "max_history < {} OR max_history > {}".format(*pool["max_history"]),
    ):
        assert bound in flat(text)


# How a context query reads the graph's one split: a membership of the ready split in a
# partition up to the phase.
MEMBERSHIP = (
    "-(Entity_In_Training_Scope>:membership)- Temporal_Training_Scope:r WHERE r.ready AND "
    "membership.partition >= 1 AND membership.partition <= visibility_phase ACCUM"
)


def test_a_phase_of_the_one_split_or_no_split_decides_what_a_context_sees(text: str) -> None:
    # No scope id: phase 0 (the default) is no split, 1 to 3 the phases of the graph's one
    # ready split, and any other phase is refused with the other invalid parameters.
    query = flat(text)
    assert "scope_id" not in text
    assert f"INT visibility_phase = {NO_SPLIT}," in flat(signature(text))
    assert (
        f"OR visibility_phase < {NO_SPLIT} OR visibility_phase > {max(PHASE_SPLIT)} THEN "
        'PRINT "invalid_parameters" AS status; RETURN; END;'
    ) in query
    assert (
        "IF visibility_phase > 0 THEN ScopeCatalog = {Temporal_Training_Scope.*}; "
        "RequestedScope = SELECT r FROM ScopeCatalog:r WHERE r.ready; "
        'IF RequestedScope.size() != 1 THEN PRINT "scope_not_ready" AS status; RETURN; END; END;'
    ) in query
    # Every membership read is the ready split's, and only in a phase of it.
    reads = query.count("-(Entity_In_Training_Scope>:membership)-")
    assert reads == query.count(MEMBERSHIP) > 0
    guarded = blocks(text, "IF visibility_phase > 0 THEN")
    assert sum(flat(block).count(MEMBERSHIP) for block in guarded) == reads
    # Each row's key names its phase, never a scope.
    assert query.count("visibility_phase AS visibility_phase,") == 1


def test_ids_resolve_without_runtime_errors(text: str) -> None:
    for node_type in ("Account", "Token", "Party", "Device", "IP", "Address"):
        assert f'to_vertex_set(@@requested_{node_type}, "{node_type}")' in text
    # to_vertex is used once, after the typed lookup has proven the ID exists.
    assert text.count("to_vertex(") == 1
    assert text.index('"missing_entity"') < text.index("root = to_vertex(root_id, root_type)")


def test_interpreter_compatible_text(text: str) -> None:
    assert "getAttr" not in text
    body = text.split("SYNTAX V2 {", 1)[1]
    declarations, statements = body.split("@@request_ids = node_ids;", 1)
    scalars = set(
        re.findall(
            r"^\s+(?:INT|UINT|DOUBLE|BOOL|STRING|VERTEX|PairKey)\s+(\w+)", declarations, re.M
        )
    )
    globals_ = set(re.findall(r"@@(\w+)", text))
    params = set(re.findall(r"(?:STRING|INT|UINT|BOOL|LIST<\w+>)\s+(\w+)", signature(text)))
    assert "peer_key" in scalars
    assert not scalars & globals_
    assert not params & globals_
    # A query declares exactly the accumulators it uses: no use without its declaration
    # and no declaration without a use.
    assert set(re.findall(r"@@(\w+)", declarations)) == globals_
    assert set(re.findall(r"@@(\w+)", "@@request_ids = node_ids;" + statements)) == globals_
    interpreted = as_interpreted(text)
    assert flat(interpreted).startswith("INTERPRET QUERY ( LIST<STRING> node_types")
    assert "CREATE" not in interpreted


def test_no_per_event_point_selects(text: str) -> None:
    for pattern in ("to_vertex(item.", "Event = {", "Peer = {", "@@sender_vertices"):
        assert pattern not in text
    loops = list(blocks(text, "FOREACH item IN @@events DO"))
    assert loops
    for loop in loops:
        assert " SELECT " not in flat(loop)


def test_encodings_only_on_request(text: str) -> None:
    guard = "IF emit_encodings THEN"
    calls = [loop for loop in blocks(text, guard) if f"{FOURIER_QUERY}(" in loop]
    assert len(calls) == 2
    assert flat(text).count(f"{FOURIER_QUERY}(") == 2
    assert "@@age_encoding AS age_encoding, @@gap_encoding AS gap_encoding" in flat(text)


RELATION_EDGES = (
    "Account_Sent_Zelle_Transfer>|Token_Sent_Transfer>",
    "Account_Received_Zelle_Transfer>|Token_Received_Transfer>",
    "Account_Initiated_Transaction>|Token_Sent_Transaction>",
    "Account_Received_Transaction>|Token_Received_Transaction>",
)


def scans(query: str, edge: str) -> list[int]:
    """Where a query's flat text traverses a relation from the visible roots."""
    return [m.start() for m in re.finditer(re.escape(f"Visible:s -(({edge}):e)-"), flat(query))]


def test_each_relation_is_scanned_once(text: str) -> None:
    # The training query computes no floating sum, so each relation is scanned once, by
    # the candidate SELECT.
    for edge in RELATION_EDGES:
        (scan,) = scans(text, edge)
        assert re.search(r"Candidates_\w+ = SELECT t FROM $", flat(text)[:scan])


def test_pair_history_paths(text: str) -> None:
    # Path A (chronology) for Account roots and path B (the prior-pair scan) for Token
    # roots, the only other roots with payments.
    assert 'use_chronology = root_type == "Account";' in flat(text)
    assert 'use_prior_scan = root_type == "Token";' in flat(text)
    assert 'IF root_type == "Account" OR root_type == "Token" THEN Candidates_' in flat(text)


def test_each_association_relation_is_read_once_by_its_target_type(text: str) -> None:
    # One SELECT per target type reads the root's relations to it, and e.type names the
    # relation of each row; a root has at most one relation to each target type.
    targets: dict[str, str] = {}
    for (forward, reverse), (to_forward, to_reverse) in zip(
        ASSOCIATIONS, ASSOCIATION_TARGETS, strict=True
    ):
        targets |= {forward: to_forward, reverse: to_reverse}
    selects = re.findall(r"Assoc_(\w+) = SELECT t FROM Visible:s -\(\((.*?)\):e\)- (\w+):t", text)
    read: dict[str, str] = {}
    for name, edges, target in selects:
        assert name == target
        for relation in edges.removesuffix(">").split(">|"):
            assert relation not in read, relation
            read[relation] = target
    assert read == targets
    sources = {relation.split("_", 1)[0] for relation in targets}
    for source in sources:
        reached = [targets[r] for r in targets if r.split("_", 1)[0] == source]
        assert len(reached) == len(set(reached)), source
    # Messages list a root's relations in this order of target types.
    assert [name for name, _, _ in selects] == [
        "Party",
        "Account",
        "Token",
        "Device",
        "IP",
        "Address",
    ]
    assert text.count("e.type,") == len(selects)
