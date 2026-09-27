"""Pool counts over the payment messages of a context's own candidate pool."""

from __future__ import annotations

from typing import Any

from ..contract.feature_groups import FIRST_INFLOW_BANDS, PASS_THROUGH_RATIO, PASS_THROUGH_SECONDS
from ..contract.graph_schema import RELATIONS

_INCOMING = frozenset({"zelle_in", "payment_in"})
_OUTGOING = frozenset({"zelle_out", "payment_out"})


def pool_activity(context: dict[str, Any]) -> dict[str, float]:
    """The pool groups (pool_activity, pool_internal_inflows) of one context.

    These are counts over the payment messages of the context's own candidate pool,
    at most `recent + older + distinct` per relation (`PoolPlan`), not over the account's
    whole history; the `distinct` stratum favours new counterparties. TigerGraph returns
    the messages strictly before the context cutoff and computes their pair and flow
    fields there over the whole visible history, so every value is cutoff-safe. A
    first-time inflow has no earlier payment in its directed pair
    (`pair_prior_count == 0`; the GSQL pair is relation, rail and peer), and an internal
    one has a peer that is not external. A rapid pass-through is an inflow whose next
    outflow in the visible history (the per-message `flow_*` fields; pool events are
    never paired here) follows within PASS_THROUGH_SECONDS and moves PASS_THROUGH_RATIO
    of the inflow amount. Stubs and contexts without payments get zeros.
    """
    peers: dict[str, set[tuple[str, str]]] = {r: set() for r in RELATIONS[:4]}
    counts = dict.fromkeys(RELATIONS[:4], 0)
    bands = dict.fromkeys(FIRST_INFLOW_BANDS, 0)
    first_in = first_internal = pass_through = 0
    low, high = PASS_THROUGH_RATIO
    try:
        for m in context["messages"]:
            relation = m["relation"]
            if relation not in counts:
                continue  # associations
            counts[relation] += 1
            peers[relation].add((m["node_type"], m["node_id"]))
            if relation not in _INCOMING:
                continue
            if int(m["pair_prior_count"]) == 0:
                first_in += 1
                if not m["peer_external"]:
                    first_internal += 1
                    for band in bands:
                        bands[band] += bool(m["amount_present"]) and m["amount"] >= band
            pass_through += bool(
                m["flow_present"]
                and m["flow_ratio_present"]
                and m["flow_delay_seconds"] <= PASS_THROUGH_SECONDS
                and low <= m["flow_amount_ratio"] <= high
            )
    except KeyError as error:
        raise ValueError(f"Message lacks required field {error.args[0]!r}") from None
    values: dict[str, float] = {}
    for relation in RELATIONS[:4]:
        values[f"pool_{relation}_count"] = counts[relation]
        values[f"pool_{relation}_unique"] = len(peers[relation])
    values |= {
        "pool_in_unique": len({peer for r in _INCOMING for peer in peers[r]}),
        "pool_out_unique": len({peer for r in _OUTGOING for peer in peers[r]}),
        "pool_first_in": first_in,
        "pool_pass_through_1d": pass_through,
        "pool_first_in_internal": first_internal,
        **{f"pool_first_in_internal_{band}": n for band, n in bands.items()},
    }
    return values
