"""A Python mirror of the label reveal job (reveal_mule_labels).

`plan` recomputes the job's discovery channel and discovery time for every internal
mule, and the mules it reveals (every one discovered before its split's cutoff), from
the rows the reveal's inputs query prints (tigergraph.oracle.REVEAL_INPUTS_QUERY), with
the job's own hash (contract.discovery.reveal_uniforms) and constants
(contract.discovery.REVEAL_DEFAULTS). It reads nothing itself:
tests/integration/test_label_reveal.py compares it with a dry run of the installed
job (dry_run_differences), and diagnostics.reveal_spread runs it over many salts.
"""

from __future__ import annotations

import math
from typing import Any

from ..contract.discovery import REVEAL_DEFAULTS, reveal_uniforms
from ..contract.graph_schema import PHASE_SPLIT

DAY = 86400000.0
NEVER = 1.0e15


def lognormal(median: float, sigma: float, u1: float, u2: float) -> float:
    return math.exp(
        math.log(median) + sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
    )


def available_ms(mule: dict[str, Any], data_end_ms: int) -> int:
    """The availability clock the job writes for a mule of `plan`.

    The end of the discovery's UTC day, never after the last event and never before
    the account's first observation.
    """
    day_end = (int(mule["t"] // DAY) + 1) * 86400000 - 1
    return max(min(day_end, data_end_ms), int(mule["first"]))


def plan(inputs: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    """The job's reveal for query parameters `params`, and its constants unless they say.

    Returns every mule (split `part`, discovery time `t`, `channel`) and the revealed
    account IDs: every mule discovered before its split's cutoff. A mule that sent no
    event gets its internal vertex ID as draw key in the job; the inputs cannot show that
    ID, so such a mule keeps the query's sentinel key here.
    """
    p = {**REVEAL_DEFAULTS, **params}
    salt = int(p["salt"])
    cutoffs = {1: p["train_cutoff_ms"], 2: p["validation_cutoff_ms"], 3: p["test_cutoff_ms"]}
    mules: dict[str, dict[str, Any]] = {}
    for row in inputs[0]["M"]:
        a = row["attributes"]
        mules[row["v_id"]] = {
            "part": a["M.@part"],
            "key": a["M.@key"],
            "first": a["M.first_seen_ts_ms"],
            "inflows": [tuple(int(x) for x in s.split(":")) for s in a["M.@inflows"]],
        }
    links: list[tuple[int, int, list[str]]] = []
    for name in ("zelle_links", "payment_links"):
        rows = next(r[name] for r in inputs if name in r)
        for row in rows:
            a = row["attributes"]
            prefix = "LZ" if name == "zelle_links" else "LP"
            links.append(
                (a[f"{prefix}.event_seq"], a[f"{prefix}.event_ts_ms"], sorted(a[f"{prefix}.@ends"]))
            )
    for m in mules.values():
        m["cutoff"] = cutoffs.get(m["part"], 0)
        reports = []
        for seq, available in m["inflows"]:
            u = reveal_uniforms(seq, salt, 11)
            if not u[0] < p["p_report"]:
                continue
            n1 = math.sqrt(-2 * math.log(u[2])) * math.cos(2 * math.pi * u[3])
            n2 = math.sqrt(-2 * math.log(u[6])) * math.cos(2 * math.pi * u[7])
            n3 = math.sqrt(-2 * math.log(u[9])) * math.cos(2 * math.pi * u[10])
            if u[1] < 0.69:
                report = math.exp(1.3 * n1)
            elif u[1] < 0.79:
                report = math.exp(math.log(14.0) + n1)
            else:
                report = math.exp(math.log(60.0) + n1)
            fast = 1.0 if u[4] < 0.85 else 0.0
            notify = fast * u[5] + (1 - fast) * math.exp(math.log(7.0) + n2)
            reports.append(
                (available + (report + notify) * DAY, u[8], math.exp(math.log(5.0) + n3))
            )
        first = min((r[0] for r in reports), default=NEVER)
        react = NEVER
        for when, action, confirm in reports:
            if action < (p["p_action_first"] if when <= first else p["p_action_later"]):
                react = min(react, when + confirm * DAY)
        pro = NEVER
        if p["proactive_per_day"] > 0:
            u_pro = reveal_uniforms(m["key"], salt + 1000, 1)[0]
            pro = m["first"] + (-math.log(u_pro) / p["proactive_per_day"]) * DAY
        m["t"], m["channel"] = (react, "victim_report") if react <= pro else (pro, "monitoring")
    for _ in range(3):
        trace: dict[str, dict[int, tuple[int, float]]] = {}
        for seq, ts, ends in links:
            carriers = {
                mules[e]["key"]: mules[e]["t"]
                for e in ends
                if mules[e]["t"] < NEVER and ts < mules[e]["t"]
            }
            for b in ends:
                for source, found in carriers.items():
                    if source == mules[b]["key"]:
                        continue
                    best = trace.setdefault(b, {}).get(source)
                    trace[b][source] = (
                        min(best[0], seq) if best else seq,
                        min(best[1], found) if best else found,
                    )
        for b, sources in trace.items():
            candidate = NEVER
            for source, (seq, found) in sources.items():
                u = reveal_uniforms(seq * 31 + source, salt + 3000, 3)
                if u[0] < p["trace_probability"]:
                    candidate = min(candidate, found + lognormal(30.0, 1.0, u[1], u[2]) * DAY)
            if candidate < mules[b]["t"]:
                mules[b]["t"], mules[b]["channel"] = candidate, "network_trace"
    revealed = {k for k, m in mules.items() if 1 <= m["part"] <= 3 and m["t"] < m["cutoff"]}
    return {"mules": mules, "revealed": revealed}


def counts_by_split(result: dict[str, Any]) -> dict[int, int]:
    """Per scope partition 1..3, how many mules plan's result reveals."""
    return {
        part: sum(1 for k in result["revealed"] if result["mules"][k]["part"] == part)
        for part in PHASE_SPLIT
    }


def dry_run_differences(expected: dict[str, Any], dry_run: dict[str, Any]) -> list[str]:
    """Where a dry run of the installed job disagrees with plan's result; empty if nowhere.

    ``dry_run`` is the job's merged output with apply = FALSE: its revealed mules with
    their discovery channel and availability clock (``known_ts_ms``), the data end
    and the revealed mules per split. The job's map has no entry for a split without
    revealed mules.
    """
    actual = {row["account_id"]: row for row in dry_run["revealed_mules"]}
    failures: list[str] = []
    if set(actual) != expected["revealed"]:
        failures.append(
            f"revealed sets differ: only GSQL {sorted(set(actual) - expected['revealed'])[:5]}, "
            f"only Python {sorted(expected['revealed'] - set(actual))[:5]}"
        )
    data_end = int(dry_run["data_end_ts_ms"])
    for account, row in actual.items():
        mule = expected["mules"].get(account)
        if mule is None:
            continue
        known = available_ms(mule, data_end)
        if row["channel"] != mule["channel"] or row["known_ts_ms"] != known:
            failures.append(
                f"{account}: GSQL {row['channel']} {row['known_ts_ms']}, "
                f"Python {mule['channel']} {known}"
            )
    revealed = counts_by_split(expected)
    gsql_revealed = {part: int(dry_run["revealed"].get(str(part), 0)) for part in revealed}
    if revealed != gsql_revealed:
        failures.append(f"revealed counts differ: GSQL {gsql_revealed}, Python {revealed}")
    return failures
