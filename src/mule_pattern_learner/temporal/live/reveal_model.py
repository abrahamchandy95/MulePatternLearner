"""A Python mirror of the label reveal job (temporal_reveal_mule_labels).

`plan` recomputes the job's discovery channel, discovery time, eligibility and
revealed set for every internal mule from the rows INPUTS_QUERY prints, with the
job's own hash (labels.reveal_uniforms) and parameter defaults
(labels.REVEAL_DEFAULTS). It reads nothing itself: scripts/temporal/verify_label_reveal.py
compares it with the installed job, and scripts/temporal/simulate_label_reveal.py
runs it over many salts.
"""

from __future__ import annotations

import math
from typing import Any

from .contract import PHASE_SPLIT
from .labels import REVEAL_DEFAULTS, reveal_uniforms

DAY = 86400000.0
NEVER = 1.0e15
# Read-only interpreted query: every input of the reveal, with the job's traversals.
# The first result holds the mules (split, draw key, first observation and the
# "event_seq:label_available_ts_ms" fraud-labelled Zelle inflows), the next two the
# Zelle and payment events between two mules.
INPUTS_QUERY = """
INTERPRET QUERY (STRING scope_id) FOR GRAPH Mule_Pattern_Learner {
  MaxAccum<INT> @part;
  MinAccum<INT> @key;
  OrAccum @mule;
  ListAccum<STRING> @inflows;
  SetAccum<STRING> @ends;
  Scopes = {Temporal_Training_Scope.*};
  Ready = SELECT r FROM Scopes:r WHERE r.scope_id == scope_id;
  M = {Account.*};
  M = SELECT a FROM M:a WHERE a.is_mule == 1 AND NOT a.is_external POST-ACCUM a.@mule += TRUE;
  S = SELECT a FROM Ready:r -(Training_Scope_Has_Entity>:e)- Account:a WHERE a.@mule ACCUM a.@part += e.partition;
  K1 = SELECT t FROM M:a -(Account_Initiated_Transaction>:e)- Payment_Transaction:t ACCUM a.@key += t.event_seq;
  K2 = SELECT z FROM M:a -(Account_Sent_Zelle_Transfer>:e)- Zelle_Transfer:z ACCUM a.@key += z.event_seq;
  F = SELECT z FROM M:a -(Account_Received_Zelle_Transfer>:e)- Zelle_Transfer:z
      WHERE z.fraud_label == 1 AND z.label_known
      ACCUM a.@inflows += (to_string(z.event_seq) + ":" + to_string(z.label_available_ts_ms));
  LZ = SELECT z FROM M:a -((Account_Sent_Zelle_Transfer>|Account_Received_Zelle_Transfer>):e)- Zelle_Transfer:z
       ACCUM z.@ends += a.id;
  LP = SELECT t FROM M:a -((Account_Initiated_Transaction>|Account_Received_Transaction>):e)- Payment_Transaction:t
       ACCUM t.@ends += a.id;
  LZ = SELECT z FROM LZ:z WHERE z.@ends.size() > 1;
  LP = SELECT t FROM LP:t WHERE t.@ends.size() > 1;
  PRINT M[M.id, M.first_seen_ts_ms, M.@part, M.@key, M.@inflows];
  PRINT LZ[LZ.event_seq, LZ.event_ts_ms, LZ.@ends] AS zelle_links;
  PRINT LP[LP.event_seq, LP.event_ts_ms, LP.@ends] AS payment_links;
}
"""


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
    """The job's reveal for query parameters `params` (absent ones take its defaults).

    Returns every mule (split `part`, discovery time `t`, `channel`, reports seen by
    the cutoff), the eligible and the revealed account IDs. A mule that sent no event
    gets its internal vertex ID as draw key in the job; the inputs cannot show that ID,
    so such a mule keeps the query's sentinel key here.
    """
    p = {**REVEAL_DEFAULTS, **params}
    salt, budget = int(p["salt"]), int(p["budget"])
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
        m["seen"] = sum(1 for r in reports if r[0] < m["cutoff"])
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
    eligible = {k: m for k, m in mules.items() if 1 <= m["part"] <= 3 and m["t"] < m["cutoff"]}
    revealed: set[str] = set()
    for part in (1, 2, 3):
        group = {k: m for k, m in eligible.items() if m["part"] == part}
        if not group:
            continue
        ev = {k: math.log(1 + m["seen"]) for k, m in group.items()}
        mean = sum(ev.values()) / len(ev)
        var = sum(v * v for v in ev.values()) / len(ev) - mean * mean
        w: dict[str, float] = {}
        for k, v in ev.items():
            z = (v - mean) / math.sqrt(var) if var > 1e-12 else 0.0
            w[k] = p["propensity_floor"] + (1 - p["propensity_floor"]) / (
                1 + math.exp(-p["propensity_slope"] * z)
            )
        quota_total = min(budget, len(group))
        if quota_total == 0:
            continue
        capped: dict[str, float] = {}
        while True:
            free = {k: v for k, v in w.items() if k not in capped}
            total = sum(free.values())
            new = {
                k: 1.0 for k, v in free.items() if (quota_total - len(capped)) * v / total >= 1.0
            }
            if not new:
                lam = {
                    **{k: (quota_total - len(capped)) * v / total for k, v in free.items()},
                    **capped,
                }
                break
            capped.update(new)
        q: dict[str, float] = {}
        for k in group:
            u = reveal_uniforms(mules[k]["key"], salt + 2000, 1)[0]
            q[k] = -1.0 if lam[k] >= 1.0 else u * (1 - lam[k]) / (lam[k] * (1 - u))
        revealed |= set(sorted(group, key=lambda k: q[k])[:quota_total])
    return {"mules": mules, "eligible": set(eligible), "revealed": revealed}


def counts_by_split(result: dict[str, Any], accounts: str) -> dict[int, int]:
    """Per scope partition 1..3, how many of result[accounts] ("eligible" or "revealed")."""
    return {
        part: sum(1 for k in result[accounts] if result["mules"][k]["part"] == part)
        for part in PHASE_SPLIT
    }
