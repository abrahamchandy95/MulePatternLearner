from __future__ import annotations

from mule_pattern_learner.pyg.fanout import NeighborFanout, NeighborFanoutError


def main() -> int:
    print("=" * 66)
    print("STEP 1: default NeighborFanout (validated defaults)")
    print("=" * 66)
    fanout = NeighborFanout()
    print("  per-relation neighbor counts, by round:")
    print(f"  {'relation':22s} {'round 1':>9s} {'round 2':>9s}")
    rows = [
        ("HAS_PAID", fanout.has_paid, fanout.has_paid_round2),
        ("Account_Account", fanout.account_account, fanout.account_account_round2),
        ("Account->Party", fanout.acct_party, fanout.acct_party_round2),
        ("Party->Entity", fanout.party_entity, fanout.party_entity_round2),
        ("Entity->Party", fanout.entity_party, fanout.entity_party_round2),
        ("Party->Account", fanout.party_acct, fanout.party_acct_round2),
    ]
    for label, r1, r2 in rows:
        print(f"  {label:22s} {r1:>9d} {r2:>9d}")
    print("  note: round 2 <= round 1 everywhere (explosion control).")
    print("  clean access: fanout.has_paid  (not fanout.fanout_has_paid)")

    print()
    print("=" * 66)
    print("STEP 2: as_query_params() -> the GSQL parameter dict")
    print("=" * 66)
    params = fanout.as_query_params()
    print(f"  {len(params)} params; GSQL's 'fanout_*' / '_2' spelling lives only here:")
    for k, v in params.items():
        print(f"    {k:28s} = {v}")

    print()
    print("=" * 66)
    print("STEP 3: custom overrides (clean, de-smurfed kwargs)")
    print("=" * 66)
    custom = NeighborFanout(has_paid=25, has_paid_round2=10)
    cp = custom.as_query_params()
    print("  NeighborFanout(has_paid=25, has_paid_round2=10)")
    print(f"    -> fanout_has_paid       = {cp['fanout_has_paid']}  (overridden)")
    print(f"    -> fanout_has_paid_2     = {cp['fanout_has_paid_2']}  (overridden)")
    print(f"    -> fanout_account_account= {cp['fanout_account_account']}  (still default)")

    print()
    print("=" * 66)
    print("STEP 4: validation rejects bad values (fail fast)")
    print("=" * 66)
    try:
        _ = NeighborFanout(has_paid=0)
        print("  zero: NOT rejected (unexpected!)")
    except NeighborFanoutError as e:
        print(f"  zero fanout: rejected -> {e}")
    try:
        _ = NeighborFanout(account_account=-5)
        print("  negative: NOT rejected (unexpected!)")
    except NeighborFanoutError as e:
        print(f"  negative fanout: rejected -> {e}")

    print()
    print("  A wrong fanout would silently break sampling; NeighborFanout")
    print("  refuses to construct, so the bug can't reach GSQL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
