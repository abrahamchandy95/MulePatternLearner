import sys
from typing import cast

from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings
from mule_pattern_learner.tigergraph.gsql_paths import gsql_path


def _gsql(client: Client, statement: str) -> str:
    return cast(str, client.conn.gsql(statement))


def _run(client: Client, name: str, params: dict[str, object]) -> list[object]:
    return cast(list[object], client.conn.runInstalledQuery(name, params))


def _scalar(raw: list[object], key: str) -> object:
    for block in raw:
        if isinstance(block, dict) and key in block:
            return cast(dict[str, object], block)[key]
    return None


def main() -> int:
    settings = Settings()
    client = Client(settings)
    print(f"graph: {settings.graphname}\n")

    path = gsql_path("diagnose_sentinels")
    if not path.is_file():
        print(f"missing: {path}")
        return 1

    text = path.read_text(encoding="utf-8")
    out = _gsql(client, text + "\nINSTALL QUERY diagnose_sentinels\n")
    if any(m in out.lower() for m in ("error", "fail", "could not", "cannot")):
        print("install failed:\n" + out)
        return 1
    print("installed diagnose_sentinels ... ok\n")

    raw = _run(client, "diagnose_sentinels", {"page_size": 5000})

    total = _scalar(raw, "total_accounts")
    dsl = _scalar(raw, "days_since_last_eq_neg1")
    age = _scalar(raw, "account_age_eq_neg1")
    mit = _scalar(raw, "mean_inter_eq_neg1")
    neg1_zero = _scalar(raw, "neg1_AND_zero_txns")
    mit_one = _scalar(raw, "mean_inter_neg1_AND_one_txn")
    span0 = _scalar(raw, "activity_span_is_zero")
    spann = _scalar(raw, "activity_span_nonzero")
    cmin = _scalar(raw, "min_com_size")
    cmax = _scalar(raw, "max_com_size")

    print(f"accounts sampled                         : {total}")
    print()
    print("--- the -1 sentinel ---")
    print(f"days_since_last_txn == -1                : {dsl}")
    print(f"account_age_days == -1                   : {age}")
    print(f"mean_inter_txn_days == -1                : {mit}")
    print(f"  of which (-1 AND zero transactions)    : {neg1_zero}")
    print(f"  mean_inter==-1 AND exactly 1 txn       : {mit_one}")
    print("  -> if days_since_last==-1 count equals the zero-txn count,")
    print("     then -1 means 'account had no transactions'.")
    print()
    print("--- activity_span_days ---")
    print(f"activity_span_days == 0                  : {span0}")
    print(f"activity_span_days != 0                  : {spann}")
    print("  -> if nonzero count is 0, the feature is dead (always 0).")
    print()
    print("--- com_size spread ---")
    print(f"min com_size                             : {cmin}")
    print(f"max com_size                             : {cmax}")
    print("  -> if min == max, com_size is constant across all sampled accounts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
