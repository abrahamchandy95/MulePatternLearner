from typing import cast

from mule_pattern_learner.tigergraph.client import Client

_QUERY_NAME = "get_split_accounts"
_COL_ACCOUNT_ID = "account_id"
_COL_PU_LABEL = "pu_label"


class SplitSeeds:
    """
    One split's seed account ids, plus an in-memory lookup of each account's
    pu_label attribute (1 = revealed positive, 0 = unlabeled), read from the graph.

    pu_label lives on the account vertex; this just caches it by id so the
    training loop reads labels from memory instead of re-querying per batch.
    """

    account_ids: tuple[str, ...]
    pu_label_of: dict[str, int]

    def __init__(self, account_ids: tuple[str, ...], pu_label_of: dict[str, int]) -> None:
        self.account_ids = account_ids
        self.pu_label_of = pu_label_of

    @property
    def num_positives(self) -> int:
        return sum(1 for v in self.pu_label_of.values() if v == 1)

    @property
    def num_unlabeled(self) -> int:
        return sum(1 for v in self.pu_label_of.values() if v == 0)


def _parse_rows(raw: list[object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for block in raw:
        if not (isinstance(block, dict) and "result" in block):
            continue
        vertices = cast(dict[str, object], block)["result"]
        if not isinstance(vertices, list):
            continue
        for vertex in cast(list[object], vertices):
            if not isinstance(vertex, dict):
                continue
            attrs = cast(dict[str, object], vertex).get("attributes")
            if isinstance(attrs, dict):
                rows.append(cast(dict[str, object], attrs))
    return rows


def fetch_split_seeds(client: Client, split: str) -> SplitSeeds:
    """
    Fetch one split's accounts + pu_label from the graph.

    split is "train" | "val" | "test". Runs get_split_accounts and returns the
    seed ids plus the pu_label lookup the trainer feeds to the nnPU loss.
    """
    raw = cast(
        list[object],
        client.conn.runInstalledQuery(_QUERY_NAME, {"split": split}),
    )
    rows = _parse_rows(raw)
    account_ids: list[str] = []
    pu_label_of: dict[str, int] = {}
    for attrs in rows:
        account_id = str(attrs[_COL_ACCOUNT_ID])
        account_ids.append(account_id)
        pu_label_of[account_id] = int(cast(int, attrs[_COL_PU_LABEL]))
    return SplitSeeds(account_ids=tuple(account_ids), pu_label_of=pu_label_of)
