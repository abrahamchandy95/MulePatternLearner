import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
from pyTigerGraph import TigerGraphConnection

from mule_pattern_learner.data.pu_masking import (
    Bucket,
    MaskConfig,
    MaskRecord,
    apply_pu_mask,
    resolve_account_rings,
)
from mule_pattern_learner.data.splitting import (
    Split,
    SplitConfig,
    SplitRecord,
    split_accounts,
)
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings

_VERTEX_ACCOUNT = "Account"
_EXTRACT_QUERY = "get_masking_inputs"
_WRITEBACK_COLUMNS = ("is_train", "is_val", "is_test", "pu_label")

_TXN_SRC = "src_acct"
_TXN_DST = "dst_acct"
_TXN_RING = "ring_id"

# Large result set: lift the response size cap well above the default.
_EXTRACT_SIZE_LIMIT = 512 * 1024 * 1024  # 512 MB


@dataclass(frozen=True, slots=True)
class _Account:
    """One extracted account: identity, truth, ring, and leak-safe group key."""

    account_id: str
    is_mule: bool
    group: str | None
    ring_id: int


@dataclass(frozen=True, slots=True)
class _Args:
    transactions: Path
    out_dir: Path
    reveal_prevalence: float
    dark_ring_fraction: float
    val_fraction: float
    test_fraction: float
    seed: int
    dry_run: bool


def _parse_args() -> _Args:
    p = argparse.ArgumentParser(
        description="Apply PU masking + party-grouped split; write back to TigerGraph."
    )
    _ = p.add_argument("--transactions", type=Path, default=Path("data/transactions.csv"))
    _ = p.add_argument("--out-dir", type=Path, default=Path("data/masks"))
    _ = p.add_argument("--reveal-prevalence", type=float, default=0.04)
    _ = p.add_argument("--dark-ring-fraction", type=float, default=0.30)
    _ = p.add_argument("--val-fraction", type=float, default=0.15)
    _ = p.add_argument("--test-fraction", type=float, default=0.15)
    _ = p.add_argument("--seed", type=int, default=1337)
    _ = p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and save artifacts but do NOT write back to the graph.",
    )
    ns = p.parse_args()
    return _Args(
        transactions=cast(Path, ns.transactions),
        out_dir=cast(Path, ns.out_dir),
        reveal_prevalence=cast(float, ns.reveal_prevalence),
        dark_ring_fraction=cast(float, ns.dark_ring_fraction),
        val_fraction=cast(float, ns.val_fraction),
        test_fraction=cast(float, ns.test_fraction),
        seed=cast(int, ns.seed),
        dry_run=cast(bool, ns.dry_run),
    )


def _coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _coerce_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _group_key(party_id: str, entity_id: int) -> str | None:
    # Leak-safe split grouping: resolved entity (E:) > owning party (P:) > solo.
    # Whole groups never straddle train/val/test. Tier prefixes prevent an
    # entity id colliding with a party id of the same string.
    if entity_id >= 0:
        return f"E:{entity_id}"
    if party_id:
        return f"P:{party_id}"
    return None


def _extract_accounts(conn: TigerGraphConnection) -> list[_Account]:
    raw: object = cast(
        object,
        conn.runInstalledQuery(_EXTRACT_QUERY, usePost=True, sizeLimit=_EXTRACT_SIZE_LIMIT),
    )
    return _records_from_query_result(raw)


def _records_from_query_result(raw: object) -> list[_Account]:
    records: list[_Account] = []
    if not isinstance(raw, list):
        return records
    for block in cast(list[object], raw):
        if not isinstance(block, dict):
            continue
        for value in cast(dict[str, object], block).values():
            if not isinstance(value, list):
                continue
            for item in cast(list[object], value):
                if not isinstance(item, dict):
                    continue
                attrs_obj = cast(dict[str, object], item).get("attributes")
                if not isinstance(attrs_obj, dict):
                    continue
                attrs = cast(dict[str, object], attrs_obj)
                account_id = _coerce_str(attrs.get("account_id"))
                if not account_id:
                    continue
                is_mule = _coerce_int(attrs.get("is_fraud")) == 1
                party = _coerce_str(attrs.get("party_id"))
                entity = _coerce_int(attrs.get("resolved_entity_id"), default=-1)
                records.append(
                    _Account(
                        account_id=account_id,
                        is_mule=is_mule,
                        group=_group_key(party, entity),
                        ring_id=0,
                    )
                )
    return records


def _read_ring_endpoints(transactions_csv: Path) -> list[tuple[str, int]]:
    endpoints: list[tuple[str, int]] = []
    with transactions_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ring = _coerce_int(row.get(_TXN_RING))
            src = _coerce_str(row.get(_TXN_SRC))
            dst = _coerce_str(row.get(_TXN_DST))
            if src:
                endpoints.append((src, ring))
            if dst:
                endpoints.append((dst, ring))
    return endpoints


def _write_back(
    conn: TigerGraphConnection,
    account_ids: tuple[str, ...],
    splits: tuple[int, ...],
    pu_label: tuple[int, ...],
) -> int:
    # Bulk-write is_train/is_val/is_test AND pu_label onto Accounts. pu_label is
    # the PU training target read from the graph; it must never enter the
    # model's input feature list.
    df = pd.DataFrame(
        {
            "account_id": list(account_ids),
            "is_train": [1 if s == int(Split.TRAIN) else 0 for s in splits],
            "is_val": [1 if s == int(Split.VAL) else 0 for s in splits],
            "is_test": [1 if s == int(Split.TEST) else 0 for s in splits],
            "pu_label": list(pu_label),
        }
    )
    n: int = conn.upsertVertexDataFrame(
        df=df,
        vertexType=_VERTEX_ACCOUNT,
        v_id="account_id",  # pyright: ignore[reportArgumentType]
        attributes={c: c for c in _WRITEBACK_COLUMNS},
    )
    return n


def _save_artifacts(
    out_dir: Path,
    account_ids: tuple[str, ...],
    pu_label: tuple[int, ...],
    true_label: tuple[int, ...],
    bucket: tuple[int, ...],
    split: tuple[int, ...],
    seed: int,
    reveal_prevalence: float,
) -> tuple[Path, Path]:
    # The parquet is the EVALUATION artifact: it carries true_label and bucket
    # (the synthetic answer key) so the model can be scored on revealed vs.
    # hidden positives vs. dark-ring accounts. The graph only needs pu_label +
    # the split masks.
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.DataFrame(
        {
            "account_id": list(account_ids),
            "pu_label": list(pu_label),
            "true_label": list(true_label),
            "bucket": list(bucket),
            "split": list(split),
        }
    )
    parquet_path = out_dir / f"pu_labels_seed{seed}_p{reveal_prevalence}.parquet"
    labels.to_parquet(parquet_path, index=False)

    summary_rows: list[tuple[str, float | int]] = [
        ("reveal_prevalence", reveal_prevalence),
        ("seed", seed),
        ("accounts", len(account_ids)),
        ("revealed_pos", sum(1 for b in bucket if b == int(Bucket.REVEALED_POS))),
        ("hidden_pos", sum(1 for b in bucket if b == int(Bucket.HIDDEN_POS))),
        ("unlabeled_neg", sum(1 for b in bucket if b == int(Bucket.UNLABELED_NEG))),
        ("train", sum(1 for s in split if s == int(Split.TRAIN))),
        ("val", sum(1 for s in split if s == int(Split.VAL))),
        ("test", sum(1 for s in split if s == int(Split.TEST))),
    ]
    summary_path = out_dir / f"summary_seed{seed}_p{reveal_prevalence}.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)
    return parquet_path, summary_path


def main() -> None:
    args = _parse_args()
    mask_config = MaskConfig(
        reveal_prevalence=args.reveal_prevalence,
        dark_ring_fraction=args.dark_ring_fraction,
        seed=args.seed,
    )
    split_config = SplitConfig(
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    print("Connecting to TigerGraph ...")
    conn: TigerGraphConnection = Client(Settings()).conn

    print(f"Extracting accounts via '{_EXTRACT_QUERY}' ...")
    accounts = _extract_accounts(conn)
    print(f"  {len(accounts)} accounts pulled.")
    n_entity = sum(1 for a in accounts if a.group is not None and a.group.startswith("E:"))
    n_party = sum(1 for a in accounts if a.group is not None and a.group.startswith("P:"))
    n_solo = sum(1 for a in accounts if a.group is None)
    print(f"  grouping: {n_entity} by entity, {n_party} by party, {n_solo} unowned-singleton.")

    print(f"Resolving rings from {args.transactions} (local ledger; not in graph) ...")
    endpoints = _read_ring_endpoints(args.transactions)
    ring_of = resolve_account_rings(endpoints)
    print(f"  {sum(1 for v in ring_of.values() if v > 0)} accounts in a ring.")

    # ── step 1: masking (pu_label, true_label, bucket, forced_test) ──
    print("Applying PU mask ...")
    mask_records = [
        MaskRecord(account_id=a.account_id, is_mule=a.is_mule, ring_id=ring_of.get(a.account_id, 0))
        for a in accounts
    ]
    mask = apply_pu_mask(mask_records, mask_config)
    for k, v in mask.summary().items():
        print(f"  {k}: {v}")

    # ── step 2: party-grouped split, pinning the dark rings (forced_test) and
    # stratifying the revealed positives so val/test are guaranteed coverage ──
    print("Assigning train/val/test split (party-grouped, dark rings pinned to test) ...")
    revealed_positives = frozenset(aid for aid, y in zip(mask.account_ids, mask.pu_label) if y == 1)
    split_records = [SplitRecord(account_id=a.account_id, party_id=a.group) for a in accounts]
    split = split_accounts(
        split_records,
        split_config,
        force_test=mask.forced_test,
        stratify=revealed_positives,
    )
    for k, v in split.summary().items():
        print(f"  {k}: {v}")

    # masking and splitting are aligned to the same input order, so their
    # account_ids match; use one as the canonical id vector
    parquet_path, summary_path = _save_artifacts(
        args.out_dir,
        mask.account_ids,
        mask.pu_label,
        mask.true_label,
        mask.bucket,
        split.split,
        args.seed,
        args.reveal_prevalence,
    )
    print(f"Saved labels  -> {parquet_path}")
    print(f"Saved summary -> {summary_path}")

    if args.dry_run:
        print("Dry run: nothing written back to the graph.")
        return

    print("Writing splits + pu_label back to the graph ...")
    n = _write_back(conn, mask.account_ids, split.split, mask.pu_label)
    print(f"  {n} Account vertices updated (is_train/is_val/is_test/pu_label).")
    print("Done.")


if __name__ == "__main__":
    main()
