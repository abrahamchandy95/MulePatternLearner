#!/usr/bin/env python3
"""
temporal_flow_aggs.py

Build ACCOUNT_FLOW_AGG.csv (directed account-to-account flow aggregates with a
fixed-width temporal bin sequence) from PhantomLedger's raw transactions.csv.

Design rationale (research-grounded)
------------------------------------
Mule detection is fundamentally temporal. The literature shows two distinct
mule time signals: a micro-tempo (rapid in->out "flipping", minutes-to-hours)
and a macro-evolution (dormant -> burst -> abandoned, over weeks). Because the
data is already aggregated per account-pair, the micro-tempo is unrecoverable;
the macro-evolution is what we can model — and it is exactly the "pattern over
weeks" we want the GNN to learn.

We therefore attach to each HAS_PAID edge a FIXED-WIDTH time-bin sequence:
amount and transaction count per bin, where each bin is a fixed calendar span
(default 14 days / bi-weekly). The number of bins floats with the dataset span
(a 180-day sim -> ~13 bins; a 365-day sim -> ~26), but the bin WIDTH is fixed so
temporal patterns are directly comparable across datasets of different lengths.

This is the discrete-time "binned dt" representation production fraud GNNs use,
expressed as a per-edge sequence (NOT calendar-period vertices, which create
message-passing hubs and manufacture false similarity between accounts that
merely share a calendar period). A downstream sequence encoder (GRU / Markov-
style) can consume the ordered bins; flat consumption also works.

Bin 0 = OLDEST span (dataset start), bin N-1 = MOST RECENT.

The all-time aggregates (total_amount, txn_count, first/last) match the C++
derivation. The bin sequences replace the kit's fixed 30d/90d columns.

Output columns
--------------
  from_id, to_id, total_amount, txn_count, first_txn_date, last_txn_date,
  span_days, num_bins, bin_days,
  amount_bins, count_bins
where amount_bins / count_bins are semicolon-separated, comma-free strings
(e.g. "0.0;340.5;1200.0") — parsed into LIST<DOUBLE> / LIST<INT> by
TigerGraph's SPLIT(col, ";") in the loading job.

Datetime format: "%Y-%m-%d %H:%M:%S" (PhantomLedger formatTimestamp output).

Usage
-----
  python temporal_flow_aggs.py \
      --in  /path/to/transactions.csv \
      --out /path/to/ACCOUNT_FLOW_AGG.csv \
      --bin-days 14
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone

DATETIME_FMT: str = "%Y-%m-%d %H:%M:%S"
SECONDS_PER_DAY: int = 86_400


def parse_ts(s: str) -> int:
    """Parse a PhantomLedger timestamp string to epoch seconds (UTC)."""
    dt: datetime = datetime.strptime(s.strip(), DATETIME_FMT).replace(
        tzinfo=timezone.utc
    )
    return int(dt.timestamp())


def fmt_ts(epoch: int) -> str:
    """Render epoch seconds back to the PhantomLedger datetime string."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(DATETIME_FMT)


@dataclass(slots=True)
class PairAgg:
    """Aggregates for one directed (src, dst) pair."""

    total_amount: float = 0.0
    txn_count: int = 0
    first_ts: int = 0
    last_ts: int = 0
    amount_bins: list[float] = field(default_factory=list)
    count_bins: list[int] = field(default_factory=list)
    _seen: bool = False

    def ensure_bins(self, n: int) -> None:
        if not self.amount_bins:
            self.amount_bins = [0.0] * n
            self.count_bins = [0] * n

    def add_base(self, amount: float, ts: int) -> None:
        """All-time totals + first/last timestamp (C++-faithful)."""
        self.total_amount += amount
        if not self._seen:
            self.first_ts = ts
            self.last_ts = ts
            self._seen = True
        else:
            if ts < self.first_ts:
                self.first_ts = ts
            if ts > self.last_ts:
                self.last_ts = ts
        self.txn_count += 1

    def add_bin(self, amount: float, bin_index: int) -> None:
        self.amount_bins[bin_index] += amount
        self.count_bins[bin_index] += 1


@dataclass(slots=True)
class LedgerRow:
    src: str
    dst: str
    amount: float
    ts: int


def iter_rows(path: str) -> Iterator[LedgerRow]:
    """Yield typed rows from transactions.csv."""
    with open(path, newline="") as fh:
        reader: csv.DictReader[str] = csv.DictReader(fh)
        fields: list[str] = list(reader.fieldnames or [])
        required: set[str] = {"src_acct", "dst_acct", "amount", "ts"}
        missing: set[str] = required - set(fields)
        if missing:
            raise SystemExit(
                f"transactions.csv missing required columns {sorted(missing)}; found {fields}"
            )
        for row in reader:
            yield LedgerRow(
                src=row["src_acct"],
                dst=row["dst_acct"],
                amount=float(row["amount"]),
                ts=parse_ts(row["ts"]),
            )


def bin_index(ts: int, min_ts: int, bin_seconds: int, num_bins: int) -> int:
    """Map a timestamp to its fixed-width bin index (0 = oldest).

    Bins tile forward from min_ts in steps of bin_seconds. The final
    timestamp lands in the last bin (clamped).
    """
    if bin_seconds <= 0:
        return num_bins - 1
    raw: int = (ts - min_ts) // bin_seconds
    if raw < 0:
        return 0
    if raw >= num_bins:
        return num_bins - 1
    return raw


def fmt_float_list(xs: list[float], decimals: int) -> str:
    """Render a float list as a semicolon-separated, comma-free string.

    Semicolon (not comma) so the value is a single CSV cell, and bracket-free
    so TigerGraph's SPLIT(col, ";") parses it directly into a LIST<DOUBLE>.
    """
    return ";".join(f"%.{decimals}f" % x for x in xs)


def fmt_int_list(xs: list[int]) -> str:
    """Render an int list as a semicolon-separated, comma-free string for
    TigerGraph SPLIT(col, ";") into a LIST<INT>."""
    return ";".join(str(x) for x in xs)


def build(in_path: str, out_path: str, bin_days: int, money_decimals: int) -> None:
    if bin_days < 1:
        raise SystemExit(f"--bin-days must be >= 1, got {bin_days}")

    bin_seconds: int = bin_days * SECONDS_PER_DAY
    aggs: dict[tuple[str, str], PairAgg] = {}
    min_ts: int = 0
    max_ts: int = 0
    n_rows: int = 0

    # ── Pass 1: base aggregates + global min/max timestamp ──
    for r in iter_rows(in_path):
        key: tuple[str, str] = (r.src, r.dst)
        agg: PairAgg | None = aggs.get(key)
        if agg is None:
            agg = PairAgg()
            aggs[key] = agg
        agg.add_base(r.amount, r.ts)
        if n_rows == 0:
            min_ts = r.ts
            max_ts = r.ts
        else:
            if r.ts < min_ts:
                min_ts = r.ts
            if r.ts > max_ts:
                max_ts = r.ts
        n_rows += 1

    if n_rows == 0:
        raise SystemExit("No transactions found in input; nothing to write.")

    span_seconds: int = max_ts - min_ts
    # Number of fixed-width bins needed to cover the span (at least 1).
    num_bins: int = max(1, (span_seconds // bin_seconds) + 1)

    # ── Pass 2: per-bin aggregation (needs the global min_ts / num_bins) ──
    for r in iter_rows(in_path):
        agg = aggs[(r.src, r.dst)]
        agg.ensure_bins(num_bins)
        idx: int = bin_index(r.ts, min_ts, bin_seconds, num_bins)
        agg.add_bin(r.amount, idx)

    # ── Write output ──
    money_fmt: str = f"%.{money_decimals}f"
    header: list[str] = [
        "from_id",
        "to_id",
        "total_amount",
        "txn_count",
        "first_txn_date",
        "last_txn_date",
        "span_days",
        "num_bins",
        "bin_days",
        "amount_bins",
        "count_bins",
    ]

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)

        def emit(fields_row: list[str]) -> None:
            writer.writerow(fields_row)
            return None

        emit(header)
        for (src, dst), agg in aggs.items():
            agg.ensure_bins(num_bins)
            pair_span_days: float = (agg.last_ts - agg.first_ts) / SECONDS_PER_DAY
            emit(
                [
                    src,
                    dst,
                    money_fmt % agg.total_amount,
                    str(agg.txn_count),
                    fmt_ts(agg.first_ts),
                    fmt_ts(agg.last_ts),
                    f"{pair_span_days:.4f}",
                    str(num_bins),
                    str(bin_days),
                    fmt_float_list(agg.amount_bins, money_decimals),
                    fmt_int_list(agg.count_bins),
                ]
            )

    span_days_total: float = span_seconds / SECONDS_PER_DAY
    summary: str = (
        f"Wrote {len(aggs):,} directed account-pair edges from {n_rows:,} transactions.\n"
        f"  dataset span: {fmt_ts(min_ts)} .. {fmt_ts(max_ts)} ({span_days_total:.1f} days)\n"
        f"  bins: {num_bins} x {bin_days} days (bin0=oldest, bin{num_bins - 1}=newest)\n"
        f"  -> {out_path}\n"
    )
    sys.stderr.write(summary)  # pyright: ignore[reportUnusedCallResult]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fixed-width temporal-bin flow aggregates from a raw ledger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument(
        "--in", dest="in_path", required=True, help="path to transactions.csv"
    )
    _ = parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="path to write ACCOUNT_FLOW_AGG.csv",
    )
    _ = parser.add_argument(
        "--bin-days",
        dest="bin_days",
        type=int,
        default=14,
        help="fixed bin width in days (default 14 = bi-weekly)",
    )
    _ = parser.add_argument(
        "--money-decimals",
        dest="money_decimals",
        type=int,
        default=2,
        help="decimal places for amount values (default 2)",
    )

    ns: dict[str, object] = vars(parser.parse_args())
    in_path_obj: object = ns["in_path"]
    out_path_obj: object = ns["out_path"]
    bin_days_obj: object = ns["bin_days"]
    money_decimals_obj: object = ns["money_decimals"]

    assert isinstance(in_path_obj, str)
    assert isinstance(out_path_obj, str)
    assert isinstance(bin_days_obj, int)
    assert isinstance(money_decimals_obj, int)

    build(in_path_obj, out_path_obj, bin_days_obj, money_decimals_obj)


if __name__ == "__main__":
    main()
