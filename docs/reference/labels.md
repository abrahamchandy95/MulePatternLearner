# Labels

The Account label contract: mule ground truth and the label training may read. Labels
are never features (no feature, cutoff or hub query reads them; the population query
reads only `pu_label` and its availability), and the model trains only on revealed
positives. Zelle transfer fraud is a separate target ([Zelle transfer
labels](#zelle-transfer-labels)).

## The Account attributes

In load column order (`contract.graph_schema.ACCOUNT_LOAD_COLUMNS`): PhantomLedger's
Account file. Positions 5 to 14 are its ten supervision fields; the schema keeps no
attribute for four of them, which `load_graph` skips.

| Position | Column | Account attribute | Type | Default | Meaning |
|---|---|---|---|---|---|
| 0 | `id` | `id`, the primary id | STRING | | |
| 1 | `account_type` | `account_type` | STRING | `unknown` | |
| 2 | `is_external` | `is_external` | BOOL | false | |
| 3 | `first_seen_seq` | `first_seen_seq` | UINT | | |
| 4 | `first_seen_ts_ms` | `first_seen_ts_ms` | UINT | | |
| 5 | `is_mule` | `is_mule` | INT | 0 | Ground truth: 1 mule, 0 non-mule when known |
| 6 | `mule_label_known` | `mule_label_known` | BOOL | false | Truth supplied; false means unknown, not legitimate |
| 7 | `is_mule_masked` | none, skipped | | | The generator's mask, which `pu_label` carries |
| 8 | `pu_label` | `pu_label` | INT | 0 | 1 revealed positive, 0 unlabelled (positive-unlabelled learning) |
| 9 | `mule_label_effective_seq` | none, skipped | | | When the target holds; nothing reads it |
| 10 | `mule_label_effective_ts_ms` | none, skipped | | | The same in UTC epoch milliseconds |
| 11 | `mule_label_available_seq` | none, skipped | | | When the label becomes available, as a sequence; nothing reads it |
| 12 | `mule_label_available_ts_ms` | `mule_label_available_ts_ms` | UINT | 0 | When the label becomes available, in UTC epoch milliseconds |
| 13 | `mule_ring_id` | `mule_ring_id` | INT | -1 | Mule ring; 0 is a valid ring, -1 none or unknown |
| 14 | `mule_label_source` | `mule_label_source` | STRING | empty | Generator and version, or the reveal's record and channel |

`pu_label = 1` only for a known mule (`mule_label_known`, `is_mule = 1`) that the reveal
revealed. Revealing changes `pu_label`, never the ground truth.

| Case | `is_mule` | `mule_label_known` | `pu_label` |
|---|---:|---|---:|
| Revealed mule | 1 | true | 1 |
| Hidden mule | 1 | true | 0 |
| Labelled non-mule | 0 | true | 0 |
| Unknown account | 0 | false | 0 |

- An unknown account has `is_mule = 0` as a placeholder and ring -1, and is never
  evaluated as a negative.
- One ring per account; overlapping memberships would need their own representation.

## What training reads

Only revealed positives, through [`list_scope_accounts`](queries.md#list_scope_accounts).
An observed positive is exactly `pu_label == 1`, and only it gets a discovery time,
`known_from_ms`, from `mule_label_available_ts_ms`. Every other account, hidden mules and
labelled non-mules included, gets `observed_positive` false and `known_from_ms` 0, so
neither field reveals a withheld label or which accounts are labelled.

- The prepared dataset keeps them in `observed_labels.parquet`: `account_id`,
  `known_positive`, `known_from_ms` (`data.observed_labels.LABEL_COLUMNS`). An unlisted
  or zero account is unlabelled, not confirmed legitimate.
- A positive is usable at a cutoff only when its availability precedes it, so a current
  `pu_label = 1` is no positive for an earlier example.
- The label interface refuses oracle columns (`is_mule`, ring ids, label sources);
  training cannot import the code that reads them.
- Another label source, as in production, writes its known positives and discovery times
  into this contract, with no change to the model or loss. There `is_mule = 0` means
  unlabelled unless adjudicated negative, and unknown evaluation truth is absent or -1,
  never silently a legitimate account.

## What the audits read

The ground-truth audits and the diagnostics read every Account's truth through
[`read_ground_truth`](queries.md#read_ground_truth) as `contract.graph_schema.TRUTH_COLUMNS`:
`account_id`, `is_mule` (1, 0, or -1 when not known), `ring_id`, `label_source`. An
unknown label counts as unknown, never as a negative.

## Who writes the labels

A fresh PhantomLedger load masks every mule. Preparing a dataset runs
[`reveal_mule_labels`](queries.md#reveal_mule_labels) to reveal the mules a bank would
have discovered, unless the graph already holds that reveal's labels
([Label reveal](../explanation/label-reveal.md) explains the discovery model). It writes
every internal Account:

- `mule_label_known = true`;
- for a mule, availability at its simulated discovery: the end of its discovery day in
  UTC, never after the last event and never before the account's first observation;
  a non-mule's availability is its first observation;
- `pu_label = 1` for a revealed mule, 0 for every other account;
- `mule_label_source`: for a mule, the reveal's record,
  `phantomledger_role;reveal_v1;salt=<salt>;budget=1000`, plus `;revealed:<channel>` for a
  revealed mule or `;hidden`; for a non-mule, `phantomledger_role`. The record is the one
  earlier code wrote, whose `budget=1000` meant no cap, so the labels it wrote are this
  reveal's.

External accounts stay unknown: the generator does not calibrate external mule roles.

## Loading accounts

The loading job `load_graph` (`gsql/schema/loading.gsql`,
[Set up a graph](../how-to/set-up-a-graph.md#2-load-the-data)) reads PhantomLedger's
Account file, a CSV with exactly this header, loading each kept attribute from its
same-named column and skipping the other four.

```text
id,account_type,is_external,first_seen_seq,first_seen_ts_ms,is_mule,mule_label_known,is_mule_masked,pu_label,mule_label_effective_seq,mule_label_effective_ts_ms,mule_label_available_seq,mule_label_available_ts_ms,mule_ring_id,mule_label_source
```

A Kafka loading job, or a mapping drawn in the loading UI of GraphStudio or TigerGraph
Cloud, maps each column to the Account attribute of the same name
([The Account attributes](#the-account-attributes)) and skips the four without one. Map
every other column: an unmapped attribute keeps its schema default. Unmapped label
columns leave every account unknown, with no truth to reveal or audit. An unmapped
`mule_ring_id` leaves every account at -1, no ring, which no check refuses; the audits
then resample every mule alone instead of with its ring, treating one ring's mules as
independent.

A positional mapping, such as a loading job's `VALUES` list, follows the schema's
attribute order, which declares `is_mule` last, after `mule_label_source`: by position
the values are `$0`, `$1`, `$2`, `$3`, `$4`, `$6`, `$8`, `$12`, `$13`, `$14`, then `$5`.
`load_graph` lists them in that order, taking each column by its header name.

- Use integer `0` and `1` for `is_mule`. TigerGraph reads a flag in a CSV file as `true`,
  `True`, `TRUE` or `1`, else `false`, `False`, `FALSE` or `0`, so PhantomLedger's
  `True` and `False` load as they are.
- Keep the header for server-file loading; the REST++ streaming interface
  (`runLoadingJobWithData`, `runLoadingJobWithFile`) takes data rows without it.
- Derive the ground truth from the simulated account's role, never from an account
  sending or receiving a fraudulent payment.
- Give generated non-mules explicit 0 labels, unknown external accounts
  `mule_label_known = false`, and hidden mules their truth, so their recovery can be
  evaluated.

After loading, read a page of the oracle export; a page's last `account_id` is the next
page's `after_id`:

```gsql
RUN QUERY read_ground_truth("", 100)
```

## Zelle transfer labels

`Zelle_Transfer` holds its own supervision: `fraud_label` (-1 unknown, never negative),
`label_known`, `label_available_ts_ms`. They describe the transfer, not the account, and
no feature, sampling or aggregate reads them. Only the label reveal reads `fraud_label`,
as "this payment was a scam", and the diagnostics' replay of it, through the reveal's
inputs query (`tigergraph.oracle`). The verdict arrives at the payment instant, so the
reveal simulates the reporting and investigation delays.
