# Labels

The Account label contract: the attributes that hold each account's mule ground truth,
its masking and the label available to training, and how training, the audits and the
loader use them. Labels are never features: no feature, population, cutoff or hub query
reads them, and the model trains only on the revealed positives. Zelle transfer fraud and
account mule status are separate targets.

## The Account attributes

Besides `id`, `account_type`, `is_external`, `first_seen_seq` and `first_seen_ts_ms`, an
Account holds ten supervision fields:

| Attribute | Type and default | Meaning |
|---|---|---|
| `is_mule` | INT, 0 | Ground truth: 1 for a mule, 0 for a non-mule when the label is known |
| `mule_label_known` | BOOL, false | The ground truth is explicitly supplied; false means unknown, not legitimate |
| `is_mule_masked` | BOOL, true | The positive label is withheld from training |
| `pu_label` | INT, 0 | 1 for a revealed positive; 0 is unlabelled for positive-unlabelled learning |
| `mule_label_effective_seq` | UINT, 0 | The first sequence at which the target holds, as the source defines it |
| `mule_label_effective_ts_ms` | UINT, 0 | The same in UTC epoch milliseconds |
| `mule_label_available_seq` | UINT, 0 | The sequence at which the label becomes available |
| `mule_label_available_ts_ms` | UINT, 0 | The same in UTC epoch milliseconds |
| `mule_ring_id` | INT, -1 | The account's ring of mules; 0 is a valid ring, -1 means none or unknown |
| `mule_label_source` | STRING, empty | Where the label came from: the generator and version, or the reveal's version, salt and channel |

`pu_label = 1` exactly when `mule_label_known AND is_mule == 1 AND NOT is_mule_masked`.
Masking changes the mask and `pu_label`, never the ground truth, and both change
together.

| Case | `is_mule` | `mule_label_known` | `is_mule_masked` | `pu_label` |
|---|---:|---|---|---:|
| Revealed mule | 1 | true | false | 1 |
| Hidden mule | 1 | true | true | 0 |
| Labelled non-mule | 0 | true | true | 0 |
| Unknown account | 0 | false | true | 0 |

A labelled non-mule may also have `is_mule_masked = false`; its PU label stays 0.
Unknown accounts need the masked state, `is_mule = 0` as a placeholder and ring -1, and
must never be evaluated as negatives.

Every known label needs positive effective clocks and availability clocks at or after
them, in the sequence domain the payments and association changes share; zero clocks mean
unspecified and are a violation for a known label. A current `pu_label = 1` is a
positive for an earlier example only when its availability precedes that example's
cutoff. The schema holds one ring per account; overlapping memberships would need a
membership representation of their own.

## What training reads

Training reads only the revealed positives, through the scope population query with
`include_observed = TRUE` ([`list_scope_accounts`](queries.md#list_scope_accounts)): an
account is an observed positive exactly when `pu_label == 1 AND is_mule == 1 AND
mule_label_known AND NOT is_mule_masked`, and only such an account has a discovery time,
`known_from_ms`, from `mule_label_available_ts_ms`. Every other account, masked mules
and labelled non-mules included, comes back with `observed_positive` false and
`known_from_ms` 0, so neither field reveals a withheld label or which accounts are
labelled.

The prepared dataset keeps them in `observed_labels.parquet` with the columns
`account_id`, `known_positive` and `known_from_ms` (`data.observed_labels.LABEL_COLUMNS`).
An unlisted or zero account is unlabelled, not a confirmed legitimate account. A positive
is usable at a cutoff only when it was known before it. Oracle columns (`is_mule`, the
mask, ring ids) are refused by the label interface, and training cannot import the code
that reads them.

A production system with another label source writes its known positives and their
discovery times into this contract; the model and the loss need no change. There
`is_mule = 0` must mean unlabelled unless a case was adjudicated negative, and unknown
evaluation truth must be absent or -1, never silently a legitimate account.

## What the audits read

The ground-truth audits and the diagnostics read every Account's truth through
[`read_ground_truth`](queries.md#read_ground_truth), as the table
`contract.graph_schema.TRUTH_COLUMNS`: `account_id`, `is_mule` (1 or 0, and -1 where the
label is not known), `ring_id` and `label_source`. An account whose label is not known
counts as unknown, never as a negative.

## Who writes the labels

A fresh PhantomLedger load masks every mule, so the first preparation reveals the
mules a bank would have discovered, once, with
[`reveal_mule_labels`](queries.md#reveal_mule_labels). It writes every internal Account:

- `mule_label_known = true`, with effective clocks at its first observation;
- for a mule, its simulated discovery as availability: the end of its discovery day in
  UTC and the last sequence at or before it;
- for a revealed mule, `is_mule_masked = false` and `pu_label = 1`; the others stay
  masked;
- `mule_label_source`: the reveal's version and salt, and for a revealed mule the
  channel that found it.

External accounts stay unknown, because the generator does not calibrate external mule
roles. [Label reveal](../explanation/label-reveal.md) explains the discovery model.

## Loading accounts

The loading job `load_accounts` (`gsql/schema/account_loading.gsql`) reads an
Account CSV with a header in exactly this column order
(`contract.graph_schema.ACCOUNT_LOAD_COLUMNS`):

```text
id,account_type,is_external,first_seen_seq,first_seen_ts_ms,is_mule,mule_label_known,is_mule_masked,pu_label,mule_label_effective_seq,mule_label_effective_ts_ms,mule_label_available_seq,mule_label_available_ts_ms,mule_ring_id,mule_label_source
```

Use integer `0` and `1` for `is_mule` and lowercase `true` and `false` for the flags. A
header-less PSV export with the same fifteen columns, separated by `|`, works the same.
Keep the header for server-file loading; the REST++ streaming interface
(`runLoadingJobWithData`, `runLoadingJobWithFile`) takes data rows without it.

Generate the ground truth from the simulated account's role, never because an account
sent or received a fraudulent payment. Supply explicit 0 labels for generated non-mules,
mark unknown external accounts with `mule_label_known = false`, and keep the truth of
masked mules so their recovery can be evaluated. For synthetic mules, use the first
simulated mule activity as effectiveness; for known synthetic non-mules the account's
creation may serve as both clocks.

After loading, check the contract (every violation count must be zero) and read a page of
the oracle export; the last `account_id` of a page is the `after_id` of the next:

```gsql
RUN QUERY validate_label_contract()
RUN QUERY read_ground_truth("", 100)
```

The populated reference graph reached this contract through two one-off migrations, kept
in git history. The first added the fields and left the existing accounts unknown and
masked; the second turned an earlier boolean label into the integer `is_mule`. TigerGraph
appends a replaced attribute to storage, so `is_mule` is stored last, and the loading
job maps the unchanged CSV order onto that storage order. The graph also keeps an older
five-column job, `mt_load_account`, for compatibility; it skips the label fields, which
only `load_accounts` loads.

## Zelle transfer labels

`Zelle_Transfer` holds its own supervision: `fraud_label` (-1 unknown, never negative),
`label_known`, `label_available_seq` and `label_available_ts_ms`. They describe the
transfer, not the account, and no feature, sampling or aggregate reads them. Only the
label reveal reads `fraud_label`, as "this payment was a scam"; the verdict arrives at the
payment instant, so the reveal simulates the reporting and investigation delays itself.
