# Labels

The Account label contract: mule ground truth, its masking and the label training may
read. Labels are never features (no feature, population, cutoff or hub query reads them),
and the model trains only on revealed positives. Zelle transfer fraud is a separate
target ([Zelle transfer labels](#zelle-transfer-labels)).

## The Account attributes

In load column order (`contract.graph_schema.ACCOUNT_LOAD_COLUMNS`). Positions 5 to 14 are
the ten supervision fields.

| Position | Column | Account attribute | Type | Default | Meaning |
|---|---|---|---|---|---|
| 0 | `id` | `id`, the primary id | STRING | | |
| 1 | `account_type` | `account_type` | STRING | | |
| 2 | `is_external` | `is_external` | BOOL | | |
| 3 | `first_seen_seq` | `first_seen_seq` | UINT | | |
| 4 | `first_seen_ts_ms` | `first_seen_ts_ms` | UINT | | |
| 5 | `is_mule` | `is_mule` | INT | 0 | Ground truth: 1 mule, 0 non-mule when known |
| 6 | `mule_label_known` | `mule_label_known` | BOOL | false | Truth supplied; false means unknown, not legitimate |
| 7 | `is_mule_masked` | `is_mule_masked` | BOOL | true | Positive label withheld from training |
| 8 | `pu_label` | `pu_label` | INT | 0 | 1 revealed positive, 0 unlabelled (positive-unlabelled learning) |
| 9 | `mule_label_effective_seq` | `mule_label_effective_seq` | UINT | 0 | First sequence at which the target holds, as the source defines it |
| 10 | `mule_label_effective_ts_ms` | `mule_label_effective_ts_ms` | UINT | 0 | The same in UTC epoch milliseconds |
| 11 | `mule_label_available_seq` | `mule_label_available_seq` | UINT | 0 | Sequence at which the label becomes available |
| 12 | `mule_label_available_ts_ms` | `mule_label_available_ts_ms` | UINT | 0 | The same in UTC epoch milliseconds |
| 13 | `mule_ring_id` | `mule_ring_id` | INT | -1 | Mule ring; 0 is a valid ring, -1 none or unknown |
| 14 | `mule_label_source` | `mule_label_source` | STRING | empty | Generator and version, or the reveal's version, salt and channel |

`pu_label = 1` exactly when `mule_label_known AND is_mule == 1 AND NOT is_mule_masked`.
Masking changes the mask and `pu_label` together, never the ground truth.

| Case | `is_mule` | `mule_label_known` | `is_mule_masked` | `pu_label` |
|---|---:|---|---|---:|
| Revealed mule | 1 | true | false | 1 |
| Hidden mule | 1 | true | true | 0 |
| Labelled non-mule | 0 | true | true | 0 |
| Unknown account | 0 | false | true | 0 |

- A labelled non-mule may also have `is_mule_masked = false`; its PU label stays 0.
- An unknown account is masked, with `is_mule = 0` as a placeholder and ring -1, and is
  never evaluated as a negative.
- A known label needs positive effective clocks, and availability clocks at or after
  them, in the sequence domain payments and association changes share. Zero means
  unspecified: a violation for a known label.
- A current `pu_label = 1` is a positive for an earlier example only when its
  availability precedes that example's cutoff.
- One ring per account; overlapping memberships would need their own representation.

## What training reads

Only revealed positives, through [`list_scope_accounts`](queries.md#list_scope_accounts)
with `include_observed = TRUE`. An observed positive is exactly `pu_label == 1 AND
is_mule == 1 AND mule_label_known AND NOT is_mule_masked`, and only it gets a discovery
time, `known_from_ms`, from `mule_label_available_ts_ms`. Every other account, masked
mules and labelled non-mules included, gets `observed_positive` false and `known_from_ms`
0, so neither field reveals a withheld label or which accounts are labelled.

- The prepared dataset keeps them in `observed_labels.parquet`: `account_id`,
  `known_positive`, `known_from_ms` (`data.observed_labels.LABEL_COLUMNS`). An unlisted
  or zero account is unlabelled, not confirmed legitimate.
- A positive is usable at a cutoff only when known before it.
- The label interface refuses oracle columns (`is_mule`, the mask, ring ids); training
  cannot import the code that reads them.
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

A fresh PhantomLedger load masks every mule. The first preparation runs
[`reveal_mule_labels`](queries.md#reveal_mule_labels) once to reveal the mules a bank
would have discovered ([Label reveal](../explanation/label-reveal.md) explains the
discovery model). It writes every internal Account:

- `mule_label_known = true`, with effective clocks at its first observation;
- for a mule, availability at its simulated discovery: the end of its discovery day in
  UTC, and the last sequence at or before it;
- for a revealed mule, `is_mule_masked = false` and `pu_label = 1`; others stay masked;
- `mule_label_source`: the reveal's version and salt, plus the channel that found a
  revealed mule.

External accounts stay unknown: the generator does not calibrate external mule roles.

## Loading accounts

The loading job `load_accounts` (`gsql/schema/account_loading.gsql`) reads an Account CSV
with exactly this header. PhantomLedger's mule-temporal export writes its Account table
in this order, so `load_accounts` reads it as it is.

```text
id,account_type,is_external,first_seen_seq,first_seen_ts_ms,is_mule,mule_label_known,is_mule_masked,pu_label,mule_label_effective_seq,mule_label_effective_ts_ms,mule_label_available_seq,mule_label_available_ts_ms,mule_ring_id,mule_label_source
```

A Kafka loading job, or a mapping drawn in the loading UI of GraphStudio or TigerGraph
Cloud, maps each column to the Account attribute of the same name
([The Account attributes](#the-account-attributes)). Map every column: an unmapped
attribute keeps its schema default. Unmapped label columns leave every account unknown,
with no truth to reveal or audit. An unmapped `mule_ring_id` leaves every account at
-1, no ring, which no check refuses; the audits then resample every mule alone instead
of with its ring, treating one ring's mules as independent.

A positional mapping, such as a loading job's `VALUES` list, follows the schema's
attribute order, which declares `is_mule` last, after `mule_label_source`: by position
the values are `$0`, `$1`, `$2`, `$3`, `$4`, `$6` to
`$14`, then `$5`. `load_accounts` lists them in that order, taking each column by its
header name.

- Use integer `0` and `1` for `is_mule` and lowercase `true` and `false` for the flags.
- A header-less PSV export of the same fifteen columns, separated by `|`, works the same.
- Keep the header for server-file loading; the REST++ streaming interface
  (`runLoadingJobWithData`, `runLoadingJobWithFile`) takes data rows without it.
- Derive the ground truth from the simulated account's role, never from an account
  sending or receiving a fraudulent payment.
- Give generated non-mules explicit 0 labels, unknown external accounts
  `mule_label_known = false`, and masked mules their truth, so their recovery can be
  evaluated.
- A synthetic mule's effectiveness is its first simulated mule activity; a known
  synthetic non-mule may use the account's creation for both clocks.

After loading, check the contract (every violation count must be zero) and read a page
of the oracle export; a page's last `account_id` is the next page's `after_id`:

```gsql
RUN QUERY validate_label_contract()
RUN QUERY read_ground_truth("", 100)
```

## Zelle transfer labels

`Zelle_Transfer` holds its own supervision: `fraud_label` (-1 unknown, never negative),
`label_known`, `label_available_seq`, `label_available_ts_ms`. They describe the
transfer, not the account, and no feature, sampling or aggregate reads them. Only the
label reveal reads `fraud_label`, as "this payment was a scam". The verdict arrives at
the payment instant, so the reveal simulates the reporting and investigation delays.
