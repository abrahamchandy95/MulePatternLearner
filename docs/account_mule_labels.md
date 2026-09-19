# Account mule labels and masking

The temporal schema now stores mule ground truth on **Account**. Removing that
supervision together with label-derived features was incorrect. The canonical
schema and the live graph have been corrected for the next data generation/load.
Zelle transfer fraud and account mule status are separate targets.

The old `main` branch loaded the source `mule` column into `Account.is_fraud`
(an integer), then wrote `pu_label` for masking. The new schema uses the explicit
integer attribute `is_mule` and preserves the distinction between truth and the label
available to training. It adds `is_mule_masked` as a directly inspectable boolean.
The legacy training split flags are not required by the temporal pipeline.

## Account attributes

The existing `id`, `account_type`, `is_external`, `first_seen_seq` and
`first_seen_ts_ms` fields remain. These ten attributes are appended:

| Attribute | Type / default | Meaning |
|---|---|---|
| `is_mule` | INT / 0 | Ground truth: 1 = mule, 0 = non-mule when the label is known |
| `mule_label_known` | BOOL / false | Ground truth is explicitly supplied; false means unknown, not legitimate |
| `is_mule_masked` | BOOL / true | Withhold the positive label from training |
| `pu_label` | INT / 0 | 1 = revealed positive; 0 = unlabeled in PU learning |
| `mule_label_effective_seq` | UINT / 0 | First sequence at which the target is true/negative as defined by the source |
| `mule_label_effective_ts_ms` | UINT / 0 | Corresponding UTC epoch milliseconds |
| `mule_label_available_seq` | UINT / 0 | Sequence when the supervision becomes available |
| `mule_label_available_ts_ms` | UINT / 0 | Corresponding UTC epoch milliseconds |
| `mule_ring_id` | INT / -1 | Optional ground-truth ring; 0 is valid, -1 means none/unknown |
| `mule_label_source` | STRING / empty | Generator/version or adjudication provenance |

`pu_label = 1` exactly when `mule_label_known AND is_mule == 1 AND NOT is_mule_masked`.
Masking changes the mask and `pu_label`, **never ground truth**. Both changes must
be submitted together. Their values must also be gated by availability and
validity at each training cutoff; a current `pu_label=1` is not automatically a
positive for an earlier example.

| Case | is_mule | mule_label_known | is_mule_masked | pu_label |
|---|---|---|---|---:|
| Revealed mule | 1 | true | false | 1 |
| Hidden mule | 1 | true | true | 0 |
| Labeled non-mule | 0 | true | true | 0 |
| Unknown account | 0 | false | true | 0 |

A labeled non-mule can also have `is_mule_masked=false`; its PU label remains 0.
Confirmed negatives are used as evaluation ground truth, while nnPU treats the
marginal training population as unlabeled. Unknown accounts require a masked
state, `is_mule=0` as a placeholder, and ring -1. Do not evaluate those
placeholders as negatives.

For every known label, supply positive effective clocks and availability clocks
at or after effectiveness. For synthetic mules, use the first simulated mule
activity as effectiveness and an explicit simulated discovery delay for
availability. If the generator deliberately makes an account a mule from its
creation, use its creation clocks. For known synthetic non-mules, the source may
use account creation for both clocks. Use the same sequence domain as payments
and association changes. Zero clocks mean unspecified and are rejected for
known labels. Ring IDs and all ten fields are supervision metadata, never model
features or neighborhood-selection criteria. This schema supports one ring ID
per account; overlapping memberships need a separate membership representation.

## Generation and upload contract

Generate ground truth directly from the simulated **account's role**. Do not infer
mule status merely because an account sends/receives a fraudulent payment.
Supply explicit integer 0 labels for generated non-mules, and distinguish unknown
external accounts with `mule_label_known=false`. Persist the truth for masked
mules so their recovery can be evaluated afterwards.

The installed `load_temporal_accounts` job consumes an Account CSV with a header
and **this exact column order**. Its user-defined header assigns names by position:

```text
id,account_type,is_external,first_seen_seq,first_seen_ts_ms,is_mule,mule_label_known,is_mule_masked,pu_label,mule_label_effective_seq,mule_label_effective_ts_ms,mule_label_available_seq,mule_label_available_ts_ms,mule_ring_id,mule_label_source
```

Use integer `0`/`1` for `is_mule`, and lowercase `true`/`false` for boolean flags.
The corresponding manifest-based PSV export uses the same fifteen columns,
separated by `|`, without a header. The
feature stager accepts both the earlier five-column Account export and this new
fifteen-column form. In the new form it writes supervision to
`account_labels.parquet` plus metadata **separately** from `nodes.parquet`;
none of the label columns enter the entity feature table.

For server-file loading, keep the CSV header. For the REST++ streaming interface
(`runLoadingJobWithData` / `runLoadingJobWithFile`), send data rows without the
header, as required by that API. The live integration check exercises this path.

Only the Account loading contract changes. Zelle transfers, tokens, payment
participation, association tenures and Fourier encoding fields remain as defined
in [the temporal schema](temporal_schema.md). Refresh the exporter manifest and
loader verification for the regenerated dataset, and stage it into a new output
directory. Old snapshot caches/checkpoints belong to the old dataset and must not
be reused as though they were trained on the new labels.

After loading, run:

```gsql
RUN QUERY temporal_validate_account_supervision()
RUN QUERY temporal_get_account_supervision("", 100)
```

The validation query reports counts of known labels, true mules, masked mules,
revealed positives and five violation counters, including `invalid_mule` for
values outside 0/1. Every violation counter should be zero. The paginated export
is an oracle-supervision endpoint, not a feature
endpoint. Use the last returned `account_id` as `after_id` for the next page.

The additive live migration preserves existing data and leaves old accounts
unknown/masked until the new load supplies truth. It does not relabel the old
corpus as legitimate and does not invent labels. The old five-column
`mt_load_account` job is preserved for compatibility and skips the new fields;
use the new account loader or update the regenerated exporter's loading job to
actually populate them.

The earlier boolean deployment is converted with
`scripts/temporal/convert_mule_label_to_integer.py`. It backs up supervision before
replacing the attribute and verifies every label afterwards. TigerGraph appends
the replacement integer attribute in storage; the loader maps the unchanged CSV
column order to that storage order. The canonical schema uses the same final order.

## Legacy snapshot training with the stored mask

`configs/temporal/mule_graph_mask.toml` sets `mask_source="graph"`. It reads the
staged `pu_label`/mask fields and additionally gates label availability and train
membership. Its `reveal_fraction=1` means no second masking pass; it does **not**
unmask accounts whose stored mask is true.

`configs/temporal/mule_pu.toml` sets `mask_source="resample"`. It preserves graph
truth, computes independent experiment masks locally, and supports the
10%/25%/50%/100% label-budget comparison without overwriting graph state. Run it
only when intentionally conducting that experiment. The model's input features
are identical under both modes.

The default label path for both configurations is now
`artifacts/temporal/stage/account_labels.parquet`. Update paths and chronological
cutoffs to match the regenerated dataset before training. There is no need for
the laundering-intermediary proxy importer for correctly generated mule labels.

## Files

- [Canonical schema](../gsql/schema/temporal_schema.gsql)
- [Existing-graph migration](../gsql/schema/migrations/account_mule_supervision.gsql)
- [Account loading job](../gsql/schema/temporal_account_loading.gsql)
- [Supervision export and validation queries](../gsql/temporal/account_supervision.gsql)


## Production training and local experiments

The live trainer uses an observed-label provider, not the schema's synthetic
masking fields. Complete synthetic truth is used by a separate local setup
utility and by post-training evaluation only. Production observed positives
require discovery-time semantics; a zero is unlabeled for nnPU.
See [the current training contract](live_temporal_training.md).
