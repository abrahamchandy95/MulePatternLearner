# Feature plan v4 (draft): mule detection with the memoryless TGAT

Draft produced 20 September 2026 from a code and schema review plus a judged design pass. Every feature below is a hypothesis until the experiment matrix says otherwise. Sources are cited as repo paths with line numbers.

## Headline

Keep the memoryless TGAT, fix history before features: one installed GSQL query with runtime group flags computes everything in a single ascending pass over the root's own adjacency (pair gaps, incoming-to-outgoing delay, decayed sums, peer maps), a quota sampler with rank-based older strata replaces top-2 round-robin, the model splits into an event path and an optional summary path, and every comparison runs from one cached superset with a noise floor, an nnPU positive_weight fix and a pre-registered adoption rule before any feature group is declared useful.

## Corrections carried forward

- Owner correction 1: the AP results in docs/experiments/temporal_training_results.md measure prediction of future Zelle activity on an older dataset. They say nothing about which features detect mules, so no feature in this plan is justified by them; every group must earn its place in the experiment matrix below under the mule labels and the locked evaluation.
- Owner correction 2: a memoryless TGAT does not require window aggregates. It learns from ordered historical interactions with amount, direction, rail and exact timestamps (already on every message: src/mule_pattern_learner/temporal/live/batching.py:72-89, contract.py:27-30). Zero node features in the TGAT papers is a dataset configuration, not a requirement, so the plan carries an explicit zero-node-feature cell and a minimal-metadata cell, and treats the 40 rolling fields, the 2 window ratios and the 3 message pair-window counts as an optional summary group that is removed entirely in the graph-without-windows cell.
- Related owner point folded into the features: the existing 1d and 7d out/in amount ratio (queries.py:232-244, floor 1.0 and cap 100.0 at contract.py:15-16) does not measure how quickly received money leaves, and even the incoming-to-outgoing matching proposed here is a timing and amount coincidence, not funds attribution, because the schema has no balances or transaction linkage (gsql/schema/temporal_schema.gsql:90-131).

## Feature groups

Path 1 groups attach to sampled events and relationships. Path 2 groups are optional behavioural summaries of the root account. Every group toggles end to end (contract, GSQL, batching, model).

### message_core (window free, always on)

Path: event and relationship. Optional: no.

- **amount_log1p, amount_present, is_event**. Per sampled payment message: log1p(amount), the amount_present flag (missing amounts stay distinct from zeros) and the event-versus-association indicator. Existing message indices 0 to 2 (batching.py:75-78).
  - Behaviour: Amount coincidence between received and sent payments; structuring.
  - GSQL: Exists. EventRow carries t.amount and t.amount_present (queries.py:185-186); schema fields at temporal_schema.gsql:92, 99, 114, 116.
  - Cost: low. Schema change: no.
  - Leakage: History predicate t.event_seq < seed_seq AND t.event_ts_ms <= seed_ts_ms (queries.py:174-175, 180); scope-blocked events excluded before the heap (queries.py:181); validate_context enforces 0 < event_seq < cutoff_seq (source.py:84-88).
- **direction_embedding, rail_embedding**. Relation name encodes direction (zelle_out, zelle_in, payment_out, payment_in, contract.py:27) and rail has seven values (contract.py:30); both are learned nn.Embedding lookups added to the projected message vector (model.py:61-62, 69-77).
  - Behaviour: Rail hopping and directional asymmetry; the owner requirement to keep amount, direction and rail on messages is already met by the contract.
  - GSQL: Exists; rail literal zelle for Zelle_Transfer, t.payment_rail for Payment_Transaction (queries.py:156-158).
  - Cost: low. Schema change: no.
  - Leakage: None beyond event visibility.
- **payment_age_fourier64**. 64 fixed Fourier coordinates of age_ms = seed_ts_ms - event_ts_ms in basis log1p_s_400d_32x_sincos_v1 (queries.py:352-354; gsql/features/temporal_fourier64.gsql:9-23). Message indices 7 to 70 today (batching.py:86).
  - Behaviour: Recency of each event relative to the assessment moment; lets attention weigh a burst against older behaviour.
  - GSQL: Exists; client checks parity with the Python basis at 1e-5 (source.py:96-101).
  - Cost: low. Schema change: no.
  - Leakage: age_ms must equal cutoff_ms - event_ts_ms exactly (source.py:90-91).
- **pair_gap_fourier64, gap_present (computed in the one-pass walk)**. Gap to the previous event of the same directed pair and rail, Fourier encoded, plus gap_present (batching.py:79, 87-88). gap_present false is the message-level first-observed-pair signal, with the caveat that the first retained event does not prove no earlier one exists (docs/temporal_encoding.md:77-79). NEW implementation: computed inside one ascending pass over the root's own scanned events keyed by (direction, rail, peer id) with MapAccum<STRING, MaxAccum<UINT>> last_ts, instead of a sender-adjacency scan per selected event (queries.py:326-349).
  - Behaviour: Cadence within a pair; new-recipient novelty at the message level.
  - GSQL: INFERENCE from the canonical pair definition (docs/temporal_encoding.md:52-60) and role cardinality checks (queries.py:311-315): every same-pair predecessor of a message on an Account root lies on the root's own sent or received adjacency, so the per-event PriorCandidates scans are redundant for Account roots. Must be proven by a parity test against zelle_pair_time64 and payment_pair_time64 on live accounts (docs/leakage_and_scaling.md:80-81) before the sender scans are removed. For Token roots the sender scan path stays.
  - Cost: low. Schema change: no.
  - Leakage: Inherits the root's scope-filtered pre-cutoff event set (queries.py:179-181).

### event_channel

Path: event and relationship. Optional: yes.

- **channel_embedding**. Learned embedding of the event's channel string, added to the message embedding next to rail: add channel to EventRow and MessageRow (queries.py:49-56), a CHANNELS tuple in contract.py next to RAILS (contract.py:30), nn.Embedding(len(CHANNELS)) next to self.rail (model.py:62), and a vocabulary check in validate_context like the rail check (source.py:82-83) that rejects unknown values rather than mapping them to zero.
  - Behaviour: Cash-out versus onward transfer versus purchase. README.md:210-214 calls the atm_withdrawal versus card_purchase versus p2p distinction arguably the most important available for mule detection (written about the legacy path; the field now exists on both event vertices).
  - GSQL: channel exists on Payment_Transaction (temporal_schema.gsql:95) and Zelle_Transfer (temporal_schema.gsql:118) and is read by nothing in gsql/temporal or src/mule_pattern_learner/temporal/live (grep, zero hits). Emit t.channel in the EventRow at queries.py:185-186. Distinct values in the live graph are unknown; a one-time MapAccum<STRING, SumAccum<INT>> vocabulary scan is a pre-training data check.
  - Cost: low. Schema change: no.
  - Leakage: Immutable event attribute fixed at event time; no cutoff interaction.

### entity_meta

Path: event and relationship. Optional: yes.

- **type_onehot, is_external, is_deposit, age_days**. Six type indicators, is_external, account_type == deposit and log1p(age_days) for the root (queries.py:146-153; batching.py:47, 51-53), plus the 9-dim base vector for outermost peers (queries.py:359-373; batching.py:57-69). This is the minimal metadata variant.
  - Behaviour: Entity identity and age. On this dataset age is nearly constant: 99.2 percent of the 24,059 prepared accounts have first_seen_ts_ms 2024-01-01 and the latest opening is 2024-05-25 (verified with pandas on artifacts/temporal/phantomledger_2024_seed42_snapshot_20260919/accounts.parquet), and all 233 mules share the single value 2024-01-01 (local_experiments/strict_mule_v1/positive_oracle.parquet, verified). Keep in the production contract; expect no lift here.
  - GSQL: Exists.
  - Cost: low. Schema change: no.
  - Leakage: first_seen is an immutable first observation (docs/temporal_schema.md:153-155). Root uses first_seen_seq <= seed_seq while association endpoints use <= state_seq (queries.py:138-141 versus 259-261); unify to state_seq (pre-training fix).
- **visible_event_count_log, history_lt_5_events**. NEW. SumAccum of the root's scope-filtered visible payment events in both directions (log1p, capped at 10,000) and a flag when fewer than 5. Emitted for the root context only. Age-based insufficient-history flags are dropped because age is near-constant on this dataset.
  - Behaviour: Too little history for baselines; lets the model read absent summaries as absent rather than calm.
  - GSQL: One SumAccum inside the existing Events ACCUM (queries.py:179-187). Zero extra scan.
  - Cost: low. Schema change: no.
  - Leakage: Counts scope-filtered visible events only; derived from visible history so it is cutoff-safe, unlike outdegree-derived flags.

### sampler_meta

Path: event and relationship. Optional: yes.

- **stratum_embedding, hop_direction_embedding, scan_truncated flag**. NEW. Learned embeddings for why a message was kept (recent, older_rank_stratum, distinct_peer, association) and for the direction of the edge through which a child context was reached (out, in, association); plus a per-context flag set from visible-history counts when the ordered heap capacity max_scan was hit.
  - Behaviour: Lets attention weigh a deliberately older sample differently from a recent burst; sampler metadata, not a behaviour feature.
  - GSQL: Stratum tag is a new MessageRow member set at pop time; hop direction is Python only (first_relation is available at batching.py:132). scan_truncated is a SumAccum comparison in GSQL.
  - Cost: low. Schema change: no.
  - Leakage: Never label derived. The truncation flag is derived from the visible pre-cutoff count, not from all-time outdegree, so it does not encode post-cutoff activity.

### pass_through_message

Path: event and relationship. Optional: yes.

- **forward_delay (log seconds or Fourier64), forward_present, forward_censored**. NEW. For each sampled INCOMING message e at the root: time to the root's earliest OUTGOING event (any recipient, any rail) with event_seq > e.event_seq, event_ts_ms >= e.event_ts_ms and event_seq < seed_seq. forward_present true when such an event exists; forward_censored true when none exists before the cutoff (money may still be there or may leave after the cutoff; never read as no forwarding). Two encodings tested: 2 scalars (log1p seconds plus flags) and the 64-dim Fourier variant via temporal_fourier64_values.
  - Behaviour: Money received then soon sent onward (relay, layering hop). The owner's top candidate.
  - GSQL: Both directions carry event_ts_ms and event_seq on vertices and participation edges with named reverse types (temporal_schema.gsql:193-263). In the one-pass ascending walk over the root's merged in and out events: keep a ListAccum of pending incoming EventRows (bounded 64); on each outgoing event assign it as the forward event of every pending incoming and clear. O(M). Selected messages look up their delay in a MapAccum<STRING, UINT> keyed by event_id. No sender or recipient adjacency is touched.
  - Cost: low. Schema change: no.
  - Leakage: Both events precede the context cutoff. A child context cannot see forwarding after the reaching payment because its cutoff is that payment (batching.py:16-24; queries.py:174-175), so the signal is defined at the root and at children reached through incoming edges only; document the asymmetry in the contract. A flow proxy without funds attribution.
- **forward_amount_ratio, forward_same_rail**. NEW. min(next_out.amount / max(e.amount, 1.0), 100.0) with presence (absent when either amount_present is false), and a flag whether the forward event is on the same rail. 3 dims on incoming messages.
  - Behaviour: Whole-amount pass-through versus fee-skimmed relay; rail hopping.
  - GSQL: Same pass; store amount, amount_present and rail of the forward event in the same map.
  - Cost: low. Schema change: no.
  - Leakage: As forward_delay. Missing amounts produce absent, not zero (docs/temporal_encoding.md:77-79 style).
- **backward_delay, backward_amount_ratio (on outgoing messages)**. NEW. Mirror of the forward fields for each sampled OUTGOING message: time since the root's most recent incoming event with a smaller event_seq, and out/in amount ratio with presence, so attention sees the pass-through pair from the debit side.
  - Behaviour: Pass-through viewed from the debit side; dispersal after collection.
  - GSQL: Free in the same pending-incoming walk (track last incoming seq, ts, amount).
  - Cost: low. Schema change: no.
  - Leakage: Pure history.

### pair_history_window_free

Path: event and relationship. Optional: yes.

- **pair_prior_count_log**. NEW. log1p of the total number of prior events in the same directed pair and rail before the sampled event, no window.
  - Behaviour: Repeated transfers to one destination versus a new recipient.
  - GSQL: One SumAccum per pair key in the one-pass walk (MapAccum<STRING, SumAccum<INT>>); if the parity gate fails, one SumAccum alongside the existing pair window counters (queries.py:343-347).
  - Cost: low. Schema change: no.
  - Leakage: Prior events only; window free, so it survives the no-windows cell.
- **pair_first_seen_age (log seconds or Fourier64), pair_first_seen_present**. NEW. Sampled event time minus the earliest prior same-pair event time; absent when no prior pair event exists.
  - Behaviour: Directed-pair first-seen (counterparty novelty) without the stored first_txn_date the legacy HAS_PAID edge had (gsql/schema/schema.gsql:90-103) and the temporal schema dropped.
  - GSQL: MapAccum<STRING, MinAccum<UINT>> per pair in the same walk. Earliest visible means earliest in retained, scope-filtered history, not true first contact.
  - Cost: low. Schema change: no.
  - Leakage: Same as pair_prior_count. Do not resolve Token recipients to accounts with the present-day binding (docs/temporal_encoding.md:62-66).

### pair_window_counts (legacy, kept only to reproduce the current hybrid)

Path: event and relationship. Optional: yes.

- **pair_count_1h, pair_count_1d, pair_count_7d**. log1p of prior same-pair counts within 1h, 1d and 7d of the sampled event (batching.py:80-82; queries.py:345-347). Window derived; removed in every no-window cell together with rolling_windows and amount_ratios.
  - Behaviour: Short-horizon repeated payments to one destination.
  - GSQL: Exists. Served from the existing sender-scan path inside the cached superset; NOT reimplemented as a per-pair ListAccum (that map would grow with distinct pairs on hub accounts).
  - Cost: low. Schema change: no.
  - Leakage: Strict boundary in temporal_training_context (queries.py:345-347) versus inclusive in gsql/features/zelle_pair_time64.gsql:122-124; unified before training. The existing no_fourier variant keeps these columns (model.py:69-77) and is retired.

### device_ip_context (conditional on populated role edges)

Path: event and relationship. Optional: yes.

- **device_age_at_event, ip_age_at_event (+ present flags)**. NEW, message level. log1p((event_ts_ms - device.first_seen_ts_ms) / 1000) with device_present, and the same for IP. 4 dims. Exact milliseconds because Device and IP carry first_seen_ts_ms (temporal_schema.gsql:70-81) and payments link to them by Transfer_Used_Device, Transfer_Used_IP, Transaction_Used_Device, Transaction_Used_IP (temporal_schema.gsql:217-227, 253-263).
  - Behaviour: A new device shortly before transfers, measured in hours without association timestamps.
  - GSQL: One edge hop from the selected event vertex in the per-event block (next to queries.py:305-309). Population of these four edge types is unverified: no row counts exist in the repo and no live query traverses them. A count query gates the group.
  - Cost: low. Schema change: no.
  - Leakage: Require device.first_seen_seq <= item.seq and first_seen_ts_ms <= item.ts (extend the role check pattern at queries.py:305-309). A device observation is not proof of a continuing association (docs/temporal_schema.md:116-120).
- **device_shared_accounts_at_event (second wave, conditional)**. NEW. Number of Accounts with an Account_Uses_Device tenure valid at item.seq - 1 for the payment's device, capped at 16 with a truncation flag. Second wave only: all 233 mules have 233 distinct ownership groups (positive_oracle.parquet, verified), so ring co-ownership is absent by construction here, and the herder-device signal is a hypothesis for real data.
  - Behaviour: Shared device across mule accounts.
  - GSQL: Device_Used_By_Account reverse edge (temporal_schema.gsql:173) with the validity predicate (temporal_schema.gsql:19-20); cost is the device's tenure degree.
  - Cost: medium. Schema change: no.
  - Leakage: Held-out accounts on the same device must be excluded via the scope membership check (queries.py:12-19), otherwise a held-out mule leaks through the count.

### causal_paths (bounded, root only)

Path: event and relationship. Optional: yes.

- **direction_aware_second_hop (sampler policy, no tensor column)**. NEW policy in select_messages for child contexts. A child reached through a root OUTGOING payment has a cutoff equal to that payment (batching.py:16-24), so its visible history precedes the payment and can only show prior behaviour: prefer its INCOMING relations 3:1 (is the recipient a collector fed by many senders). A child reached through a root INCOMING payment: prefer its INCOMING relations 3:1 (did the sender itself just receive, i.e. upstream relay). Quotas are hypotheses. Correction to the typology design: preferring a recipient's outgoing relations does not produce a money-flow path, because nothing after the reaching payment is visible from that child.
  - Behaviour: Collection then dispersal; relay chains upstream.
  - GSQL: Python only (batching.py:27-43 replaced by quota selection).
  - Cost: low. Schema change: no.
  - Leakage: Child cutoffs unchanged.
- **neighbour_clock arm (policy): child_key at the root's cutoff**. NEW experimental arm. child_key uses the root's (cutoff_seq, cutoff_ms) instead of the connecting event's clocks (batching.py:16-24). The recipient's context then includes what it did after receiving from the root, up to the root's assessment time, with zero new GSQL, and per-date child contexts collapse into shared cache entries.
  - Behaviour: Onward hop from the recipient side; the cheapest route to relay evidence.
  - GSQL: Python only; the server predicate event_seq < seed_seq is unchanged.
  - Cost: low. Schema change: no.
  - Leakage: Leakage safe: every child event still satisfies event_seq < root seed_seq. It changes what a neighbour representation means (state at assessment time rather than at interaction time), so it is compared against the root-only walks rather than assumed.
- **relay_forward_at_root: onward_delay_min (log seconds or Fourier64), onward_count_log, onward_present**. NEW, root level only. For the k = 4 most recent outgoing payments to Account recipients R: the delay from the payment to R's first outgoing event with event_seq > pay_seq and event_seq < seed_seq, the count of such events among R's first 8 onward events, and presence. No fixed 24-hour horizon: the model learns the horizon from the delay encoding (correction to the implementation design's hard window).
  - Behaviour: Money keeps moving one hop beyond the root (layering out). This aggregate is the only channel for recipient behaviour after receipt under the TGAT clock, since child contexts stop at the payment.
  - GSQL: Bounded walk R -(Account_Sent_Zelle_Transfer|Account_Initiated_Transaction)- event with per-hop clocks in the style of queries.py:174-175 and scope_events (queries.py:22-32), HeapAccum(8, seq ASC) per R. Opens up to 4 recipient adjacencies per root; never for children.
  - Cost: high. Schema change: no.
  - Leakage: R's post-payment activity precedes the root's cutoff, so it is legitimate root history; held-out R are removed by scope_events. Operational hub guard: skip R when its all-time outdegree of the two send edge types exceeds 8,192 (STATS OUTDEGREE_BY_EDGETYPE, temporal_schema.gsql:42, 60); the skip is logged in a diagnostic column that is NOT a model input, because all-time outdegree encodes post-cutoff activity. If the skip rate among scorable peers exceeds 1 percent the group is redesigned.
- **collect_backward_at_root: upstream_delay_min, upstream_count_log, chain_flag**. NEW, root level only. For the k = 4 most recent incoming payments from Account senders S: delay from S's most recent received event before pay_seq to the payment, count of S's received events among its 8 most recent before the payment, and chain_flag = onward_present AND upstream_present.
  - Behaviour: Upstream collection feeding the root; relay chain of length 3 seen from the middle.
  - GSQL: Mirror walk over Account_Received_Zelle_Transfer|Account_Received_Transaction with HeapAccum(8, seq DESC). Same guard and scope filter.
  - Cost: high. Schema change: no.
  - Leakage: As above.

### rolling_windows (legacy)

Path: behavioural and summary. Optional: yes.

- **{1h,1d,7d,30d} x {out,in} x {count, amount, missing, zelle, unique} (40 fields)**. The existing hard-window group (contract.py:13, 31-42, 46; queries.py:188-195, 206-213). Kept unchanged so the current hybrid is reproducible; disabled in every no-window cell.
  - Behaviour: Volume and breadth of recent activity; order free.
  - GSQL: Exists. Disabling removes the four window IF blocks and the eight Unique_* joins per context; the adjacency scan remains because it feeds the heaps (INFERENCE from queries.py:179-219).
  - Cost: medium. Schema change: no.
  - Leakage: Strict window membership (queries.py:190). Distinct counts are exact SetAccum sizes over every visible event (queries.py:64, 203-213).

### amount_ratios (legacy, current-hybrid cell only)

Path: behavioural and summary. Optional: yes.

- **1d_out_in_amount_ratio, 7d_out_in_amount_ratio**. min(out_amount / max(in_amount, 1.0), 100.0) per window (queries.py:232-244; contract.py:14-17). Appears only in the reproduced current-hybrid cell.
  - Behaviour: Coarse outflow-versus-inflow proxy; does not measure how fast received money leaves (owner), and floor plus cap collapse every no-incoming account to one value.
  - GSQL: Exists. validate_context hard-requires these keys (source.py:71-76); the check becomes conditional on the group flag.
  - Cost: low. Schema change: no.
  - Leakage: Derived from window sums.

### recency

Path: behavioural and summary. Optional: yes.

- **out_recency_days, in_recency_days, out_recency_present, in_recency_present**. Days since the most recent visible outgoing and incoming event plus presence (queries.py:223-230; contract.py:47). Not window derived.
  - Behaviour: Dormancy versus activity at the cutoff.
  - GSQL: Exists (MaxAccum @@out_last, @@in_last, queries.py:66, 187).
  - Cost: low. Schema change: no.
  - Leakage: None beyond event visibility.

### association_counts

Path: behavioural and summary. Optional: yes.

- **{7 families x 2 directions} x {active, ended} (28 counts)**. Tenure counts at state_seq = seed_seq - 1 (queries.py:254-285). Counts tenures, not distinct entities (docs/gsql_feature_catalog.md:51-54).
  - Behaviour: Identity breadth and churn (ordinal only).
  - GSQL: Exists. For an Account root only Account_Owned_By_Party, Account_Bound_From_Token and Account_Uses_Device are non-empty (contract.py:18-26 with queries.py:245-253). Whether the ended half is ever nonzero is unrecorded (data check).
  - Cost: low. Schema change: no.
  - Leakage: Known time is absent (temporal_schema.gsql:24-26; docs/temporal_schema.md:127-142), so a backdated tenure appears as if known.

### decayed_activity

Path: behavioural and summary. Optional: yes.

- **decayed_{out,in}_{count, amount, log_amount}_{1d,7d,30d,90d} (24 features)**. NEW. For each direction and half-life h: sum w, sum w * amount, sum w * log1p(amount) with w = pow(0.5, (seed_ts_ms - t.event_ts_ms) / h_ms). Smooth replacement for hard windows. Half-life sets are hypotheses: compare {1d,7d,30d,90d} against {1d,7d,30d}; a sub-day set (6h) is added only if the timestamp granularity check shows intraday event clocks (account first_seen values are exact midnights for 99.2 percent of the cohort).
  - Behaviour: Quiet account becomes active; amount level shift; smooth activity intensity.
  - GSQL: pow and log already used in installed GSQL (gsql/features/temporal_fourier64.gsql:13, 17; gsql/features/fastrp.gsql:48). Three @@features += lines per half-life inside the existing Events ACCUM (queries.py:179-196). No extra scan.
  - Cost: low. Schema change: no.
  - Leakage: Same visibility as rolling windows.
- **burst_and_shift_ratios (4 features, Python side)**. NEW. count_1d/count_30d, amount_7d/amount_90d, mean log amount over 1d minus mean log amount over 90d, in/out decayed count ratio, with presence guards when denominators are zero. Declared in the contract as derived so GSQL and Python stay consistent.
  - Behaviour: Rate change relative to the account's own baseline.
  - GSQL: Computed in node_features (batching.py:46-54) from the raw sums; fixed transforms, no dataset-level fit.
  - Cost: low. Schema change: no.
  - Leakage: Cannot be validated as an onset signal: no mule effective clocks are loaded (docs/experiments/live_training_readiness.json:31-43, known_labels 0, invalid_unknown 233) and all mules exist from day one. Reported as exploratory.

### pass_through_summary

Path: behavioural and summary. Optional: yes.

- **forwarded_within_{1h,24h,72h}_decayed30d, incoming_decayed30d**. NEW. Decayed (h = 30d) count of incoming events whose forward_delay is at most 1h, 24h, 72h, plus the decayed incoming count so shares are computable. Thresholds are delay thresholds on the pass-through pairing, not cutoff windows; the 1h threshold is dropped if event clocks are day granular.
  - Behaviour: Habitual relay behaviour over the account's history.
  - GSQL: Accumulated in the same pending-incoming pass.
  - Cost: low. Schema change: no.
  - Leakage: As pass_through_message; censored incoming events do not count as unforwarded.
- **min_forward_delay_log, forwarded_amount_share_24h**. NEW. log1p of the minimum forward delay over the 64 most recent incoming events plus presence; decayed sum over incoming events forwarded within 24h of min(in.amount, out.amount) divided by decayed incoming amount.
  - Behaviour: Fastest observed pass-through and the share of received value that left within a day (proxy, not attribution).
  - GSQL: MinAccum and SumAccum in the same pass.
  - Cost: low. Schema change: no.
  - Leakage: Amount matching is coincidence, not attribution; say so in the catalog.

### counterparty_structure

Path: behavioural and summary. Optional: yes.

- **out_hhi, in_hhi, out_top1_share, in_top1_share, out_peer_count_log, in_peer_count_log, in_to_out_peer_ratio**. NEW. From MapAccum<STRING, SumAccum<DOUBLE>> of decayed (h = 30d) amount per peer per direction: Herfindahl index, largest share, distinct peer counts and their ratio. Peer key is Account:id or Token:token_id with the existing fallback rule (queries.py:197-213).
  - Behaviour: Many senders feeding few recipients; repeated transfers to one destination.
  - GSQL: Replace the SetAccum @@distinct (queries.py:64) with peer-keyed MapAccums populated in the existing Unique_* joins (queries.py:203-213), then one FOREACH like queries.py:220-222. Memory grows with distinct peers; cap at 10,000 keys with a truncation flag derived from the visible count. The existing @@distinct already has this shape and is unprofiled.
  - Cost: medium. Schema change: no.
  - Leakage: Held-out peers vanish because their events are scope-blocked before the join (queries.py:179-181), which is the documented intent (docs/leakage_and_scaling.md:27-31).
- **new_out_peers_7d, new_in_peers_7d, out_token_share, decayed_distinct_mass_{out,in}_{7d,30d}**. NEW. Peers whose earliest visible event with the root is younger than 7 days (MapAccum<STRING, MinAccum<UINT>> first_ts); share of decayed outgoing amount to unresolved Token recipients; and decayed distinct-counterparty mass = sum over peers of pow(0.5, (seed_ts_ms - latest_event_ts_of_peer) / h) for h in {7d, 30d}.
  - Behaviour: Counterparty novelty breadth; cash-out to unresolved external recipients; smooth breadth of recent counterparties.
  - GSQL: Same joins and maps.
  - Cost: medium. Schema change: no.
  - Leakage: Earliest visible is bounded by retained history; a stored directed-pair first_seen would make novelty exact (see schema changes).
- **reciprocal_peer_count_log, min_return_delay_log (+ present)**. NEW. Peers present in both direction maps, and the shortest out-then-in return delay from the same peer (from per-peer last-out and first-in-after maps). Window free (replaces the implementation design's cycle_7d_count and the typology design's per-message return_to_root hop).
  - Behaviour: Circular transfers and two-node cycles as a root summary.
  - GSQL: From the peer maps built in the root's own scan; no walk.
  - Cost: low. Schema change: no.
  - Leakage: Root history only. Generator cycle length is unknown, so this is a hypothesis.

### association_timing_bounds

Path: behavioural and summary. Optional: yes.

- **newest_tenure_start_bounds (12 features)**. NEW. For each account-side family (Account_Owned_By_Party, Account_Bound_From_Token, Account_Uses_Device): upper_bound_age = seed_ts_ms minus the timestamp of the root's earliest visible event with event_seq > newest valid_from_seq; lower_bound_age = seed_ts_ms minus the timestamp of the root's latest event with event_seq < newest valid_from_seq; each as log1p seconds plus presence. States a bracket in real milliseconds, never a point, and never sequence differences as time (docs/temporal_training.md:48-55; docs/gsql_feature_catalog.md:77-79).
  - Behaviour: Token rebinding or new device shortly before transfers, bounded from the root's own witnessed events.
  - GSQL: Collect association tenures before the event scans (reorder the renderer), keep up to 8 newest start sequences per family in a ListAccum, and inside the Events ACCUM update MapAccum<UINT, MinAccum<UINT>> hi and MapAccum<UINT, MaxAccum<UINT>> lo per tracked start. Constant-size FOREACH per scanned event; no index needed. Sequence-only inputs (temporal_schema.gsql:135-189).
  - Cost: low. Schema change: no.
  - Leakage: Bounds use only the root's own pre-cutoff events. Known time absent (temporal_schema.gsql:24-26). Bracket width depends on the root's activity; report the width distribution.
- **tenure_changes_within_last_10_events (6 features)**. NEW. Count of tenure starts and of tenure ends (valid_to_seq nonzero and <= state_seq) whose sequence exceeds the event_seq of the root's tenth most recent visible event, per account-side family. Ordinal; labelled in the contract as sequence position.
  - Behaviour: Identity churn interleaved with recent payments.
  - GSQL: Needs the tenth-most-recent seq from the recent heap (capacity 16) and a comparison in the Assoc ACCUM (queries.py:267-275). Conditional on the closed-tenure data check showing ended tenures exist.
  - Cost: low. Schema change: no.
  - Leakage: Must not be described as elapsed time.
- **exact_tenure_age (schema change alternative)**. With valid_from_ts_ms and valid_to_ts_ms on all seven association families, association messages carry a real age encoding and the bounds features collapse to exact values. The witnessed-sequence version proposed by the typology and experiment designs is dropped: it needs a sequence-to-timestamp lookup with no index (the only precedent is the full scan in gsql/temporal/training_cutoffs.gsql:14-23) and an unmeasured hit rate.
  - Behaviour: Same as above, exact.
  - GSQL: Loader must supply valid-time timestamps. Whether TigerGraph permits ALTER ADD ATTRIBUTE on non-discriminator edge attributes is external knowledge, not a repo fact; docs/temporal_schema.md:89-91 only states discriminators cannot be changed that way.
  - Cost: medium. Schema change: yes.
  - Leakage: Timestamps must be valid time, not load time; a backfill must not invent times for unwitnessed rows.

## Sampler

GSQL side: one installed temporal_training_context with new parameters replacing per_relation: k_recent, k_old, k_div, max_scan, plus include_<group> BOOL flags (statement-level IF is already used for scope blocks, queries.py:13-19). HeapAccum capacity CAN be a query parameter: gsql/features/pagerank.gsql:9,16 declares HeapAccum<Vertex_Score>(top_k, score DESC) with top_k an INT parameter, so no literal-capacity fallback is needed (correction to two designs). For each of the four payment relations the existing adjacency scan (queries.py:174-187) fills: (1) @@recent_{rel}: HeapAccum(k_recent, seq DESC) as today; (2) @@ordered_{rel}: HeapAccum(max_scan, seq DESC) popped into a ListAccum and walked from the end to obtain ascending order; this single pass computes pair gaps and counts, pair first-seen, forward and backward delays, decayed sums, peer maps and association bounds; if the relation has more than max_scan visible events, scan_truncated_{rel} is set from the visible count and gaps for events outside the retained window are gap_present false with the truncated flag (never an invented zero, docs/temporal_encoding.md:77-79); (3) older stratum by RANK, not by age or hash: from the ordered list beyond the k_recent most recent events, take the events at fixed rank fractions (one quarter, one half, three quarters of the remaining visible history), k_old = 3. This keeps a burst from displacing earlier behaviour, is deterministic, needs no string hash (none exists under gsql/, and getvid is snapshot specific per training_scope.gsql:5), and puts no hard time boundary back into which events the no-window model sees (correction to the age-bucketed strata in two designs); (4) @@diverse_{rel}: in the Unique_* peer join that already exists (queries.py:203-213) accumulate MapAccum<STRING, MaxAccum<UINT>> peer -> latest seq, then keep the latest event of the k_div = 2 most recently active distinct peers not already present in recent. Association relations keep k_assoc = 2 most recent active tenures (queries.py:276-280). Every row carries a stratum tag. Currency becomes a per-event exclusion with a counter instead of RETURN (queries.py:286-289), which today aborts the whole 16-context request (the RETURN sits inside the FOREACH over requests) and makes checked_rows fail the batch (source.py:55-60). Root-only groups (causal_paths walks) run only when a per-request BOOL marks the row as a root. Python side: select_messages (batching.py:27-43) becomes quota selection: layer-1 fanout F1 = 16 as {zelle_out 4, zelle_in 4, payment_out 3, payment_in 3, associations 2}, each payment quota filled recent first, then rank strata, then diverse, then backfill from any payment relation; layer-2 fanout F2 = 4 payments only, direction aware as described in causal_paths. make_live_batch stays two-hop (batching.py:100). lite_external_children: peers with is_external or account_type != deposit (static attributes) get the 9-dim base vector only and no fetched context, under a config flag; this needs layer 1 to accept a mix of encoded contexts and base-encoded stubs. Current-sampler emulation: the legacy top-2 round-robin selection is a subset of the superset response whenever k_recent >= 2; because the legacy policy at fanout 8 takes position 0 of every non-empty relation (up to 3 association children) while the improved quota keeps only 2 association slots, the cache build computes children under BOTH policies and fetches the union, so no offline miss occurs (ContextStore raises on a miss without an executor, source.py:174-178). Causality unchanged: child cutoffs are the connecting event (batching.py:16-24) except in the neighbour-clock arm, and the server applies event_seq < seed_seq AND event_ts_ms <= seed_ts_ms at every hop (queries.py:174-175).

### Budget

Current sampler facts: per_relation 2 with HeapAccum(8) (queries.py:58-60, 214-219) and relation round-robin (batching.py:27-43, RELATIONS order contract.py:27-29) give an Account root typically 5 payment events at fanout 8 (4 most recent, one per payment relation, plus one second zelle_out when all three account-side association families return a message) and at most 8 when no tenure is visible; the data hold about 147 events per account by event count, 147 to about 294 participations per account depending on how many recipients are Tokens (115,741,341 events over 788,283 accounts, docs/experiments/live_training_readiness.json:4-13; arithmetic is inference). Measured baseline: 64 roots, 448 unique contexts, 29 database calls, 51.07 s fetch, 1,501,184 tensor bytes, 674 MB peak RSS (artifacts/temporal/batch_readiness.json); about 1.76 s per 16-context request and 0.114 s per context (inference). Improved budget S1: k_recent 4, k_old 3, k_div 2, k_assoc 2, max_scan 2,048, F1 16, F2 4. Per Account context at most 4 x 9 = 36 payment messages plus 3 x 2 association messages = 42, so validate_context's bound becomes the sum of caps instead of len(RELATIONS) * 8 = 144 (source.py:69-70). Per 64-root batch: contexts <= 64 x 17 = 1,088; requests = 4 for roots plus at most 64 for children = 68 (versus 36 nominal and 29 measured today); naive projection about 120 s per batch at today's per-request cost, 2.3x, before the per-context growth from more emitted events and before the savings from removing the per-event sender scans (8 per context today, 36 under S1 if kept) and from lite_external_children; both effects are hypotheses to measure. Tensor memory is not the constraint: 1.43 MiB of 64 MiB today (memory.py:23); at F1 16, F2 4, 64 roots and about 140 message dims the estimate is about 4 MiB; max_contexts 2,048 (memory.py:22) still holds. Stress budget S2: F1 24, F2 6, 1,600 contexts, about 7 MiB, about 100 requests. Training wall clock: 100 steps x 30 epochs (configs/temporal/live_tgat.toml) is 3,000 batches per run, about 42 h streamed at 51 s, so the matrix runs only from the SQLite cache. One-time cache cost per sampler variant for the 24,059-account cohort (accounts.parquet, verified): 409k contexts at F1 16, about 25.6k requests, about 12.5 h sequential (ContextStore.fetch is sequential today, source.py:169-191) or about 3.5 h at concurrency 4 if the server scales, which is unmeasured. Acceptance gate for any budget: median server time per context at most 0.2 s, per-batch stream fetch at most 3x the 51 s baseline, no admission failure, and a p95 request time well under the 300 s client timeout (source.py:45-52).

### Cost controls

- Single-scan rule enforced in the renderer: strata heaps, peer maps, decayed sums, pass-through pairing and association bounds all fill inside the existing Events_* and Unique_* passes (queries.py:179-213); code review rejects any new SELECT over the root's payment adjacency.
- Remove the per-selected-event PriorCandidates sender scans (queries.py:326-349) for Account roots by computing pair predecessors in the one-pass walk, gated by a parity test against zelle_pair_time64 and payment_pair_time64 (docs/leakage_and_scaling.md:80-81).
- Operational hub guard (not a model input): before scanning a root or walking a peer, read the O(1) all-time outdegree of the four participation edge types (STATS OUTDEGREE_BY_EDGETYPE, temporal_schema.gsql:42, 60, 108, 131); above 8,192 skip the scans and return metadata only, logging the skip in a diagnostic column excluded from tensors. Outdegree counts all-time edges, so this is conservative and must never be emitted as a feature.
- max_scan cap (2,048) on the ordered heap with scan_truncated flags from visible counts; the adjacency scan still visits every edge (GSQL cannot stop early), so the cap bounds accumulator memory and heap work, not scan (docs/gsql_feature_catalog.md:182-187; docs/leakage_and_scaling.md:179-184).
- Root-only two-hop walks (causal_paths) with k = 4 peers, HeapAccum(8) per peer, degree guard, never for child contexts.
- lite_external_children: no fetched context for is_external or non-deposit peers; about 470,000 of 788,283 accounts are outside the scorable cohort (317,840 scorable per docs/experiments/mps_known60_readiness.json), so this is potentially the largest saving; the share of counterparties that are external is unknown.
- Runtime include_<group> flags so a graph-only variant does not pay for Unique_* joins, ratio block, decayed sums or walks; per-group server cost measured by subtraction with benchmark_live_batch.py extended to per-request timing (today it records only aggregate fetch_seconds, database_calls and tensor_bytes, scripts/temporal/benchmark_live_batch.py:86-101). Honest limit: disabling rolling removes arithmetic and joins but not the Candidates and Events adjacency scans, which also feed the heaps.
- Raise StreamingContextSource capacity from 64 toward the allowed 256 (source.py:248, 254-259) and pass it from open_context_source (source.py:300-308). Note: within one batch, keys are deduplicated before requests (source.py:274; batching.py:107-112), so the LRU only affects cross-batch reuse; the claim that reuse is currently zero is an inference, not a measurement.
- Cache identity: key ContextStore metadata on an extraction fingerprint (dataset_id, source_counts, query_hashes, extraction contract version, sampler parameters, scope_id) instead of config_sha256 (dataset.py:163-170; source.py:127-131; pipeline.py:33-36 rejects any config change today and the config includes the observed_labels path, configs/local/live_tgat.toml:4), so one cache serves every label draw, seed and Python-side ablation; add concurrency up to 4 to ContextStore.fetch.
- Keep request_batch_size at 16 (queries.py:100-102; source.py:122-123, 254-259; REST sizeLimit 32 MB at source.py:51) until per-request timing is measured; keep Fourier vectors optional on the wire (send scalar delays, expand on the GPU, docs/gsql_feature_catalog.md:161-163) if JSON size becomes the bottleneck at 42 messages per context.
- Derive every admission constant from the contract: the 83 and 135 literals in memory.py:31-32, the 135/7 literal in model.py:63, the 135 in batching.py:74, and the message bound in source.py:69-70; re-review the 24x working-memory multiplier (memory.py:39-50) once two encoders exist.

## Model input paths

Two independently testable input paths with one head, replacing the single projection at model.py:57-58 and the single-path encode at model.py:79-94. Path 1, event and relationship: node_event = Linear over the entity_meta columns of every context (constant zeros in the zero-node-feature cell); messages carry message_core plus any enabled message groups (event_channel, sampler_meta, pass_through_message, pair_history_window_free, pair_window_counts, device_ip_context) as a contract-derived width instead of the 135 literal, plus relation, rail, channel, stratum and hop-direction embeddings; the two existing AttentionBlocks (model.py:30-42) run over second-hop base vectors then first-hop encoded vectors with the root as the single query; root-only causal_paths aggregates (relay_forward_at_root, collect_backward_at_root) are appended to the root's event-path vector because they are relationship facts about the root's own payments. Path 2, behavioural summaries (optional): node_summary = MLP over the enabled summary groups of the root only (rolling_windows, amount_ratios, recency, association_counts, decayed_activity, pass_through_summary, counterparty_structure, association_timing_bounds); its output is concatenated with the attention output and the first head Linear widens from hidden to 2 x hidden (model.py:65-67). Ablation by omission, not zeroing: a disabled path or group is absent from the tensors and from the parameter count, so parameter counts match inputs. FEATURE_GROUPS and MESSAGE_GROUPS are declared once in contract.py with ordered names and fixed transforms (replacing the x[8:] log1p slice at batching.py:53 with per-group transform specs); FEATURE_NAMES becomes the concatenation of enabled groups; two fingerprints are kept: an extraction fingerprint (GSQL superset, sampler parameters, scope) for cache identity and an input fingerprint (enabled groups, model paths) for the checkpoint (training.py:249; predictor.py:42-44); both, plus sampler parameters, enter the train/manifest agreement list (training.py:56-67). The renderer wraps each group's GSQL block in IF include_<group> THEN ... END so unused groups skip their joins and arithmetic. Summaries-only is Path 2 alone (today's tabular variant at model.py:79-82, which still pays the child fetch because make_live_batch always fetches first-hop contexts, batching.py:105-113; a batching flag skips the fetch when Path 1 is off). Graph-only is Path 1 alone. Hybrid is both. The no_fourier variant is retired because it keeps the pair window counts (model.py:69-77) and is neither a no-window nor a no-time cell. A single-path control cell (all groups through one projection, today's architecture) is kept so the split is tested rather than assumed.

## Experiment matrix

- **X0 cost sweep and parity gate (no training)**. What does each sampler parameter and each feature group cost in server time per context, and does the one-pass predecessor computation reproduce the pair queries exactly?
  - Config: benchmark_live_batch.py extended with per-request timing and emitted-events-per-root histograms; sweep k_recent {2,4,8}, k_old {0,3}, k_div {0,2,4}, max_scan {512,2048}, F1 {8,16,24}, lite_external_children on/off; toggle each include_<group>; 3 batches of 64 roots from the train date; parity of pair gaps and window counts against zelle_pair_time64 and payment_pair_time64 with the seed at message.event_seq + 1 (scripts/temporal/verify_live_training.py:66-93 style). Output: seconds per context per group; fixes the S1 budget and the GSQL hash before any training.
- **X1 nnPU sanity and noise floor**. Does the textbook positive weight (equal to the prior; training.py:70 passes only prior, loss.py:50-61 documents ranking inversion at pi about 0.001, loss.py:86 sets the default) invert rankings, and how large is validation proxy AP spread under label draw and seed alone?
  - Config: Current hybrid, current sampler emulated from the cache. Grid: class_prior {0.0005, 0.001, 0.002} (cohort prevalence 233 / 317,840 = 0.00073, inference from docs/experiments/mps_known60_readiness.json) x positive_weight {prior, 0.1, 0.5} (legacy trainer uses 0.5, src/mule_pattern_learner/training/train.py:79, 257). Then 3 train reveal draws x 2 init seeds at the chosen setting for the noise floor. Reveal draws change only the train split by adding a per-split seed to local_experiments/prepare_live_labels.py:31 (today one seed applies to all splits, so this is a script change); validation and test observed sets stay fixed. Decision: freeze the loss setting and publish the floor; differences below it are never reported as findings.
- **X2 summaries only**. How far do order-free behavioural summaries alone get on the observed-label proxy?
  - Config: Path 2 only with entity_meta + rolling_windows + amount_ratios + recency + association_counts (today's tabular variant, with the child fetch skipped by the new batching flag); second arm with decayed_activity replacing rolling_windows + amount_ratios. 6 paired repeats (3 draws x 2 seeds), frozen loss setting, strict_inductive, scope strict_mule_v1, dates 2024-07-01 / 2024-10-01 / 2025-01-01, evaluation_unlabeled_limit 2000 for validation selection.
- **X3 current hybrid, both samplers**. Reproduce the shipped contract (83 node, 135 message) as the reference, and test whether richer history alone changes the result with features fixed.
  - Config: message_core + pair_window_counts messages with entity_meta + rolling_windows + amount_ratios + recency + association_counts, single-path control and two-path head, x {S-cur emulated (per_relation 2, fanouts 8/4, round-robin), S1 (k_recent 4, k_old 3, k_div 2, F1 16 quota, F2 4 direction aware)}. 6 paired repeats each.
- **X4 graph without window-derived features, both samplers**. Can the event path learn mule behaviour with every window-derived number removed?
  - Config: message_core only in messages (132 dims: indices 4 to 6 dropped) with entity_meta + recency + association_counts on Path 2; rolling_windows, amount_ratios and pair_window_counts disabled; validate_context ratio assertions conditional (source.py:71-76); x {S-cur, S1}. 6 paired repeats.
- **X5 graph with minimal entity metadata and graph with zero node features, both samplers**. Do recency and association counts add anything over pure event history plus type/external/deposit/age, and is a zero-node-feature TGAT competitive (the owner's point that zero node features is a dataset configuration)?
  - Config: Cell A: message_core + entity_meta, Path 2 off. Cell B: message_core only, node encoder receives a constant vector, peers via base vector only. x {S-cur, S1}. 6 paired repeats. X3 to X5 fix the best graph variant (chosen on validation proxy AP under the adoption rule) and the sampler used from X7 onward.
- **X6 neighbour clock arm**. Does representing neighbours at the root's assessment cutoff (still all events < seed_seq) beat the interaction-time convention, and does it reduce unique contexts per batch?
  - Config: Best graph cell on S1 with child_key at the root's cutoff (batching.py:16-24). 6 paired repeats. Report unique contexts, cache size and AP; this arm is the cheap comparator for the root-only walks in X12.
- **X7 best graph + event_channel**. Does the payment channel (cash-out versus p2p versus purchase) add signal over rail?
  - Config: Precondition: channel vocabulary scan on both event types. Best graph + channel embedding versus best graph. 6 paired repeats.
- **X8 best graph + pass_through_message, then + pass_through_summary**. Does incoming-to-outgoing delay at the root detect this generator's intermediaries, and does a summary of it add anything over the per-event form?
  - Config: Cells: best graph + pass_through_message (scalar encoding); same with the Fourier64 delay encoding; best graph + pass_through_message + pass_through_summary. Censored flag reported. 6 paired repeats each.
- **X9 best graph + pair_history_window_free versus + pair_window_counts**. Do all-time pair count and pair first-seen age recover what the removed 1h/1d/7d pair counts provided, without windows?
  - Config: Best graph + pair_history_window_free; comparison cell best graph + pair_window_counts re-added. 6 paired repeats each.
- **X10 best graph + decayed_activity with half-life validation, versus hard windows**. Do smooth decayed summaries add to the graph path, which half-lives matter, and do they replace hard windows without loss?
  - Config: Best graph + decayed_activity with sets {1d,7d,30d,90d} and {1d,7d,30d}, plus a sub-day set {6h,1d,7d,30d} only if the granularity check shows intraday event clocks; comparison cell best graph + rolling_windows + amount_ratios re-added. 6 paired repeats each.
- **X11 best graph + counterparty_structure**. Do concentration, novelty breadth, token share and reciprocity summaries add over the sampler's distinct-peer coverage?
  - Config: Best graph + counterparty_structure, conditional on the MapAccum memory check in X0 passing on the highest-degree child contexts; log map sizes per context. 6 paired repeats.
- **X12 best graph + causal_paths walks**. Do bounded root-only relay and collection aggregates add detection at acceptable cost, beyond direction-aware second-hop selection and the neighbour clock arm?
  - Config: Best graph + relay_forward_at_root + collect_backward_at_root (k = 4, degree guard operational only); ablation with walks off and direction-aware selection on; compare against X6. Report per-batch latency delta and skip rate. 6 paired repeats.
- **X13 best graph + association_timing_bounds (conditional)**. Do sequence-derived tenure timing brackets carry signal, how wide are they, and is the exact-time schema change needed?
  - Config: Precondition: association row counts and closed-tenure counts per family, and the witness hit rate of valid_from_seq against event_seq and first_seen_seq. Best graph + association_timing_bounds. Report bracket width distribution. 6 paired repeats.
- **X14 best graph + device_ip_context (conditional)**. Are the event-to-device and event-to-IP edges populated, and does device or IP age at event add detection?
  - Config: Precondition: row counts and per-event density of Transfer_Used_Device, Transfer_Used_IP, Transaction_Used_Device, Transaction_Used_IP. Best graph + device_ip_context (age features only; device_shared_accounts is second wave). 6 paired repeats.
- **X15 combined winners and prior sensitivity**. What is the combined effect of every group that passed the adoption rule, and how sensitive is every conclusion to the nnPU prior?
  - Config: All adopted groups on S1; class_prior {0.0005, 0.001, 0.002} at the frozen positive_weight; 3 draws x 2 seeds plus 2 extra draws for the final cell (5 x 2).
- **X16 locked hidden-truth evaluation (once)**. On the frozen cohort containing all test-split mules, how do the pre-registered cells rank, with uncertainty?
  - Config: Run once after the matrix is frozen. Truth-holder cohort: all test-split mules (40 in strict_mule_v1 per positive_oracle.parquet, verified) plus 6,000 uniformly sampled test non-mules with inverse-probability weights, replacing the roughly one hidden mule that the 2,000-account uniform sample holds (sampling.py:45-57; about 20 hidden test mules among 47,749 test accounts per the snapshot manifest). Scorer sees IDs only; evaluate_predictions (evaluation.py:27-57, currently unweighted, evaluation.py:44) gains a weighted mode; report weighted precision at 1 and 5 percent review budgets, weighted AP, and 200-draw ownership-group bootstrap intervals via grouped_ap_interval (src/mule_pattern_learner/temporal/metrics.py:47), which the live trainer does not call today (training.py:16 imports only evaluate and select_threshold). All cells reported, no configuration change afterwards. Post-hoc interpretation only: typology probes (forward delay, amount match, collection index, onward-hop rate) on the 140 hidden train-partition mules, never on validation or test mules, and never as a go/no-go before this step.

## Order of work

1. Label-blind data checks on a uniform account sample (no training, no truth): event_ts_ms modulo 86,400,000 for intraday granularity; non-USD event count; share of events with an unresolved Token recipient; channel vocabulary on both event types; distinct Account.account_type values; row counts and per-event density of the four Used_Device and Used_IP edge families; association row counts and closed-tenure counts per family; witness hit rate of valid_from_seq against event_seq and first_seen_seq; global max event_seq and event_ts_ms after 2025-01-01.
2. Pre-training fixes that change the GSQL hash, landed together in one regeneration via scripts/temporal/render_training_queries.py: boundary unification, state_seq visibility, currency per-event exclusion, include_<group> and sampler parameters, one-pass walk, channel and stratum fields, per-request reset of vertex-attached scope accumulators. Parity test against the pair queries before the sender scans are removed.
3. Contract and pipeline plumbing: FEATURE_GROUPS and MESSAGE_GROUPS registry, extraction versus input fingerprints, contract-derived widths in memory.py, model.py, batching.py and source.py, conditional ratio checks, quota select_messages with direction-aware second hop and legacy emulation, lite_external_children, two-path LiveTGAT, positive_weight exposed in the live trainer, weighted evaluator and grouped bootstrap in the live path, per-split reveal seed in prepare_live_labels.py, run_live_experiments.py gains group, sampler and positive_weight axes and asserts scope_id.
4. X0 cost sweep and parity gate; fix S1 and freeze the query hash.
5. Build the SQLite superset cache once for S1 with children under both selection policies, at concurrency 4; record wall clock and per-request server time.
6. X1 nnPU sanity and noise floor; freeze the loss setting and publish the floor.
7. History before features: X2 summaries only, X3 current hybrid under both samplers, X4 graph without windows under both samplers, X5 minimal metadata and zero node features under both samplers, X6 neighbour clock. Choose the best graph variant and sampler on validation proxy AP under the adoption rule.
8. Feature additions one group at a time on the best graph variant: X7 channel, X8 pass-through, X9 window-free pair history, X10 decayed activity and half-lives, X11 counterparty structure, X12 causal path walks, X13 association timing bounds (conditional), X14 device and IP context (conditional).
9. X15 combined winners and prior sensitivity.
10. Freeze the matrix; X16 locked hidden-truth evaluation once; post-hoc typology interpretation on hidden train-partition mules only.
11. Decide the schema change (association timestamps) from the X13 bracket widths and the witness hit rate; request generator changes (onset clocks, ring IDs, mule accounts opened after the training cutoff) for the next dataset.

## Pre-training fixes

- Unify boundary conventions at five sites and pick one: window membership strict (seed_ts_ms - t.event_ts_ms < W, queries.py:190, 208) versus inclusive (age_ms <= W, gsql/features/zelle_pair_time64.gsql:122-124 and payment_pair_time64.gsql); pair counts relative to the payment (queries.py:345-347) versus relative to the seed cutoff (zelle_pair_time64.gsql:120-124); history selection strict on sequence and inclusive on timestamp (queries.py:174-175); root entity visibility at seed_seq versus association endpoints at state_seq (queries.py:138-141 versus 259-261); legacy snapshot strict timestamp boundary (src/mule_pattern_learner/temporal/snapshots.py:86-103). Recommendation: strict age comparison everywhere, strict seq plus inclusive ts for history, state_seq for all entity visibility; change the two pair queries and scripts/temporal/verify_time_encoding.py:251-276 (which asserts inclusive) to match, and make scripts/temporal/verify_live_training.py:82-93 compare temporal_training_context pair counts against the pair query's count_1h/count_24h/count_7d directly instead of recomputing them in Python. The owner decides the convention (open question); the fix lands before any cell runs.
- Choose assessment timestamps that match deployment: sample_keys pins ms = timestamp(date) - 1 (dataset.py:135) and manifest cutoff_seqs are keyed by date (dataset.py:228-242); predictor.py scores the same way. Keep midnight cutoffs only if scoring is a UTC-midnight batch. Otherwise no parser change is needed (timestamp() already accepts ISO datetimes, src/mule_pattern_learner/temporal/common.py:8-12); only manifest keying (dataset.py:239-241) and the eligibility comparisons (training.py:80-82 uses strict <, contexts use <= at source.py:87) change, and temporal_training_cutoffs accepts up to 24 cutoffs per call (gsql/temporal/training_cutoffs.gsql:11-13). Align the seed eligibility comparison with the context comparison either way.
- Expose positive_weight in the live trainer: training.py:70 builds NonNegativePULoss(prior=prior) with no positive weight; the default equals the prior (loss.py:86) and the docstring (loss.py:50-61) warns of ranking inversion at pi about 0.001; the legacy trainer passes 0.5 (training/train.py:79, 257). Do not cite artifacts/temporal/strict_readiness/model_run/metrics.json test ROC AUC 0.325 as evidence: that run has epochs 2, steps_per_epoch 2, batch_size 8 and evaluation_unlabeled_limit 4 (its config.json), so it is a smoke run.
- Make currency a per-event exclusion with a counter: the RETURN at queries.py:286-289 executes inside the FOREACH over requests, so one non-USD event in any root's visible history aborts all 16 contexts of the request and checked_rows fails the batch (source.py:55-60); the invalid_event_roles RETURN at queries.py:311-315 has the same blast radius. A deeper sampler raises the probability of hitting both.
- Introduce FEATURE_GROUPS and MESSAGE_GROUPS in contract.py; build FEATURE_NAMES from enabled groups; split contract_fingerprint (contract.py:85-96) into extraction and input fingerprints; add enabled groups and sampler parameters to the train/manifest agreement list (training.py:56-67); replace the literals 83 and 135 (memory.py:31-32), 135/7 (model.py:63), 135 (batching.py:74) and the len(RELATIONS) * 8 bound (source.py:69-70) with contract-derived values; replace x[8:] (batching.py:53) with per-group transform specs; make the ratio presence and cap checks conditional (source.py:71-76).
- Add runtime include_<group> BOOL parameters and sampler parameters (k_recent, k_old, k_div, max_scan) to the query signature (queries.py:44-48) with parameterised HeapAccum capacities (precedent gsql/features/pagerank.gsql:9, 16); regenerate once; keep installation.REVIEW_FILES single-file (installation.py:9-13); accept that query_hashes (dataset.py:23-33) invalidates every prepared dataset once. Update the three per_relation guards (queries.py:100-102; source.py:122-123, 254-259).
- Replace round-robin select_messages (batching.py:27-43) with quota selection by relation class and stratum, add the direction-aware second-hop policy and the legacy emulation path, and update tests that assume the interleave (tests/temporal/test_live_pipeline.py:100-123).
- Split LiveTGAT into node_event and node_summary encoders with a 2 x hidden head (model.py:57-67, 79-97); add a batching flag that skips the first-hop fetch when Path 1 is off (make_live_batch always fetches today, batching.py:105-113); retire the no_fourier variant; re-review memory.validate_model's 24x multiplier (memory.py:39-50).
- Decouple cache identity from config_sha256 (dataset.py:163-170; source.py:127-131; pipeline.py:33-36) so one cache per sampler variant serves every label draw, seed and model variant; add concurrency to ContextStore.fetch (source.py:169-191).
- Fix the evaluation cohort and add uncertainty: evaluation_indices (sampling.py:45-57) at limit 2000 holds about one hidden test mule, so evaluation.py:52-56 unlabeled_accounts is a coin flip; add the weighted truth-holder cohort and call grouped_ap_interval (src/mule_pattern_learner/temporal/metrics.py:47) from the live trainer and evaluator; record scope_id and per-split truth counts in every metrics file because the two label artifacts disagree: 160/33/40 mules and 222,337/47,754/47,749 accounts in local_experiments/strict_mule_v1 and the snapshot manifest versus 163/25/45 and 222,578/47,564/47,698 in docs/experiments/mps_known60_readiness.json (both total 233 and 317,840; two scope partitions of one population).
- Add a per-split reveal seed to local_experiments/prepare_live_labels.py:27-35 (today stable_score(value, 42, 'reveal') applies one seed to all three splits) so train reveal draws vary while validation and test observed sets stay fixed.
- Reset the vertex-attached @scope_allowed and @scope_blocked accumulators per request inside the FOREACH (queries.py:71, 126-130) or keep the single scope and phase invariant (source.py:207-208; batching.py:102-103) as an explicit test, because root-only walks add new vertex-attached state.
- Run the parity test between the one-pass predecessor computation and the pair queries before removing the sender scans (docs/leakage_and_scaling.md:80-81), and confirm equal-timestamp ordering in live data (docs/temporal_encoding.md:71-79; docs/temporal_schema.md:112-114): if sequences within an equal-timestamp group are not authoritative, the pass-through and onward features must use a strict timestamp boundary rather than a sequence tie-break.
- Extend the per-event role validity check to devices and IPs (first_seen_seq <= item.seq and first_seen_ts_ms <= item.ts, pattern at queries.py:305-309) before any device feature is emitted.
- Ask the generator owner for supervision clocks before interpreting behaviour features: mule_label_effective_ts_ms, mule_label_available_ts_ms, mule_label_known and mule_ring_id for the 233 mules (docs/account_mule_labels.md:22-33); the live audit shows known_labels 0 and invalid_unknown 233 (docs/experiments/live_training_readiness.json:31-43), all 233 mules have first_seen_ts_ms 2024-01-01 (verified), and docs/leakage_and_scaling.md:57-61 says to change the simulation rather than move old accounts into a cold-start cohort.

## Schema and data changes recommended

- Add valid_from_ts_ms UINT and valid_to_ts_ms UINT (0 = unknown or open) to all seven association edge families (Party_Owns_Account, Party_Uses_Token, Token_Bound_To_Account, Party_Uses_Device, Account_Uses_Device, Party_Uses_IP, Party_Has_Address; temporal_schema.gsql:135-189), populated by the loader as valid time. This is the only change that makes identity-change-to-payment timing measurable in hours; sequences alone cannot (docs/temporal_training.md:48-55). Decide after X13 reports bracket widths and after the witness hit rate is measured. Whether ALTER ADD ATTRIBUTE works on non-discriminator edge attributes without a reload is external TigerGraph knowledge to verify; docs/temporal_schema.md:89-91 only rules out changing discriminators.
- Optional, second wave: a directed pair summary edge (Account or Token recipient, per rail) carrying first_seen_seq, first_seen_ts_ms, last_seen_seq and count, maintained in ingestion order, to make counterparty novelty exact and cheap instead of bounded by retained history; the legacy HAS_PAID edge carried first_txn_date (gsql/schema/schema.gsql:90-103). Only if the one-pass MinAccum proves too costly on hub accounts or if truncation rates are material.
- No schema change is needed for incoming-to-outgoing delay, causal path walks, decayed summaries, counterparty concentration, channel, or device and IP age at event: every event vertex and participation edge already carries event_ts_ms and event_seq with named reverse types (temporal_schema.gsql:90-131, 193-263), and channel plus device and IP role edges exist but are unread today.
- Data, not schema: the generator should populate the existing Account supervision fields (mule_label_effective_ts_ms, mule_label_available_ts_ms, mule_label_known, mule_ring_id; temporal_schema.gsql:44-60), open some mule accounts after the training cutoff, and document mule mechanics (hop delay, fan-out, cycle length, amount policy, dormancy) so onset-relative and ring-aware evaluation become possible.
- Outstanding work already named in the docs and not part of this plan: time-bucketed adjacency or a temporal index and directed-pair predecessor state (docs/leakage_and_scaling.md:179-184), which are the only way to bound server scan rather than returned rows.

## Deck outline (schema first, then one slide per feature idea)

- **The graph we already have**. People, accounts, phones and emails, devices, IP addresses and postal addresses are the nodes; every Zelle transfer and every other payment is its own node with an exact time, an order number, an amount, a currency, a rail and a channel; the links between people and their accounts, tokens, devices and addresses are intervals with a start and an end.
  - Visual: One simple diagram: a circle for each entity type, a diamond for the two payment types, arrows labelled with the words sent, received, uses, owns; no boxes with icons, no card grid.
- **Why this schema is good for mule detection**. Nothing is aggregated away: every payment can be reached from the sender and from the recipient in both directions, with its exact time and order, so we can compute how fast money left, whether a recipient is new, and who paid whom before whom, without a second data source.
  - Visual: A single horizontal timeline of one account's payments with arrows in and out, each stamped with its time; a caption reading exact time on every payment, in both directions.
- **What the schema cannot tell us yet**. Associations (a phone bound to an account, a device used by an account) carry an order number but no clock time, so we can say a device was added before payment number 40 and after payment number 39, but not that it was added two hours before a transfer; adding two timestamp fields to those seven link types fixes this.
  - Visual: The same timeline with a bracket drawn between two payments labelled device added somewhere here; a second faint timeline below showing the exact point once timestamps exist.
- **How the model reads the graph**. Two separate inputs: the event path reads the ordered payments and links around an account (amount, direction, rail, channel, how old, how long since the last payment to the same counterparty, and the neighbours reached through them); the optional summary path reads a handful of numbers about the account as a whole; a small head combines them, and either path can be switched off to see what it contributes.
  - Visual: Two plain arrows converging on one box labelled score: the upper arrow drawn as a sequence of small payment marks, the lower as a short list of numbers; a switch symbol on the lower arrow.
- **Better history first**. Today the model typically sees five recent payments per account (at most eight) out of roughly one hundred and fifty; the new sampler shows the most recent payments in each direction, a few payments spread across the account's older history so a burst does not hide what came before, the latest payment to each distinct counterparty, and the account's people, phones and devices; every lookup still stops at its own historical cutoff.
  - Visual: Two rows of dots on a shared time axis: the top row with five highlighted dots clustered at the right edge, the bottom row with highlights at the right edge, spread across the middle, and marked by counterparty; plain text labels, no cards.
- **Feature idea: money in, money out**. For each payment an account receives, measure how long until the account next sends money and how the amounts compare; mules tend to forward quickly and nearly whole; this is a timing and amount coincidence, not proof the same dollars moved, and we say so.
  - Visual: A timeline with a green arrow in and a red arrow out shortly after, a bracket between them labelled delay, and two bars of nearly equal height for the two amounts.
- **Feature idea: how the money left**. Every payment already records a channel such as ATM withdrawal, card purchase or peer-to-peer transfer; cash-out looks different from onward transfer, and nothing reads this field today.
  - Visual: Three small pictograms (ATM, card, phone-to-phone) each above a short arrow leaving the same account, with the channel word beneath each.
- **Feature idea: who the money goes to**. Is the recipient new to this account, how much of the outgoing money goes to a single destination, and how many senders feed how few recipients; these describe collection and dispersal without any fixed time window.
  - Visual: A funnel shape: many small arrows entering an account from the left, one or two thick arrows leaving to the right; a small first-time tag on a new recipient.
- **Feature idea: change from the account's own past**. Compare recent activity with the account's own longer history using smoothly decaying counts and amounts instead of hard windows; a quiet account that turns busy, or whose amounts jump, stands out against itself; on the current synthetic data every mule exists from day one and has no recorded onset, so this idea cannot yet be validated and is marked exploratory.
  - Visual: A gently decaying curve over a payment timeline, with a recent spike rising above the curve; a small note in the corner reading validation needs onset dates from the generator.
- **Feature idea: identity changes near payments**. A new device or a re-bound phone shortly before transfers is a classic takeover or mule pattern; today we can bracket when the change happened between two of the account's own payments, in real time units, and count how many changes fell among the last ten payments; exact hours need the two new timestamp fields.
  - Visual: The account timeline with a device symbol placed between two payment marks and a bracket labelled between these two payments; beside it a faded exact-point version labelled after schema change.
- **Feature idea: devices and addresses on the payments themselves**. Payments can carry the device and the IP address used; if those links are populated, we can measure how new the device was at the moment of the payment in exact time, and later how many accounts share it.
  - Visual: A payment mark with two small tags hanging from it, a device tag and an IP tag, each with a small age label; a footnote that population of these links is being checked.
- **Feature idea: time-ordered patterns around the account**. Bounded, causally ordered look-arounds: did the recipient move the money on after receiving it, did the sender just receive before paying in, did money come back from the same counterparty; limited to a handful of counterparties per account and never a whole-graph search.
  - Visual: A three-node chain drawn left to right with arrows labelled first, second, third in time order, and a small loop arrow returning to the first node.
- **How we will decide**. With about twenty revealed mules per split, single runs are noise; every configuration runs several times with different label draws and seeds, a change is adopted only under a rule fixed in advance, history improvements are tested before feature additions, and the hidden truth is opened once, at the end, on a cohort that contains every test mule.
  - Visual: A plain ladder of steps from top to bottom: data checks, fixes, cost sweep, noise floor, summaries only, current hybrid, graph without windows, minimal metadata, one feature at a time, locked evaluation.
- **What we need from the data side**. Timestamps on the seven association link types, mule onset and ring identifiers from the generator, some mule accounts opened after the training cutoff, and a decision on when in the day scoring will run so training uses the same clock.
  - Visual: A short plain checklist with four lines and empty check boxes.

## Open questions for the owner

- Deployment clock: will scoring run as a UTC-midnight batch, or at another fixed hour or continuously? This fixes the assessment timestamps for training (today midnight minus one millisecond, dataset.py:135) and decides whether the manifest must be keyed by full timestamps.
- Boundary convention: do you want strict (age < window, as the renderer does at queries.py:190, 208, 345-347) or inclusive (age <= window, as the installed pair queries do at gsql/features/zelle_pair_time64.gsql:122-124) as the single authoritative rule? The plan recommends strict; both verifiers and the pair queries change to match.
- Which scope partition is canonical for all cells: strict_mule_v1 (160/33/40 mules, 222,337/47,754/47,749 accounts per local_experiments/strict_mule_v1 and the snapshot manifest) or the partition recorded in docs/experiments/mps_known60_readiness.json (163/25/45, 222,578/47,564/47,698)?
- Can the generator and the loader be changed for the next dataset: association valid_from_ts_ms and valid_to_ts_ms supplied as valid time, mule_label_effective_ts_ms and mule_ring_id populated, some mule accounts opened after 2024-07-01, and a written description of the mule mechanics (hop delay, fan-out, cycle length, amount policy, dormancy)? Without this, onset, cold-start and ring-aware claims stay unvalidatable.
- Budget: what wall clock is acceptable for the one-time superset cache build per sampler variant (estimated 3.5 to 12.5 hours depending on server concurrency) and for the experiment matrix (roughly 150 training runs from the cache), and may the cache build run at concurrency 4 against the live TigerGraph instance?
