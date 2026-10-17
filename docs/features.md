# The features in every batch

Every number that reaches the model in a batch, with its exact definition, encoding, width
and leak guard, and the PhantomLedger gates it depends on. It is part of
[the design](design.md), written for the owner's request "Figure out all the features that we
get in during every batch" under his rule that leaks are absolutely not allowed, and it was
checked by two adversarial reviews, one hunting leaks only.

- Why each query and field exists, with the mule behaviour and the ordinary look-alike it
  separates, is in the design's [query sections](design.md#the-queries).
- The rules every query shares (time bounds, visible and current links, known starts, ties,
  caps, money, self-payments) are in the design's [Time](design.md#time); hubs, crowds,
  leaves and the sharing rule are in its [Shared rules](design.md#shared-rules).
- Guard names in the tables below are those of the design's
  [leak rules](design.md#leak-rules-and-how-tests-enforce-them).
- Encodings are those of the design's [Encoding](design.md#encoding). "Presence + 64 Fourier"
  is a duration: a presence bit and 64 Fourier numbers, 65 wide.

## In short

- A batch is a set of examples. An example is an (account, Monday t) pair with its own copy of
  a two-hop subgraph, so no node is shared between examples.
- The model receives four node feature matrices (account, customer, device, IP), sixteen
  relations (eight and their reverses), each with an edge index and, for fourteen of them,
  edge attributes, one community row per example, the index of each scored account, and, in
  training only, a label and a weight per example.
- Widths after encoding: account 309, customer 11, device 12, IP 12; payment edge 401;
  ownership edge none; enrolment edge 130 (customer level and the account-level copy);
  seen-on-payment edge 131; community block 8. The head reads 64 + 8 = 72 numbers.
- Every duration is a presence bit plus 64 Fourier numbers. The cache and the built batch keep
  raw integer milliseconds; the GPU encodes them each step.

## What one example's subgraph holds

Names used below: s is the scored account, C_s its customer, D_s the devices and IPs C_s is
enrolled on (kept), S_s the devices and IPs seen on s's payments (kept).

1. **s**, with its full row, even when it is a hub.
2. **Its payments:** s's newest 32 payments sent and newest 32 received in the lookback.
3. **Hop-1 accounts:** the other ends of those payments.
   - A hop-1 account hub gets a flag-only row and is not expanded.
   - Every other hop-1 account is expanded: its own newest 16 payments sent and 16 received
     (deduplicated by payment), its customer, its owner-enrolment edges to its customer's kept
     devices and IPs, and its seen edges.
4. **Hop-2 accounts:** the other ends of hop-1's payments. They get rows (flag-only if hubs)
   and are not expanded.
5. **Other accounts of C_s:** up to 8 that exist at t with a visible ownership link, open or
   closed, most recently active first (latest payment in the lookback), then accounts with no
   payment in the lookback, ties by account id. They get rows and are not expanded. They are
   never ordered by ownership start, an opening date.
6. **Devices and IPs of C_s (D_s):** C_s's visible enrolment links, up to 8 of each kind,
   ordered by their newest evidence: the later of the link's known start and the latest
   payment seen on the node from an account C_s owns. Links with neither come last; ties by
   node id. They join as enrolment edges from C_s and as owner-enrolment edges from s and from
   every account of C_s in the subgraph.
7. **Seen devices and IPs:** for s (S_s) and for each expanded hop-1 account, the devices and
   IPs seen on its payments sent in the lookback, up to 8 of each, latest use first, ties by
   node id. Leaves seen this way stay as flag-only nodes.
8. **Co-users of s's devices and IPs.** From each non-leaf node N in D_s or S_s:
   - Candidates are the customers counted in N's `customers_90d` other than C_s: a visible
     enrolment link to N, or a deposit account that exists at t, with visible ownership, that
     sent a payment seen on N in the lookback. One candidate per customer.
   - Up to 48 per scored account are kept, ordered by their newest evidence on any of s's
     nodes (the later of a known link start and the latest seen use; neither counts as
     `t - 90 d`), ties by customer id. No order reads a link end.
   - Each kept customer brings one account: among its deposit accounts that exist at t with
     visible ownership, the one most recently seen sending on s's nodes, or, if none was seen,
     the one with the latest payment in the lookback, then accounts with no payment, ties by
     account id.
   - A kept customer with a visible link to N joins as a customer node with its enrolment
     edge, and its account joins with an owner-enrolment edge to N. A customer seen only
     through payments joins through its account's seen edge; its customer node would be three
     hops away and is not fetched.
9. **Closure around s's own nodes.** For every non-leaf node one hop from s, every enrolment,
   owner-enrolment and seen edge between it and a customer or account already in the subgraph
   is included.
10. **Ownership edges** join s and each expanded hop-1 account to its customer, and every
    account of the subgraph that C_s owns to C_s. Other ownership edges would join two nodes
    two hops away and are not fetched.

Every account node exists at t: payment ends by the loader's check that no exported payment
comes before its accounts open, and accounts reached through C_s or a co-user by the explicit
`opened_ms < t` and visible-ownership tests above. Every account node gets B1 to B3 rows
(flag-only for a hub that is not s), every customer node a customer row, and every device and
IP node a node row.

## What reaches the score in two layers

The scored account's final representation depends on the node rows within two hops and on the
attributes of edges into s or into its one-hop nodes.

- **One hop:** s's payment counterparties (with payment edges), C_s, D_s through the
  owner-enrolment edges, S_s through the seen edges.
- **Two hops:** hop-2 accounts and their payment edges into hop-1; hop-1 accounts' customers,
  owner-enrolled devices and IPs, and seen devices and IPs; the other accounts of C_s;
  co-users of s's devices and IPs, their customers through enrolment edges and their accounts
  through owner-enrolment or seen edges, with their B1 to B3 rows.
- **Three or more hops:** not fetched.

So an account with no payments that shares a ring device with an active mule (the owner's
Account Z) reads the mule account's velocity, counterparty and activity rows at two hops (s,
the device, the mule account), beside its customer's co-user counts, the device's counts and
the community block.

## Account node: 309 columns

**B1's walk** (`read_money_flow`), the basis of B1's columns and of the per-payment dwell. Its
GSQL form is in the design's [B1 section](design.md#read_money_flow-b1-velocity).

- **Lists.** The inflows are the newest 256 payments received in `[t - 30 d, t)` with positive
  cents; the outflows are the newest 256 sent then.
- **Span.** If neither list was cut, the span is `[t - 30 d, t)`. Otherwise b is the latest
  second, over the cut lists, of the newest payment left out of a list, and the span is
  `(b, t)`: every payment of either list at or before b is dropped. Both lists are then
  complete inside the span, so the walk is exactly last in, first out over the whole 30 days
  for every inflow it keeps.
- **Lots.** A lot is the set of inflows of one second inside the span. Within one second the
  lot comes before every outflow; the order among same-second outflows changes nothing.
- **Matching.** Last in, first out: an outflow of c cents spends from the most recent lot with
  unspent cents, then the next, until c is spent or none is left. The rest of the outflow spent
  money from before the span and is not recorded.
- **Matched portion.** Each match records (lot, cents, dwell = outflow time minus lot time, the
  outflow's rail).
- **R** is the cents received in the span.

| Col | Feature | Query | Definition as of t | Temporal | Kind | Encoding | Width | Leak guard |
|---|---|---|---|---|---|---|---|---|
| 1 | `is_external` | read_payment_events | the account belongs to another bank (export attribute) | no | structure | flag | 1 | a static attribute, 0 on every scored account; Gates (the share of victims at the same bank) |
| 2 | `hub` | read_hubs | account hub at t | no | structure | flag | 1 | Before t; Same rule for all |
| 3 | `hub_customers_90d` | read_hubs | on a hub row, distinct customers owning the other accounts of its payments in `[t - 90 d, t)`; 0 on other rows | yes | structure | log1p count | 1 | Before t; the all-time prefilter changes cost, never an answer |
| 4 to 6 | `in_amount_7d`, `_30d`, `_90d` | read_money_flow | cents received in the last 7, 30, 90 days (all payments, uncapped) | yes | behaviour | log1p dollars | 3 | Before t; Lookback; Gates |
| 7 to 9 | `out_amount_7d`, `_30d`, `_90d` | read_money_flow | cents sent in the last 7, 30, 90 days | yes | behaviour | log1p dollars | 3 | as above |
| 10 | `flow_present` | read_money_flow | R > 0 | yes | behaviour | flag (presence) | 1 | as above |
| 11 to 16 | `left_15m`, `left_1h`, `left_6h`, `left_1d`, `left_3d`, `left_7d` | read_money_flow | matched cents with dwell < w, divided by R; an outflow at or after t never matches | yes | behaviour | share | 6 | Before t; Lookback; Order-free ties; Same rule for all; Gates |
| 17 | `held_share` | read_money_flow | (R minus all matched cents) / R | yes | behaviour | share | 1 | as above; Gates (every mule keeps 5% to 10% of every credit) |
| 18 | `median_dwell` | read_money_flow | portions in ascending dwell: the dwell at which cumulative cents first satisfy `2 x cum >= R`; absent when less than half left before t | yes | behaviour | presence + 64 Fourier | 65 | as `left_*` |
| 19 to 21 | `left_by_transfer`, `left_by_card`, `left_by_cash` | read_money_flow | of matched cents with dwell < 1 day, the share whose outflow rail is transfer (`ach`, `check`, `internal`, `zelle`), `card`, `cash`; `unknown` counts in the denominator only | yes | behaviour | share | 3 | as `left_*` |
| 22 | `payers_30d` | read_counterparties | distinct accounts that paid it in the last 30 days | yes | structure | log1p count | 1 | Before t; Lookback |
| 23 | `payees_30d` | read_counterparties | distinct accounts it paid in the last 30 days | yes | structure | log1p count | 1 | Before t; Lookback |
| 24, 25 | `new_internal_payers_7d`, `_30d` | read_counterparties | distinct payers with `is_external` 0 whose earliest payment to it in the lookback is in the last 7 or 30 days | yes | structure change | log1p count | 2 | Lookback; No dates of record; Gates |
| 26, 27 | `new_external_payers_7d`, `_30d` | read_counterparties | the same for payers with `is_external` 1 | yes | structure change | log1p count | 2 | as above |
| 28, 29 | `new_payees_7d`, `_30d` | read_counterparties | distinct payees whose earliest payment from it in the lookback is in the last 7 or 30 days | yes | structure change | log1p count | 2 | as above |
| 30 | `new_payer_share_30d` | read_counterparties | cents received in the last 30 days from payers new over 30 days, divided by all cents received then | yes | behaviour | share | 1 | as above |
| 31 | `max_new_payers_24h` | read_counterparties | over the first-payment times of its payers new over 30 days (one per payer, the newest 256 kept by the whole-second rule), the most in `[x, x + 24 h)` with x one of them | yes | behaviour | log1p count | 1 | as above; Clock; Order-free ties |
| 32 | `top_payee_share_30d` | read_counterparties | the most cents sent to one payee in the last 30 days, divided by all cents sent then | yes | behaviour | share | 1 | Before t; Lookback |
| 33 | `hub_payee_share_30d` | read_counterparties | cents sent in the last 30 days to account hubs, divided by all cents sent then | yes | behaviour | share | 1 | Before t (hubs as of t) |
| 34 to 36 | `in_7d`, `in_30d`, `in_90d` | read_activity_change | payments received in the last 7, 30, 90 days | yes | behaviour | log1p count | 3 | Before t; Lookback; Gates |
| 37 to 39 | `out_7d`, `out_30d`, `out_90d` | read_activity_change | payments sent in the last 7, 30, 90 days | yes | behaviour | log1p count | 3 | as above |
| 40, 41 | `burst_in_1h`, `burst_in_24h` | read_activity_change | over its newest 256 payments received in the last 30 days (whole seconds), the most in `[x, x + w)` with x one of them; same-second payments all count | yes | behaviour | log1p count | 2 | Clock; Order-free ties; Same rule for all; Gates |
| 42, 43 | `burst_out_1h`, `burst_out_24h` | read_activity_change | the same over payments sent | yes | behaviour | log1p count | 2 | as above |
| 44 | `shortest_in_gap` | read_activity_change | over the same kept inflows, the smallest `x - x'` with x' the latest kept inflow strictly before x; absent when no kept inflow has a strictly earlier one | yes | behaviour | presence + 64 Fourier | 65 | Clock; Gates |
| 45 | `since_last_in` | read_activity_change | t minus its latest payment received in the lookback; absent with none | yes | behaviour | presence + 64 Fourier | 65 | No identification closure; Clock |
| 46 | `since_last_out` | read_activity_change | t minus its latest payment sent in the lookback; absent with none | yes | behaviour | presence + 64 Fourier | 65 | as above |
| 47 to 53 | degree columns | the built batch | see [Degree columns](#degree-columns) | no | structure | log1p count | 7 | Same rule for all |

Width: 3 + 82 (B1) + 12 (B2) + 205 (B3) + 7 (degrees) = 309. Stored in the cache before
encoding: 50 (the four durations kept as presence and milliseconds); the degrees are counted
when the batch is built.

**Flag-only rows.** An account hub that is not the scored account carries `is_external`,
`hub` = 1, `hub_customers_90d` and its degree columns. Every other column is 0 and every
presence bit 0. It is never expanded, its B1 to B3 are never computed, and the payment edges
touching it carry no gap or dwell at its end.

## Degree columns

Attention's softmax averages over neighbours and cannot count them, so every node carries, for
each relation that sends messages into its type, `log(1 + the number of edges of that relation
ending at the node in the example's subgraph)`. They are counted when the batch is built,
from the edge lists the queries returned and after any ablation switch, because they describe
the batch's own structure rather than a value of the graph; the same caps and rules build every
subgraph, so they mean the same for every example.

| Node | Relations counted (edges arriving) | Width |
|---|---|---|
| account | `pays` (payments received), `rev_pays` (payments sent), `rev_owned_by`, `rev_owner_enrolled_on_device`, `rev_owner_enrolled_on_ip`, `rev_seen_on_device`, `rev_seen_on_ip` | 7 |
| customer | `owned_by`, `rev_enrolled_on_device`, `rev_enrolled_on_ip` | 3 |
| device | `enrolled_on_device`, `owner_enrolled_on_device`, `seen_on_device` | 3 |
| IP | `enrolled_on_ip`, `owner_enrolled_on_ip`, `seen_on_ip` | 3 |

## Customer node: 11 columns (`read_shared_identity`)

| Col | Feature | Definition as of t | Temporal | Kind | Encoding | Width | Leak guard |
|---|---|---|---|---|---|---|---|
| 1 | `accounts_owned` | accounts with `opened_ms < t` and a visible ownership link to it, open or closed | no | structure | log1p count | 1 | No dates of record (open or closed alike); the ownership end is read only by the visible test, so a churned account counts for 90 days after it closes; Gates |
| 2 | `devices_30d` | devices it has a current enrolment link to (held at some time in the last 30 days) | no | structure | log1p count | 1 | No link ends as values |
| 3 | `ips_30d` | the same for IPs | no | structure | log1p count | 1 | No link ends as values |
| 4 | `new_devices_30d` | device links with a known start in the last 30 days | yes | structure change | log1p count | 1 | Known starts; Gates |
| 5 | `new_ips_30d` | the same for IPs | yes | structure change | log1p count | 1 | Known starts; Gates |
| 6 | `co_users` | distinct customers sharing at least one non-leaf device or IP with it | yes | structure | log1p count | 1 | leaves excluded; No label conditioning; Gates |
| 7 | `new_co_users_30d` | co-users whose pair began sharing in the last 30 days | yes | structure change | log1p count | 1 | Known starts (an unknown start counts as `t - 90 d`) |
| 8 | `co_users_both` | co-users sharing at least one device and at least one IP with it | yes | structure | log1p count | 1 | Gates (a near-pure ring marker until PhantomLedger's households step) |
| 9 to 11 | degree columns | see [Degree columns](#degree-columns) | no | structure | log1p count | 3 | Same rule for all |

## Device and IP nodes: 12 columns each (`read_shared_identity`, flags from `read_hubs`)

| Col | Feature | Definition as of t | Temporal | Kind | Encoding | Width | Leak guard |
|---|---|---|---|---|---|---|---|
| 1 | `customers_30d` | customers sharing it under the sharing rule | no | structure | log1p count | 1 | Before t; No link ends as values; Same rule for all |
| 2 | `customers_90d` | distinct customers with a visible enrolment link to it or owning a deposit account that exists at t and sent a payment seen on it in the lookback (the hub measure) | yes | structure | log1p count | 1 | Before t; Lookback |
| 3, 4 | `new_customers_7d`, `_30d` | customers whose sharing start on it is in the last 7 or 30 days | yes | structure change | log1p count | 2 | Known starts; Gates |
| 5 | `accounts_seen_30d` | distinct accounts seen sending payments on it in the last 30 days | yes | behaviour | log1p count | 1 | Gates |
| 6 | `payments_seen_30d` | payments seen on it in the last 30 days | yes | behaviour | log1p count | 1 | Gates |
| 7 | `observed_only` | no customer has a visible enrolment link to it | no | structure | flag | 1 | Gates (every observed endpoint gets usage edges) |
| 8 | `crowd` | `customers_90d` > 8 (a leaf) | no | structure | flag | 1 | Same rule for all |
| 9 | `hub` | `customers_90d` > 200 | no | structure | flag | 1 | Same rule for all |
| 10 to 12 | degree columns | see [Degree columns](#degree-columns) | no | structure | log1p count | 3 | Same rule for all |

A leaf's row fills only `customers_90d`, `observed_only`, `crowd`, `hub` and its degrees; the
rest are 0. A non-leaf node has at most 8 customers.

## Payment edges: 401 attributes

`pays` runs from sender to receiver, and its reverse `rev_pays` carries the same values.

| Attr | Feature | Query | Definition as of t | Temporal | Kind | Encoding | Width | Leak guard |
|---|---|---|---|---|---|---|---|---|
| 1 | `amount` | read_payment_events | the payment's cents | no | behaviour | log1p dollars | 1 | integer cents; Gates |
| 2 to 8 | `rail` | read_payment_events | `payment_rail`, or `zelle` for a `Zelle_Transfer` | no | behaviour | one-hot | 7 | Gates (`unknown` honest only after the rail fixes) |
| 9 | `age` | read_payment_events | t minus the payment's time | yes | behaviour | presence (always 1) + 64 Fourier | 65 | Before t; Clock |
| 10 | `pair_prior_count` | read_payment_events | payments from the same sender to the same receiver with `t - 90 d <= ts' < ts` | yes | structure | log1p count | 1 | Lookback; strictly earlier, so same-second pair payments do not count each other |
| 11 | `pair_gap` | read_payment_events | ts minus the latest of those; absent when there are none (its absent bit is "first in pair") | yes | structure change | presence + 64 Fourier | 65 | as above |
| 12 | `pair_age` | read_payment_events | ts minus the earliest of those; absent likewise | yes | structure change | presence + 64 Fourier | 65 | as above |
| 13 | `sender_gap` | read_payment_events | ts minus the sender's latest payment sent to anyone with `t - 90 d <= ts' < ts`; absent at a hub sender or with none | yes | behaviour | presence + 64 Fourier | 65 | Before t; Lookback; Same rule for all; Gates |
| 14 | `receiver_gap` | read_payment_events | ts minus the receiver's latest payment received from anyone, same bounds; absent at a hub receiver or with none | yes | behaviour | presence + 64 Fourier | 65 | as above |
| 15 | `dwell_tracked` | read_money_flow | the payment belongs to a lot of its receiver's B1 walk (receiver not a hub, payment inside the span) | yes | behaviour | flag (presence) | 1 | as B1's `left_*` |
| 16 | `left_24h` | read_money_flow | of its lot's cents, the share matched to the receiver's outflows within 24 hours and before t; 0 when not tracked | yes | behaviour | share | 1 | as B1's `left_*` |
| 17 | `dwell` | read_money_flow | from ts until its lot's matched cents first satisfy `2 x cum >= lot cents`; absent when not tracked or less than half left before t | yes | behaviour | presence + 64 Fourier | 65 | as B1's `left_*` |

Width 401. Stored before encoding: 23. Each payment row depends only on the payment, the date
and the hub set, never on which account is scored, so it can be cached per payment; for the
same reason gaps and dwell are absent at every hub end, the scored hub's included. Payments of
one lot share `left_24h` and `dwell`, so the order inside one second never matters.

## Ownership edges: no attributes

`owned_by` (account to customer) and `rev_owned_by` are built with `edge_dim` None: the edge
itself is the information (a shared owner, a customer's other accounts). Leak guard: Before t;
one owner and one ownership tenure per account, refused otherwise by the loader; the ownership
end is read only by the visible test.

## Enrolment edges: 130 attributes

Two relations carry these fields:

- `enrolled_on_device`, `enrolled_on_ip` (customer to node) and their reverses: C_s's kept
  links, and co-user customers' links to s's nodes.
- `owner_enrolled_on_device`, `owner_enrolled_on_ip` (account to node) and their reverses: the
  same links copied onto the accounts of the subgraph whose customer holds them (s, the other
  accounts of C_s, expanded hop-1 accounts, co-users' accounts), with caps counted per
  customer, so every account of one customer has the same set. This puts a co-user's account
  two hops from s.

| Attr | Feature | Definition as of t | Temporal | Kind | Encoding | Width | Leak guard |
|---|---|---|---|---|---|---|---|
| 1 | `since_start` | `t - valid_from_ms`, present only with a known start | yes | structure change (relationship age) | presence + 64 Fourier | 65 | Before t; Known starts; Clock; Gates |
| 2 | `since_last_seen` | t minus the latest payment in the lookback, sent by an account the customer owns (visible ownership), that was seen on the node; absent with none | yes | behaviour | presence + 64 Fourier | 65 | Lookback; No identification closure; Clock |

Width 130. Stored before encoding: 4. `active`, `since_end` and `held` are not fields: they
read `valid_to_ms` against t (see the design's
[B5 section](design.md#read_shared_identity-b5-the-device-and-ip-graph)).

## Seen-on-payment edges: 131 attributes

`seen_on_device` and `seen_on_ip` run from account to node, with their reverses.

| Attr | Feature | Definition as of t | Temporal | Kind | Encoding | Width | Leak guard |
|---|---|---|---|---|---|---|---|
| 1 | `uses_90d` | payments it sent in the lookback seen on the node | yes | behaviour | log1p count | 1 | Lookback; Gates |
| 2 | `since_first_use` | t minus the first of them | yes | behaviour | presence (always 1) + 64 Fourier | 65 | Lookback (the first use within 90 days, never a date of record); Clock |
| 3 | `since_last_use` | t minus the last of them | yes | behaviour | presence (always 1) + 64 Fourier | 65 | as above |

Width 131. Stored before encoding: 5.

## The community block: 8 numbers per example

`read_community` builds the members as the design's
[B6 section](design.md#read_community-b6-how-the-mule-network-formed) says (rounds 0 to 2 by
sharing, then the payment round, at most 64). The block joins the scored account at the head.

| Col | Feature | Definition | Temporal | Kind | Encoding | Width | Leak guard |
|---|---|---|---|---|---|---|---|
| 1 | `members` | members kept, the owner included | yes | structure | log1p count | 1 | Before t; leaves excluded |
| 2 | `capped` | a candidate of any round was left out | yes | structure | flag | 1 | Same rule for all |
| 3 | `shared_nodes` | non-leaf devices and IPs shared by at least two members | yes | structure | log1p count | 1 | No link ends as values |
| 4 | `density` | links / (members x shared nodes), a link being a member sharing a shared node; 0 with no shared node | yes | structure | share | 1 | as above |
| 5 | `new_link_share_30d` | links whose sharing start is at or after `t - 30 d`, divided by links; 0 with none | yes | structure change | share | 1 | Known starts |
| 6 | `internal_amount_30d` | cents of payments in the last 30 days between accounts of two different members | yes | behaviour | log1p dollars | 1 | Before t |
| 7 | `isolation` | internal / (internal + external) payment counts of the last 30 days, external meaning the other account belongs to no member (payments within one member's own accounts count as neither); 0 with none | yes | structure | share | 1 | Before t |
| 8 | `member_payments_30d` | internal plus external payment counts of the last 30 days | yes | behaviour | log1p count | 1 | Before t |

A member's accounts are those with `opened_ms < t` and a visible ownership link, open or closed,
leaving out account hubs, as their flag-only rows do. The block is never built from a library
algorithm or from the demo query.

## The other tensors of a batch

| Tensor | Shape and type | Training batch | Scoring batch |
|---|---|---|---|
| `x[account]`, `x[customer]`, `x[device]`, `x[ip]` | [nodes, 309 / 11 / 12 / 12], float32 after encoding | yes | yes |
| `edge_index` for 16 relations: `pays`, `owned_by`, `enrolled_on_device`, `enrolled_on_ip`, `owner_enrolled_on_device`, `owner_enrolled_on_ip`, `seen_on_device`, `seen_on_ip`, each with `rev_` | [2, edges], int64 | yes | yes |
| `edge_attr` for the 14 relations other than `owned_by` and its reverse | [edges, 401 / 130 / 131] | yes | yes |
| `scored_index` | [B], int64: the position of each example's scored account among the account nodes | yes | yes |
| `community` | [B, 8], row b for example b | yes (zeros and unused in the ablations without it) | yes |
| `label` | [B], 1 for a positive, 0 for unlabelled | yes | no |
| `weight` | [B] | yes | no |
| presence bits | inside the feature matrices: every duration's bit, `flow_present`, `dwell_tracked` | yes | yes |
| example keys (world, t, account id) | a CPU list, never on the model | yes | yes |

`label` and `weight` follow the design's [Training world](design.md#training-world); labels and
weights never join a node row. A scoring batch (selection, test, `mule score`) is built by the
same queries and settings without `label` and `weight`, so a scored account's subgraph is
exactly what it would be in training.

## Ablation and comparison switches

Each is a switch on built batches; none needs a new fetch. Degree columns are counted after the
switch.

- **Scored account alone:**
  - keeps s's row and s's own payment edges (`pays` and `rev_pays`), drops every other edge and
    zeroes every other node's features;
  - zeroes the fields measured at the other end: `dwell_tracked`, `left_24h`, `dwell` and
    `receiver_gap` on s's outflows, `sender_gap` on s's inflows; keeps `amount`, `rail`, `age`
    and the pair fields;
  - turns the community block off.
- **Without devices and IPs:**
  - drops device and IP nodes and the six enrolment, owner-enrolment and seen relations with
    their reverses, so co-users' accounts reached only through them drop out;
  - zeroes columns 2 to 8 on every customer node, leaving `accounts_owned`;
  - turns the community block off.
- **Without community:** the head is Linear(64, 32).
- **Without per-payment timing:** zeroes every payment-edge duration and its bit (`age`,
  `pair_gap`, `pair_age`, `sender_gap`, `receiver_gap`, `dwell`), and `dwell_tracked` and
  `left_24h`; keeps `amount`, `rail` and `pair_prior_count`.

**XGBoost inputs.**

- **Own behaviour:** the scored account's own row as stored, columns 1 to 46 with each
  duration as presence and seconds: 50 numbers.
- **Graph features:** the same, plus its customer's columns 1 to 8, the mean and the maximum
  of each column of the device rows of D_s and of the IP rows of D_s (zeros when there are
  none), and the community block: 50 + 8 + 18 + 18 + 8 = 102 numbers.

## Coverage of the owner's proposed features

| Proposed | Where |
|---|---|
| transaction velocity and frequency | B1 `left_*`, `median_dwell`; B3 counts and bursts |
| time between transactions | `shortest_in_gap`; edge `pair_gap`, `sender_gap`, `receiver_gap` |
| new counterparties | B2 new payers (internal and external) and payees, `new_payer_share_30d`, `max_new_payers_24h`; `pair_gap` absent (first in pair) |
| relationship age | edge `pair_age`; enrolment `since_start` |
| changes in graph connectivity | `new_devices_30d`, `new_ips_30d`, `new_co_users_30d`, device and IP `new_customers_*`, `new_link_share_30d` |
| rapid movement of funds | `left_15m` to `left_1d`; per-payment `dwell`, `left_24h` |
| formation of mule networks; clustered connections with isolated communities | the community block; `co_users_both`; co-users' accounts at two hops |
| money in against money out, encoded as time | B1's dwell curve, `median_dwell`, per-payment `dwell` |
| device sharing | customer co-user counts, device and IP node rows, enrolment and seen edges, owner-enrolment edges |

## Deliberately not a feature

- **Labels and the answer key:** `pu_label`, `is_mule`, `mule_ring_id`, `mule_onset_ms`,
  `mule_last_ms`. They make labels, weights and evaluation only.
- **Anything about known or caught neighbours:** no label inputs, no rule ranker, no trees
  with links.
- **Any evaluation output,** time to first flag included.
- **`account_type`:** it chooses the population and the deposit-only co-user accounts; the rail
  on each payment says what kind of payment it was.
- **Dates of record and ages:** `opened_ms`, `joined_ms`, `closed_ms` as values or orders;
  account and customer age; the start of any link that began at sign-up or before the
  lookback; a device's, IP's or token's first sighting before the lookback. PhantomLedger
  clamps them, and the owner ruled first-seen time out.
- **Link ends:** `active`, `since_end`, `held`, and any count or order of links active at t.
- **Closure:** a closed flag, open and closed account counts, the ownership end as a value.
- **Clock:** hour, weekday, day of month, second of the minute, the Monday's index, days since
  the data start, the world or its role. A keyed per-date offset of t was considered and
  rejected: a fixed-clock salary reveals it, and it breaks the Monday 00:00 rule.
- **Payment id order:** used only to break ties at subgraph caps, and only once ids are
  pseudonyms.
- **Token and its links**, `token_present`, `token_age`: each token copies one account, and
  Zelle use shows as the rail.
- **Address**, until households exist.
- **Ids** of accounts, customers, devices, IPs and payments: used only to join rows and break
  ties.
- **365-day windows:** the lookback is 90 days.
- **`next_outflow`, `passed_on_1h`, `passed_on_24h`:** replaced by B1's matched dwell.
- **Global community algorithms:** they have no as-of time.
- **Self-payments.**
- **The behaviour of a hub that is not the scored account:** flag-only.
- **The customer's 7-day new device and IP counts, and payments over 7 days on device and IP
  rows:** finer versions exist.

## Widths

| Vector | Encoded width | Stored before encoding |
|---|---|---|
| account node | 309 | 50, plus 7 degrees counted at build |
| customer node | 11 | 8, plus 3 degrees |
| device node | 12 | 9, plus 3 degrees |
| IP node | 12 | 9, plus 3 degrees |
| `pays`, `rev_pays` edge | 401 each | 23 |
| `owned_by`, `rev_owned_by` edge | none | none |
| `enrolled_on_device`, `enrolled_on_ip` and reverses | 130 each | 4 |
| `owner_enrolled_on_device`, `owner_enrolled_on_ip` and reverses | 130 each | 4 |
| `seen_on_device`, `seen_on_ip` and reverses | 131 each | 5 |
| community block | 8 | 8 |
| head input | 64 + 8 = 72 | |

## Generator gates each feature depends on

A reported run refuses a world unless every entry below that reaches a feature has passed its
gate on that world or on its gate world (see the design's [Gates](design.md#gates)). Entry ids
are those of the shortcut dossier, `generator_shortcuts_2026-10-17.md`.

**Timing and clocks** (every one reaches the durations measured to t through the clock):

| Entry | Cue | Features it reaches |
|---|---|---|
| S-TIM-1 | victim, entry, organizer, camouflage and deferred rows at second 0 | every duration measured to t; `dwell`, `median_dwell`, gaps, `left_15m` (whole minutes) |
| S-TIM-2 | a debit keeps its credit's second; exact 24-hour ATM repeats; sibling rows share a second | `dwell`, `median_dwell`, `left_15m` to `left_1d`, `sender_gap` of exactly 86,400 s, bursts, B1 lots |
| S-TIM-3 | victim session clock unlike the victim's own | `age` and `receiver_gap` of victim credits, `burst_in_*`, `max_new_payers_24h` |
| S-TIM-4 | self-operated mules all wake at 07:00 | `age`, `since_last_out`, `dwell`, `left_*` |
| S-TIM-5 | organizer payments at any hour; invoice weekly lattice | `age`, `pair_gap`, `sender_gap`, `receiver_gap` |
| S-TIM-6 | night ATM withdrawals only by mules | `age` and `since_last_out` with rail cash, `left_by_cash` |
| S-TIM-7 | pure-mule onsets on a shared 30-day lattice | `age`, `pair_age`, `since_first_use`, `since_start`, community timing |
| S-TIM-8 | batch-mates first credited the same day, then paying each other | pair fields between members, `new_payer_share_30d`, `internal_amount_30d`, `isolation` |
| S-TIM-9 | ATO and impostor rail on a clock of its own | `age`, `receiver_gap` of pushes into unlabelled payees |
| S-TIM-10 | hour and weekday of mule cash-outs unmeasured | `age`, `since_last_out` of outflows |

**Amounts:**

| Entry | Cue | Features it reaches |
|---|---|---|
| S-AMT-1 | slow-type victims pay the type mean to the cent | `amount`, `in_amount_*` |
| S-AMT-2 | victims pay business-pooled means | `amount`, `in_amount_*`, piece sizes |
| S-AMT-3 | transfer pieces of one size (about USD 662) | `amount` on outflows, `out_amount_*`, out counts, `burst_out_*` |
| S-AMT-4 | card pieces never above about USD 201, 59% remainders under USD 67 | `amount` with rail card (kept on edges to flag-only merchants), `left_by_card` |
| S-AMT-5 | round transfer amounts only from mules | `amount` |
| S-AMT-6 | structuring amounts not price-scaled (real behaviour, no fix) | `amount` on cash; no gate |
| S-AMT-7 | structuring above the reporting threshold, cents on cash | `amount` with rail cash, `in_amount_*` |
| S-AMT-8 | every mule keeps 5% to 10% of every credit | `held_share`, `left_7d`, `left_24h`, `dwell`, `median_dwell` |
| S-AMT-9 | other playbooks' victims pay one lognormal, once | `amount`, `pair_prior_count`, `new_payer_share_30d` |
| S-AMT-10 | no chunking under alert thresholds (declared) | `amount` |
| S-AMT-11 | legitimate P2P amounts too small | `amount`, `in_amount_*`, `out_amount_*` |

**Rails and channels:**

| Entry | Cue | Features it reaches |
|---|---|---|
| S-RAIL-1 | card cash-out on rail unknown | `rail`, `left_by_*` |
| S-RAIL-2 | own-account move on rail unknown, and its residual: an internal same-owner move minutes after a non-salary credit is mule-only | `rail`, ownership edges, `dwell`, `left_24h`, `left_15m`, `left_by_transfer` |
| S-RAIL-3 | structuring as customer-to-customer cash | `rail` cash between two customer accounts, `amount` |
| S-RAIL-4 | accounts paying themselves | none, since no query reads a self-payment; the loader counts them |
| S-RAIL-5 | ATM amounts off the ordinary menu and exact daily limits | `amount` with rail cash, `sender_gap` of exactly a day, `out_amount_*` |
| S-RAIL-6 | legitimate cash and internal rails never on the 29th to 31st | `age` of cash and internal edges, once a monthly payment fixes the day of month |
| S-RAIL-7 | ATM and P2P from a non-primary account only by mules | `rail` on other accounts of a customer, their B1 to B3 |
| S-RAIL-8 | devices and IPs on cash rows | seen edges, `accounts_seen_30d`, `payments_seen_30d` |
| S-RAIL-9 | victims never pay by check or wire | `rail` of credits |
| S-RAIL-10 | mule money never reaches crypto or other banks | `is_external` of payees, `new_payees_*`, `top_payee_share_30d` |

**Devices, IPs and identity:**

| Entry | Cue | Features it reaches |
|---|---|---|
| S-ID-1 | household device and IP groups are random strangers | `co_users`, `co_users_both`, `new_co_users_30d`, community block |
| S-ID-2 | fraud-flagged persons left out of household sharing | `co_users` and community of organizers and collectors |
| S-ID-3 | ring tenures start before planned use; tenures from planned payments | `new_devices_30d`, `new_ips_30d`, `since_start`, `new_customers_*`, `new_link_share_30d`, early warning, and the No link ends guard |
| S-ID-4 | ring sharing independent of who operates the account | seen edges, `observed_only`, `co_users` |
| S-ID-5 | ring endpoints carry the holder's ordinary rows | `uses_90d`, `accounts_seen_30d`, `payments_seen_30d` |
| S-ID-6 | cross-household sharing ring-only | `co_users`, `co_users_both`, community block |
| S-ID-7 | one ring IP that never renumbers | IP `since_start`, `customers_90d`, `new_customers_*`, `since_first_use` |
| S-ID-8 | ordinary device and IP links never short | `since_start`, `new_devices_30d`, `new_ips_30d`, `new_customers_*` |
| S-ID-9 | only some endpoints get usage edges (settled: all do) | `observed_only`, `crowd`, `customers_90d` |

**Ring membership and roles:**

| Entry | Cue | Features it reaches |
|---|---|---|
| S-MEM-1 | camouflage cover traffic | `age`, `payees_30d`, `new_payees_*`, `hub_payee_share_30d`, `new_external_payers_*` (a second payroll) |
| S-MEM-2 | organizers outnumber mules | labels; community `members` |
| S-MEM-3 | organizer accounts are money sinks | `top_payee_share_30d`, collectors' `payers_30d` and hub status |
| S-MEM-4 | organizers labelled from their own money | labels |
| S-MEM-5 | ATO drops and impostor payees labelled 0 | labels; their B1 to B3 as unlabelled look-alikes |
| S-MEM-6 | ring members drawn uniformly | `co_users`, community `density`, `isolation`, `internal_amount_30d` |
| S-MEM-7 | victims drawn uniformly, type after the victim | new internal and external payers, payers' rows |
| S-MEM-8 | victim lists run out | in counts, `flow_present`, `held_share` |
| S-MEM-9 | funnel relays unpinned | `top_payee_share_30d` |

**Lifecycle and identification:**

| Entry | Cue | Features it reaches |
|---|---|---|
| S-LIFE-1 | no history before the window | Known starts, `mule prepare`'s lookback check |
| S-LIFE-2 | laundering-free warm-up (settled: a 365-day burn-in) | none beyond the burn-in check |
| S-LIFE-3 | ring activity anchored to the window's edges | `age`, `pair_age`, positives per Monday |
| S-LIFE-4 | no purpose-opened mule accounts, and short ordinary dormancy (7 to 45 days) | `accounts_owned`, newness from counts (`in_90d` against `in_30d`), new payers, `pair_age`, `since_first_use` |
| S-LIFE-5 | closure only at death, almost no arrivals | `accounts_owned`, the population, the newness lift |
| S-LIFE-6 | ring rows outside a member's membership dropped | labels, joiner mules' counts |
| S-LIFE-7 | positional ring budget cut | `held_share` 1 on received-only mules, `flow_present`, `left_*`, empty tenures |
| S-LIFE-8 | a ring stops at its first member's death | `since_last_in`, `since_last_out` jumping together across members, `internal_amount_30d` |
| S-LIFE-9 | rings never form or end | `new_link_share_30d`, `since_start` of ring links |
| S-IDN-1 to S-IDN-5 | label completeness, dates, range, roles, masking | labels and evaluation only; S-IDN-2 settled (identification closes nothing) |

**Shortcuts noted but not yet researched**, which reach features:

| Residual | Features it reaches |
|---|---|
| the burst model runs only in pure-mule rings | counts, bursts, `since_last_*` |
| a declined ring row may post again at a different second | `dwell`, `age` |
| mule own-account moves sized like onward pieces (about USD 662) against LN(USD 120, 0.75) for ordinary self-transfers | `amount` on internal same-owner edges |
| multi-ring reuse drawn uniformly | community block |
| legitimate hour profiles uncited | every duration measured to t |
| the consumer cash-deposit cap | `amount` with rail cash |
| no legitimate second jobs | `new_external_payers_*` |
| the attacker clock of ATO drains | `age` |

**Gates MulePatternLearner adds**, each a check on every world with its ceiling declared
before the run. The ones that read `is_mule` are evaluation code, run by `mule audit`:

| Gate | Why |
|---|---|
| no ring tenure has a quiet stretch longer than 21 days, and tenures are built from attempted payments | keeps the visible and current tests decided by the past |
| links made at sign-up start exactly at `joined_ms` (export contract and loader check) | otherwise every joiner's first device reads new |
| no exported payment comes before its accounts open | every account node exists at t |
| payment ids are pseudonyms whose order does not follow event time | subgraph cap ties are class-free |
| no payment between two external accounts | a bank cannot see them |
| the `is_mule` lift of "no payment in the first 60 days of the lookback, some after" among scored account-Mondays stays under its ceiling, and the share of customers new to the bank within a year is at least 0.033 | account newness rebuilt from behaviour must be as strong as in reality, not more |
| the `is_mule` lift of "an internal move to an account of the same owner within 1 hour of a non-payroll credit", and the distance between mule and ordinary same-owner move amounts, stay under their ceilings | the same-owner move residual and own-account move sizes |
| the `is_mule` lift of "a link to a node shared with another customer, started in the last 14 days" stays under its ceiling | every ring tenure is followed by ring money in PhantomLedger |
| the share of victims at the same bank is pinned in the dossier | `is_external` and the internal or external split of new payers |
| the clock probe | durations measured to t |
| on development worlds only: no positive is an account hub, and the share of ring devices and IPs that are leaves on their ring's active Mondays is near 0 | hub handling and the crowd threshold must not hide mule evidence |
| on development worlds only: busy ordinary accounts' card spending does not fill `left_15m` to `left_1d` | else the dwell shares count transfer and cash outflows only |
