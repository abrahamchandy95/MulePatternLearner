# Future design work

Ideas the owner has raised for a later version of the design. None of them is decided:
[the design](design.md) is what the code builds, and it stays so until the owner agrees to
a change and the design is updated. When one of these ideas bears on the work in hand, say
so before building anything. When one is agreed, move it into the design and delete it
here.

The ideas listed here on 2026-10-16 (labels from mule activity rather than from the catch,
splitting by account, no first-seen time as a feature, community features) were settled in
the grilling session of 2026-10-17 and are now in the design. What stays open is below.

## HIGH PRIORITY: the device and IP of each payment

Raised by the owner on 2026-11-01, to come first after the presentation of 2026-11-02 (no time
for a refactor before it):

> device and ip sharing should be huge signal for mule behavior! ... just keep this on HIGH
> PRIORITY.

In TigerGraph a payment is a vertex linked to its sender, its receiver, and the device and IP
seen on it. The model loses that link: the batch builder turns each payment into one link
from sender to receiver, and read_shared_identity summarizes device and IP use per account
(uses, first and last use), so the model sees which customers share a device or IP, but not
which device or IP made which payment.

Options to weigh:

- Keep payments as nodes in the model, linked to sender, receiver, device and IP, as in the
  schema. Two layers then reach only the counterparties, so matching today's reach takes
  four.
- Keep payments as links and put the device and IP of each payment on it.
- Check the generated worlds first: in them a ring device reaches only about 18% of mules, and
  the device and IP features separate weakly, so the model cannot learn a signal the data does
  not carry.

## Guilt by association, as its own experiment

Deferred by the owner on 2026-10-17, when the design dropped every input about known or
caught neighbours:

> correct, because we don't want the knowledge of caught mules to affect some scores.

The design learns behaviour only: no feature says whether a neighbour was identified, and
there is no rule ranker and no trees-with-links control. Guilt by association (scoring an
account higher because it pays, or shares a device or IP with, an account the bank has
already caught) can return later as an experiment of its own, compared with the behavioural
model rather than mixed into it.

Starting points: the label-input variant of the October 17 revision (commit ed62329,
`docs/design.md`, "Label-input variant") and its model code at fc63ad9
(`src/mule_pattern_learner/model/hetero.py`): a confirmed flag and the time since
confirmation on each account node, confirmations masked at random during training, the
scored account's own customer always read as unknown, and a rule ranker and trees with link
counts as controls.

Open questions:

- When a mule became known. The design reads no identification date, and PhantomLedger writes
  none. A confirmation can only count before t, so the experiment needs an identification
  time per identified mule, which is a new export column and a change to the contract.
- Whether identification stays free of device and IP sharing. The design rules it out so that
  labels do not lean on the evidence the model is tested on; an experiment on guilt by
  association would have to keep that rule, or measure what breaking it inflates.
- What it is compared with: the behavioural GNN, the GNN with label inputs, a rule ranker,
  and XGBoost with link counts, on the same worlds.

## Zelle token re-registration

Raised by Claude on 2026-10-17, not yet discussed as a design change. The owner asked how
Zelle uses tokens:

> I am not sure how fraudsters move a single token between multiple transactions, i thougth
> tokens were completely handled by zelle. it was my assumption that we get one token per
> transaction.

A Zelle token is not per transaction. It is the recipient's email address or US mobile
number, enrolled once and linked to one bank account at a time; Early Warning's directory
resolves it to the deposit account, and the money moves between the two deposit accounts.
Moving a token means re-enrolling the same number or address at another account or bank, so
that later payments land there. The CFPB and New York Attorney General complaints against
Early Warning Services describe fraudsters doing this (pointing one number at different mule
accounts, a new token after a report, takeovers of a victim's token), but no source measures
how often mules move tokens compared with ordinary users, who also move them when they change
banks.

Today PhantomLedger gives each enrolled account one token that never moves, so the token
copies its account and the design reads none; Token stays in the export because it is how
Zelle addresses money. To make re-registration useful later:

- PhantomLedger would need several tokens per account, ordinary token moves at a declared
  rate (otherwise only rings ever move tokens, a shortcut), declared ring behaviours, and,
  for external counterparties, the token rather than the account as what the bank sees.
- MulePatternLearner would then read Token like Device and IP: known starts within the
  lookback, sharing in sequence, no first-seen age.

## Address, once households exist

Address and its links left the export: rings share none, and PhantomLedger's address tokens
collide between strangers by chance. Once PhantomLedger's households step gives one address
per dwelling, Address could return as a sharing signal beside devices and IPs.

Open questions:

- What it adds over device and IP sharing, measured as an ablation, never assumed.
- Whether it helps tell a household from a ring, which `co_users_both` must do once
  households share devices and IPs too.
- Its rules would follow devices and IPs: known starts only, no first-seen age, crowds
  (apartment blocks, shelters, mail drops) as leaves.

## Whether a mule label belongs to the account or the customer

Raised by the owner on 2026-10-16:

> i think a caught mule must be labeled MULE permanently (maybe write a note to research this
> later)

Part of this is settled in the design: an identified mule is positive on the Mondays it is
active and dropped on every other Monday, never drawn as unlabelled, and identification closes
nothing, so a caught mule never disappears from the world. What stays open is who carries the
label.

How the design treats it today:

- The label sits on the account that carried the money (PhantomLedger labels by the account's
  ring payments, not its holder's other accounts), and the scored unit is an account: the
  owner, on 2026-10-17, "we want to detect mule accounts. not customers."
- PhantomLedger's identification identifies every mule account of a customer together, but a
  customer's other, non-mule accounts stay unlabelled.

What research found so far: a Cifas marker is recorded against the customer and kept six years
from filing; UK firms offboarded 194,084 mules from January 2022 to September 2023 and filed 37%
of them; FinCEN says no Bank Secrecy Act rule requires closing an account after a
suspicious activity report; GOTCHA decays a caught fraudster's weight rather than relabelling
it.

Open questions:

- Whether a customer-level marker should exist at all, and if so whether it is a label (every
  account of a caught customer positive) or an input (guilt by association, above).
- How it fits with labels that follow activity: a customer's other accounts may never have
  moved ring money.
- What evaluation does with a caught customer's other accounts.

## Daily evaluation, after the benchmark

Settled on 2026-10-27 and now in the design: each scored week is scored at one time inside it,
drawn from the world's seed, the queries accept any whole second, and training stays one time
per week. Production can therefore score daily. What stays open is evaluating the test world
daily. A burst between two scoring times is flagged at best at the next one, up to almost 14
days after it ends; scoring every day would catch it during the burst, at about seven times
the cost: the whole-world test at 300,000 people is already about 23 million account-weeks.
Decide once `mule bench` shows the cost.

Open questions:

- Whether daily evaluation is affordable for the test world, or only for a sample.
- How the label rule, early warning and time to first flag read when the test world is scored
  every day rather than once a week.
