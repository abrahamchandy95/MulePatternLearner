# Score new accounts

Score any accounts with the built-in run's model, including accounts that never appeared
in training: the model has no per-account parameters, so it scores any account from its
history and its counterparties' at a date.

## Score a file of accounts

Write the account ids to a file, one per line, and give a date (an ISO date; history is
what happened before its midnight UTC). Without a date the model's test cutoff is used.

```bash
mule score new_accounts.txt 2025-02-01
```

The scores go to `results/baseline/seed-42/scores/new_accounts_2025-02-01.parquet`, and
the ids TigerGraph rejected, if any, to `scores/new_accounts_2025-02-01_rejected.txt`
beside them. The command refuses to overwrite either file, so move them aside to score
the same file and date again. It ends with how many accounts it scored and where, how many
TigerGraph rejected by status, and how many child contexts it left out. Its `score`
event in the run's `events.jsonl` has the whole result: `rejected` (roots not scored),
`rejected_roots_by_status`, `rejected_children` (child contexts masked out),
`rejected_children_by_status`, `stub_children` and `rejection_events_by_status`.

## What the scores mean

- **History as an operational scorer sees it.** The command reads the history visible
  before the date, without the experiment scope, and computes the hub registry for that
  cutoff. It needs neither the training dataset nor any label. A date before the graph's
  first visible event is refused.
- **Rejected accounts are not scored.** An id that is missing, not yet visible at the date
  or over the history cap goes to the rejected file instead.
- **Scores rank accounts.** Under the balanced nnPU weight they are not probabilities, and
  only the model's threshold, chosen on validation, gives them a cut-off. Scores are
  float64, so the highest-scored accounts do not tie.
- **Quality on new accounts is unmeasured.** The audit measures withheld existing
  ownership groups. An account with no history is scored from its metadata alone, and how
  well that works needs its own chronological sample
  ([Leakage and scaling](../explanation/leakage-and-scaling.md#three-evaluation-protocols)).

The graph need not be the training dataset's frozen source, so scoring has no context
cache and requests every context from TigerGraph, with the model's retry budgets. The
installed queries must still be the repository's.

## Score with another run's model

`mule score` uses `results/baseline/seed-42/model.pt`. Another run's model scores from
Python:

```python
from pathlib import Path

from mule_pattern_learner.paths import RunPaths
from mule_pattern_learner.pipeline.score import score_accounts

score_accounts(RunPaths.of("no_slot_sum", 43), Path("new_accounts.txt"), "2025-02-01")
```
