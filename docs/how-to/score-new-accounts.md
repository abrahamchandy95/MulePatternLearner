# Score new accounts

The built-in run's model scores any account, including ones never seen in training: it has
no per-account parameters, so it scores an account from its history and its
counterparties' at a date ([mule score](../reference/cli.md#mule-score-accounts-date)).

## Score a file of accounts

Put the account ids in a file, one per line, and give an ISO date (history is what
happened before its midnight UTC; the default is the model's test cutoff):

```bash
mule score new_accounts.txt 2025-02-01
```

Scores go to `results/baseline/seed-42/scores/new_accounts_2025-02-01.parquet`, rejected
ids to `scores/new_accounts_2025-02-01_rejected.txt` beside it; to score the same file and
date again, move both aside. The `score` event in the run's `events.jsonl` holds the whole
result: `rejected` (roots not scored), `rejected_roots_by_status`, `rejected_children`
(child contexts masked out), `rejected_children_by_status`, `stub_children` and
`rejection_events_by_status`.

## What the scores mean

- **History as an operational scorer sees it:** no experiment scope, training dataset or
  label; the hub registry is computed for the date.
- **Rejected accounts are not scored:** an id missing, not yet visible at the date or over
  the history cap goes to the rejected file.
- **Scores rank accounts.** Under the balanced nnPU weight they are not probabilities;
  only the model's validation-chosen threshold gives a cut-off. They are float64, so the
  top accounts do not tie.
- **Quality on new accounts is unmeasured.** The audit measures withheld existing
  ownership groups; an account with no history is scored from its metadata alone, which
  needs its own chronological sample to measure ([Leakage and
  scaling](../explanation/leakage-and-scaling.md#three-evaluation-protocols)).
- **No context cache:** the graph need not be the dataset's frozen source, so every
  context is requested from TigerGraph, with the model's retry budgets.

## Score with another run's model

`mule score` uses `results/baseline/seed-42/model.pt`; another run's model scores from
Python:

```python
from pathlib import Path

from mule_pattern_learner.paths import RunPaths
from mule_pattern_learner.pipeline.score import score_accounts

score_accounts(RunPaths.of("no_slot_sum", 43), Path("new_accounts.txt"), "2025-02-01")
```
