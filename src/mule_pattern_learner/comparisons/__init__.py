"""The two XGBoost comparisons (docs/design.md, "Comparisons"): XGBoost on the scored account's
own row, and on its row with its customer's, its devices' and IPs' and its community block,
read from the same built batches as the GNN and trained on the same examples, weights and
labels, their round count chosen by training's selection score.

Every XGBoost fit and prediction, these two and the clock probe's, runs in a Python process of
its own that never imports torch (boosting is the process's code, process starts it, and the
parent holds a booster as its saved bytes), because the OpenMP runtimes of torch and XGBoost
crash a process that holds both.

It never imports evaluation or testing, as the import contracts of pyproject.toml check."""
