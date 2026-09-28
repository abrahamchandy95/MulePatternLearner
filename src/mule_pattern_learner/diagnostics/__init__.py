"""Analyses of a dataset and its runs against the ground truth; training never imports them.

They read the graph only through ports and truth only through evaluation.truth, and
report tables in long format (split, subset, metric, value). `mule diagnose` runs them
(the diagnostics step adds the command and the other analyses).
"""
