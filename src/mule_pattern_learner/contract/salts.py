"""The salts of the seeded draws, frozen.

Each salt feeds the hash behind a draw, so its value never changes: another value
would select other accounts or train on other steps. The values keep the words they
had before the layered restructure (tests/test_naming.py allows them).
"""

# The ranks of the seed reservoirs (data.accounts): which accounts a dataset selects.
RESERVOIR_SALT = "marginal_cohort"
# The seed of each training step (training.schedule.step_seed).
STEP_SALT = "temporal_live_step"
