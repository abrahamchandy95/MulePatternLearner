"""What every test shares: one OpenMP thread on macOS.

On macOS the pip wheels of torch and XGBoost each bring an OpenMP runtime of their own (torch
its bundled libomp, XGBoost Homebrew's), and XGBoost crashes the process when it runs beside
torch's runtime with more than one thread. The comparisons' tests train XGBoost in the process
that runs the GNN's tests, so on macOS every test runs with one OpenMP thread, set here before
any test imports torch; comparisons.rankers refuses to run on macOS otherwise.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
