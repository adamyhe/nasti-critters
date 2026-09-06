#!/usr/bin/env python3
"""Enumerate BPNet benchmark jobs for all experiments.

A thin wrapper over src/launcher.py, like the fit and attribution launchers.
The job unit is the EXPERIMENT: benchmark_predictions.py scores every fold's
model on its own held-out loci in one run, and pools their predictions for the
genome-wide block, which could not be computed if the folds were split across
jobs.

    python src/bpnet/benchmark/launch.py --dry-run
    python src/bpnet/benchmark/launch.py -e D.melanogaster-S2_PROcap
    python src/bpnet/benchmark/launch.py --print-commands | bash
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from launcher import run_benchmark  # noqa: E402

if __name__ == "__main__":
    run_benchmark(
        "bpnet",
        REPO_ROOT / "src" / "bpnet" / "benchmark" / "benchmark_predictions.py",
    )
