#!/usr/bin/env python3
"""Enumerate GC-matched negative jobs, one per experiment.

A thin wrapper over src/launcher.py, like every other launcher. Lives beside
make_negatives.py rather than under a model family because negatives are shared
by both families and produced outside the Snakemake DAG.

    python src/launch_negatives.py --dry-run
    python src/launch_negatives.py --print-commands | bash
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from launcher import run_negatives  # noqa: E402

if __name__ == "__main__":
    run_negatives(REPO_ROOT / "src" / "make_negatives.py")
