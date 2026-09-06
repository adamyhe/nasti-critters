#!/usr/bin/env python3
"""Enumerate `modisco motifs` jobs for BPNet attributions.

A thin wrapper over src/launcher.py, like the other launchers. One job per
(experiment x attribute type), consuming that run's attribution npz plus the
experiment's one-hot npz -- tfmodisco needs both, which is why
filter_nonACGT_regions.py writes the OHE.

Pipeline order: launch_filter.py -> attribute/launch.py -> here ->
launch_report.py.

    python src/bpnet/modisco/launch.py --dry-run
    python src/bpnet/modisco/launch.py --attribute-type profile --attribute-type counts
    python src/bpnet/modisco/launch.py --time 2-00:00:00 --mem 64G --cpus-per-task 32
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from launcher import run_modisco  # noqa: E402

if __name__ == "__main__":
    run_modisco("bpnet")
