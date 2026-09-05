#!/usr/bin/env python3
"""Enumerate non-ACGT locus-filtering jobs, one per experiment.

A thin wrapper over src/launcher.py, like the fit and attribution launchers.
Separate from launch.py for two reasons: the job unit is one per experiment
rather than one per attribute type, and these jobs are CPU-only, so they request
no GPU and should not queue behind training.

The step is optional -- attribute.py reads the experiment's peaks unless pointed
at the filtered BED with --loci.

    python src/bpnet/attribute/launch_filter.py --dry-run
    python src/bpnet/attribute/launch_filter.py --print-commands | bash
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from launcher import run_filter  # noqa: E402

if __name__ == "__main__":
    run_filter(
        "bpnet",
        REPO_ROOT / "src" / "bpnet" / "attribute" / "filter_nonACGT_regions.py",
    )
