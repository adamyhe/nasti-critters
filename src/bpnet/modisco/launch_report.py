#!/usr/bin/env python3
"""Enumerate `modisco report` jobs for completed BPNet modisco runs.

Run after launch.py. The MEME motif database is resolved PER SPECIES from
config/genomes.yaml's `jaspar_collection`, not hardcoded to vertebrates the way
procap-atlas can afford to be -- see experiments.motif_db_path().

    python src/bpnet/modisco/launch_report.py --dry-run
    python src/bpnet/modisco/launch_report.py --print-commands | bash
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from launcher import run_modisco_report  # noqa: E402

if __name__ == "__main__":
    run_modisco_report("bpnet")
