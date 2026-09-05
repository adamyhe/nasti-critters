#!/usr/bin/env python3
"""Enumerate BPNet attribution jobs for all experiments.

A thin wrapper over src/launcher.py, like the fit launchers. The job unit is
(experiment x attribute type) rather than (experiment x fold): attribute.py
loops every fold internally and averages, so one job covers all folds.

    python src/bpnet/attribute/launch.py --dry-run
    python src/bpnet/attribute/launch.py --attribute-type profile --attribute-type counts
    python src/bpnet/attribute/launch.py --print-commands | bash
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from launcher import run_attribute  # noqa: E402

if __name__ == "__main__":
    run_attribute("bpnet", REPO_ROOT / "src" / "bpnet" / "attribute" / "attribute.py")
