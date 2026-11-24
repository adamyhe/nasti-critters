#!/usr/bin/env python3
"""Enumerate `modisco report` jobs for Cherimoya attributions.

A thin wrapper over src/launcher.py, identical to the BPNet one but for the
family. CPU-only, and needs no cherimoya at all -- it reads the attribution and
one-hot npz files and shells out to the modisco CLI -- so it runs from the
torch-free venv; no APPTAINER_IMAGE needed.

    python src/cherimoya/modisco/launch_report.py --dry-run
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from launcher import run_modisco_report  # noqa: E402

if __name__ == "__main__":
    run_modisco_report("cherimoya")
