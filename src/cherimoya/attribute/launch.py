#!/usr/bin/env python3
"""Enumerate Cherimoya attribution jobs for all experiments.

A thin wrapper over src/launcher.py, identical in shape to the BPNet one: the
job unit is (experiment x attribute type), and attribute.py loops every fold
internally. Needs cherimoya >= 0.3.1, so point APPTAINER_IMAGE at the
cherimoya-0.3.1 image (not the 0.2.1 image used for training).

    python src/cherimoya/attribute/launch.py --dry-run
    python src/cherimoya/attribute/launch.py --attribute-type profile --attribute-type counts
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from launcher import run_attribute  # noqa: E402

if __name__ == "__main__":
    run_attribute("cherimoya", REPO_ROOT / "src" / "cherimoya" / "attribute" / "attribute.py")
