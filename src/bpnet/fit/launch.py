#!/usr/bin/env python3
"""Enumerate BPNet training jobs for all experiments and folds.

A thin wrapper over src/launcher.py, which holds the selection rule shared with
src/cherimoya/fit/launch.py. See that module for the three emission modes
(--print-commands, --dry-run, submit) and why there is one implementation
rather than two.

    python src/bpnet/fit/launch.py --dry-run
    python src/bpnet/fit/launch.py --time 12:00:00 --mem 32G
    python src/bpnet/fit/launch.py --print-commands | bash
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from launcher import run  # noqa: E402

if __name__ == "__main__":
    # BPNet takes bias-control tracks; cherimoya does not.
    run("bpnet", REPO_ROOT / "src" / "bpnet" / "fit" / "fit_bpnet.py",
        supports_controls=True)
