#!/usr/bin/env python3
"""Enumerate Cherimoya training jobs for all experiments and folds.

A thin wrapper over src/launcher.py, which holds the selection rule shared with
src/bpnet/fit/launch.py. See that module for the three emission modes
(--print-commands, --dry-run, submit) and why there is one implementation
rather than two.

Cherimoya is NOT restricted to one experiment or one species: fit_cherimoya.py
takes -e and resolves paths, species and folds through src/experiments.py
exactly as fit_bpnet.py does, and config/cherimoya_params.json holds no
species-specific value. This launcher existing is what makes that usable at
corpus scale -- before it, the only cherimoya entry points were a single-job
hand-written slurm.sh (since deleted) and a bare fit invocation, which made the
family look dm-only.

Per src/cherimoya/README.md the models are not deployment-ready; this is the
training workflow, not a deployment pipeline.

    python src/cherimoya/fit/launch.py --dry-run
    python src/cherimoya/fit/launch.py -e D.melanogaster-S2_PROcap
    python src/cherimoya/fit/launch.py --print-commands | bash
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from launcher import run  # noqa: E402

if __name__ == "__main__":
    # No supports_controls: fit_cherimoya.py has no --controls flag and
    # config/cherimoya_params.json pins "controls": null.
    run("cherimoya", REPO_ROOT / "src" / "cherimoya" / "fit" / "fit_cherimoya.py")
