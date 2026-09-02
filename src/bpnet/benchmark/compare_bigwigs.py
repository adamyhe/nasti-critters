"""
Compare base-resolution signal tracks over configured loci and folds.

Loci, sequences and folds resolve through src/experiments.py from an experiment
ID; --signals-a/--signals-b supply the two track sets being compared (defaulting
to the experiment's own signals).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

sys.path.insert(0, str(REPO_ROOT / "src"))
from experiments import IGNORE, Experiment, load_params  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e", "--experiment", type=str, required=True,
        help="experiment ID as it appears in config/experiment_config.yaml",
    )
    parser.add_argument("--signals-a", nargs="+", default=None)
    parser.add_argument("--signals-b", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    try:
        exp = Experiment.load(args.experiment)
    except (KeyError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    if exp.missing:
        for m in exp.missing:
            print(f"Error: missing {m}", file=sys.stderr)
        sys.exit(1)

    params = load_params("bpnet")
    signals = [str(x) for x in exp.signals]
    params.update({
        "loci": str(exp.peaks),
        "sequences": str(exp.sequences),
        "signals_a": args.signals_a or signals,
        "signals_b": args.signals_b or signals,
        "output": str(Path(args.output).resolve()) if args.output else None,
    })
    # Compare across every fold's held-out chromosomes.
    chroms = [c for f in exp.all_folds("bpnet") for c in f["test_chroms"]]

    import torch
    from bpnetlite.performance import (
        jensen_shannon_distance,
        pearson_corr,
        spearman_corr,
    )
    from tangermeme.io import extract_loci

    _, a, b = extract_loci(
        loci=params["loci"],
        sequences=params["sequences"],
        chroms=chroms,
        signals=params["signals_a"],
        in_signals=params["signals_b"],
        in_window=params["in_window"],
        out_window=params["out_window"],
        verbose=params["verbose"],
        ignore=IGNORE,
    )
    a = torch.abs(a)
    b = torch.abs(b)

    profile_corr = pearson_corr(b.reshape(b.shape[0], -1), a.reshape(a.shape[0], -1))
    profile_jsd = jensen_shannon_distance(
        torch.nn.functional.log_softmax(b.reshape(b.shape[0], -1), dim=-1),
        a.reshape(a.shape[0], -1),
    )
    counts_pearson = pearson_corr(b.sum(dim=(-1, -2)), a.sum(dim=(-1, -2)))
    log_counts_pearson = pearson_corr(
        torch.log1p(b.sum(dim=(-1, -2))), torch.log1p(a.sum(dim=(-1, -2)))
    )
    counts_spearman = spearman_corr(b.sum(dim=(-1, -2)), a.sum(dim=(-1, -2)))

    print(f"Profile Pearson correlation median: {profile_corr.median()}")
    print(f"Profile JSD median: {profile_jsd.median()}")
    print(f"Counts Pearson correlation {counts_pearson}")
    print(f"Log counts Pearson correlation: {log_counts_pearson}")
    print(f"Counts Spearman correlation: {counts_spearman}")

    if params["output"] is not None:
        Path(params["output"]).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            params["output"],
            **{
                "profile_corr": profile_corr,
                "profile_jsd": profile_jsd,
                "counts_pearson": counts_pearson,
                "log_counts_pearson": log_counts_pearson,
                "counts_spearman": counts_spearman,
            },
        )


if __name__ == "__main__":
    main()
