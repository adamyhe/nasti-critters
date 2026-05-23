#!/usr/bin/env python3
"""Generate random train/val/test splits for S. pombe peaks.

S. pombe has only 3 chromosomes, making chromosome-based cross-validation
impractical for 5-fold CV. This script instead randomly assigns individual
peaks to 5 folds, with the constraint that peaks whose training windows
could overlap are always assigned to the same fold (to prevent data leakage).

Two peaks are grouped together if their centers are within
  threshold = in_window + 2 * max_jitter  (default: 2114 + 400 = 2514 bp)
of each other on the same chromosome. Groups are then assigned to folds
using greedy bin-packing so fold sizes stay roughly equal.

For fold i:
  - test  = peaks assigned to fold i
  - val   = peaks assigned to fold (i+1) % n_folds
  - train = peaks assigned to all remaining folds

Output: configs/splits/S.pombe_random_fold_assignments.csv
  columns: chrom, start, end, fold

Usage:
    python src/data_preprocessing/make_pombe_random_splits.py
    python src/data_preprocessing/make_pombe_random_splits.py --n-folds 5 --seed 47
    python src/data_preprocessing/make_pombe_random_splits.py --peaks path/to/peaks.bed.gz
    python src/data_preprocessing/make_pombe_random_splits.py --window 2114 --jitter 200
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PEAKS = REPO_ROOT / "data" / "GSE179468_spombe-tsr.bed.gz"
DEFAULT_OUTPUT = (
    REPO_ROOT / "configs" / "splits" / "S.pombe_random_fold_assignments.csv"
)


def assign_groups(peaks: pd.DataFrame, threshold: int) -> pd.Series:
    """Assign a group ID to each peak via linear sweep on sorted (chrom, center).

    Consecutive peaks on the same chromosome within `threshold` bp of each
    other (center-to-center) are placed in the same group. Because the peaks
    are sorted, this correctly captures all transitive connections.

    Returns a Series of integer group IDs aligned to `peaks` index.
    """
    centers = ((peaks["start"] + peaks["end"]) // 2).rename("center")
    df = peaks[["chrom"]].join(centers)
    df = df.sort_values(["chrom", "center"])

    group_ids = np.empty(len(df), dtype=int)
    group_id = 0
    prev_chrom: str | None = None
    prev_center: int = 0

    for i, (_, row) in enumerate(df.iterrows()):
        if row["chrom"] != prev_chrom or row["center"] - prev_center > threshold:
            group_id += 1
        group_ids[i] = group_id
        prev_chrom = row["chrom"]
        prev_center = row["center"]

    result = pd.Series(group_ids, index=df.index, name="group")
    return result.reindex(peaks.index)  # restore original row order


def assign_folds(peaks: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Assign fold IDs to peaks via greedy bin-packing over groups.

    Groups are shuffled randomly then assigned one-by-one to the fold with
    the current fewest peaks, keeping all peaks in a group together.
    """
    group_sizes = peaks.groupby("group").size()
    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(group_sizes.index)

    fold_totals = np.zeros(n_folds, dtype=int)
    group_fold: dict[int, int] = {}
    for g in shuffled_groups:
        f = int(np.argmin(fold_totals))
        group_fold[g] = f
        fold_totals[f] += group_sizes[g]

    return peaks["group"].map(group_fold).rename("fold")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--peaks",
        type=Path,
        default=DEFAULT_PEAKS,
        help="path to S. pombe peaks BED file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="number of folds (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=47,
        help="random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=2114,
        help="model input window size in bp (default: %(default)s)",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=200,
        help="max jitter during training in bp (default: %(default)s)",
    )
    args = parser.parse_args()

    peaks = pd.read_csv(
        args.peaks,
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )
    n_peaks = len(peaks)
    print(f"Loaded {n_peaks} peaks from {args.peaks}")

    threshold = args.window + 2 * args.jitter
    print(f"Proximity threshold: {threshold} bp  (window={args.window}, jitter={args.jitter})")

    peaks["group"] = assign_groups(peaks, threshold)

    n_groups = peaks["group"].nunique()
    n_grouped = (peaks.groupby("group")["group"].transform("size") > 1).sum()
    max_group = peaks.groupby("group").size().max()
    print(f"Grouped {n_peaks} peaks into {n_groups} clusters "
          f"({n_grouped} peaks share a cluster; max cluster size: {max_group})")

    peaks["fold"] = assign_folds(peaks, args.n_folds, args.seed)

    fold_counts = peaks["fold"].value_counts().sort_index()
    print("Peak counts per fold:")
    for fold, count in fold_counts.items():
        print(f"  fold {fold}: {count} peaks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    peaks[["chrom", "start", "end", "fold"]].to_csv(args.output, index=False)
    print(f"Saved fold assignments to {args.output}")

    print("\nSplit summary (fold i = test, fold (i+1)%n = val, rest = train):")
    for i in range(args.n_folds):
        test_fold = i
        val_fold = (i + 1) % args.n_folds
        train_folds = [f for f in range(args.n_folds) if f not in (test_fold, val_fold)]
        n_test = (peaks["fold"] == test_fold).sum()
        n_val = (peaks["fold"] == val_fold).sum()
        n_train = peaks["fold"].isin(train_folds).sum()
        print(
            f"  fold {i}: train={n_train}, val={n_val} (fold {val_fold}), "
            f"test={n_test} (fold {test_fold})"
        )


if __name__ == "__main__":
    main()
