#!/usr/bin/env python3
"""Generate random, peak-level train/val/test splits.

Originally written for S. pombe alone (and named make_pombe_random_splits.py
until a second species needed it), but the logic was always species-agnostic.
Defaults still target S. pombe; pass --experiment and --output for any other:

    python src/data_preprocessing/make_random_splits.py \
        -e S.moellendorffii-stemleaf_5GRO \
        -o config/splits/S.moellendorffii_random_fold_assignments.csv

src/experiments.py picks the result up automatically: it looks for
config/splits/{species}_random_fold_assignments.csv and switches to peak-level
folds when one exists, for any species.

Peak-level folds are for species where CHROMOSOME-level folds are not
available, of which there are two kinds here:

  * too few chromosomes -- S. pombe has 3, which cannot give 5 balanced folds;
  * none at all -- S. moellendorffii v1.0 is a 759-scaffold draft with an
    empty Ensembl karyotype, so there are no chromosomes to assign.

Note this is NOT the right tool for a species that merely lacks a fold
assignment yet. C. griseus has 12 chromosome-scale units and should get a
peak-matched chromosome-level entry in config/chrom_splits.yaml instead.

The script randomly assigns individual peaks to 5 folds, with the constraint
that peaks whose training windows could overlap are always assigned to the same
fold (to prevent data leakage).

Two peaks are grouped together if their centers are within
  threshold = in_window + 2 * max_jitter  (default: 2114 + 400 = 2514 bp)
of each other on the same chromosome. Groups are then assigned to folds
using greedy bin-packing so fold sizes stay roughly equal.

For fold i:
  - test  = peaks assigned to fold i
  - val   = peaks assigned to fold (i+1) % n_folds
  - train = peaks assigned to all remaining folds

Default output: config/splits/S.pombe_random_fold_assignments.csv
  columns: chrom, start, end, fold

Usage:
    python src/data_preprocessing/make_random_splits.py
    python src/data_preprocessing/make_random_splits.py --n-folds 5 --seed 47
    python src/data_preprocessing/make_random_splits.py --peaks path/to/peaks.bed.gz
    python src/data_preprocessing/make_random_splits.py --window 2114 --jitter 200
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT / "src"))
from experiments import Experiment  # noqa: E402

DEFAULT_EXPERIMENT = "S.pombe_PROcap"
DEFAULT_OUTPUT = (
    REPO_ROOT / "config" / "splits" / "S.pombe_random_fold_assignments.csv"
)


def peaks_digest(peaks: pd.DataFrame) -> str:
    """sha256 over the peak coordinates, in file order.

    Order-sensitive on purpose: the fold CSV is written in the same row order
    as its input and src/experiments.py joins the two positionally, so a peak
    set with identical content but different row order is NOT interchangeable.
    """
    h = hashlib.sha256()
    for chrom, start, end in peaks[["chrom", "start", "end"]].itertuples(index=False):
        h.update(f"{chrom}\t{start}\t{end}\n".encode())
    return h.hexdigest()


def provenance(args, peaks: pd.DataFrame) -> dict:
    return {
        "generated_by": "src/data_preprocessing/make_random_splits.py",
        "peaks": str(args.peaks),
        "peaks_sha256": peaks_digest(peaks),
        "n_peaks": len(peaks),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "window": args.window,
        "jitter": args.jitter,
    }


def read_provenance(path: Path) -> dict:
    """Parse the `#` header written by write_splits(); {} if absent."""
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if ":" in line:
                k, _, v = line[1:].partition(":")
                meta[k.strip()] = v.strip()
    return meta


def write_splits(path: Path, table: pd.DataFrame, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        table.to_csv(f, index=False)


def assign_groups(peaks: pd.DataFrame, threshold: int) -> pd.Series:
    """Assign a group ID to each peak via linear sweep on sorted (chrom, center).

    Consecutive peaks on the same chromosome within `threshold` bp of each
    other (center-to-center) are placed in the same group. Because the peaks
    are sorted, this correctly captures all transitive connections.

    Returns a Series of integer group IDs aligned to `peaks` index.
    """
    centers = ((peaks["start"] + peaks["end"]) // 2).rename("center")
    df = peaks[["chrom"]].join(centers)
    # mergesort is the stable option; ties on (chrom, center) then keep
    # input order instead of whatever quicksort happens to produce.
    df = df.sort_values(["chrom", "center"], kind="mergesort")

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
        "-e",
        "--experiment",
        type=str,
        default=DEFAULT_EXPERIMENT,
        help="experiment whose peaks define the folds (default: %(default)s)",
    )
    parser.add_argument(
        "--peaks",
        type=Path,
        default=None,
        help="override the peaks BED resolved from --experiment",
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing split file still matches the current peaks "
             "and parameters; exit nonzero on drift, write nothing",
    )
    args = parser.parse_args()

    # Resolve peaks through experiments.py so this tracks whatever the ENCODE
    # pipeline actually wrote, rather than a hard-coded filename.
    if args.peaks is None:
        try:
            args.peaks = Experiment.load(args.experiment).peaks
        except (KeyError, ValueError) as err:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)
    if not Path(args.peaks).exists():
        print(f"Error: peaks not found: {args.peaks}\n"
              "Run the PRO-cap pipeline for this experiment first "
              "(snakemake -c16 --config tier=conditional).", file=sys.stderr)
        sys.exit(1)

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

    meta = provenance(args, peaks)
    table = peaks[["chrom", "start", "end", "fold"]]

    if args.check:
        if not args.output.exists():
            print(f"Error: {args.output} does not exist", file=sys.stderr)
            sys.exit(1)
        have = read_provenance(args.output)
        drift = [f"  {k}: file={have.get(k, '<missing>')} current={v}"
                 for k, v in meta.items()
                 if k != "peaks" and str(have.get(k)) != str(v)]
        if drift:
            print(f"Error: {args.output.name} is stale:", file=sys.stderr)
            print("\n".join(drift), file=sys.stderr)
            print("Regenerate it by rerunning this script without --check.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"{args.output.name} is up to date with {args.peaks}")
        return

    write_splits(args.output, table, meta)
    print(f"Saved fold assignments to {args.output}")
    print(f"  peaks sha256: {meta['peaks_sha256'][:16]}...  "
          f"(seed={args.seed}, n_folds={args.n_folds})")

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
