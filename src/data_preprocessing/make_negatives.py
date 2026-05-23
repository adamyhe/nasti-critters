#!/usr/bin/env python3
"""Generate negative control regions for BPNet training.

Reads experiment paths from configs/experiment_config.yaml, builds an unstranded
csRNA BigWig per experiment, and calls `bpnet negatives` to sample genomic windows
with low transcription signal.

Minus-strand BigWigs are always abs-valued before merging, which correctly
handles both UCSC-format tracks (stored as negative values) and direct-format
tracks (already positive).

Intermediate files are written to a per-experiment temporary directory that is
cleaned up automatically. Chrom sizes are derived on-the-fly from the FASTA
.fai index.

Usage:
    python src/data_preprocessing/make_negatives.py
    python src/data_preprocessing/make_negatives.py -e S.pombe_PROcap S.cerevisiae_PROcap
    python src/data_preprocessing/make_negatives.py --dry-run
    python src/data_preprocessing/make_negatives.py -e D.melanogaster-S2_PROcap --force
"""

import argparse
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"

_fai_locks: dict[Path, threading.Lock] = {}
_fai_locks_mutex = threading.Lock()

# Chromosome exclusion regex patterns per experiment (grep -vE equivalent).
# Filters out unplaced scaffolds, organellar chromosomes, and other artifacts
# before passing peaks to bpnet negatives.
CHROM_EXCLUDE: dict[str, str] = {
    "S.cerevisiae_PROcap": r"Mito",
    "S.pombe_PROcap": r"MT|AB",
    "D.melanogaster-S2_PROcap": r"_|2110000|Y|rDNA",
}

# Per-experiment alpha threshold passed as -a to bpnet negatives.
ALPHA: dict[str, float] = {
    "S.pombe_PROcap": 0.7,
}


def run(cmd: list, dry_run: bool, check: bool = True) -> None:
    print(" ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run([str(c) for c in cmd], check=check)


def make_chrom_sizes(sequences: Path, tmp: Path, dry_run: bool) -> Path:
    """Return path to a 2-column chrom.sizes file derived from the FASTA .fai."""
    fai = Path(str(sequences) + ".fai")
    if not fai.exists():
        with _fai_locks_mutex:
            if fai not in _fai_locks:
                _fai_locks[fai] = threading.Lock()
        with _fai_locks[fai]:
            if not fai.exists():
                run(["samtools", "faidx", sequences], dry_run)
    chrom_sizes = tmp / "chrom.sizes"
    if not dry_run:
        fai_df = pd.read_csv(fai, sep="\t", header=None, usecols=[0, 1])
        fai_df.to_csv(chrom_sizes, sep="\t", header=False, index=False)
    else:
        print(f"  # (chrom sizes: {fai} → {chrom_sizes})")
    return chrom_sizes


def bw_to_abs_bg(bw: Path, out_bg: Path, dry_run: bool) -> None:
    """Convert a BigWig to a BedGraph with all values made positive (abs)."""
    if dry_run:
        print(f"  # bigWigToBedGraph {bw} | abs → {out_bg}")
        return
    raw_bg = out_bg.with_suffix(".raw.bg")
    subprocess.run(["bigWigToBedGraph", str(bw), str(raw_bg)], check=True)
    df = pd.read_csv(
        raw_bg, sep="\t", header=None, names=["chrom", "start", "end", "value"]
    )
    df["value"] = df["value"].abs()
    df.to_csv(out_bg, sep="\t", header=False, index=False)
    raw_bg.unlink()


def make_unstranded_bw(
    pl_bw: Path, mn_bw: Path, out_bw: Path, chrom_sizes: Path, tmp: Path, dry_run: bool
) -> None:
    """Merge plus/minus BigWigs into a sorted unstranded BigWig."""
    mn_abs_bg = tmp / "mn_abs.bg"
    mn_abs_bw = tmp / "mn_abs.bw"
    us_bg = tmp / "us.bg"
    us_sorted_bg = tmp / "us.sorted.bg"

    bw_to_abs_bg(mn_bw, mn_abs_bg, dry_run)
    run(["bedGraphToBigWig", mn_abs_bg, chrom_sizes, mn_abs_bw], dry_run)
    run(["bigWigMerge", pl_bw, mn_abs_bw, us_bg], dry_run)

    if dry_run:
        print(f"  # sort {us_bg} → {us_sorted_bg}")
    else:
        df = pd.read_csv(
            us_bg, sep="\t", header=None, names=["chrom", "start", "end", "value"]
        )
        df.sort_values(["chrom", "start"]).to_csv(
            us_sorted_bg, sep="\t", header=False, index=False
        )

    run(["bedGraphToBigWig", us_sorted_bg, chrom_sizes, out_bw], dry_run)


def filter_peaks(
    peaks: Path, exclude_pattern: str, out_path: Path, dry_run: bool
) -> None:
    """Write a chromosome-filtered gzipped BED to out_path."""
    if dry_run:
        print(f"  # zcat {peaks} | grep -vE '{exclude_pattern}' | bgzip > {out_path}")
        return
    result = subprocess.run(
        ["zcat", str(peaks)], capture_output=True, text=True, check=True
    )
    pat = re.compile(exclude_pattern)
    filtered = "\n".join(
        line for line in result.stdout.splitlines() if not pat.search(line)
    )
    with open(str(out_path), "wb") as f_out:
        subprocess.run(
            ["bgzip", "-c"], input=filtered.encode(), stdout=f_out, check=True
        )


def process_experiment(exp_id: str, exp: dict, force: bool, dry_run: bool) -> bool:
    """Process one experiment. Returns True if processed, False if skipped."""
    processed = exp.get("processed", {})

    out_path = REPO_ROOT / processed["gc_negatives"]
    if out_path.exists() and not force:
        print(
            f"SKIP {exp_id}: output exists (use --force to overwrite): {out_path.name}"
        )
        return False

    pl_bw = REPO_ROOT / processed["pl_bigwig"]
    mn_bw = REPO_ROOT / processed["mn_bigwig"]
    peaks = REPO_ROOT / processed["peaks"]
    sequences = REPO_ROOT / processed["sequences"]

    for path, label in [
        (pl_bw, "pl_bigwig"),
        (mn_bw, "mn_bigwig"),
        (peaks, "peaks"),
        (sequences, "sequences"),
    ]:
        if not path.exists():
            print(f"SKIP {exp_id}: missing {label}: {path}", file=sys.stderr)
            return False

    print(f"\n=== {exp_id} ===")

    with tempfile.TemporaryDirectory(prefix=f"negatives_{exp_id}_") as tmp_str:
        tmp = Path(tmp_str)
        chrom_sizes = make_chrom_sizes(sequences, tmp, dry_run)

        us_bw = tmp / "csrna.us.bw"
        print("Building unstranded csRNA BigWig...")
        make_unstranded_bw(pl_bw, mn_bw, us_bw, chrom_sizes, tmp, dry_run)

        exclude = CHROM_EXCLUDE.get(exp_id)
        if exclude:
            peaks_input = tmp / "peaks.filtered.bed.gz"
            print(f"Filtering peaks (excluding chromosomes matching: {exclude})...")
            filter_peaks(peaks, exclude, peaks_input, dry_run)
        else:
            peaks_input = peaks

        print("Running bpnet negatives...")
        neg_cmd: list = [
            "bpnet",
            "negatives",
            "-v",
            "-i",
            peaks_input,
            "-f",
            sequences,
            "-o",
            out_path,
            "-b",
            us_bw,
        ]
        alpha = ALPHA.get(exp_id)
        if alpha is not None:
            neg_cmd += ["-a", alpha]
        run(neg_cmd, dry_run)

    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="+",
        metavar="EXP",
        default=None,
        help="experiment IDs to process (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing output files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without executing them",
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        default=1,
        metavar="N",
        help="number of experiments to process in parallel (default: %(default)s)",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = config["experiments"]

    if args.experiments is not None:
        unknown = [e for e in args.experiments if e not in experiments]
        if unknown:
            print(
                f"Error: unknown experiment(s): {', '.join(unknown)}", file=sys.stderr
            )
            sys.exit(1)
        exp_ids = args.experiments
    else:
        exp_ids = list(experiments.keys())

    run_one = partial(
        process_experiment, force=args.force, dry_run=args.dry_run
    )

    n_processed = n_skipped = 0
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(run_one, exp_id=exp_id, exp=experiments[exp_id]): exp_id
            for exp_id in exp_ids
        }
        for future in as_completed(futures):
            exp_id = futures[future]
            try:
                if future.result():
                    n_processed += 1
                else:
                    n_skipped += 1
            except Exception as exc:
                print(f"ERROR {exp_id}: {exc}", file=sys.stderr)
                n_skipped += 1

    action = "Would process" if args.dry_run else "Processed"
    print(f"\n{action} {n_processed} experiments, skipped {n_skipped}")


if __name__ == "__main__":
    main()
