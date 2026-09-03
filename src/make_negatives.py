#!/usr/bin/env python3
"""Generate negative control regions for BPNet training.

Reads experiment paths from config/experiment_config.yaml, builds an unstranded
csRNA BigWig per experiment, and calls `bpnet negatives` to sample genomic windows
with low transcription signal.

Minus-strand BigWigs are always abs-valued before merging, which correctly
handles both UCSC-format tracks (stored as negative values) and direct-format
tracks (already positive).

Peaks, signal and chrom.sizes are all restricted to the species'
`main_chromosomes` from config/genomes.yaml -- the same allow-list the bigWig
and PINTS steps use -- so negatives are drawn from exactly the space the peaks
occupy, and never from organelles or unplaced scaffolds.

Intermediate files are written to a per-experiment temporary directory that is
cleaned up automatically. Chrom sizes are derived on-the-fly from the FASTA
.fai index.

Usage:
    python src/make_negatives.py
    python src/make_negatives.py -e S.pombe_PROcap S.cerevisiae_PROcap
    python src/make_negatives.py --dry-run
    python src/make_negatives.py -e D.melanogaster-S2_PROcap --force
"""

import argparse
import gzip
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent   # src/ -> repo root
CONFIG_PATH = REPO_ROOT / "config" / "experiment_config.yaml"
GENOMES_PATH = REPO_ROOT / "config" / "genomes.yaml"

_fai_locks: dict[Path, threading.Lock] = {}
_fai_locks_mutex = threading.Lock()

# Which chromosomes negatives may be drawn from: `main_chromosomes` for the
# experiment's species, from config/genomes.yaml -- the same allow-list the
# bigWig and PINTS steps use, so negatives live in exactly the space the peaks
# do.
#
# This replaced a hand-written per-EXPERIMENT exclusion regex
# (`CHROM_EXCLUDE = {"S.cerevisiae_PROcap": "Mito", "S.pombe_PROcap": "MT|AB",
# "D.melanogaster-S2_PROcap": "_|2110000|Y|rDNA"}`). Three problems with that,
# all of which get worse as species are added:
#   * it covered 3 of the 38 experiments then defined, so 35 filtered nothing;
#   * the chrom.sizes it fed to bedGraphToBigWig came from the whole FASTA
#     .fai, so negatives could be sampled on organelles and unplaced scaffolds
#     -- 757 of them in S. moellendorffii, 637 in C. griseus;
#   * being a substring regex over the whole BED line it was fragile: `_`
#     matches any dm6 scaffold but also anything else with an underscore
#     anywhere in the row.
# It was also keyed by experiment when the thing it describes is a property of
# the SPECIES.

# Per-experiment alpha threshold passed as -a to bpnet negatives.
ALPHA: dict[str, float] = {
    "S.pombe_PROcap": 0.7,
}


def run(cmd: list, dry_run: bool, check: bool = True) -> None:
    print(" ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run([str(c) for c in cmd], check=check)


def main_chromosomes(species: str) -> list[str]:
    """The species' main_chromosomes from config/genomes.yaml, as strings.

    str() matters: bare-numeric names (A.thaliana 1-5, C.reinhardtii 1-17,
    P.patens 1-27, C.griseus 2-10) parse out of YAML as ints, while every
    chromosome name read back out of a .fai or a BED is text. Comparing the two
    directly matches nothing and fails silently, which is the exact class of bug
    config/genomes.yaml's chrom_style note warns about.
    """
    with open(GENOMES_PATH) as f:
        genomes = yaml.safe_load(f)["species"]
    return [str(c) for c in genomes[species]["main_chromosomes"]]


def make_chrom_sizes(
    sequences: Path, keep: list[str], tmp: Path, dry_run: bool
) -> Path:
    """chrom.sizes from the FASTA .fai, restricted to `keep`.

    Restricting here is what keeps the two bedGraphToBigWig calls below and
    `bpnet negatives` inside the same chromosome space as the peaks.
    """
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
        fai_df = pd.read_csv(
            fai, sep="\t", header=None, usecols=[0, 1],
            names=["chrom", "size"], dtype={"chrom": str},
        )
        kept = fai_df[fai_df["chrom"].isin(keep)]
        # A name mismatch between genomes.yaml and the FASTA would otherwise
        # produce an empty (or short) chrom.sizes and a baffling downstream
        # error. Fail here instead, naming the missing contigs.
        absent = sorted(set(keep) - set(fai_df["chrom"]))
        if absent:
            raise SystemExit(
                f"main_chromosomes not found in {fai.name}: {absent[:10]}"
                f"{' ...' if len(absent) > 10 else ''}. config/genomes.yaml and "
                "the FASTA disagree about chromosome naming."
            )
        kept.to_csv(chrom_sizes, sep="\t", header=False, index=False)
    else:
        print(f"  # (chrom sizes: {fai} → {chrom_sizes}, {len(keep)} chromosomes)")
    return chrom_sizes


def bw_to_abs_bg(bw: Path, out_bg: Path, keep: list[str], dry_run: bool) -> None:
    """BigWig -> BedGraph, values abs-valued, rows restricted to `keep`.

    abs() is the load-bearing part and must stay: it makes UCSC-convention
    minus-strand tracks (negative values) and direct tracks (already positive)
    behave identically, and it is what lets bigWigMerge below work at all --
    bigWigMerge defaults -threshold to 0 and DROPS values at or below it, so a
    negative minus track would merge to nothing.

    Filtering to `keep` is what stops bedGraphToBigWig aborting on a contig the
    restricted chrom.sizes no longer lists.
    """
    if dry_run:
        print(f"  # bigWigToBedGraph {bw} | abs | keep main chroms → {out_bg}")
        return
    raw_bg = out_bg.with_suffix(".raw.bg")
    subprocess.run(["bigWigToBedGraph", str(bw), str(raw_bg)], check=True)
    df = pd.read_csv(
        raw_bg, sep="\t", header=None, names=["chrom", "start", "end", "value"],
        dtype={"chrom": str},
    )
    df["value"] = df["value"].abs()
    df[df["chrom"].isin(keep)].to_csv(out_bg, sep="\t", header=False, index=False)
    raw_bg.unlink()


def make_unstranded_bw(
    pl_bw: Path, mn_bw: Path, out_bw: Path, chrom_sizes: Path,
    keep: list[str], tmp: Path, dry_run: bool,
) -> None:
    """Merge plus/minus BigWigs into a sorted unstranded BigWig over `keep`."""
    mn_abs_bg = tmp / "mn_abs.bg"
    mn_abs_bw = tmp / "mn_abs.bw"
    us_bg = tmp / "us.bg"
    us_sorted_bg = tmp / "us.sorted.bg"

    bw_to_abs_bg(mn_bw, mn_abs_bg, keep, dry_run)
    run(["bedGraphToBigWig", mn_abs_bg, chrom_sizes, mn_abs_bw], dry_run)
    run(["bigWigMerge", pl_bw, mn_abs_bw, us_bg], dry_run)

    if dry_run:
        print(f"  # sort {us_bg}, keep main chroms → {us_sorted_bg}")
    else:
        df = pd.read_csv(
            us_bg, sep="\t", header=None, names=["chrom", "start", "end", "value"],
            dtype={"chrom": str},
        )
        # pl_bw is filtered here rather than up front: bigWigMerge takes it
        # directly, so anything outside `keep` that it carries would otherwise
        # reach the restricted chrom.sizes below and abort the conversion.
        df = df[df["chrom"].isin(keep)]
        df.sort_values(["chrom", "start"]).to_csv(
            us_sorted_bg, sep="\t", header=False, index=False
        )

    run(["bedGraphToBigWig", us_sorted_bg, chrom_sizes, out_bw], dry_run)


def filter_peaks(
    peaks: Path, keep: list[str], out_path: Path, dry_run: bool
) -> int:
    """Write a gzipped BED holding only rows on a chromosome in `keep`.

    Matches on COLUMN 0 exactly, not as a substring of the whole line. The
    regex this replaced was a `grep -vE` over the entire row, so dm6's `_`
    pattern would drop any peak whose name, class or confidence field happened
    to contain an underscore, not just the scaffolds it was aimed at.

    Returns the number of rows kept, so the caller can refuse to hand an empty
    or near-empty BED to `bpnet negatives`.
    """
    if dry_run:
        print(f"  # {peaks} | keep {len(keep)} main chroms | bgzip > {out_path}")
        return -1
    # gzip.open, not `zcat`: bgzip output is valid gzip, and macOS `zcat` only
    # handles .Z, so shelling out made this function untestable off-cluster for
    # no benefit. Writing still goes through bgzip, since the output must be
    # BGZF rather than plain gzip.
    allowed = set(keep)
    with gzip.open(peaks, "rt") as fh:
        lines = [
            line for line in (raw.rstrip("\n") for raw in fh)
            if line and not line.startswith("#")
            and line.split("\t", 1)[0] in allowed
        ]
    with open(str(out_path), "wb") as f_out:
        subprocess.run(
            ["bgzip", "-c"], input=("\n".join(lines) + "\n").encode(),
            stdout=f_out, check=True,
        )
    return len(lines)


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
        keep = main_chromosomes(exp["species"])
        chrom_sizes = make_chrom_sizes(sequences, keep, tmp, dry_run)

        us_bw = tmp / "csrna.us.bw"
        print("Building unstranded csRNA BigWig...")
        make_unstranded_bw(pl_bw, mn_bw, us_bw, chrom_sizes, keep, tmp, dry_run)

        # Always filter, for every species. Peaks from the pipeline are already
        # restricted to main_chromosomes, so this is normally a no-op -- but it
        # is the guard that makes that true rather than assumed, and it matters
        # for any peak set not produced by workflow/Snakefile.
        peaks_input = tmp / "peaks.filtered.bed.gz"
        print(f"Filtering peaks to {len(keep)} main chromosomes...")
        n_kept = filter_peaks(peaks, keep, peaks_input, dry_run)
        if n_kept == 0:
            print(
                f"SKIP {exp_id}: no peaks on {exp['species']} main_chromosomes. "
                "Check config/genomes.yaml naming against the peak file.",
                file=sys.stderr,
            )
            return False

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
