#!/usr/bin/env python3
"""Generate negative control regions for BPNet training.

Reads experiment paths from config/experiment_config.yaml, builds an unstranded
initiation-signal BigWig per experiment, and GC-matches genomic windows with low
transcription signal against the peak set.

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
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pybigtools
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent   # src/ -> repo root
CONFIG_PATH = REPO_ROOT / "config" / "experiment_config.yaml"
GENOMES_PATH = REPO_ROOT / "config" / "genomes.yaml"


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

    Restricting here is what keeps the merged bigWig below and the negative
    sampling space inside the same chromosome space as the peaks.
    """
    fai = Path(str(sequences) + ".fai")
    if not fai.exists() and not dry_run:
        # pyfaidx, not `samtools faidx`: samtools is an environment.yml binary
        # and this script runs from the uv venv, so shelling out here is the
        # same cross-environment bug as the old bigWigToBedGraph call. pyfaidx
        # writes a byte-identical .fai.
        #
        # A last-resort fallback only. main() builds every index serially before
        # the pool starts, because workers are PROCESSES and a lock here would
        # not be shared between them. Reaching this line means an index went
        # missing mid-run.
        import pyfaidx
        pyfaidx.Faidx(str(sequences))
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


#: Window sizes, matching the rest of the repo (and, as it happens, the
#: `bpnet negatives` CLI defaults). Passed explicitly rather than relied on:
#: these are a project convention, not tangermeme's to change.
IN_WINDOW = 2114
OUT_WINDOW = 1000

#: Per-species candidate-tiling width for GC matching ONLY. Absent = IN_WINDOW.
#:
#: `extract_matching_loci` tiles each chromosome into NON-OVERLAPPING blocks of
#: `in_window` and masks out every block containing a peak, so the entire
#: candidate pool is `genome / in_window`. That is ~5,700 blocks for a 12 Mb
#: yeast genome against 23,642 peaks -- four peaks per block -- and almost
#: nothing survives: the yeasts get 0.01-0.07 negatives per peak where every
#: large genome gets 1.00.
#:
#: Shrinking the tiling width for those species places candidate MIDPOINTS more
#: finely. It does not shrink the training window: both `_resize_coords_generator`
#: here and `extract_loci` in the training loader resize to the same midpoint, so
#: the model still sees IN_WINDOW. The written BED intervals are this width,
#: which is why nothing downstream may depend on their width.
#:
#: The cost is real and is why `pct_peak_overlap` is reported on every run: a
#: negative whose narrow block is peak-free can still contain peaks once expanded
#: to IN_WINDOW. In S. cerevisiae there is a peak every ~780 bp, so a 2114 bp
#: window holds ~2.7 of them on average and no large clean set exists to find.
#: That is NOT a labelling error -- the loader extracts real measured signal for
#: backgrounds, so an overlapping negative is a low-contrast example, not a
#: mislabelled one -- but it does weaken the contrast negatives are there to
#: provide. Set a value here only with the measured overlap in front of you.
NEGATIVE_WINDOW: dict[str, int] = {}

#: Interval values below this are treated as absent when summing the two
#: strands. Counts are integers stored exactly in float32 and accumulated in
#: float64, so the sweep is exact for real data; the epsilon only guards
#: against float residue if a track ever holds non-integer values.
_ZERO = 1e-9

#: One bigWig interval, as a numpy record. Used with np.fromiter so a whole
#: chromosome never becomes a Python list of tuples -- a deep mouse or hamster
#: chromosome runs to millions of intervals, and the tuples cost far more than
#: the numbers do.
_IVAL = np.dtype([("start", np.int64), ("end", np.int64), ("value", np.float64)])


def _records(reader, chrom: str) -> np.ndarray:
    """Every interval on one chromosome, or empty if the track lacks it.

    A chromosome present in chrom.sizes can be absent from a strand's bigWig --
    `bedGraphToBigWig` records only contigs that appear in its input, which is
    the same asymmetry that makes PINTS reject mismatched pl/mn contig sets.
    pybigtools raises KeyError there rather than returning nothing.
    """
    try:
        return np.fromiter(reader.records(chrom), dtype=_IVAL)
    except KeyError:
        return np.empty(0, dtype=_IVAL)


def _sum_strands(pl: np.ndarray, mn: np.ndarray) -> np.ndarray:
    """Sum two piecewise-constant interval sets into non-overlapping intervals.

    A coordinate sweep: each interval contributes +v at its start and -v at its
    end, events are sorted, and the running level gives the value on each gap
    between consecutive event positions. This is what `bigWigMerge` did, minus
    its trap -- see make_unstranded_bw.

    Values are abs()-ed by the caller, so the level is non-negative and a level
    at or below _ZERO means "no coverage here", which is exactly the bedGraph
    convention of omitting zeros.
    """
    if not len(pl) and not len(mn):
        return np.empty(0, dtype=_IVAL)
    pos = np.concatenate([pl["start"], pl["end"], mn["start"], mn["end"]])
    delta = np.concatenate([pl["value"], -pl["value"], mn["value"], -mn["value"]])
    order = np.argsort(pos, kind="mergesort")
    pos, delta = pos[order], delta[order]
    # Collapse events that share a coordinate before accumulating, so a level
    # is only ever read between distinct positions.
    edges, first = np.unique(pos, return_index=True)
    level = np.cumsum(np.add.reduceat(delta, first))
    out = np.empty(len(edges) - 1, dtype=_IVAL)
    out["start"], out["end"], out["value"] = edges[:-1], edges[1:], level[:-1]
    return out[out["value"] > _ZERO]


def make_unstranded_bw(
    pl_bw: Path, mn_bw: Path, out_bw: Path, chrom_sizes: Path,
    keep: list[str], dry_run: bool,
) -> None:
    """Merge plus/minus BigWigs into an unstranded BigWig over `keep`.

    Done in-process with pybigtools rather than by shelling out. The old route
    was `bigWigToBedGraph | abs | bedGraphToBigWig | bigWigMerge | sort |
    bedGraphToBigWig` -- four UCSC binaries and four temporary files -- and it
    is the reason this script failed with `No such file or directory:
    'bigWigToBedGraph'`. Those binaries live in `environment.yml`, but
    make_negatives.py is the one thing here that needs bpnet-lite, so it runs
    from the uv venv where they are not on PATH. pybigtools is already a venv
    dependency via tangermeme, so reading and writing directly removes the
    cross-environment dependency instead of papering over it.

    abs() on the minus strand is still the load-bearing part and must stay: it
    makes UCSC-convention tracks (negative values) and direct tracks (already
    positive) behave identically. It used to ALSO be what made `bigWigMerge`
    work at all, since that tool defaults `-threshold` to 0 and drops values at
    or below it, so a negative minus track merged to nothing. That trap is gone
    with the tool, but the abs() is not optional -- without it the two strands
    would cancel rather than sum.

    Output intervals are written in chrom.sizes order with ascending starts,
    which is what the bigWig format requires.
    """
    if dry_run:
        print(f"  # merge {pl_bw.name} + |{mn_bw.name}| -> {out_bw.name} "
              f"(pybigtools, {len(keep)} chromosomes)")
        return

    sizes = pd.read_csv(
        chrom_sizes, sep="\t", header=None, names=["chrom", "size"],
        dtype={"chrom": str},
    )
    keep_set = set(keep)

    pl_reader = pybigtools.open(str(pl_bw))
    mn_reader = pybigtools.open(str(mn_bw))
    try:
        def intervals():
            for chrom, _size in sizes.itertuples(index=False):
                if chrom not in keep_set:
                    continue
                mn = _records(mn_reader, chrom)
                mn["value"] = np.abs(mn["value"])
                merged = _sum_strands(_records(pl_reader, chrom), mn)
                for start, end, value in merged:
                    yield chrom, int(start), int(end), float(value)

        writer = pybigtools.open(str(out_bw), "w")
        try:
            writer.write(dict(zip(sizes["chrom"], sizes["size"])), intervals())
        finally:
            writer.close()
    finally:
        pl_reader.close()
        mn_reader.close()


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
        print(f"  # {peaks} | keep {len(keep)} main chroms | gzip > {out_path}")
        return -1
    # gzip.open for BOTH directions, no shelling out. Reading: bgzip output is
    # valid gzip and macOS `zcat` only handles .Z. Writing: this used to call
    # `bgzip` on the claim that the output "must be BGZF rather than plain
    # gzip", which is wrong -- nothing tabix-indexes this temp file. Traced the
    # consumer to be sure: `bpnet negatives` hands the path to
    # tangermeme.match.extract_matching_loci, which does
    # `pandas.read_csv(loci, sep='\t', ...)`, and pandas reads plain gzip and
    # BGZF identically. bgzip is also an environment.yml binary, so calling it
    # from the uv venv was the same bug as the old bigWigToBedGraph call --
    # and it ran unconditionally, so it was the very next failure.
    allowed = set(keep)
    with gzip.open(peaks, "rt") as fh:
        lines = [
            line for line in (raw.rstrip("\n") for raw in fh)
            if line and not line.startswith("#")
            and line.split("\t", 1)[0] in allowed
        ]
    with gzip.open(out_path, "wt") as f_out:
        f_out.write("\n".join(lines) + "\n")
    return len(lines)


def median_window_signal(bigwig: Path, regions: pd.DataFrame, window: int) -> float:
    """Median total signal in `window` bp around each region's midpoint.

    Reported for the negatives and for the peaks so the two are comparable. It
    is the check that matters when the signal filter is off: without it a
    "negative" is only a GC-matched, peak-free-by-tile window, and this says
    whether those are genuinely quiet or merely uncalled.
    """
    if not len(regions):
        return float("nan")
    totals = []
    with pybigtools.open(str(bigwig)) as bw:
        sizes = bw.chroms()
        for chrom, sub_df in regions.groupby("chrom", sort=False):
            if chrom not in sizes:
                continue
            mid = (sub_df["start"].to_numpy() + sub_df["end"].to_numpy()) // 2
            lo = np.clip(mid - window // 2, 0, sizes[chrom])
            hi = np.clip(mid + (window + 1) // 2, 0, sizes[chrom])
            for a, b in zip(lo, hi):
                if b > a:
                    totals.append(float(np.nansum(bw.values(chrom, int(a), int(b)))))
    return float(np.median(totals)) if totals else float("nan")


def peak_overlap_fraction(matched: pd.DataFrame, loci: pd.DataFrame,
                          window: int) -> float:
    """Share of negatives whose IN_WINDOW training window overlaps any peak.

    The negatives BED holds whatever width the tiling used, but training resizes
    each one to IN_WINDOW around its midpoint -- so this measures the window the
    model will actually see, not the interval on disk. With the default tiling
    this is ~0 by construction; it is the number that makes a narrower
    NEGATIVE_WINDOW an informed choice rather than a hopeful one.
    """
    if not len(matched) or not len(loci):
        return 0.0
    hit = 0
    for chrom, neg in matched.groupby("chrom", sort=False):
        pk = loci[loci["chrom"] == chrom]
        if not len(pk):
            continue
        # Sort peaks by start and carry a running max of their ends, so a
        # single searchsorted answers "does any peak starting at or before this
        # point still extend past it".
        order = np.argsort(pk["start"].to_numpy(), kind="mergesort")
        ps = pk["start"].to_numpy()[order]
        pe = np.maximum.accumulate(pk["end"].to_numpy()[order])
        mid = (neg["start"].to_numpy() + neg["end"].to_numpy()) // 2
        s, e = mid - window // 2, mid + (window + 1) // 2
        idx = np.searchsorted(ps, e, side="left") - 1
        ok = idx >= 0
        hit += int(np.count_nonzero(ok & (pe[np.clip(idx, 0, None)] > s)))
    return hit / len(matched)


def sample_negatives(
    peaks: Path, sequences: Path, bigwig: Path, out_path: Path,
    keep: list[str], alpha: float | None, species: str, dry_run: bool,
    signal_filter: bool = True,
) -> None:
    """GC-matched negatives, restricted to `keep`.

    Calls `tangermeme.match.extract_matching_loci` directly instead of shelling
    out to `bpnet negatives`, and the reason is a correctness bug rather than a
    dependency one.

    That CLI is three lines, and the first is
    `chroms = list(pyfaidx.Fasta(args.fasta).keys())` -- **the whole assembly**.
    `chroms` is not just a filter on the input loci; `extract_matching_loci`
    builds its candidate space from it (`chrom_sizes = {key: len(fa[key]) for
    key in chroms}`), so negatives were being sampled from organelles, unplaced
    scaffolds and decoy-adjacent contigs. Peaks and signal were both restricted
    to `main_chromosomes` and the sampling space silently was not, which is
    exactly the invariant this script exists to enforce.

    It surfaced as a crash rather than as bad data only because the merged
    bigWig *is* restricted: `_counts_from_coords` then asked it for a contig it
    does not contain and pybigtools raised
    `KeyError: 'No chromomsome with name `scaffold_37` found.'` (also seen as
    `Mito` on S. cerevisiae). A less restricted bigWig would have returned
    counts and the negatives would have been quietly wrong.

    Everything else mirrors the CLI exactly -- same defaults, same `n_jobs=1`,
    same `to_csv` -- so this is a one-argument divergence, not a fork. Keep it
    that way; if bpnet-lite ever grows a `--chroms` flag, go back to the CLI.
    """
    tile = NEGATIVE_WINDOW.get(species, IN_WINDOW)
    # out_window scaled to keep the flank proportion; the assertion inside
    # extract_matching_loci is `in_window >= out_window`, so both must move.
    out_tile = max(1, round(tile * OUT_WINDOW / IN_WINDOW))
    if tile != IN_WINDOW:
        print(f"  tiling GC candidates at {tile} bp (not {IN_WINDOW}) for {species}")
    if dry_run:
        print(f"  # extract_matching_loci({peaks.name}, chroms={len(keep)} main, "
              f"tile={tile}) -> {out_path.name}")
        return

    from tangermeme.match import extract_matching_loci

    # Read the BED OURSELVES, with chrom forced to str, and pass the DataFrame.
    # extract_matching_loci accepts either a path or a DataFrame, and its path
    # branch is `pandas.read_csv(..., names=['chrom','start','end'])` with no
    # dtype -- so a purely numeric chromosome column infers as int64. It then
    # does `numpy.isin(loci['chrom'], chroms)` against our all-string `chroms`,
    # every comparison is False, and EVERY PEAK IS SILENTLY DROPPED.
    #
    # That is why A.thaliana (1-5), C.reinhardtii (1-17) and P.patens (1-27) --
    # the three species with purely numeric names -- produced no negatives at
    # all, while C.griseus survived on the strength of having an `X`, which
    # makes the column object dtype. Roman-numeral and prefixed names are safe
    # for the same reason.
    #
    # Pre-existing, not introduced by dropping the CLI: `bpnet negatives` passed
    # pyfaidx keys, which are also strings, so it hit the same mismatch.
    loci = pd.read_csv(
        peaks, sep="\t", usecols=[0, 1, 2], header=None, index_col=False,
        names=["chrom", "start", "end"], dtype={0: str},
    )

    # The empty-match check below is NOT reachable on its own: with verbose=True
    # tangermeme prints diagnostics before returning, and one of them is
    # `_counts_from_coords(...).max()`, which raises on an empty array first.
    # So the ValueError has to be caught here and translated.
    # bigwig=None disables the SIGNAL restriction and nothing else: in
    # `_extract_and_filter_chrom` the threshold and the `values <=
    # signal_threshold` mask both sit behind `if bigwig is not None`, while GC
    # matching, the max_n_perc filter and the peak-tile mask are unconditional.
    # Worth having for the dense genomes, where the surviving tile count is the
    # binding constraint and the signal filter only cuts it further -- but a
    # negative is then merely GC-matched and peak-free BY TILE, so the reported
    # median signal is what says whether it is quiet or just uncalled.
    try:
        matched = extract_matching_loci(
            loci=loci,
            fasta=str(sequences),
            bigwig=str(bigwig) if signal_filter else None,
            chroms=list(keep),
            in_window=tile,
            out_window=out_tile,
            verbose=True,
            n_jobs=1,
            **({"signal_beta": alpha} if alpha is not None else {}),
        )
    except ValueError as exc:
        if "zero-size array" not in str(exc):
            raise
        matched = loci.iloc[:0]
    if len(matched) == 0:
        # tangermeme's own verbose path calls .max() on the matched counts and
        # dies with "zero-size array to reduction operation maximum", which says
        # nothing about the cause. The cause IS printed, in the GC-bin table
        # immediately above, so point at it.
        raise SystemExit(
            f"\nno GC-matched negatives for {peaks.name} over {len(keep)} "
            f"chromosomes.\n"
            "Read the 'GC Bin / Background Count / Peak Count' table printed "
            "just above:\n"
            "  * Background Count all zero -> every candidate window was "
            "rejected, either by\n    max_n_perc (N content) or by the signal "
            "threshold, which is the 1st percentile\n    of peak signal times "
            "signal_beta. Raise this experiment's ALPHA in make_negatives.py.\n"
            "  * Background nonzero but concentrated in bins where Peak Count "
            "is zero -> the\n    peaks sit at a GC content the rest of the "
            "genome does not offer. Widen\n    gc_bin_width.\n"
            "  * Both columns near-empty -> too few input peaks, or a "
            "chromosome-naming\n    mismatch between the peaks and the FASTA."
        )
    overlap = peak_overlap_fraction(matched, loci, IN_WINDOW)
    neg_sig = median_window_signal(bigwig, matched, IN_WINDOW)
    pk_sig = median_window_signal(bigwig, loci, IN_WINDOW)
    matched.to_csv(out_path, header=False, sep="\t", index=False)
    ratio = (neg_sig / pk_sig) if pk_sig else float("nan")
    print(f"  wrote {len(matched):,} negatives "
          f"({len(matched) / max(len(loci), 1):.2f} per peak, "
          f"{overlap:.1%} of their {IN_WINDOW} bp windows overlap a peak, "
          f"median signal {neg_sig:,.0f} vs {pk_sig:,.0f} in peaks = {ratio:.1%})")


def process_experiment(exp_id: str, exp: dict, force: bool, dry_run: bool,
                       signal_filter: bool = True) -> bool:
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

        us_bw = tmp / "unstranded.bw"
        print("Building unstranded signal BigWig (plus + |minus|)...")
        make_unstranded_bw(pl_bw, mn_bw, us_bw, chrom_sizes, keep, dry_run)

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

        print(f"Matching GC-content negatives over {len(keep)} chromosomes...")
        sample_negatives(peaks_input, sequences, us_bw, out_path, keep,
                         ALPHA.get(exp_id), exp["species"], dry_run,
                         signal_filter=signal_filter)

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
        "--no-signal-filter",
        action="store_true",
        help="drop the signal restriction on candidate background windows "
             "(passes bigwig=None to extract_matching_loci). GC matching, the "
             "N-content filter and peak-tile masking still apply. For dense "
             "genomes where the surviving tile count, not signal, is what binds",
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
        process_experiment, force=args.force, dry_run=args.dry_run,
        signal_filter=not args.no_signal_filter,
    )

    # Build every .fai serially first. It used to be created inside the worker
    # under a threading.Lock; processes do not share that lock, so two workers
    # on the same species could race writing one index. Doing it here is also
    # cheap and idempotent.
    if not args.dry_run:
        for exp_id in exp_ids:
            ref = (experiments[exp_id].get("processed") or {}).get("sequences")
            if not ref:
                continue
            seq = REPO_ROOT / ref
            if seq.exists() and not Path(str(seq) + ".fai").exists():
                import pyfaidx
                print(f"Indexing {seq.name}...")
                pyfaidx.Faidx(str(seq))

    n_processed = n_skipped = 0
    # PROCESSES, not threads. -j was a ThreadPoolExecutor, which was fine while
    # the work happened in `bigWigToBedGraph`/`bpnet` subprocesses that release
    # the GIL. Now that the strand merge and the GC matching both run
    # in-process, threads serialise on the GIL and `-j 42` used about two cores.
    # Each experiment is independent, so one process each restores the old
    # concurrency without the old cross-environment subprocess calls.
    #
    # Note this is a genome per worker in memory, so a large -j on the
    # multi-gigabase species is memory-hungry -- the same exposure the previous
    # N-concurrent-`bpnet` model had.
    with ProcessPoolExecutor(max_workers=args.threads) as pool:
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
