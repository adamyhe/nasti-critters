"""Filter attribution loci to windows that are pure ACGT, and optionally
one-hot encode them.

The one-hot encoding lives here rather than in attribute.py, where `--save-ohe`
used to be, because it is a property of the LOCUS SET and not of any model: it
depends only on (loci, sequences, in_window). Written from attribute.py it was
recomputed and rewritten on every attribution run -- once per head, per
reference mode, per experiment -- all producing the same array.

**This script is standalone — nothing invokes it.** Not attribute.py, not the
launchers, not the Snakemake DAG. It is a preprocessing step you run by hand,
and its output only reaches attribution if you point `attribute.py --loci` at
the filtered BED. Without that flag attribute.py reads `exp.peaks` and this
script's output has no reader at all.

**It is not redundant with `extract_loci(ignore=...)`, which does something
different.** `ignore` sets an all-zero column at a non-ACGT position and KEEPS
the locus (verified: a window containing one N comes back as 1 row with exactly
1 blank column). This script DROPS the locus. So use it where a blanked position
would be a problem — variant-effect loci, say, where the original `snp_bed`
naming came from — and skip it where zero columns are acceptable.

**Given `--loci`, the rows line up with the output BED and with attribute.py's
X.** Three things make that true and all three have to hold:

* the OHE is built by the same `tangermeme.io.extract_loci` call attribute.py
  makes, not by a hand-rolled encoder, so the channel order and `ignore`
  handling cannot drift;
* `extract_loci` preserves input row order -- its chromosome filter is
  `df[numpy.isin(df['chrom'], chroms)]`, a boolean mask (checked against
  tangermeme 1.4.1);
* attribute.py passes `chroms=` the union of every fold's test chromosomes,
  which covers all of `main_chromosomes` -- `config/write_split_csvs.py --check`
  fails if any main chromosome is in no fold -- so for a peak set already
  restricted to `main_chromosomes` that filter drops nothing.

If you ever attribute over a locus set whose chromosomes are NOT all in the fold
assignment, the OHE written here will be longer than the attributions. Nothing
detects that automatically.
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx
import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from experiments import IGNORE, load_params  # noqa: E402


def _open_text(path):
    """Open a BED that may or may not be bgzipped."""
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def filter_nonACGT_regions(bed_fp, fa_fp, in_window=2114, verbose=False):
    """Drop loci whose `in_window` window contains a non-ACGT base.

    Returns `(kept_lines, coords)`: the surviving input lines VERBATIM, and a
    BED3 DataFrame of their coordinates for save_ohe(). Both are in input order
    and the same length.

    **Read line by line rather than with pandas, because a PINTS peak file is
    RAGGED.** `combine_peaks` concatenates the unidirectional and bidirectional
    calls, which carry different numbers of columns, so
    `pd.read_csv(bed_fp, sep="\t", header=None)` dies with
    `Expected 6 fields in line 2, saw 9`. Adding `usecols=[0, 1, 2]` fixes the
    read -- that is why load_bed() elsewhere in this repo is unaffected -- but
    would be wrong here, because this function's output is written back out as a
    BED and upstream deliberately keeps PINTS' strand/confidence/class/summit
    columns rather than cutting to BED3; the summit column is what makes
    `extract_loci(summits=True)` possible. Keeping the raw line preserves every
    column whatever their number, and sidesteps pandas' ragged-file handling.

    Coordinate arithmetic matches
    procap-atlas/src/preprocess/_filter_nonACGT_regions.py: pyfaidx (0-based,
    exclusive end, as tangermeme uses), and `end` derived from the CLAMPED start
    so a region near a contig start is shifted rather than dropped for being
    short.
    """
    with _open_text(bed_fp) as fh:
        lines = [line.rstrip("\n") for line in fh
                 if line.strip() and not line.startswith(("#", "track", "browser"))]

    fa = pyfaidx.Fasta(fa_fp)
    chroms = set(fa.keys())
    kept, coords = [], []
    for line in tqdm.tqdm(lines, disable=not verbose):
        fields = line.split("\t")
        # str() on the chromosome for the same reason load_bed forces dtype:
        # A.thaliana, C.reinhardtii and P.patens name chromosomes with digits.
        chrom, lo, hi = str(fields[0]), int(fields[1]), int(fields[2])
        center = (lo + hi) // 2
        start = max(0, center - in_window // 2)
        end = start + in_window
        if chrom in chroms:
            seq = str(fa[chrom][start:end]).upper()
            ok = len(seq) == in_window and all(c in "ACGT" for c in seq)
        else:
            ok = False
        if ok:
            kept.append(line)
            coords.append((chrom, lo, hi))

    print(
        f"Filtered out {len(lines) - len(kept)} of {len(lines)} due to non-ACGT "
        f"characters, length != {in_window}, or invalid chromosome."
    )
    return kept, pd.DataFrame(coords, columns=["chrom", "start", "end"])


def save_ohe(loci, fa_fp, out_fp, in_window=2114, out_window=1000, verbose=False):
    """One-hot encode `loci` and write a compressed npz.

    Uses extract_loci rather than encoding by hand, so this is byte-identical to
    the X attribute.py builds. uint8 because the array is one-hot: float32 would
    be 4x the size for no information.
    """
    import torch
    from tangermeme.io import extract_loci

    # tangermeme's DataFrame branch is `df.iloc[:, cols].copy()` and does NOT
    # rename the columns, while the rest of extract_loci addresses them by name
    # ('chrom'/'start'/'end'). A BED read with header=None has integer column
    # labels, so normalise here rather than relying on which code paths happen
    # to avoid name-based access. Also drops any extra BED columns.
    if isinstance(loci, pd.DataFrame):
        loci = loci.iloc[:, :3].copy()
        loci.columns = ["chrom", "start", "end"]
        loci["chrom"] = loci["chrom"].astype(str)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tangermeme_compat import patch_numeric_chroms

    # Numeric chromosome names (A.thaliana 1-5, C.reinhardtii 1-17,
    # P.patens 1-27) hit a dtype bug in tangermeme's BED reading. Self-retiring
    # no-op once tangermeme is fixed -- see src/tangermeme_compat.py.
    patch_numeric_chroms(verbose=verbose)

    X = extract_loci(
        loci=loci,
        sequences=fa_fp,
        in_window=in_window,
        out_window=out_window,
        max_jitter=0,
        verbose=verbose,
        min_counts=None,
        max_counts=None,
        ignore=IGNORE,
    ).to(torch.float32)

    Path(out_fp).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_fp, X.to(torch.uint8).numpy())
    print(f"Wrote {X.shape[0]:,} x {X.shape[1]} x {X.shape[2]} one-hot to {out_fp}")
    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-b", "--bed_fp", type=str, required=True)
    parser.add_argument("-f", "--fa_fp", type=str, required=True)
    parser.add_argument("-o", "--out_fp", type=str, required=True)
    parser.add_argument("-w", "--in_window", type=int, default=2114)
    parser.add_argument(
        "--save-ohe", type=str, default=None, metavar="PATH",
        help="also write the one-hot encoding of the FILTERED loci here (npz, "
             "uint8). Moved from attribute.py --save-ohe: the encoding depends "
             "only on (loci, sequences, in_window), so producing it once here "
             "beats rewriting the same array on every attribution run. Rows "
             "correspond 1:1 to the output BED -- see the module docstring for "
             "why that also matches attribute.py's X",
    )
    parser.add_argument(
        "--out-window", type=int, default=None,
        help="out_window for --save-ohe only (default: from "
             "config/bpnet_params.json). Does not affect X, which is sequence, "
             "but is passed so the extract_loci call matches attribute.py's",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    kept_lines, coords = filter_nonACGT_regions(
        args.bed_fp, args.fa_fp, args.in_window, args.verbose
    )
    Path(args.out_fp).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_fp, "w") as fh:
        fh.write("".join(line + "\n" for line in kept_lines))

    if args.save_ohe is not None:
        out_window = args.out_window
        if out_window is None:
            out_window = load_params("bpnet")["out_window"]
        save_ohe(coords, args.fa_fp, args.save_ohe,
                 in_window=args.in_window, out_window=out_window,
                 verbose=args.verbose)
