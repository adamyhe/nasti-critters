#!/usr/bin/env python3
"""Join PINTS class and confidence onto a per-locus benchmark table.

`benchmark_predictions.py --per-locus-tsv` writes coordinates, so the peak
file's own columns can be joined back on rather than recomputed -- which is the
whole reason it writes coordinates. Adds `pints_class` and `pints_conf`:

    pints_class  "uni" or "bi", from the field count
    pints_conf   the unidirectional q-value (numeric, smaller = more confident)
                 or the bidirectional label (`Relaxed` / `Stringent(qval)`)

The class/field-count mapping and the readers come from stratify_peaks, imported
rather than reimplemented so the two cannot disagree about which width is which
-- that mapping is NOT the way round it looks (unidirectional is the 9-field
row) and has been got wrong once already.

The join is on (chrom, start, end) exactly. Coordinates in the per-locus table
are columns 0-2 of the peak file as `extract_loci` interleaved them, so they are
the peak file's own start/end, unmodified. Peaks whose coordinates are not
unique are dropped from the annotation rather than joined arbitrarily, and the
count is reported.

Usage:
    python src/analysis/annotate_per_locus.py -e G.hirsutum-ovule_GROcap \\
        qc/stratified/G.hirsutum-ovule_GROcap/per_locus.tsv
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import Experiment  # noqa: E402
from stratify_peaks import (  # noqa: E402
    BI_CONFIDENCE_COL,
    BIDIRECTIONAL_FIELDS,
    UNI_QVAL_COL,
    UNIDIRECTIONAL_FIELDS,
    read_by_width,
)


def build_annotation(peaks_path: Path):
    """(chrom, start, end) -> (class, confidence), or None where ambiguous.

    Extracted from main() so it can be exercised directly against a fixture;
    the coordinate join is the part that can silently go wrong.
    """
    groups = read_by_width(peaks_path)
    annot = {}
    dupes = 0
    for width, rows in groups.items():
        if width == UNIDIRECTIONAL_FIELDS:
            cls, conf_col = "uni", UNI_QVAL_COL
        elif width == BIDIRECTIONAL_FIELDS:
            cls, conf_col = "bi", BI_CONFIDENCE_COL
        else:
            raise ValueError(
                f"field width {width} matches neither class; run "
                f"stratify_peaks.py --inspect")
        for r in rows:
            f = r.split("\t")
            key = (f[0], int(f[1]), int(f[2]))
            if key in annot:
                dupes += 1
                annot[key] = None          # ambiguous; dropped on join
                continue
            annot[key] = (cls, f[conf_col])
    return annot, dupes


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv", type=Path, help="a --per-locus-tsv table")
    ap.add_argument("-e", "--experiment", required=True)
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="default: alongside the input, as *_annotated.tsv")
    args = ap.parse_args()

    try:
        exp = Experiment.load(args.experiment)
    except (KeyError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    table = pd.read_csv(args.tsv, sep="\t", dtype={"chrom": str})
    for col in ("chrom", "start", "end"):
        if col not in table.columns:
            print(f"Error: {args.tsv} has no {col!r} column; expected a "
                  f"--per-locus-tsv table", file=sys.stderr)
            return 1

    try:
        annot, dupes = build_annotation(Path(exp.peaks))
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    ambiguous = {k for k, v in annot.items() if v is None}
    keys = list(zip(table["chrom"], table["start"], table["end"]))
    table["pints_class"] = [annot.get(k, (None, None))[0]
                            if annot.get(k) else None for k in keys]
    table["pints_conf"] = [annot.get(k, (None, None))[1]
                           if annot.get(k) else None for k in keys]

    matched = table["pints_class"].notna().sum()
    print(f"{exp.id}")
    print(f"  peak file  {exp.peaks}")
    print(f"  {len(annot):,} coordinate keys "
          f"({len(ambiguous):,} ambiguous, from {dupes:,} duplicate rows)")
    print(f"  joined {matched:,} / {len(table):,} loci "
          f"({matched/len(table):.2%})")
    print(f"  classes: {dict(Counter(table['pints_class'].dropna()))}")
    if matched < len(table):
        print(f"  {len(table) - matched:,} loci did not join -- ambiguous "
              f"coordinates, or a peak file that has changed since the "
              f"benchmark ran", file=sys.stderr)
    if matched == 0:
        print("  ERROR: nothing joined. The per-locus table and the peak file "
              "do not describe the same loci.", file=sys.stderr)
        return 1

    # Numeric confidence for the unidirectional class only; the bidirectional
    # label has no numeric equivalent, so it stays in pints_conf.
    uni = table["pints_class"] == "uni"
    table["uni_qval"] = pd.to_numeric(table["pints_conf"].where(uni),
                                      errors="coerce")
    n_q = table["uni_qval"].notna().sum()
    print(f"  uni_qval numeric for {n_q:,} loci "
          f"({n_q/max(uni.sum(),1):.1%} of unidirectional)")

    out = args.output or args.tsv.with_name(args.tsv.stem + "_annotated.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, sep="\t", index=False, float_format="%.6g")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
