#!/usr/bin/env python3
"""Split a PINTS peak file back into its unidirectional and bidirectional calls.

For stratified evaluation: `benchmark_predictions.py --loci` scores a subset, so
"are the bad loci the low-confidence unidirectional calls?" can be answered with
the same metric definition the canonical JSON uses, rather than a second copy of
it.

**The class is recoverable from the FIELD COUNT, and nothing else records it.**
`combine_peaks` is a plain `cat` of PINTS' `*unidirectional_peaks.bed` and
`*bidirectional_peaks.bed` followed by a coordinate sort -- it adds no class
column -- and the two files carry different numbers of columns. That is the same
raggedness that makes `pd.read_csv` die on this file with "Expected 6 fields in
line 2, saw 9", which is why filter_nonACGT_regions.py reads it line by line and
why this does too. Reading it with `usecols=[0,1,2]` would succeed and silently
lose the distinction.

**The field-count -> class mapping is VERIFIED, not assumed.** `--inspect` prints
the histogram with an example row per width, and the split is checked against
`experiment_stats.py`'s independently derived unidirectional percentage: the
cottons are known to be 94% (G. arboreum) and 96% (G. hirsutum) unidirectional,
so a split coming out near 50/50 means the mapping is wrong and the run aborts.

Usage:
    python src/analysis/stratify_peaks.py -e G.hirsutum-ovule_GROcap --inspect
    python src/analysis/stratify_peaks.py -e G.hirsutum-ovule_GROcap
"""

import argparse
import gzip
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from experiments import Experiment  # noqa: E402

#: Measured from the real files 2026-09-07, and NOT the way round it looks.
#: UNIDIRECTIONAL rows are the WIDE ones -- 9 fields, carrying a name, a
#: q-value and a real strand:
#:     NC_053424.1  263  278  NC_053424.1-6  0.0141816  -  108  264  26
#: BIDIRECTIONAL rows are 6 fields, carrying a confidence LABEL and TWO
#: summits, which is what a divergent pair looks like:
#:     NC_053424.1  55691  55926  Relaxed  55922  55692
#: An earlier version of this file had these swapped, on the assumption that
#: unidirectional meant plain BED6. The percentage guard below caught it.
UNIDIRECTIONAL_FIELDS = 9
BIDIRECTIONAL_FIELDS = 6

#: Column holding the confidence measure, per class. Unidirectional carries a
#: numeric q-value (smaller = more confident); bidirectional carries PINTS'
#: label, `Relaxed` or `Stringent(qval)`.
UNI_QVAL_COL = 4
BI_CONFIDENCE_COL = 3

#: Strand values that confirm a row really is the unidirectional class, rather
#: than the field count coinciding.
STRAND_COL, STRAND_VALUES = 5, {"+", "-", "."}

#: Every species in this corpus is unidirectional-dominated, so a split coming
#: out below this means the field-count reading is wrong for this file.
MIN_PLAUSIBLE_UNI_FRAC = 0.50


def open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def read_by_width(path: Path) -> dict[int, list[str]]:
    """Raw lines grouped by field count, preserving order within each group."""
    groups: dict[int, list[str]] = defaultdict(list)
    with open_maybe_gzip(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            groups[line.rstrip("\n").count("\t") + 1].append(line.rstrip("\n"))
    return groups


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-e", "--experiment", required=True)
    ap.add_argument("--outdir", type=Path, default=None,
                    help="default qc/stratified/{experiment}")
    ap.add_argument("--inspect", action="store_true",
                    help="print the field-count histogram and exit")
    args = ap.parse_args()

    try:
        exp = Experiment.load(args.experiment)
    except (KeyError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    peaks = Path(exp.peaks)
    if not peaks.exists():
        print(f"Error: no peak file at {peaks}", file=sys.stderr)
        return 1

    groups = read_by_width(peaks)
    total = sum(len(v) for v in groups.values())
    print(f"{exp.id}\n  {peaks}\n  {total:,} peaks, {len(groups)} distinct widths")
    for width in sorted(groups):
        rows = groups[width]
        label = {UNIDIRECTIONAL_FIELDS: "unidirectional",
                 BIDIRECTIONAL_FIELDS: "bidirectional"}.get(width, "UNKNOWN")
        print(f"    {width:>2} fields  {len(rows):>8,}  {len(rows)/total:>6.1%}  "
              f"({label})")
        print(f"               e.g. {rows[0][:110]}")
    if args.inspect:
        return 0

    uni = groups.get(UNIDIRECTIONAL_FIELDS, [])
    bi = groups.get(BIDIRECTIONAL_FIELDS, [])
    unknown = {w: len(r) for w, r in groups.items()
               if w not in (UNIDIRECTIONAL_FIELDS, BIDIRECTIONAL_FIELDS)}
    if unknown:
        print(f"\nERROR: field widths {unknown} match neither class. Re-run "
              f"--inspect and identify them before splitting.", file=sys.stderr)
        return 1

    uni_frac = len(uni) / total if total else 0.0
    if uni_frac < MIN_PLAUSIBLE_UNI_FRAC:
        print(f"\nERROR: only {uni_frac:.1%} of rows are being read as "
              f"unidirectional ({UNIDIRECTIONAL_FIELDS} fields). Every species "
              f"in this corpus is unidirectional-dominated (the cottons are "
              f"94-96%), so the field-count reading is wrong for this file. "
              f"Re-run with --inspect and check which width is which.",
              file=sys.stderr)
        return 1

    # Structural confirmation, independent of the percentages: a unidirectional
    # row must carry a real strand.
    strands = {r.split("\t")[STRAND_COL] for r in uni[:1000]} if uni else set()
    if uni and not strands <= STRAND_VALUES:
        print(f"\nERROR: column {STRAND_COL} of the "
              f"{UNIDIRECTIONAL_FIELDS}-field rows holds {sorted(strands)[:5]}, "
              f"not a strand. The layout has changed; re-run --inspect.",
              file=sys.stderr)
        return 1

    groups_out: dict[str, list[str]] = {"uni": uni, "bi": bi}

    # Unidirectional q-value quartiles. This is where the resolution is: 96% of
    # calls are unidirectional, so a binary uni/bi split leaves the dominant
    # class unresolved. q1 is the most confident (smallest q).
    numeric = []
    for r in uni:
        try:
            numeric.append((float(r.split("\t")[UNI_QVAL_COL]), r))
        except (ValueError, IndexError):
            numeric = []
            break
    if numeric:
        numeric.sort(key=lambda t: t[0])
        n = len(numeric)
        for k in range(4):
            chunk = numeric[k * n // 4:(k + 1) * n // 4]
            groups_out[f"uniq{k + 1}"] = [r for _, r in chunk]
            print(f"    uni q{k + 1}: {len(chunk):>7,} peaks, "
                  f"qval {chunk[0][0]:.4g} to {chunk[-1][0]:.4g}")
    else:
        print(f"    column {UNI_QVAL_COL} is not numeric; skipping q-value "
              f"strata", file=sys.stderr)

    # Bidirectional confidence label, as PINTS wrote it.
    labels = Counter(r.split("\t")[BI_CONFIDENCE_COL] for r in bi)
    for label, k in labels.items():
        slug = "bi_" + "".join(c for c in label.lower() if c.isalnum())
        groups_out[slug] = [r for r in bi
                            if r.split("\t")[BI_CONFIDENCE_COL] == label]
        print(f"    bi {label!r}: {k:,} peaks -> {slug}")

    outdir = args.outdir or (REPO_ROOT / "qc" / "stratified" / exp.id)
    outdir.mkdir(parents=True, exist_ok=True)
    # Names are chosen so Path(...).name.split(".")[0] -- the stem
    # benchmark_predictions.py --loci puts in the metrics filename -- is the
    # informative part. "{exp}.uni.bed" would give a stem of just the first
    # dotted component of the experiment id.
    print()
    for name, rows in groups_out.items():
        if not rows:
            continue
        dest = outdir / f"{name}.bed"
        dest.write_text("".join(r + "\n" for r in rows))
        print(f"  wrote {dest.relative_to(REPO_ROOT)}  {len(rows):,} peaks "
              f"({len(rows)/total:.1%})")
    print(f"\n  unidirectional {uni_frac:.1%} -- compare against this "
          f"experiment's value in qc/stats/experiment_stats.tsv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
