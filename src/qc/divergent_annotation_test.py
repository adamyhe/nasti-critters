#!/usr/bin/env python3
"""Is a species' upstream antisense peak SHARED-NFR divergent initiation, or a
DIVERGENT GENE PAIR?

The summit-anchored panel in orientation_qc.py reports an upstream antisense
peak at ~-100 bp for every M. musculus and C. griseus experiment, which is the
expected tetrapod promoter architecture: divergent initiation from the SAME
nucleosome-free region as the main TSS. It reports the same feature for all four
C. elegans libraries, and worm is not a tetrapod -- so either worm has that
architecture (a real finding) or the feature has a mundane cause.

The mundane cause to rule out is genome compaction. C. elegans is ~100 Mb with
~20,000 genes, so head-to-head gene pairs whose SEPARATE minus-strand promoter
sits 100-200 bp away are common. Anchoring on a plus-strand summit then puts
that neighbour's sense signal in the antisense channel at exactly the offset a
divergent NFR would produce. PINTS would call such a locus `bidirectional`, so
the peak class does not separate them either.

WHAT SEPARATES THEM IS ANNOTATION. Shared-NFR divergent transcription is
largely UNANNOTATED -- it is upstream antisense RNA, not a gene. A divergent
gene pair's upstream antisense IS an annotated minus-strand gene start. So:

  (A) SIGNAL ATTRIBUTION. Of the upstream-band antisense signal, what share
      falls within +/- ANNOT_HALFWIDTH bp of an annotated opposite-strand
      protein-coding gene start? High share => gene pairs.

  (B) PARTITION. Split summits by whether such an annotated start exists in
      their upstream band, then compute the summit statistic separately. If the
      peak SURVIVES in the unpaired class, the feature is not annotation and
      shared-NFR initiation is the live reading. If it collapses, it was pairs.

(A) is quoted against a SHIFT NULL -- the identical attribution with every
annotated position moved by a fixed offset. That controls for annotation
density, which is the thing that makes a compact genome look "explained"
whatever is true: in a gene-dense genome a random position is near SOME gene
start, so a raw attribution share means nothing on its own. Read the ratio, not
the share.

THE NULL USES SEVERAL SHIFTS AND TAKES THE MEDIAN, because ONE shift can ALIAS
against gene spacing and silently return the observed value as its own null.
Caught on a fixture periodic at 2 kb queried with a 10 kb shift: every shifted
start landed on another start, so a case built to be 100% explained by
annotation reported 1.01x enrichment. Real genomes are not periodic, but
C. elegans genes average ~5 kb apart, which is the same order as any single
shift worth using -- so the aliasing risk is live exactly where this test is
aimed. The spread across shifts is reported; a wide one means the null itself
is unstable and the ratio should not be read.

Run mouse as the POSITIVE CONTROL: its peak is known to be shared-NFR, so it
must survive (B) and show a low (A) ratio. Run fly as the NEGATIVE control: it
has no peak to explain, so its numbers say what "nothing here" looks like. A
worm result is only interpretable next to both.

    python src/qc/divergent_annotation_test.py \
        -e C.elegans-embryo_GROcap C.elegans-L3_GROcap \
           C.elegans-L1starved_GROcap C.elegans-embryo-sdc2_GROcap \
           M.musculus-liver-young-female_ChROcap D.melanogaster-S2_PROcap \
        --tsv qc/orientation/divergent_annotation.tsv

Standalone by design -- it touches no DAG output and reads only what is already
on disk, so it needs no --forcerun and cannot trigger the fetch_fastq cascade.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import orientation_qc as oq  # noqa: E402  (reuse peak_maxima/annotation/statistics)

#: A summit's antisense signal is credited to an annotated gene start within
#: this distance. 50 bp is deliberately generous relative to the ~10-30 bp
#: scatter between an annotated start and the observed initiation site, so the
#: test errs toward EXPLAINING the feature -- the conservative direction for a
#: script whose interesting outcome is "annotation does not explain it".
ANNOT_HALFWIDTH = 50

#: Offsets for the density-matched null. Spread over an order of magnitude and
#: deliberately not multiples of one another, so no single gene-spacing
#: periodicity can alias against all of them.
NULL_SHIFTS = (2_300, 7_100, 13_700, 31_300, 71_900)


def annotated_starts(tss, strand_wanted):
    """{chrom: sorted array of positions} for one strand."""
    import numpy as np
    by = {}
    for chrom, pos, strand in tss:
        if strand == strand_wanted:
            by.setdefault(chrom, []).append(pos)
    return {c: np.sort(np.asarray(v, dtype=np.int64)) for c, v in by.items()}


def near_annotated(chrom, positions, starts, halfwidth):
    """Boolean mask: is each genomic position within halfwidth of a start?

    searchsorted rather than an interval tree -- the arrays are per-chromosome
    and already sorted, and the query is one nearest-neighbour distance.
    """
    import numpy as np
    arr = starts.get(chrom)
    if arr is None or arr.size == 0:
        return np.zeros(positions.shape, dtype=bool)
    i = np.searchsorted(arr, positions)
    left = np.clip(i - 1, 0, arr.size - 1)
    right = np.clip(i, 0, arr.size - 1)
    d = np.minimum(np.abs(positions - arr[left]), np.abs(positions - arr[right]))
    return d <= halfwidth


def run(exp, tss, flank, shifts, halfwidth):
    """One pass over the bigWigs, accumulating everything both tests need."""
    import numpy as np
    import pandas as pd
    import pybigtools

    peaks = pd.read_csv(exp.peaks, sep="\t", usecols=[0, 1, 2], header=None,
                        names=["chrom", "start", "end"], dtype={"chrom": str},
                        comment="#")
    sites = oq.peak_maxima(list(peaks.itertuples(index=False, name=None)),
                           exp.signals[0], exp.signals[1], None)

    # The antisense channel is the strand OPPOSITE the summit's, so the
    # annotated start that would explain it is also on the opposite strand.
    starts = {"+": annotated_starts(tss, "+"), "-": annotated_starts(tss, "-")}
    shifted = [{s: {c: a + k for c, a in d.items()} for s, d in starts.items()}
               for k in shifts]

    x = np.arange(-flank, flank)
    band_up = (x <= -oq.SUMMIT_ANTI_MIN_OFFSET) & (x >= -oq.SUMMIT_ANTI_MAX_OFFSET)
    band_dn = (x >= oq.SUMMIT_ANTI_MIN_OFFSET) & (x <= oq.SUMMIT_ANTI_MAX_OFFSET)

    pl = pybigtools.open(str(exp.signals[0]))
    mn = pybigtools.open(str(exp.signals[1]))
    sizes = pl.chroms()
    acc = {k: np.zeros(2 * flank) for k in
           ("sense_paired", "anti_paired", "sense_unpaired", "anti_unpaired")}
    n = {"paired": 0, "unpaired": 0, "paired_upstream": 0}
    attr = {"annot": 0.0, "total": 0.0, "null": [0.0] * len(shifts)}

    for chrom, pos, strand in sites:
        if chrom not in sizes:
            continue
        s, e = pos - flank, pos + flank
        if s < 0 or e > sizes[chrom]:
            continue                      # partial window: drop, never pad
        a = np.asarray(pl.values(chrom, s, e, fillna=0), dtype=float)
        b = np.abs(np.asarray(mn.values(chrom, s, e, fillna=0), dtype=float))
        if strand == "+":
            sense, anti = a, b
            opp = "-"
        else:
            sense, anti = b[::-1], a[::-1]
            opp = "+"

        # (B) partition. Genomic coordinates of the band, which for a
        # minus-strand summit runs the other way -- hence the flip above and
        # the sign here.
        off_up = x[band_up]
        gpos_up = pos + (off_up if strand == "+" else -off_up)
        mask_up = near_annotated(chrom, gpos_up, starts[opp], halfwidth)
        off_dn = x[band_dn]
        gpos_dn = pos + (off_dn if strand == "+" else -off_dn)
        mask_dn = near_annotated(chrom, gpos_dn, starts[opp], halfwidth)

        # THE PARTITION MUST BE SYMMETRIC. Defining "paired" on the UPSTREAM
        # side alone selects the complement to be downstream-biased: a summit
        # whose opposite-strand neighbour sits DOWNSTREAM fails the upstream
        # test, lands in "unpaired", and carries its downstream antisense in
        # with it. Measured on the real corpus, that drove one experiment's
        # unpaired class to log2 -2.54 with a peak at +164 bp -- enough to trip
        # the strand-swap flag -- where its whole-set value was about -0.4. The
        # artefact was entirely this asymmetry. So "unpaired" now means NO
        # annotated opposite-strand start anywhere in the band, which is
        # direction-neutral and is the only complement the two hypotheses can
        # be compared across.
        key = "paired" if (mask_up.any() or mask_dn.any()) else "unpaired"
        n[key] += 1
        if mask_up.any():
            n["paired_upstream"] += 1
        acc[f"sense_{key}"] += sense
        acc[f"anti_{key}"] += anti

        # (A) attribution stays UPSTREAM-only: the quantity being explained is
        # upstream antisense signal, so the upstream mask is the right one here
        # even though the partition above is symmetric.
        w = anti[band_up]
        attr["total"] += float(w.sum())
        attr["annot"] += float(w[mask_up].sum())
        for i, sh in enumerate(shifted):
            attr["null"][i] += float(
                w[near_annotated(chrom, gpos_up, sh[opp], halfwidth)].sum())

    pl.close(); mn.close()

    # Same normalisation as the panel: each site by its own window total, so one
    # deep locus cannot carry the average. Done after accumulation would be
    # wrong, so accumulate raw and report the partition metaplots as means --
    # acceptable here because the question is the SHAPE per class, and both
    # classes are large.
    out = {"n_sites": len(sites), "n_paired": n["paired"],
           "n_unpaired": n["unpaired"],
           "n_paired_upstream": n["paired_upstream"]}
    for key in ("paired", "unpaired"):
        if n[key] == 0:
            out[key] = None
            continue
        sense = acc[f"sense_{key}"] / n[key]
        anti = acc[f"anti_{key}"] / n[key]
        out[key] = oq.summit_notes(sense, anti, flank)
    out["attr"] = attr
    return out


#: Below this share of summits, the unpaired class is too small to carry a
#: metaplot and the partition is INCONCLUSIVE rather than negative. This is the
#: expected failure mode in a gene-dense genome: if almost every summit has an
#: annotated opposite-strand start in its band, there is no unannotated class
#: left to test, and an empty class must never be reported as evidence.
MIN_UNPAIRED_FRAC = 0.10
#: log2 the unpaired class must reach for its upstream weight to count as
#: surviving. Half the SUMMIT_SWAP_LOG2 magnitude, i.e. deliberately lenient --
#: the question is whether the feature is still THERE, not how strong it is.
SURVIVES_LOG2 = 0.5
#: A "peak" nearer the anchor than this is not upstream in any useful sense --
#: it is the summit's own footprint. Matches SUMMIT_ANTI_MIN_OFFSET, the band's
#: inner bound, so the two cannot drift.
SUMMIT_UPSTREAM_MIN = oq.SUMMIT_ANTI_MIN_OFFSET


def read_partition(r):
    """One-line reading of the partition test. A READING, not a proof."""
    import re
    n, un = r["n_sites"], r["n_unpaired"]
    if not n:
        return "no usable summits"
    frac = un / n
    if frac < MIN_UNPAIRED_FRAC or r["unpaired"] is None:
        return (f"INCONCLUSIVE: only {frac:.1%} of summits lack an annotated "
                "opposite-strand start in the band, so there is no unannotated "
                "class to test. Expected in a gene-dense genome; it means the "
                "partition cannot separate the two hypotheses here, NOT that "
                "annotation explains the feature.")
    note = r["unpaired"][0]
    lr = re.search(r"log2 ratio ([+-][\d.]+)", note)
    # `[+-]?`, NOT `-?`: summit_notes formats the offset with `{at:+d}`, so a
    # POSITIVE position always carries a leading '+' and `-?\d+` never matches
    # it. That silently reported "no localized peak" for a peak at +27 bp, and
    # would have let a downstream peak be announced as "the upstream peak
    # SURVIVES at +200 bp" -- hence the sign test below, which was also absent.
    pk = re.search(r"peak at ([+-]?\d+) bp", note)
    lr = float(lr.group(1)) if lr else 0.0
    at = int(pk.group(1)) if pk else None

    if at is not None and at <= -SUMMIT_UPSTREAM_MIN and lr >= SURVIVES_LOG2:
        return (f"SHARED-NFR IS LIVE: the upstream peak SURVIVES at {at:+d} bp "
                f"(log2 {lr:+.2f}) across the {un:,} summits with NO annotated "
                "opposite-strand start in the band, so annotation does not "
                "explain it.")
    if at is not None and at > -SUMMIT_UPSTREAM_MIN:
        return (f"NOT A DIVERGENT NFR: the unpaired class DOES have a localized "
                f"antisense peak, but at {at:+d} bp -- on or downstream of the "
                f"summit, not upstream (log2 {lr:+.2f} over the band). Antisense "
                "sitting on the anchor is the peak's own footprint or an "
                f"unresolved close pair, not divergent initiation. {un:,} summits.")
    if lr >= SURVIVES_LOG2:
        return (f"PARTLY SURVIVES: upstream weight holds (log2 {lr:+.2f}) in the "
                f"{un:,} unannotated summits but with no localized peak -- "
                "diffuse upstream antisense, not a divergent NFR.")
    return (f"GENE PAIRS: the upstream feature COLLAPSES (log2 {lr:+.2f}, "
            "no peak) once the "
            f"{un:,} summits with no annotated partner are taken alone, so the "
            "signal was tracking annotated divergent genes.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-e", "--experiments", nargs="+", action="extend", default=[])
    ap.add_argument("--flank", type=int, default=500)
    ap.add_argument("--halfwidth", type=int, default=ANNOT_HALFWIDTH)
    ap.add_argument("--shift", type=int, nargs="+", action="extend",
                    default=[], metavar="BP",
                    help=f"offsets for the density-matched null "
                         f"(default {', '.join(f'{s:,}' for s in NULL_SHIFTS)}); "
                         "several are used and the median taken, because one "
                         "can alias against gene spacing")
    ap.add_argument("--annotation", type=Path, default=None)
    ap.add_argument("--all-biotypes", action="store_true",
                    help="do NOT restrict to protein-coding (dilutes worm ~47%%)")
    ap.add_argument("--tsv", type=Path, default=None)
    ap.add_argument("--reverdict", type=Path, default=None, metavar="TSV",
                    help="recompute the verdict column from an existing TSV and "
                         "exit. Every input to read_partition() is stored there, "
                         "so a corrected reading costs no second bigWig pass.")
    args = ap.parse_args()

    if args.reverdict:
        with open(args.reverdict) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in rows:
            r = {"n_sites": int(row["n_sites"]),
                 "n_unpaired": int(row["n_unpaired"]),
                 "unpaired": [row["notes_unpaired"]] if row["notes_unpaired"] else None,
                 "paired": [row["notes_paired"]] if row["notes_paired"] else None}
            new = read_partition(r)
            flag = "" if new == row.get("verdict", "") else "   [CHANGED]"
            print(f"\n{row['experiment']} ({row['species']}){flag}\n  => {new}")
            row["verdict"] = new
        with open(args.reverdict, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader(); w.writerows(rows)
        print(f"\nrewrote {args.reverdict}")
        return

    if not args.experiments:
        ap.error("pass -e EXPERIMENT ... (or --reverdict TSV)")
    from experiments import Experiment
    import numpy as np
    import yaml
    shifts = tuple(args.shift) or NULL_SHIFTS
    with open(REPO_ROOT / "config" / "genomes.yaml") as f:
        genomes = yaml.safe_load(f)["species"]
    # No logomaker/matplotlib: this script draws nothing.
    oq.require_deps(("pybigtools", "numpy", "pandas"))

    rows = []
    for exp_id in args.experiments:
        exp = Experiment.load(exp_id)
        gaps = exp.missing_paths(Experiment.QC_INPUTS)
        if gaps:
            print(f"SKIP {exp_id}: missing {gaps[0]}")
            continue
        genome = genomes[exp.species]
        if genome.get("annotation_url") is None and args.annotation is None:
            print(f"SKIP {exp_id}: no annotation for {exp.species} -- this test "
                  "is annotation-based and cannot run without one")
            continue
        print(f"\n=== {exp_id} ({exp.species})")
        tss = oq.load_annotation(exp.species, genome,
                                 REPO_ROOT / "data" / "annotation",
                                 args.annotation,
                                 coding_only=not args.all_biotypes)
        print(f"  annotated starts: {len(tss):,} "
              f"({'all biotypes' if args.all_biotypes else 'protein-coding only'})")
        r = run(exp, tss, args.flank, shifts, args.halfwidth)
        a = r["attr"]
        tot = a["total"]
        share = a["annot"] / tot if tot else float("nan")
        nulls = sorted(n / tot for n in a["null"]) if tot else [float("nan")]
        null = float(np.median(nulls))
        # 0/0 is UNDEFINED, not infinite. Getting this wrong inverts the
        # interesting result: the fixture where annotation explains NOTHING has
        # share 0 and null 0, and reporting "inf" there reads as "maximally
        # explained" -- the exact opposite of the truth.
        if null > 0:
            ratio = share / null
        elif share > 0:
            ratio = float("inf")
        else:
            ratio = float("nan")
        print(f"  sites used: {r['n_sites']:,}  "
              f"paired {r['n_paired']:,} / unpaired {r['n_unpaired']:,} "
              f"({100 * r['n_paired'] / max(r['n_sites'], 1):.1f}% paired; "
              f"{r['n_paired_upstream']:,} of the paired have the partner "
              "UPSTREAM)")
        print(f"  (A) upstream antisense within {args.halfwidth} bp of an "
              f"annotated opposite-strand start: {share:.1%}")
        # Precomputed, NOT inlined into the f-string: a multi-line f-string
        # expression is PEP 701, i.e. Python 3.12+, and this repo's env is
        # 3.11. It parsed locally on 3.12 and was a SyntaxError on the cluster.
        # Compile-check with `/usr/bin/python3 -c "import ast,sys;
        # ast.parse(open(sys.argv[1]).read())" <file>` against an OLDER
        # interpreter, not the newest one available.
        enrich = ("undefined (no signal attributable either way)"
                  if ratio != ratio else f"{ratio:.2f}x")
        print(f"      shift null: median {null:.1%} "
              f"(range {nulls[0]:.1%}-{nulls[-1]:.1%} over {len(nulls)} shifts)"
              f"   ENRICHMENT {enrich}")
        if null and (nulls[-1] - nulls[0]) > 0.5 * null:
            print("      WARNING: the null spans more than half its own median, "
                  "so it is unstable here -- read (B), not this ratio")
        for key in ("paired", "unpaired"):
            print(f"  (B) {key}:")
            if r[key] is None:
                print("        no sites in this class")
                continue
            for line in r[key]:
                print(f"        {line}")
            if "<--" in " ".join(r[key]):
                print("        NOTE: that flag came from summit_notes, which is "
                      "written for a WHOLE experiment. On a partition subset it "
                      "is not an experiment-level verdict -- read it as a "
                      "property of this class only, never as a reason to "
                      "re-map.")
        print(f"  => {read_partition(r)}")
        rows.append({
            "experiment": exp_id, "species": exp.species,
            "n_sites": r["n_sites"], "n_paired": r["n_paired"],
            "n_unpaired": r["n_unpaired"],
            "n_paired_upstream": r["n_paired_upstream"],
            "pct_paired": round(100 * r["n_paired"] / max(r["n_sites"], 1), 1),
            "attr_share": round(share, 4), "attr_null": round(null, 4),
            "attr_null_min": round(nulls[0], 4), "attr_null_max": round(nulls[-1], 4),
            "attr_enrichment": round(ratio, 3),
            "verdict": read_partition(r),
            "notes_paired": " | ".join(r["paired"] or []),
            "notes_unpaired": " | ".join(r["unpaired"] or []),
        })

    if args.tsv and rows:
        args.tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.tsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.tsv}")


if __name__ == "__main__":
    main()
