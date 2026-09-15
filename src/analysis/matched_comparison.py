#!/usr/bin/env python3
"""Compare two experiments' per-locus profile correlation AT MATCHED DEPTH.

Test (b) for the G. hirsutum ceiling. Confidence stratification (test (a))
showed peak quality drives a large WITHIN-experiment gradient in both cottons
but does not close the gap between them -- and it cannot control for depth,
because a PINTS q-value ranks peaks within one experiment while G. hirsutum runs
816 reads/peak against G. arboreum's 1,473.

Reads the tables written by `benchmark_predictions.py --per-locus-tsv`, so the
scores are the same per-locus numbers the metrics JSON summarises rather than a
re-derivation.

Three things are reported, in the order they should be read:

  1. **Spearman(obs_counts, profile_pearson) WITHIN each experiment.** If depth
     does not predict per-locus correlation inside an experiment, it cannot
     explain a difference between two of them, and the rest of the output is
     moot. Read this first.
  2. **A binned comparison** on bin edges taken from the POOLED count
     distribution, so both experiments are scored on one scale.
  3. **A depth-standardised score** per experiment: the weighted mean of its
     per-bin medians using common weights (direct standardisation). Comparing
     these two removes the depth difference by construction, so the adjusted
     ratio against the raw ratio is the answer to test (b).

`profile_pearson` is the right response variable and the counts columns are NOT:
the profile metrics are per-locus, so subsetting or reweighting merely chooses
which loci to summarise, while `log_counts_pearson` in the JSON is a correlation
ACROSS loci and would be range-restricted by any of this.

Usage:
    python src/analysis/matched_comparison.py \\
        qc/stratified/G.arboreum-ovule_GROcap/per_locus.tsv \\
        qc/stratified/G.hirsutum-ovule_GROcap/per_locus.tsv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Bins with fewer loci than this in EITHER experiment are dropped from the
#: standardised score -- a median over a handful of loci is noise, and a bin
#: only one experiment occupies cannot contribute to a matched comparison.
MIN_BIN_N = 50


def load(path: Path, label, need) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(path, sep="\t")
    missing = set(need) - set(df.columns)
    if missing:
        raise SystemExit(
            f"{path}: missing column(s) {sorted(missing)}. Expected a table "
            f"from `benchmark_predictions.py --per-locus-tsv`; got "
            f"{list(df.columns)}"
        )
    df = df.dropna(subset=list(need))
    return label or path.parent.name, df


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv", nargs=2, type=Path,
                    help="two per-locus tables; the FIRST is the reference")
    ap.add_argument("--labels", nargs=2, default=None)
    ap.add_argument("--match-on", nargs="+", default=["obs_counts"],
                    metavar="COL",
                    help="columns to match on. One gives the depth-matched "
                         "comparison; two gives a joint cell-wise match, e.g. "
                         "`--match-on obs_counts uni_qval` for depth x "
                         "confidence (run annotate_per_locus.py first). Cells "
                         "are the cross product of each column's quantile "
                         "bins, so cell count grows fast -- lower --bins when "
                         "matching on more than one",
                    )
    ap.add_argument("--bins", type=int, default=10,
                    help="quantile bins PER COLUMN, on the pooled distribution")
    args = ap.parse_args()

    labels = args.labels or [None, None]
    need = ["profile_pearson", *args.match_on]
    (la, a), (lb, b) = (load(p, l, need) for p, l in zip(args.tsv, labels))

    from scipy.stats import spearmanr

    print("1. Does each covariate predict per-locus correlation WITHIN each "
          "experiment?")
    print(f"   {'experiment':32} {'n':>8} {'med prof_r':>11}  " +
          "  ".join(f"{'rho(' + c + ')':>18}" for c in args.match_on))
    for lab, df in ((la, a), (lb, b)):
        rhos = [spearmanr(df[c], df["profile_pearson"]).statistic
                for c in args.match_on]
        print(f"   {lab:32} {len(df):>8,} "
              f"{df['profile_pearson'].median():>11.4f}  " +
              "  ".join(f"{r:>18.3f}" for r in rhos))
    print("   A rho near 0 means that covariate cannot explain a "
          "between-experiment gap.")

    # Cells are the cross product of per-column quantile bins, with edges
    # taken from the POOLED distribution so both experiments are binned on one
    # scale. Matching on more columns controls more confounding but shrinks
    # cells, and an empty cell contributes nothing -- so coverage is reported
    # rather than assumed.
    def cell_index(df):
        idx = np.zeros(len(df), dtype=np.int64)
        for col in args.match_on:
            pooled = np.concatenate([a[col].values, b[col].values])
            edges = np.unique(np.quantile(pooled, np.linspace(0, 1, args.bins + 1)))
            k = np.clip(np.digitize(df[col], edges[1:-1]), 0, len(edges) - 2)
            idx = idx * (len(edges) - 1) + k
        return idx

    ia, ib = cell_index(a), cell_index(b)
    cells = sorted(set(ia) | set(ib))
    matched_on = " x ".join(args.match_on)
    print(f"\n2. Cell-wise on {matched_on} "
          f"({args.bins} bins per column, {len(cells)} occupied cells)")

    rows = []
    dropped_n = dropped_one = 0
    for c in cells:
        sa = a.loc[ia == c, "profile_pearson"]
        sb = b.loc[ib == c, "profile_pearson"]
        if not len(sa) or not len(sb):
            dropped_one += 1
            continue
        if len(sa) < MIN_BIN_N or len(sb) < MIN_BIN_N:
            dropped_n += 1
            continue
        rows.append((sa.median(), sb.median(), len(sa) + len(sb), c,
                     len(sa), len(sb)))

    print(f"   dropped {dropped_one} cells occupied by only one experiment, "
          f"{dropped_n} with n < {MIN_BIN_N}")
    if not rows:
        print(f"\n   No cell has >= {MIN_BIN_N} loci in BOTH experiments, "
              f"so there is nothing to match on. The covariate "
              f"distributions barely overlap -- lower --bins, or match on "
              f"fewer columns. Matching cannot rescue a comparison between "
              f"two non-overlapping populations.", file=sys.stderr)
        return 1

    if len(args.match_on) == 1:
        # One covariate: the full table is short enough to be worth printing.
        print(f"   {'cell':>6} | {la[:16]:>16} {'n':>8} | {lb[:16]:>16} "
              f"{'n':>8} | {'ratio':>6}")
        for ma, mb, _, c, na, nb in rows:
            print(f"   {c:>6} | {ma:>16.4f} {na:>8,} | {mb:>16.4f} {nb:>8,} | "
                  f"{ma/mb if mb else float('nan'):>6.2f}")
    else:
        ratios = np.array([ma / mb for ma, mb, *_ in rows if mb])
        print(f"   {len(rows)} usable cells; per-cell ratio "
              f"median {np.median(ratios):.2f}x, "
              f"IQR {np.percentile(ratios, 25):.2f}-"
              f"{np.percentile(ratios, 75):.2f}, "
              f"{(ratios > 1).mean():.0%} favour {la}")


    # Direct standardisation: common weights from the pooled bin occupancy.
    ma, mb, w = (np.array(x, dtype=float)
                 for x in zip(*[(r[0], r[1], r[2]) for r in rows]))
    w = w / w.sum()
    sa, sb = float((ma * w).sum()), float((mb * w).sum())
    raw_a = a["profile_pearson"].median()
    raw_b = b["profile_pearson"].median()
    frac = (sum(r[4] for r in rows) + sum(r[5] for r in rows)) / (len(a) + len(b))
    print(f"\n3. Standardised on {matched_on} ({len(rows)} usable cells, "
          f"covering {frac:.0%} of all loci)")
    print(f"   {la:34} raw {raw_a:.4f}  ->  standardised {sa:.4f}")
    print(f"   {lb:34} raw {raw_b:.4f}  ->  standardised {sb:.4f}")
    if raw_b and sb:
        print(f"\n   raw ratio        {raw_a / raw_b:.2f}x")
        print(f"   matched          {sa / sb:.2f}x")
        closed = 1 - (sa / sb - 1) / (raw_a / raw_b - 1) if raw_a / raw_b != 1 else 0
        print(f"   -> {matched_on} accounts for {closed:.0%} of the gap; "
              f"{1 - closed:.0%} survives matching")
    return 0


if __name__ == "__main__":
    sys.exit(main())
