#!/usr/bin/env python3
"""Collate BPNet and Cherimoya benchmark metrics and plot them against each other.

Reads `performance_metrics/{family}/{experiment}.json` -- written by
`benchmark_predictions.py` and `benchmark_cherimoya.py`, which share a schema --
inner-joins on experiment, writes one collated TSV, and produces one figure per
metric: a scatter with a y=x reference line and a histogram of per-experiment
deltas (Cherimoya minus BPNet), with a Wilcoxon signed-rank test on the pairs.

Ported from procap-atlas/src/analysis/compare_bpnet_cherimoya.py, with two
changes this repo forces:

* **Points are coloured by SPECIES, not by read depth.** Upstream is human-only,
  so depth is its only axis of variation; here the interesting question is
  whether one architecture wins uniformly or only on some clades, and a
  12-species corpus makes that visible. `--colour-by depth` restores the
  upstream view, reading `signal_reads` from qc/stats/experiment_stats.tsv.
* **No consolidate step.** Upstream joins two pre-consolidated TSVs; there are
  42 experiments here, so reading the per-experiment JSONs directly removes a
  stage that could go stale against them.

Cherimoya is not deployment-ready (see src/cherimoya/README.md), so read these
as a development comparison rather than a result.

Usage:
    python src/analysis/compare_bpnet_cherimoya.py
    python src/analysis/compare_bpnet_cherimoya.py --per-fold
    python src/analysis/compare_bpnet_cherimoya.py --metrics profile_jsd
    python src/analysis/compare_bpnet_cherimoya.py --colour-by depth
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from experiments import Experiment, experiment_ids  # noqa: E402

METRICS_DIR = REPO_ROOT / "performance_metrics"
STATS_TSV = REPO_ROOT / "qc" / "stats" / "experiment_stats.tsv"
DEFAULT_OUT_DIR = REPO_ROOT / "plots" / "bpnet_vs_cherimoya"

#: `lower_is_better` sets the delta sign convention (cherimoya - bpnet) and
#: which side of the diagonal counts as a Cherimoya win.
METRIC_INFO = {
    "profile_pearson": {"label": "profile Pearson", "lower_is_better": False},
    "profile_jsd": {"label": "profile JSD", "lower_is_better": True},
    "log_counts_pearson": {"label": "log-counts Pearson", "lower_is_better": False},
    "counts_spearman": {"label": "counts Spearman", "lower_is_better": False},
}


def load_family(family: str, per_fold: bool) -> pd.DataFrame:
    """One row per experiment (or per experiment x fold) for one model family."""
    rows = []
    for path in sorted((METRICS_DIR / family).glob("*.json")):
        with open(path) as f:
            payload = json.load(f)
        exp_id = payload.get("run_name", path.stem)
        if per_fold:
            for fold, metrics in payload.get("per_fold", {}).items():
                rows.append({"experiment": exp_id, "fold": int(fold), **metrics})
        else:
            rows.append({"experiment": exp_id, "fold": -1,
                         **payload.get("genome_wide", {})})
    return pd.DataFrame(rows)


def add_species(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each experiment's species, for colouring and for reading the table."""
    known = set(experiment_ids())
    species = {}
    for exp_id in df["experiment"].unique():
        if exp_id not in known:
            species[exp_id] = "unknown"
            continue
        try:
            species[exp_id] = Experiment.load(exp_id).species
        except KeyError:
            species[exp_id] = "unknown"
    return df.assign(species=df["experiment"].map(species))


def add_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Attach signal_reads from the QC table, if it has been built."""
    if not STATS_TSV.exists():
        return df.assign(signal_reads=pd.NA)
    stats = pd.read_csv(STATS_TSV, sep="\t")
    depth = dict(zip(stats["experiment"], pd.to_numeric(stats["signal_reads"],
                                                        errors="coerce")))
    return df.assign(signal_reads=df["experiment"].map(depth))


def plot_metric(df, metric, info, out_path, colour_by):
    """Scatter + delta histogram for one metric. Returns the summary row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import wilcoxon

    a, b = df[f"{metric}_bpnet"], df[f"{metric}_cherimoya"]
    delta = b - a
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))

    ax = axes[0]
    if colour_by == "depth" and df["signal_reads"].notna().any():
        sc = ax.scatter(a, b, c=np.log10(df["signal_reads"].astype(float)),
                        cmap="viridis", s=26, edgecolor="none")
        fig.colorbar(sc, ax=ax, label="log10 signal reads")
    else:
        for sp, grp in df.groupby("species", sort=True):
            ax.scatter(grp[f"{metric}_bpnet"], grp[f"{metric}_cherimoya"],
                       s=26, label=sp, edgecolor="none")
        ax.legend(fontsize=6, frameon=False, ncol=2)
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    pad = 0.04 * (hi - lo or 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, zorder=0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"BPNet {info['label']}")
    ax.set_ylabel(f"Cherimoya {info['label']}")
    better = "below" if info["lower_is_better"] else "above"
    ax.set_title(f"{info['label']} (Cherimoya better {better} the diagonal)")

    ax = axes[1]
    ax.hist(delta, bins=max(8, min(30, len(delta) // 2)), color="0.4")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel(f"delta {info['label']} (Cherimoya - BPNet)")
    ax.set_ylabel("experiments")

    # Wilcoxon needs at least one nonzero difference and is meaningless on a
    # couple of pairs; report it as unavailable rather than emitting a p-value
    # nobody should read.
    p = float("nan")
    if len(delta) >= 6 and (delta != 0).any():
        p = float(wilcoxon(a, b).pvalue)
    n_better = int((delta < 0).sum() if info["lower_is_better"] else (delta > 0).sum())
    ax.set_title(f"n={len(delta)}, Cherimoya better in {n_better}"
                 + ("" if pd.isna(p) else f", Wilcoxon p={p:.2g}"))

    fig.suptitle(f"BPNet vs Cherimoya: {info['label']}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"metric": metric, "n": len(delta), "n_cherimoya_better": n_better,
            "median_delta": float(delta.median()),
            "mean_delta": float(delta.mean()), "wilcoxon_p": p}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--metrics", nargs="+", default=None, choices=list(METRIC_INFO),
        help="metrics to plot (default: all four shared by both families)",
    )
    parser.add_argument(
        "--per-fold", action="store_true",
        help="compare per-fold values instead of the genome-wide block. More "
             "points, but they are not independent -- folds of one experiment "
             "share an architecture, a library and a peak set -- so the "
             "Wilcoxon test is only interpretable per experiment",
    )
    parser.add_argument("--colour-by", choices=("species", "depth"),
                        default="species")
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tsv", type=Path, default=None,
                        help="collated table (default: <out-dir>/collated.tsv)")
    args = parser.parse_args()

    bpnet = load_family("bpnet", args.per_fold)
    cherimoya = load_family("cherimoya", args.per_fold)
    for name, df in (("bpnet", bpnet), ("cherimoya", cherimoya)):
        if df.empty:
            print(f"Error: no metrics found in {METRICS_DIR / name}. Run that "
                  f"family's benchmark first.", file=sys.stderr)
            sys.exit(1)

    on = ["experiment", "fold"]
    merged = bpnet.merge(cherimoya, on=on, suffixes=("_bpnet", "_cherimoya"))
    if merged.empty:
        print("Error: no experiment has been benchmarked for BOTH families.\n"
              f"  bpnet: {sorted(bpnet['experiment'].unique())}\n"
              f"  cherimoya: {sorted(cherimoya['experiment'].unique())}",
              file=sys.stderr)
        sys.exit(1)
    merged = add_depth(add_species(merged))

    tsv = args.tsv or (args.out_dir / "collated.tsv")
    tsv.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values(["species", "experiment", "fold"]).to_csv(
        tsv, sep="\t", index=False)
    n_exp = merged["experiment"].nunique()
    print(f"Collated {len(merged)} rows over {n_exp} experiments -> {tsv}")

    metrics = args.metrics or [
        m for m in METRIC_INFO
        if f"{m}_bpnet" in merged and f"{m}_cherimoya" in merged
    ]
    summary = []
    for metric in metrics:
        if f"{metric}_bpnet" not in merged:
            print(f"  skipping {metric}: absent from one family's JSON",
                  file=sys.stderr)
            continue
        out = args.out_dir / f"{metric}.png"
        summary.append(plot_metric(merged, metric, METRIC_INFO[metric], out,
                                   args.colour_by))
        print(f"  wrote {out}")

    if summary:
        print()
        print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
