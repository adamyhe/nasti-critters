"""
Benchmark BPNet predictions on held-out fold chromosomes.

Data, species and folds resolve through src/experiments.py; model paths are
read from models/bpnet/{experiment}/{experiment}.fold{f}.torch.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

sys.path.insert(0, str(REPO_ROOT / "src"))
from experiments import (  # noqa: E402
    Experiment,
    IGNORE,
    load_model,
    load_params,
)


def load_bed(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e", "--experiment", type=str, required=True,
        help="experiment ID as it appears in config/experiment_config.yaml",
    )
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument(
        "--metrics-dir", type=str, default="performance_metrics/bpnet",
        help="directory for the metrics JSON (default: %(default)s). Matches "
             "benchmark_cherimoya.py, which has always written one; this script "
             "only PRINTED its metrics until 2026-09-04",
    )
    parser.add_argument(
        "--output-fname", type=str, default=None,
        help="optional joblib dump of the raw predictions and signals. "
             "Separate from --metrics-dir: this is the predictions, that is "
             "the scores",
    )
    parser.add_argument(
        "--no-progress", dest="progress", action="store_false",
        help="suppress the tqdm progress bars from extract_loci and predict. "
             "They are ON by default and go to stderr, so stdout stays clean "
             "for the printed metrics",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="verbose output beyond the progress bars",
    )
    args = parser.parse_args()

    try:
        exp = Experiment.load(args.experiment)
    except (KeyError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    if exp.missing:
        for m in exp.missing:
            print(f"Error: missing {m}", file=sys.stderr)
        sys.exit(1)

    params = load_params("bpnet", {})
    params.update({
        "loci": str(exp.peaks),
        "sequences": str(exp.sequences),
        "signals": [str(x) for x in exp.signals],
        "controls": [str(x) for x in exp.controls] if exp.controls else None,
        "blacklist": exp.blacklist,
    })

    folds = exp.all_folds("bpnet", models_dir=args.models_dir)
    absent = [f for f in folds if not f["model"].exists()]
    if absent:
        for f in absent:
            print(f"Error: model for fold {f['fold']} not found: {f['model']}",
                  file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        params["verbose"] = True
    params["output_fname"] = (
        str(Path(args.output_fname).resolve()) if args.output_fname else None
    )

    import torch
    from bpnetlite.performance import (
        jensen_shannon_distance,
        pearson_corr,
        spearman_corr,
    )
    from tangermeme.io import extract_loci
    from tangermeme_compat import patch_numeric_chroms

    # Numeric chromosome names (A.thaliana 1-5, C.reinhardtii 1-17,
    # P.patens 1-27) hit a dtype bug in tangermeme's BED reading. Self-retiring
    # no-op once tangermeme is fixed -- see src/tangermeme_compat.py.
    patch_numeric_chroms(verbose=params["verbose"])
    from tangermeme.predict import predict

    if params["n_cpus"] is not None:
        torch.set_num_threads(params["n_cpus"])
        torch.set_num_interop_threads(params["n_cpus"])

    loci = load_bed(params["loci"])

    signals = []
    preds = []
    for f in folds:
        fold, model_path, test_chroms = f["fold"], f["model"], f["test_chroms"]
        # Peak-level species (S. pombe, S. moellendorffii) have test_chroms
        # None, because their folds are assigned per PEAK rather than per
        # chromosome. Passing that straight to extract_loci means "no
        # chromosome filter", so every fold's model was being scored on ALL
        # loci -- its own training peaks included -- silently, and with
        # inflated metrics. fold_loci() applies the peak-level filter and is a
        # no-op under chromosome-level splits, where it returns every peak and
        # test_chroms does the work.
        test_loci = exp.fold_loci(loci, fold)["test_loci"]
        data = extract_loci(
            loci=test_loci,
            sequences=params["sequences"],
            chroms=test_chroms,
            signals=params["signals"],
            in_signals=params["controls"],
            in_window=params["in_window"],
            out_window=params["out_window"],
            # tangermeme's `verbose` IS the tqdm bar here, so progress is wired
            # to --no-progress rather than to -v: a long benchmark should show
            # progress without also turning on every other message.
            verbose=args.progress or params["verbose"],
            ignore=IGNORE,
        )
        if len(data) == 3:
            X, y, X_ctl = data
            X_ctl = (torch.abs(X_ctl),)
        else:
            X, y = data
            X_ctl = None
        signals.append(torch.abs(y))

        # The checkpoint IS the model -- see load_model()'s docstring.
        model = load_model(model_path)

        preds.append(
            predict(
                model=model,
                X=X,
                args=X_ctl,
                verbose=args.progress or params["verbose"],
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=params["batch_size"],
            )
        )

    profile_corr = [
        pearson_corr(
            torch.nn.functional.softmax(pred[0].reshape(pred[0].shape[0], -1), dim=-1),
            signal.reshape(pred[0].shape[0], -1),
        ).numpy()
        for pred, signal in zip(preds, signals)
    ]

    profile_jsd = [
        jensen_shannon_distance(
            torch.nn.functional.log_softmax(
                pred[0].reshape(pred[0].shape[0], -1), dim=-1
            ),
            signal.reshape(pred[0].shape[0], -1),
        ).numpy()
        for pred, signal in zip(preds, signals)
    ]

    counts_pearson = [
        pearson_corr(torch.exp(pred[1] - 1).squeeze(), signal.sum(dim=(-1, -2))).item()
        for pred, signal in zip(preds, signals)
    ]

    log_counts_pearson = [
        pearson_corr(pred[1].squeeze(), torch.log1p(signal.sum(dim=(-1, -2)))).item()
        for pred, signal in zip(preds, signals)
    ]

    counts_spearman = [
        spearman_corr(pred[1].squeeze(), signal.sum(dim=(-1, -2))).item()
        for pred, signal in zip(preds, signals)
    ]

    # Genome-wide aggregates: pooled across folds rather than averaged over
    # them, so each locus counts once regardless of how large its fold was.
    # Same construction as benchmark_cherimoya.py -- keep the two in step.
    log_counts_pearson_all = pearson_corr(
        torch.cat([pred[1].squeeze() for pred in preds]),
        torch.cat([torch.log1p(signal.sum(dim=(-1, -2))) for signal in signals]),
    ).item()
    counts_spearman_all = spearman_corr(
        torch.cat([pred[1].squeeze() for pred in preds]),
        torch.cat([signal.sum(dim=(-1, -2)) for signal in signals]),
    ).item()

    print("\nPer-fold results:\n----------------")
    print(
        f"Profile Pearson correlation: {[np.nanmedian(c).item() for c in profile_corr]}"
        f" (n_nan={[np.isnan(c).mean().item() for c in profile_corr]})"
    )
    print(
        f"Profile Jensen-Shannon distance: {[np.nanmedian(j).item() for j in profile_jsd]} "
        f"(n_nan={[np.isnan(j).mean().item() for j in profile_jsd]})"
    )
    print(f"Counts Pearson correlation: {counts_pearson}")
    print(f"Log counts Pearson correlation: {log_counts_pearson}")
    print(f"Counts Spearman correlation: {counts_spearman}")

    print("\nGenome-wide results:\n----------------")
    print(f"Profile Pearson correlation: {np.nanmedian(np.concatenate(profile_corr))}")
    print(
        f"Profile Jensen-Shannon distance: {np.nanmedian(np.concatenate(profile_jsd))}"
    )
    print(f"Log counts Pearson correlation: {log_counts_pearson_all}")
    print(f"Counts Spearman correlation: {counts_spearman_all}")

    # Metrics JSON. Same shape as benchmark_cherimoya.py's so the two families
    # are directly comparable; `counts_pearson` is extra here because this
    # script already computed it.
    metrics = {
        "run_name": exp.id,
        "model_paths": {str(f["fold"]): str(f["model"]) for f in folds},
        "per_fold": {
            str(fold): {
                "profile_pearson": np.nanmedian(profile_corr[i]).item(),
                "profile_jsd": np.nanmedian(profile_jsd[i]).item(),
                "counts_pearson": counts_pearson[i],
                "log_counts_pearson": log_counts_pearson[i],
                "counts_spearman": counts_spearman[i],
            }
            for i, fold in enumerate(x["fold"] for x in folds)
        },
        "genome_wide": {
            "profile_pearson": np.nanmedian(np.concatenate(profile_corr)).item(),
            "profile_jsd": np.nanmedian(np.concatenate(profile_jsd)).item(),
            "log_counts_pearson": log_counts_pearson_all,
            "counts_spearman": counts_spearman_all,
        },
    }
    metrics_dir = REPO_ROOT / args.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{exp.id}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")

    if params["output_fname"] is not None:
        import joblib

        Path(params["output_fname"]).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"preds": preds, "signals": signals}, params["output_fname"])
        print(f"Predictions saved to {params['output_fname']}")


if __name__ == "__main__":
    main()
