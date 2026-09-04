"""
Benchmark trained Cherimoya models across configured folds.

Data, species and folds resolve through src/experiments.py; model paths are
read from models/cherimoya/{experiment}/{experiment}.fold{f}.torch.
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
    IGNORE,
    Experiment,
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e", "--experiment", type=str, required=True,
        help="experiment ID as it appears in config/experiment_config.yaml",
    )
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--metrics-dir", type=str, default="performance_metrics/cherimoya")
    parser.add_argument("--predictions-dir", type=str, default="predictions/cherimoya")
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("-b", "--batch-size", type=int, default=None)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="enable torch.compile() during inference (default: DISABLED). "
             "Cherimoya.load() itself defaults to compile=True, so this script "
             "was compiling unconditionally; benchmarking is a single inference "
             "pass over the test set, and compilation's warmup cost is not "
             "worth it for one pass. Contrast fit_cherimoya.py, where it is on "
             "by default because 50 epochs amortise the warmup. Also gated on "
             "torch.compile being usable at all -- it raises on Python 3.14+ "
             "below torch 2.10",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
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

    params = load_params("cherimoya", {"batch_size": args.batch_size, "verbose": True if args.verbose else None})
    params.update({
        "loci": str(exp.peaks),
        "sequences": str(exp.sequences),
        "signals": [str(x) for x in exp.signals],
        "controls": [str(x) for x in exp.controls] if exp.controls else None,
        "blacklist": exp.blacklist,
    })

    folds = exp.all_folds("cherimoya", models_dir=args.models_dir)
    absent = [f for f in folds if not f["model"].exists()]
    if absent:
        for f in absent:
            print(f"Error: model for fold {f['fold']} not found: {f['model']}",
                  file=sys.stderr)
        sys.exit(1)

    import torch
    from bpnetlite.performance import (
        jensen_shannon_distance,
        pearson_corr,
        spearman_corr,
    )
    from cherimoya import Cherimoya
    from tangermeme.io import extract_loci
    from tangermeme_compat import patch_numeric_chroms

    # Numeric chromosome names (A.thaliana 1-5, C.reinhardtii 1-17,
    # P.patens 1-27) hit a dtype bug in tangermeme's BED reading. Self-retiring
    # no-op once tangermeme is fixed -- see src/tangermeme_compat.py.
    patch_numeric_chroms(verbose=params["verbose"])

    # torch.compile has no Python 3.14 support below torch 2.10; the limit is
    # Dynamo, not Triton itself. torch.__version__ is a TorchVersion, which
    # supports PEP 440-aware comparison against a plain string.
    compile_supported = args.compile and (
        sys.version_info < (3, 14) or torch.__version__ >= "2.10"
    )
    if args.compile and not compile_supported:
        print(
            "Warning: --compile requested but torch.compile is unsupported "
            f"on Python {'.'.join(map(str, sys.version_info[:3]))} with "
            f"torch {torch.__version__}; running without compilation.",
            file=sys.stderr,
        )
    from tangermeme.predict import predict

    loci = load_bed(params["loci"])

    print(f"Experiment: {exp.id} ({exp.species})")
    print(f"Models: {folds[0]['model'].parent}")

    signals = []
    preds = []
    for f in folds:
        fold, model_path, test_chroms = f["fold"], f["model"], f["test_chroms"]
        data = extract_loci(
            loci=loci,
            sequences=params["sequences"],
            chroms=test_chroms,
            signals=params["signals"],
            in_signals=params["controls"],
            in_window=params["in_window"],
            out_window=params["out_window"],
            verbose=params["verbose"],
            ignore=IGNORE,
            exclusion_lists=params["blacklist"],
        )
        if len(data) == 3:
            X, y, X_ctl = data
            X_ctl = (torch.abs(X_ctl),)
        else:
            X, y = data
            X_ctl = None
        signals.append(torch.abs(y))

        model = Cherimoya.load(
            model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compile=compile_supported,
        )
        preds.append(
            predict(
                model=model,
                X=X,
                args=X_ctl,
                verbose=params["verbose"],
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
    log_counts_pearson = [
        pearson_corr(pred[1].squeeze(), torch.log1p(signal.sum(dim=(-1, -2)))).item()
        for pred, signal in zip(preds, signals)
    ]
    counts_spearman = [
        spearman_corr(pred[1].squeeze(), signal.sum(dim=(-1, -2))).item()
        for pred, signal in zip(preds, signals)
    ]
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
    print(f"Log counts Pearson correlation: {log_counts_pearson}")
    print(f"Counts Spearman correlation: {counts_spearman}")

    print("\nGenome-wide results:\n----------------")
    print(f"Profile Pearson correlation: {np.nanmedian(np.concatenate(profile_corr))}")
    print(
        f"Profile Jensen-Shannon distance: {np.nanmedian(np.concatenate(profile_jsd))}"
    )
    print(f"Log counts Pearson correlation: {log_counts_pearson_all}")
    print(f"Counts Spearman correlation: {counts_spearman_all}")

    metrics = {
        "run_name": exp.id,
        "model_paths": {str(f["fold"]): str(f["model"]) for f in folds},
        "per_fold": {
            str(fold): {
                "profile_pearson": np.nanmedian(profile_corr[i]).item(),
                "profile_jsd": np.nanmedian(profile_jsd[i]).item(),
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

    if args.save_output:
        output_dir = REPO_ROOT / args.predictions_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{exp.id}.npz"
        scaled_preds = {
            f"predict_fold{fold}": (
                torch.nn.functional.softmax(
                    pred[0].reshape(pred[0].shape[0], -1), dim=-1
                )
                * torch.exp(pred[1])
            )
            .reshape(*pred[0].shape)
            .numpy()
            for fold, pred in zip((x["fold"] for x in folds), preds)
        }
        expts = {
            f"expt_fold{fold}": signal.numpy()
            for fold, signal in zip((x["fold"] for x in folds), signals)
        }
        np.savez_compressed(output_path, **scaled_preds, **expts)
        print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
