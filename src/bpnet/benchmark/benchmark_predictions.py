"""
Benchmark BPNet predictions on held-out fold chromosomes.

Data, species and folds resolve through src/experiments.py; model paths are
read from models/bpnet/{experiment}/{experiment}.fold{f}.torch.
"""

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e", "--experiment", type=str, required=True,
        help="experiment ID as it appears in config/experiment_config.yaml",
    )
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--output-fname", type=str, default=None)
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

    params["output_fname"] = (
        str(Path(args.output_fname).resolve()) if args.output_fname else None
    )

    import torch
    from bpnetlite.bpnet import BPNet
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
    n_control_tracks = 0 if params["controls"] is None else len(params["controls"])
    trimming = (params["in_window"] - params["out_window"]) // 2

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
        )
        if len(data) == 3:
            X, y, X_ctl = data
            X_ctl = (torch.abs(X_ctl),)
        else:
            X, y = data
            X_ctl = None
        signals.append(torch.abs(y))

        model = BPNet(
            n_filters=params["n_filters"],
            n_outputs=len(params["signals"]),
            n_control_tracks=n_control_tracks,
            count_loss_weight=params["count_loss_weight"],
            n_layers=params["n_layers"],
            trimming=trimming,
            verbose=params["verbose"],
        )
        model.load_state_dict(
            torch.load(
                model_path,
                weights_only=True,
                map_location=torch.device("cpu"),
            )
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

    print(
        f"Profile Pearson correlation: {[np.nanmedian(c) for c in profile_corr]}"
        f" (n_nan={[np.isnan(c).mean() for c in profile_corr]})"
    )
    print(
        f"Profile Jensen-Shannon distance: {[np.nanmedian(j) for j in profile_jsd]} "
        f"(n_nan={[np.isnan(j).mean() for j in profile_jsd]})"
    )
    print(f"Counts Pearson correlation: {counts_pearson}")
    print(f"Log Counts Pearson correlation: {log_counts_pearson}")
    print(f"Counts Spearman correlation: {counts_spearman}")

    if params["output_fname"] is not None:
        import joblib

        Path(params["output_fname"]).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"preds": preds, "signals": signals}, params["output_fname"])


if __name__ == "__main__":
    main()
