import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
from bpnetlite.performance import jensen_shannon_distance, pearson_corr, spearman_corr
from personal_bpnet.clipnet_pytorch import CLIPNET
from tangermeme.io import extract_loci
from tangermeme.predict import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--parameters", type=str, required=True)
    args = parser.parse_args()

    # Load parameters
    with open(args.parameters) as f:
        params = json.load(f)

    # Set defaults if not specified:
    default_params = {
        "in_window": 2114,
        "out_window": 1000,
        "batch_size": 64,
        "verbose": False,
        "controls": None,
    }

    for k, v in default_params.items():
        if k not in params:
            params[k] = v

    if "n_cpus" in params:
        torch.set_num_threads(params["n_cpus"])
        torch.set_num_interop_threads(params["n_cpus"])

    folds = pd.read_csv(params["data_fold_assignments"])
    loci = pd.read_csv(
        params["loci"],
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )

    # Predict on test sets
    signals = []
    preds = []
    for fold in folds.fold.unique()[: len(params["model_fnames"])]:
        # Load data
        test_chrom = folds[folds.fold == fold].chrom.astype(str).to_list()
        data = extract_loci(
            loci=loci,
            sequences=params["sequences"],
            chroms=test_chrom,
            signals=params["signals"],
            in_signals=params["controls"],
            in_window=params["in_window"],
            out_window=params["out_window"],
            verbose=params["verbose"],
            ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        )
        if len(data) == 3:
            X, y, X_ctl = data
            X_ctl = (torch.abs(X_ctl),)
        else:
            X, y = data
            X_ctl = None
        signals.append(torch.abs(y))

        # Load model
        n_control_tracks = 0 if params["controls"] is None else len(params["controls"])
        model = CLIPNET(n_control_tracks=n_control_tracks)
        model.load_state_dict(
            torch.load(
                params["model_fnames"][fold],
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )

        # Predict
        preds.append(
            predict(
                model=model,
                X=X,
                args=X_ctl,
                verbose=True,
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

    # Output
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

    if "output_fname" in params:
        joblib.dump(
            {"preds": preds, "signals": signals},
            params["output_fname"],
        )


if __name__ == "__main__":
    main()
