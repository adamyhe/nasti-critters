"""
Benchmark BPNet predictions on held-out fold chromosomes.

By default, parameters and data paths are read from configs/ and model paths are
derived as models/D.melanogaster-S2_PROcap_f{fold}.torch.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_PARAMS_PATH = REPO_ROOT / "configs" / "bpnet_params.json"
DEFAULT_DATA_PATHS_PATH = REPO_ROOT / "configs" / "data_paths.json"
DEFAULT_FOLD_ASSIGNMENTS_PATH = (
    REPO_ROOT / "configs" / "D.melanogaster_data_fold_assignments.csv"
)
DEFAULT_RUN_NAME = "D.melanogaster-S2_PROcap"


def load_config(path: str | Path | None) -> dict:
    if path is None:
        return {}
    with open(path) as f:
        config = yaml.safe_load(f)
    return config or {}


def resolve_path(value: str | Path | None) -> str | None:
    if value is None:
        return None

    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def resolve_path_list(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    return [resolve_path(value) for value in values]


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


def load_fold_assignments(path: str | Path) -> pd.DataFrame:
    folds = pd.read_csv(path)
    required = {"chrom", "fold"}
    missing = required - set(folds.columns)
    if missing:
        raise ValueError(
            f"Fold assignments missing required column(s): {sorted(missing)}"
        )

    folds = folds.copy()
    folds["chrom"] = folds["chrom"].astype(str)
    folds["fold"] = folds["fold"].astype(int)
    return folds


def merge_params(args: argparse.Namespace) -> dict:
    config_params = load_config(args.params)
    data_paths = load_config(args.data_paths)
    legacy_params = load_config(args.parameters)

    defaults = {
        "in_window": 2114,
        "out_window": 1000,
        "batch_size": 64,
        "verbose": False,
        "controls": None,
        "n_filters": 512,
        "n_layers": 8,
        "count_loss_weight": 100,
        "n_cpus": None,
        "output_fname": None,
    }
    params = {**defaults, **config_params, **data_paths, **legacy_params}

    params["loci"] = resolve_path(params.get("loci"))
    params["sequences"] = resolve_path(params.get("sequences"))
    params["signals"] = resolve_path_list(params.get("signals"))
    params["controls"] = resolve_path_list(params.get("controls"))
    params["output_fname"] = resolve_path(params.get("output_fname"))
    return params


def derive_model_paths(
    folds: pd.DataFrame,
    model_fnames: list[str] | None,
    models_dir: str,
    run_name: str,
) -> dict[int, str]:
    fold_ids = sorted(folds["fold"].unique())
    if model_fnames:
        if len(model_fnames) > len(fold_ids):
            raise ValueError(
                f"Got {len(model_fnames)} model paths for {len(fold_ids)} folds."
            )
        return {
            fold: resolve_path(model_fnames[i])
            for i, fold in enumerate(fold_ids[: len(model_fnames)])
        }

    models_dir = resolve_path(models_dir)
    return {
        fold: str(Path(models_dir) / f"{run_name}_f{fold}.torch")
        for fold in fold_ids
    }


def validate_paths(path_fields: list[tuple[str, str | None]]) -> None:
    missing = [
        (label, path)
        for label, path in path_fields
        if path is None or not Path(path).exists()
    ]
    if missing:
        for label, path in missing:
            print(f"Error: {label} not found: {path}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        default=None,
        help="optional legacy parameter config; overrides shared configs",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=str(DEFAULT_PARAMS_PATH),
        help="shared training parameter config",
    )
    parser.add_argument(
        "--data-paths",
        type=str,
        default=str(DEFAULT_DATA_PATHS_PATH),
        help="shared data path config",
    )
    parser.add_argument(
        "--fold-assignments",
        type=str,
        default=str(DEFAULT_FOLD_ASSIGNMENTS_PATH),
        help="CSV assigning chromosomes to folds",
    )
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--model-fnames", nargs="+", default=None)
    parser.add_argument("--output-fname", type=str, default=None)
    args = parser.parse_args()

    params = merge_params(args)
    if args.output_fname is not None:
        params["output_fname"] = resolve_path(args.output_fname)

    folds = load_fold_assignments(args.fold_assignments)
    model_paths = derive_model_paths(
        folds=folds,
        model_fnames=args.model_fnames or params.get("model_fnames"),
        models_dir=args.models_dir,
        run_name=args.run_name,
    )

    required = ["loci", "sequences", "signals"]
    missing = [key for key in required if not params.get(key)]
    if missing:
        print(f"Error: missing required config key(s): {missing}", file=sys.stderr)
        sys.exit(1)

    path_fields = [
        ("loci", params["loci"]),
        ("sequences", params["sequences"]),
    ]
    path_fields.extend((f"signals[{i}]", p) for i, p in enumerate(params["signals"]))
    if params["controls"] is not None:
        path_fields.extend((f"controls[{i}]", p) for i, p in enumerate(params["controls"]))
    path_fields.extend((f"model fold {fold}", path) for fold, path in model_paths.items())
    validate_paths(path_fields)

    import torch
    from bpnetlite.bpnet import BPNet
    from bpnetlite.performance import (
        jensen_shannon_distance,
        pearson_corr,
        spearman_corr,
    )
    from tangermeme.io import extract_loci
    from tangermeme.predict import predict

    if params["n_cpus"] is not None:
        torch.set_num_threads(params["n_cpus"])
        torch.set_num_interop_threads(params["n_cpus"])

    loci = load_bed(params["loci"])
    n_control_tracks = 0 if params["controls"] is None else len(params["controls"])
    trimming = (params["in_window"] - params["out_window"]) // 2

    signals = []
    preds = []
    for fold, model_path in model_paths.items():
        test_chroms = folds.loc[folds["fold"] == fold, "chrom"].to_list()
        data = extract_loci(
            loci=loci,
            sequences=params["sequences"],
            chroms=test_chroms,
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
