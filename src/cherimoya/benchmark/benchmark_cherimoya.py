"""
Benchmark trained Cherimoya models across configured folds.

By default, parameters and data paths are read from configs/ and model paths are
derived as models/cherimoya/D.melanogaster-S2_PROcap_f{fold}.torch.
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
DEFAULT_PARAMS_PATH = REPO_ROOT / "configs" / "cherimoya_params.json"
DEFAULT_DATA_PATHS_PATH = REPO_ROOT / "configs" / "data_paths.json"
DEFAULT_FOLD_ASSIGNMENTS_PATH = (
    REPO_ROOT / "configs" / "D.melanogaster_data_fold_assignments.csv"
)
DEFAULT_RUN_NAME = "D.melanogaster-S2_PROcap"


def load_config(path: str | Path) -> dict:
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
    parser.add_argument("--models-dir", type=str, default="models/cherimoya")
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--model-fnames", nargs="+", default=None)
    parser.add_argument("--metrics-dir", type=str, default="performance_metrics/cherimoya")
    parser.add_argument("--predictions-dir", type=str, default="predictions/cherimoya")
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("-b", "--batch-size", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    config_params = load_config(args.params)
    data_paths = load_config(args.data_paths)
    folds = load_fold_assignments(args.fold_assignments)

    params = {**config_params, **data_paths}
    params["loci"] = resolve_path(params.get("loci"))
    params["sequences"] = resolve_path(params.get("sequences"))
    params["signals"] = resolve_path_list(params.get("signals"))
    params["controls"] = resolve_path_list(params.get("controls"))
    if params["blacklist"] is not None:
        values = params["blacklist"]
        params["blacklist"] = resolve_path_list(values if isinstance(values, list) else [values])

    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.verbose:
        params["verbose"] = True

    model_paths = derive_model_paths(
        folds=folds,
        model_fnames=args.model_fnames,
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
    if params["blacklist"] is not None:
        path_fields.extend((f"blacklist[{i}]", p) for i, p in enumerate(params["blacklist"]))
    path_fields.extend((f"model fold {fold}", path) for fold, path in model_paths.items())
    validate_paths(path_fields)

    import torch
    from bpnetlite.performance import (
        jensen_shannon_distance,
        pearson_corr,
        spearman_corr,
    )
    from cherimoya import Cherimoya
    from tangermeme.io import extract_loci
    from tangermeme.predict import predict

    loci = load_bed(params["loci"])

    print(f"Run: {args.run_name}")
    print(f"Models dir: {resolve_path(args.models_dir)}")

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
            exclusion_lists=params["blacklist"],
        )
        if len(data) == 3:
            X, y, X_ctl = data
            X_ctl = (torch.abs(X_ctl),)
        else:
            X, y = data
            X_ctl = None
        signals.append(torch.abs(y))

        model = Cherimoya.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
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
        "run_name": args.run_name,
        "model_paths": {str(fold): path for fold, path in model_paths.items()},
        "per_fold": {
            str(fold): {
                "profile_pearson": np.nanmedian(profile_corr[i]).item(),
                "profile_jsd": np.nanmedian(profile_jsd[i]).item(),
                "log_counts_pearson": log_counts_pearson[i],
                "counts_spearman": counts_spearman[i],
            }
            for i, fold in enumerate(model_paths)
        },
        "genome_wide": {
            "profile_pearson": np.nanmedian(np.concatenate(profile_corr)).item(),
            "profile_jsd": np.nanmedian(np.concatenate(profile_jsd)).item(),
            "log_counts_pearson": log_counts_pearson_all,
            "counts_spearman": counts_spearman_all,
        },
    }
    metrics_dir = Path(resolve_path(args.metrics_dir))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{args.run_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")

    if args.save_output:
        output_dir = Path(resolve_path(args.predictions_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.run_name}.npz"
        scaled_preds = {
            f"predict_fold{fold}": (
                torch.nn.functional.softmax(
                    pred[0].reshape(pred[0].shape[0], -1), dim=-1
                )
                * torch.exp(pred[1])
            )
            .reshape(*pred[0].shape)
            .numpy()
            for fold, pred in zip(model_paths, preds)
        }
        expts = {
            f"expt_fold{fold}": signal.numpy()
            for fold, signal in zip(model_paths, signals)
        }
        np.savez_compressed(output_path, **scaled_preds, **expts)
        print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
