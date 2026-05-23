"""
Calculate BPNet attributions on the configured PRO-cap loci.

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


def derive_model_paths(
    folds: pd.DataFrame,
    model_fnames: list[str] | None,
    models_dir: str,
    run_name: str,
) -> list[str]:
    fold_ids = sorted(folds["fold"].unique())
    if model_fnames:
        return resolve_path_list(model_fnames)

    models_dir = resolve_path(models_dir)
    return [
        str(Path(models_dir) / f"{run_name}_f{fold}.torch")
        for fold in fold_ids
    ]


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
        help="optional legacy attribution config; overrides shared configs",
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
    parser.add_argument(
        "--attribute-type",
        choices=["counts", "profile"],
        default="profile",
    )
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--model-fnames", nargs="+", default=None)
    parser.add_argument("--output-fname", type=str, default=None)
    parser.add_argument("--save-ohe", type=str, default=None)
    args = parser.parse_args()

    config_params = load_config(args.params)
    data_paths = load_config(args.data_paths)
    legacy_params = load_config(args.parameters)

    defaults = {
        "in_window": 2114,
        "out_window": 1000,
        "n_filters": 512,
        "n_layers": 8,
        "n_outputs": None,
        "n_control_tracks": None,
        "count_loss_weight": 100,
        "batch_size": 16,
        "n_shuffles": 20,
        "random_state": None,
        "verbose": False,
        "controls": None,
        "save_ohe": None,
    }
    params = {**defaults, **config_params, **data_paths, **legacy_params}

    params["attribute_type"] = args.attribute_type
    params["loci"] = resolve_path(params.get("loci"))
    params["sequences"] = resolve_path(params.get("sequences"))
    params["signals"] = resolve_path_list(params.get("signals"))
    params["controls"] = resolve_path_list(params.get("controls"))
    params["model_fnames"] = derive_model_paths(
        folds=load_fold_assignments(args.fold_assignments),
        model_fnames=args.model_fnames or params.get("model_fnames"),
        models_dir=args.models_dir,
        run_name=args.run_name,
    )

    output_fname = (
        args.output_fname
        or params.get("output_fname")
        or f"attr/{args.run_name}_attr_{args.attribute_type}.npz"
    )
    params["output_fname"] = resolve_path(output_fname)
    params["save_ohe"] = resolve_path(args.save_ohe or params.get("save_ohe"))

    folds = load_fold_assignments(args.fold_assignments)
    chroms = folds["chrom"].to_list()

    required = ["loci", "sequences", "signals", "model_fnames", "output_fname"]
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
    path_fields.extend((f"model[{i}]", p) for i, p in enumerate(params["model_fnames"]))
    validate_paths(path_fields)

    import torch
    from bpnetlite.attribute import _ProfileLogitScaling
    from bpnetlite.bpnet import BPNet, ControlWrapper, CountWrapper, ProfileWrapper
    from tangermeme.deep_lift_shap import _nonlinear, deep_lift_shap
    from tangermeme.io import extract_loci

    loci = load_bed(params["loci"])
    X = extract_loci(
        loci=loci,
        sequences=params["sequences"],
        in_window=params["in_window"],
        out_window=params["out_window"],
        chroms=chroms,
        max_jitter=0,
        verbose=params["verbose"],
        min_counts=None,
        max_counts=None,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
    ).to(torch.float32)

    if params["save_ohe"] is not None:
        Path(params["save_ohe"]).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(params["save_ohe"], X.to(torch.uint8).numpy())

    n_outputs = params["n_outputs"] or len(params["signals"])
    n_control_tracks = (
        params["n_control_tracks"]
        if params["n_control_tracks"] is not None
        else 0 if params["controls"] is None else len(params["controls"])
    )
    trimming = (params["in_window"] - params["out_window"]) // 2

    attributions = []
    for model_path in params["model_fnames"]:
        model = BPNet(
            n_filters=params["n_filters"],
            n_outputs=n_outputs,
            n_control_tracks=n_control_tracks,
            count_loss_weight=params["count_loss_weight"],
            n_layers=params["n_layers"],
            trimming=trimming,
            verbose=params["verbose"],
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))

        model = ControlWrapper(model)
        additional_nonlinear_ops = None
        if params["attribute_type"] == "counts":
            model = CountWrapper(model)
        else:
            model = ProfileWrapper(model)
            additional_nonlinear_ops = {_ProfileLogitScaling: _nonlinear}

        attributions.append(
            deep_lift_shap(
                model,
                X,
                hypothetical=True,
                batch_size=params["batch_size"],
                n_shuffles=params["n_shuffles"],
                random_state=params["random_state"],
                verbose=params["verbose"],
                additional_nonlinear_ops=additional_nonlinear_ops,
                device="cuda" if torch.cuda.is_available() else "cpu",
                warning_threshold=0.01,
            ).numpy()
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Path(params["output_fname"]).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        params["output_fname"],
        np.stack(attributions).mean(axis=0)
        if len(attributions) > 1
        else attributions[0],
    )


if __name__ == "__main__":
    main()
