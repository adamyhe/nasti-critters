"""
Fit a BPNet model for the D. melanogaster S2 PRO-cap project.

Training parameters and data paths are read from configs/. The background is
restricted to the GC-matched negative loci specified by data_paths["negatives"].

Usage:
    python src/bpnet/fit/fit_bpnet.py -f 0
"""

import argparse
import sys
from pathlib import Path

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


def load_config(path: str | Path) -> dict:
    """Load JSON-shaped config, allowing YAML-compatible conveniences."""
    with open(path) as f:
        config = yaml.safe_load(f)
    return config or {}


def resolve_config_path(value: str | Path | None) -> str | None:
    """Resolve relative config paths from the repository root."""
    if value is None:
        return None

    path = Path(value)
    if path.is_absolute():
        return str(path)

    return str((REPO_ROOT / path).resolve())


def resolve_config_paths(params: dict) -> dict:
    resolved = dict(params)
    for key in ("loci", "sequences", "negatives", "controls"):
        if key in resolved:
            if isinstance(resolved[key], list):
                resolved[key] = [resolve_config_path(p) for p in resolved[key]]
            else:
                resolved[key] = resolve_config_path(resolved[key])

    if "signals" in resolved:
        resolved["signals"] = [resolve_config_path(p) for p in resolved["signals"]]

    if "blacklist" in resolved and resolved["blacklist"] is not None:
        if isinstance(resolved["blacklist"], list):
            resolved["blacklist"] = [
                resolve_config_path(p) for p in resolved["blacklist"]
            ]
        else:
            resolved["blacklist"] = [resolve_config_path(resolved["blacklist"])]

    return resolved


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


def split_chroms(folds: pd.DataFrame, fold: int) -> tuple[list[str], list[str], list[str]]:
    fold_ids = sorted(folds["fold"].unique())
    if fold not in fold_ids:
        raise ValueError(f"Fold {fold} not found. Available folds: {fold_ids}")

    n_folds = len(fold_ids)
    validation_fold = (fold + 1) % n_folds
    if validation_fold not in fold_ids:
        raise ValueError(
            "Fold IDs must support modulo validation split; "
            f"computed validation fold {validation_fold}, available folds: {fold_ids}"
        )

    test_chroms = folds.loc[folds["fold"] == fold, "chrom"].to_list()
    validation_chroms = folds.loc[
        folds["fold"] == validation_fold, "chrom"
    ].to_list()
    training_chroms = folds.loc[
        ~folds["fold"].isin([fold, validation_fold]), "chrom"
    ].to_list()
    return training_chroms, validation_chroms, test_chroms


def validate_paths(params: dict) -> None:
    path_fields = [
        ("loci", params["loci"]),
        ("sequences", params["sequences"]),
        ("negatives", params["negatives"]),
    ]
    path_fields.extend((f"signals[{i}]", p) for i, p in enumerate(params["signals"]))

    controls = params.get("controls")
    if controls is not None:
        path_fields.extend((f"controls[{i}]", p) for i, p in enumerate(controls))

    blacklist = params.get("blacklist")
    if blacklist is not None:
        path_fields.extend((f"blacklist[{i}]", p) for i, p in enumerate(blacklist))

    missing = [(label, path) for label, path in path_fields if not Path(path).exists()]
    if missing:
        for label, path in missing:
            print(f"Error: {label} not found: {path}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        required=True,
        help="fold to hold out for testing (validation = (fold+1) %% n_folds)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=str(DEFAULT_PARAMS_PATH),
        help="training parameter config",
    )
    parser.add_argument(
        "--data-paths",
        type=str,
        default=str(DEFAULT_DATA_PATHS_PATH),
        help="data path config",
    )
    parser.add_argument(
        "--fold-assignments",
        type=str,
        default=str(DEFAULT_FOLD_ASSIGNMENTS_PATH),
        help="CSV assigning chromosomes to folds",
    )
    parser.add_argument("-o", "--output-dir", type=str, default=None)
    parser.add_argument("--n-filters", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--count-loss-weight", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--early-stopping", type=int, default=None)
    parser.add_argument("--max-jitter", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    config_params = load_config(args.params)
    data_paths = load_config(args.data_paths)
    folds = load_fold_assignments(args.fold_assignments)
    train_chroms, valid_chroms, test_chroms = split_chroms(folds, args.fold)

    defaults = {
        "checkpoint": None,
        "in_window": 2114,
        "out_window": 1000,
        "max_jitter": 200,
        "n_filters": 512,
        "n_layers": 8,
        "count_loss_weight": 100,
        "reverse_complement": True,
        "shuffle": True,
        "batch_size": 64,
        "learning_rate": 0.0005,
        "max_epochs": 50,
        "early_stopping": None,
        "validation_iter": None,
        "random_state": None,
        "verbose": False,
        "controls": None,
        "blacklist": None,
        "negatives_ratio": 0.1,
    }
    params = {**defaults, **config_params, **data_paths}

    params.update(
        {
            "training_chroms": train_chroms,
            "validation_chroms": valid_chroms,
            "test_chroms": test_chroms,
        }
    )

    cli_overrides = {
        "n_filters": args.n_filters,
        "n_layers": args.n_layers,
        "count_loss_weight": args.count_loss_weight,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs,
        "early_stopping": args.early_stopping,
        "max_jitter": args.max_jitter,
        "random_state": args.random_state,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            params[key] = value
    if args.verbose:
        params["verbose"] = True

    output_dir = Path(args.output_dir or (REPO_ROOT / "models"))
    params["name"] = str(output_dir / f"{DEFAULT_RUN_NAME}_f{args.fold}")
    params = resolve_config_paths(params)

    required = ["loci", "sequences", "signals", "negatives"]
    missing = [key for key in required if key not in params or params[key] is None]
    if missing:
        print(f"Error: missing required config key(s): {missing}", file=sys.stderr)
        sys.exit(1)

    validate_paths(params)
    output_dir.mkdir(parents=True, exist_ok=True)

    peaks = load_bed(params["loci"])
    negatives = load_bed(params["negatives"])

    print(f"Run: {DEFAULT_RUN_NAME}")
    print(f"Fold {args.fold}: test={test_chroms}, valid={valid_chroms}")
    print(f"Training chroms: {train_chroms}")
    print(
        "GC-matched negatives: "
        f"{len(negatives):,} loci, ratio={params['negatives_ratio']}"
    )

    import torch
    from bpnetlite.bpnet import BPNet
    from data_loader import PeakGenerator
    from tangermeme.io import extract_loci
    from torch.optim import AdamW

    train_data_loader = PeakGenerator(
        peaks=peaks,
        negatives=negatives,
        sequences=params["sequences"],
        signals=params["signals"],
        controls=params["controls"],
        chroms=params["training_chroms"],
        in_window=params["in_window"],
        out_window=params["out_window"],
        max_jitter=params["max_jitter"],
        reverse_complement=params["reverse_complement"],
        shuffle=params["shuffle"],
        random_state=params["random_state"],
        batch_size=params["batch_size"],
        verbose=params["verbose"],
        negative_ratio=params["negatives_ratio"],
        exclusion_lists=params["blacklist"],
        min_counts=None,
        max_counts=None,
        pin_memory=True,
        num_workers=0,
    )

    print(f"Loading validation data (chroms: {valid_chroms})...")
    val = extract_loci(
        loci=peaks,
        sequences=params["sequences"],
        signals=params["signals"],
        in_signals=params["controls"],
        chroms=params["validation_chroms"],
        in_window=params["in_window"],
        out_window=params["out_window"],
        max_jitter=0,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        exclusion_lists=params["blacklist"],
        verbose=params["verbose"],
    )
    if len(val) == 3:
        X_valid, y_valid, X_valid_ctl = val
        X_valid_ctl = torch.abs(X_valid_ctl)
    else:
        X_valid, y_valid = val
        X_valid_ctl = None
    y_valid = torch.abs(y_valid)

    model = BPNet(
        name=params["name"],
        n_filters=params["n_filters"],
        n_outputs=len(params["signals"]),
        n_control_tracks=0 if params["controls"] is None else len(params["controls"]),
        count_loss_weight=params["count_loss_weight"],
        n_layers=params["n_layers"],
        trimming=(params["in_window"] - params["out_window"]) // 2,
        verbose=params["verbose"],
    )
    model = model.to("cuda")
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"])

    fit_kwargs = {
        "training_data": train_data_loader,
        "optimizer": optimizer,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "max_epochs": params["max_epochs"],
        "batch_size": params["batch_size"],
        "early_stopping": params["early_stopping"],
        "dtype": torch.float,
    }
    if X_valid_ctl is not None:
        fit_kwargs["X_ctl_valid"] = X_valid_ctl

    model.fit(**fit_kwargs)

    print(f"\nModel saved to {output_dir}/")


if __name__ == "__main__":
    main()
