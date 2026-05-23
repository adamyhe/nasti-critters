"""
Fit a Cherimoya model for the D. melanogaster S2 PRO-cap project.

Training parameters and data paths are read from configs/. The background is
restricted to the GC-matched negative loci specified by data_paths["negatives"].

Usage:
    python src/cherimoya/fit/fit_cherimoya.py -f 0
"""

import argparse
import sys
from pathlib import Path

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


def resolve_config_paths(params: dict) -> dict:
    resolved = dict(params)
    for key in ("loci", "sequences", "negatives"):
        if key in resolved:
            resolved[key] = resolve_path(resolved[key])

    for key in ("signals", "controls", "blacklist"):
        if key in resolved and resolved[key] is not None:
            values = resolved[key]
            resolved[key] = resolve_path_list(values if isinstance(values, list) else [values])

    return resolved


def validate_paths(params: dict) -> None:
    path_fields = [
        ("loci", params["loci"]),
        ("sequences", params["sequences"]),
        ("negatives", params["negatives"]),
    ]
    path_fields.extend((f"signals[{i}]", p) for i, p in enumerate(params["signals"]))
    if params["controls"] is not None:
        path_fields.extend((f"controls[{i}]", p) for i, p in enumerate(params["controls"]))
    if params["blacklist"] is not None:
        path_fields.extend((f"blacklist[{i}]", p) for i, p in enumerate(params["blacklist"]))

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
    parser.add_argument("-o", "--output-dir", type=str, default=None)
    parser.add_argument("--n-filters", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--early-stopping", type=int, default=None)
    parser.add_argument("--max-jitter", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--negative-ratio", type=float, default=None)
    parser.add_argument("--muon-lr", type=float, default=None)
    parser.add_argument("--muon-wd", type=float, default=None)
    parser.add_argument("--adam-lr", type=float, default=None)
    parser.add_argument("--adam-wd", type=float, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    config_params = load_config(args.params)
    data_paths = load_config(args.data_paths)
    folds = load_fold_assignments(args.fold_assignments)
    train_chroms, valid_chroms, test_chroms = split_chroms(folds, args.fold)

    params = {**config_params, **data_paths}
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
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "early_stopping": args.early_stopping,
        "max_jitter": args.max_jitter,
        "random_state": args.random_state,
        "negatives_ratio": args.negative_ratio,
        "muon_lr": args.muon_lr,
        "muon_wd": args.muon_wd,
        "adam_lr": args.adam_lr,
        "adam_wd": args.adam_wd,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            params[key] = value
    if args.verbose:
        params["verbose"] = True

    output_dir = Path(args.output_dir or (REPO_ROOT / "models" / "cherimoya"))
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
    from cherimoya import Cherimoya
    from data_loader import PeakGenerator
    from tangermeme.io import extract_loci
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    try:
        from torch.optim import Muon
    except ImportError:
        from muon import Muon

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

    n_control_tracks = 0 if params["controls"] is None else len(params["controls"])
    model = Cherimoya(
        name=params["name"],
        n_filters=params["n_filters"],
        n_outputs=len(params["signals"]),
        n_control_tracks=n_control_tracks,
        n_layers=params["n_layers"],
        trimming=(params["in_window"] - params["out_window"]) // 2,
        verbose=params["verbose"],
    )
    model = model.to("cuda")

    muon_params, adam_params = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2 and "weight" in name and name != "linear.weight":
            muon_params.append(parameter)
        else:
            adam_params.append(parameter)

    muon_optimizer = Muon(
        muon_params, lr=params["muon_lr"], weight_decay=params["muon_wd"]
    )
    adam_optimizer = AdamW(
        adam_params, lr=params["adam_lr"], weight_decay=params["adam_wd"]
    )

    num_warmup_epochs = 5
    max_epochs = params["max_epochs"]
    num_warmup_iters = len(train_data_loader) * num_warmup_epochs
    num_decay_iters = len(train_data_loader) * max(1, max_epochs - num_warmup_epochs)

    muon_scheduler = SequentialLR(
        muon_optimizer,
        schedulers=[
            LinearLR(muon_optimizer, start_factor=0.01, total_iters=num_warmup_iters),
            CosineAnnealingLR(muon_optimizer, T_max=num_decay_iters, eta_min=1e-5),
        ],
        milestones=[num_warmup_iters],
    )
    adam_scheduler = SequentialLR(
        adam_optimizer,
        schedulers=[
            LinearLR(adam_optimizer, start_factor=0.01, total_iters=num_warmup_iters),
            CosineAnnealingLR(adam_optimizer, T_max=num_decay_iters, eta_min=1e-5),
        ],
        milestones=[num_warmup_iters],
    )

    model.fit(
        training_data=train_data_loader,
        muon_optimizer=muon_optimizer,
        adam_optimizer=adam_optimizer,
        muon_scheduler=muon_scheduler,
        adam_scheduler=adam_scheduler,
        X_valid=X_valid,
        X_ctl_valid=X_valid_ctl,
        y_valid=y_valid,
        max_epochs=params["max_epochs"],
        batch_size=params["batch_size"],
        early_stopping=params["early_stopping"],
        dtype=torch.bfloat16,
    )

    print(f"\nModel saved to {output_dir}/")


if __name__ == "__main__":
    main()
