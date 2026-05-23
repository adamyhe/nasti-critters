"""
Compare base-resolution signal tracks over configured loci and folds.

The normal path reads loci, sequences, and folds from configs/. Optional
parameter JSONs or CLI signal arguments can provide separate signal sets.
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
    parser.add_argument("--signals-a", nargs="+", default=None)
    parser.add_argument("--signals-b", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config_params = load_config(args.params)
    data_paths = load_config(args.data_paths)
    legacy_params = load_config(args.parameters)
    params = {
        "in_window": 1000,
        "out_window": 1000,
        "verbose": False,
        **config_params,
        **data_paths,
        **legacy_params,
    }

    params["loci"] = resolve_path(params.get("loci"))
    params["sequences"] = resolve_path(params.get("sequences"))
    params["signals"] = resolve_path_list(params.get("signals"))
    params["signals_a"] = resolve_path_list(
        args.signals_a or params.get("signals_a") or params.get("signals")
    )
    params["signals_b"] = resolve_path_list(
        args.signals_b or params.get("signals_b") or params.get("signals")
    )
    params["output"] = resolve_path(args.output or params.get("output"))

    folds = load_fold_assignments(args.fold_assignments)

    required = ["loci", "sequences", "signals_a", "signals_b"]
    missing = [key for key in required if not params.get(key)]
    if missing:
        print(f"Error: missing required config key(s): {missing}", file=sys.stderr)
        sys.exit(1)

    path_fields = [
        ("loci", params["loci"]),
        ("sequences", params["sequences"]),
    ]
    path_fields.extend((f"signals_a[{i}]", p) for i, p in enumerate(params["signals_a"]))
    path_fields.extend((f"signals_b[{i}]", p) for i, p in enumerate(params["signals_b"]))
    validate_paths(path_fields)

    import torch
    from bpnetlite.performance import (
        jensen_shannon_distance,
        pearson_corr,
        spearman_corr,
    )
    from tangermeme.io import extract_loci

    _, a, b = extract_loci(
        loci=params["loci"],
        sequences=params["sequences"],
        chroms=folds["chrom"].to_list(),
        signals=params["signals_a"],
        in_signals=params["signals_b"],
        in_window=params["in_window"],
        out_window=params["out_window"],
        verbose=params["verbose"],
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
    )
    a = torch.abs(a)
    b = torch.abs(b)

    profile_corr = pearson_corr(b.reshape(b.shape[0], -1), a.reshape(a.shape[0], -1))
    profile_jsd = jensen_shannon_distance(
        torch.nn.functional.log_softmax(b.reshape(b.shape[0], -1), dim=-1),
        a.reshape(a.shape[0], -1),
    )
    counts_pearson = pearson_corr(b.sum(dim=(-1, -2)), a.sum(dim=(-1, -2)))
    log_counts_pearson = pearson_corr(
        torch.log1p(b.sum(dim=(-1, -2))), torch.log1p(a.sum(dim=(-1, -2)))
    )
    counts_spearman = spearman_corr(b.sum(dim=(-1, -2)), a.sum(dim=(-1, -2)))

    print(f"Profile Pearson correlation median: {profile_corr.median()}")
    print(f"Profile JSD median: {profile_jsd.median()}")
    print(f"Counts Pearson correlation {counts_pearson}")
    print(f"Log counts Pearson correlation: {log_counts_pearson}")
    print(f"Counts Spearman correlation: {counts_spearman}")

    if params["output"] is not None:
        Path(params["output"]).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            params["output"],
            **{
                "profile_corr": profile_corr,
                "profile_jsd": profile_jsd,
                "counts_pearson": counts_pearson,
                "log_counts_pearson": log_counts_pearson,
                "counts_spearman": counts_spearman,
            },
        )


if __name__ == "__main__":
    main()
