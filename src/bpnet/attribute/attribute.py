"""
Calculate BPNet attributions on the configured PRO-cap loci.

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
    parser.add_argument(
        "--attribute-type",
        choices=["counts", "profile"],
        default="profile",
    )
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--output-fname", type=str, default=None)
    parser.add_argument("--save-ohe", type=str, default=None)
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

    params["attribute_type"] = args.attribute_type
    params["model_fnames"] = [str(f["model"]) for f in folds]
    chroms = [c for f in folds for c in f["test_chroms"]]
    params["output_fname"] = str(
        REPO_ROOT / (args.output_fname
                     or f"attr/{exp.id}_attr_{args.attribute_type}.npz")
    )
    params["save_ohe"] = str(REPO_ROOT / args.save_ohe) if args.save_ohe else None

    import torch
    from bpnetlite.attribute import _ProfileLogitScaling
    from bpnetlite.bpnet import ControlWrapper, CountWrapper, ProfileWrapper
    from tangermeme.deep_lift_shap import _nonlinear, deep_lift_shap
    from tangermeme.io import extract_loci
    from tangermeme_compat import patch_numeric_chroms

    # Numeric chromosome names (A.thaliana 1-5, C.reinhardtii 1-17,
    # P.patens 1-27) hit a dtype bug in tangermeme's BED reading. Self-retiring
    # no-op once tangermeme is fixed -- see src/tangermeme_compat.py.
    patch_numeric_chroms(verbose=params["verbose"])

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
        ignore=IGNORE,
    ).to(torch.float32)

    if params["save_ohe"] is not None:
        Path(params["save_ohe"]).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(params["save_ohe"], X.to(torch.uint8).numpy())

    attributions = []
    for model_path in params["model_fnames"]:
        # The checkpoint IS the model -- see load_model()'s docstring. The block
        # this replaced also read params["n_outputs"] and
        # params["n_control_tracks"], which are set NOWHERE: not in
        # config/bpnet_params.json, not in the params.update() above, and behind
        # no CLI flag. So this script raised KeyError before it ever reached the
        # load.
        model = load_model(model_path)

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
