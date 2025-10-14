"""
Wrapper script to calculate attributions for a CLIPNET pytorch model.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from bpnetlite.attribute import _ProfileLogitScaling
from bpnetlite.bpnet import ControlWrapper, CountWrapper, ProfileWrapper
from personal_bpnet.clipnet_pytorch import CLIPNET
from tangermeme.deep_lift_shap import _nonlinear, deep_lift_shap
from tangermeme.io import extract_loci


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
        "n_filters": 512,
        "n_layers": 8,
        "n_outputs": 2,
        "n_control_tracks": 2,
        "batch_size": 16,
        "n_shuffles": 20,
        "random_state": None,
        "verbose": False,
        "save_ohe": None,
    }

    for k, v in default_params.items():
        if k not in params:
            params[k] = v

    if "output_fname" not in params:
        raise ValueError("output_fname must be specified")

    if "data_fold_assignments" in params:
        folds = pd.read_csv(params["data_fold_assignments"])
        chroms = folds.chrom.astype(str).to_list()
    else:
        chroms = None

    loci = pd.read_csv(
        params["loci"],
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )

    # Load attribution data
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
        np.savez_compressed(params["save_ohe"], X.to(torch.uint8).numpy())

    attributions = []
    for f in params["model_fnames"]:
        # Load model inside of for loop to prevent VRAM leak
        model = CLIPNET(
            n_filters=params["n_filters"],
            n_outputs=params["n_outputs"],
            n_control_tracks=params["n_control_tracks"],
            n_layers=params["n_layers"],
            trimming=(params["in_window"] - params["out_window"]) // 2,
        )
        model.load_state_dict(torch.load(f, weights_only=True))

        model = ControlWrapper(model)
        additional_nonlinear_ops = None
        # Wrap models depending on args.attribute_type
        if params["attribute_type"] == "counts":
            model = CountWrapper(model)
        elif params["attribute_type"] == "profile":
            model = ProfileWrapper(model)
            additional_nonlinear_ops = {_ProfileLogitScaling: _nonlinear}
        else:
            raise ValueError(
                f"Unknown attribute_type: {params['attribute_type']}."
                "Must be one of ['counts', 'profile']"
            )

        # Calculate and log attributions
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
                device="cuda",
                warning_threshold=0.01,
            ).numpy()
        )

        # clear VRAM
        del model
        torch.cuda.empty_cache()

    # Save
    np.savez_compressed(
        params["output_fname"],
        np.stack(attributions).mean(axis=0)
        if len(attributions) > 1
        else attributions[0],
    )


if __name__ == "__main__":
    main()
