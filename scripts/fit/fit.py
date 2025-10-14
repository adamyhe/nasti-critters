"""
Wrapper script to fit a CLIPNET pytorch model. Note that this script is
designed for fitting to a single reference genome + PRO-cap dataset.
"""

import argparse
import json

import pandas as pd
import torch
from data import PeakGenerator
from personal_bpnet.clipnet_pytorch import CLIPNET
from torch.optim import AdamW


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--parameters", type=str, required=True)
    args = parser.parse_args()

    # Load parameters
    with open(args.parameters) as f:
        params = json.load(f)

    # Set defaults if not specified:
    default_params = {
        "checkpoint": None,
        "in_window": 2114,
        "out_window": 1000,
        "n_filters": 512,
        "n_layers": 8,
        "alpha": 100,
        "reverse_complement": True,
        "jitter": True,
        "batch_size": 64,
        "learning_rate": 0.0005,
        "validation_iter": 500,
        "max_epochs": 100,
        "early_stopping": 20,
        "verbose": False,
        "controls": None,
        "negatives": None,
        "negatives_ratio": 1 / 10,
    }

    for k, v in default_params.items():
        if k not in params:
            params[k] = v

    # Load Training Data
    peaks = pd.read_csv(
        params["loci"],
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )
    # Load negatives
    if params["negatives"] is not None:
        negatives = pd.read_csv(
            params["negatives"],
            sep="\t",
            usecols=[0, 1, 2],
            header=None,
            index_col=False,
            names=["chrom", "start", "end"],
            dtype={"chrom": str},
        )
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
        random_state=params["random_state"],
        batch_size=params["batch_size"],
        verbose=params["verbose"],
        min_counts=None,
        max_counts=None,
        pin_memory=True,
        num_workers=0,
    )

    # Load Validation Data
    val_data_loader = PeakGenerator(
        peaks=peaks,
        negatives=negatives,
        sequences=params["sequences"],
        signals=params["signals"],
        controls=params["controls"],
        chroms=params["validation_chroms"],
        in_window=params["in_window"],
        out_window=params["out_window"],
        max_jitter=params["max_jitter"],
        reverse_complement=params["reverse_complement"],
        random_state=params["random_state"],
        batch_size=params["batch_size"],
        verbose=params["verbose"],
        min_counts=None,
        max_counts=None,
        pin_memory=True,
        num_workers=0,
    )

    # Initialize model and optimizer
    model = CLIPNET(
        name=params["name"],
        n_filters=params["n_filters"],
        n_outputs=len(params["signals"]),
        n_control_tracks=0 if params["controls"] is None else len(params["controls"]),
        alpha=params["alpha"],
        n_layers=8,
        trimming=(params["in_window"] - params["out_window"]) // 2,
    )
    model = model.to("cuda")
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"])

    # Load checkpoint if provided
    if params["checkpoint"] is not None:
        model.load_state_dict(torch.load(params["checkpoint"]))
        chkpt = torch.load(params["checkpoint"].replace(".torch", ".checkpoint.torch"))
        print(f"Restarting from checkpoint: {params['checkpoint']}")
        print(f"Epoch: {chkpt['epoch']}")
        optimizer.load_state_dict(chkpt)
        max_epochs = params["max_epochs"] - chkpt["epoch"]
    else:
        max_epochs = params["max_epochs"]

    # Fit model
    model.fit(
        training_data=train_data_loader,
        optimizer=optimizer,
        valid_data=val_data_loader,
        max_epochs=max_epochs,
        batch_size=params["batch_size"],
        early_stopping=params["early_stopping"],
        validation_iter=params["validation_iter"],
        verbose=params["verbose"],
    )


if __name__ == "__main__":
    main()
