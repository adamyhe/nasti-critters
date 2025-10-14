import argparse
import json

import numpy as np
import pandas as pd
import torch
from bpnetlite.performance import jensen_shannon_distance, pearson_corr, spearman_corr
from tangermeme.io import extract_loci


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--parameters", type=str, required=True)
    args = parser.parse_args()

    # Load parameters
    with open(args.parameters) as f:
        params = json.load(f)

    # Set defaults if not specified:
    default_params = {"in_window": 1000, "out_window": 1000, "verbose": False}

    for k, v in default_params.items():
        if k not in params:
            params[k] = v

    folds = pd.read_csv(params["data_fold_assignments"])

    # Extract signals
    _, a, b = extract_loci(
        loci=params["loci"],
        sequences=params["sequences"],
        chroms=folds.chrom.astype(str).to_list(),
        signals=params["signals_a"],
        in_signals=params["signals_b"],
        in_window=1000,
        verbose=params["verbose"],
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
    )
    a = torch.abs(a)
    b = torch.abs(b)

    # Calculate comparisons
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

    # Output
    print(f"Profile Pearson correlation median: {profile_corr.median()}")
    print(f"Profile JSD median: {profile_jsd.median()}")
    print(f"Counts Pearson correlation {counts_pearson}")
    print(f"Log counts Pearson correlation: {log_counts_pearson}")
    print(f"Counts Spearman correlation: {counts_spearman}")
    if "output" in params:
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
