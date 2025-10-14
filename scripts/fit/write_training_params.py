import json
import sys

import pandas as pd

biosample = sys.argv[1]

with open(f"{biosample}_template.json", "r") as f:
    params = json.load(f)

species = biosample.split("-")[0]
folds = pd.read_csv(f"../../data_folds/{species}_data_fold_assignments.csv")
folds.chrom = folds.chrom.astype(str)

n_folds = len(folds.fold.unique())
for i in range(n_folds):
    params["test_chroms"] = folds[folds.fold == i % n_folds].chrom.to_list()
    params["validation_chroms"] = folds[folds.fold == (i + 1) % n_folds].chrom.to_list()
    params["training_chroms"] = folds[
        ~folds.fold.isin([i, (i + 1) % n_folds])
    ].chrom.to_list()
    params["name"] = f"../../models/{biosample}_f{i}"

    with open(f"{biosample}_f{i}_fit.json", "w") as f:
        json.dump(params, f, indent=4)
