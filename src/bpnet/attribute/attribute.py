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
    ATTR_DIR,
    Experiment,
    attribution_path,
    filtered_loci_path,
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


def nucleotide_frequency_references(X, n=1, random_state=None):
    """Soft PFM references from each sequence's own observed base frequencies.

    Ported from procap-atlas. One reference per input sequence: the sequence's
    A/C/G/T frequencies repeated at every position, so the baseline is
    composition-matched but carries no positional information at all.

    Why this rather than dinucleotide shuffling, which is bpnet-lite's and
    tangermeme's default: upstream's locus diagnostics found that **dinucleotide
    shuffles sometimes produce ACTIVE references** -- sequences with cryptic
    promoter-like signal, occasionally as active as or more active than the
    genomic input. That makes the baseline reference-sensitive rather than
    neutral, which is the one property a DeepLIFT reference has to have.

    The argument is stronger in this repo than upstream, because it is
    multi-species and several of these genomes are far denser than human. In
    S. cerevisiae there are 1.2-4.1 peaks per 2114 bp training window, so almost
    every window contains a promoter; a shuffle that preserves local
    dinucleotide composition there is correspondingly more likely to reassemble
    something initiation-competent. It is the same reasoning that made the
    initiator PWM measure use relative entropy against LOCAL base composition
    rather than a uniform background -- promoter context is not genome average.

    Shape is (N, n, 4, L), which is what tangermeme's reference interface wants.
    Passed as a CALLABLE, so references are built per batch and never go through
    tangermeme's tensor-reference one-hot validator, which would reject a soft
    (non-one-hot) tensor.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    frequencies = X.float().mean(dim=-1, keepdim=True)
    return (
        frequencies.expand(-1, -1, X.shape[-1])
        .unsqueeze(1)
        .expand(-1, n, -1, -1)
        .clone()
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
    parser.add_argument(
        "--loci", type=str, default=None, metavar="BED",
        help="attribute over this BED instead of the experiment's peaks. This "
             "is what consumes filter_nonACGT_regions.py's output -- without "
             "it that script's filtered BED and its --save-ohe array have no "
             "reader, since attributions would still come from exp.peaks. When "
             "given, the loci filename's stem goes into the default output name "
             "so a run over a different locus set cannot overwrite the "
             "peaks-based one",
    )
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument(
        "--reference-mode",
        choices=("frequency", "dinucleotide"),
        default="frequency",
        help="DeepLIFT reference baseline (default: %(default)s). 'frequency' "
             "uses one soft nucleotide-frequency reference per sequence; "
             "'dinucleotide' uses bpnet-lite/tangermeme's dinucleotide "
             "shuffles, which upstream found can be ACTIVE at some loci -- see "
             "nucleotide_frequency_references()",
    )
    parser.add_argument(
        "--n-shuffles", type=int, default=None,
        help="number of dinucleotide shuffles; only used with "
             "--reference-mode dinucleotide (default: from "
             "config/bpnet_params.json, 20). Frequency mode forces 1, since "
             "that reference is deterministic and repeats would be identical",
    )
    parser.add_argument("--output-fname", type=str, default=None)
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
        "loci": str(Path(args.loci).resolve() if args.loci else exp.peaks),
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
    if args.n_shuffles is not None:
        params["n_shuffles"] = args.n_shuffles
    # The reference mode is in the default filename because it changes the
    # numbers: without it, a frequency-mode run silently overwrites a
    # dinucleotide-mode one and nothing on disk records which produced it.
    # Upstream's path omits it; this is a deliberate small divergence.
    if args.output_fname:
        params["output_fname"] = str(REPO_ROOT / args.output_fname)
    elif args.loci:
        # A custom locus set gets its own name for the same reason the reference
        # mode does: different loci, different numbers, and nothing else on disk
        # would record which set produced the file.
        stem = Path(args.loci).name.split(".")[0]
        params["output_fname"] = str(
            ATTR_DIR / f"{exp.id}_{stem}_attr_{args.attribute_type}"
                        f"_{args.reference_mode}.npz")
    else:
        params["output_fname"] = str(
            attribution_path(exp.id, args.attribute_type, args.reference_mode))

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

    # deep_lift_shap REFUSES a sequence with an unknown base -- `ValueError: X
    # must be one-hot encoded. and cannot have unknown characters.` -- and
    # `ignore=IGNORE` above is precisely what creates one: it keeps the locus and
    # zeroes that column rather than dropping it (verified against tangermeme
    # 1.4.1). Both reference modes fail; the frequency reference does not rescue
    # it, since the check is in deep_lift_shap itself, not in the shuffle.
    #
    # Caught here because the library's message says nothing about which loci or
    # what to do, and the remedy is a whole separate script.
    blank = (X.sum(dim=1) == 0).any(dim=-1)
    if blank.any():
        n = int(blank.sum())
        first = [int(i) for i in blank.nonzero()[:5, 0]]
        print(
            f"Error: {n:,} of {len(X):,} loci contain a non-ACGT base "
            f"(rows {first}{'...' if n > 5 else ''}). deep_lift_shap cannot "
            f"attribute these.\n"
            f"  Filter them out first, then attribute the filtered set:\n"
            f"    python src/bpnet/attribute/launch_filter.py -e {exp.id}\n"
            f"    python src/bpnet/attribute/attribute.py -e {exp.id} "
            f"--loci {filtered_loci_path(exp.id)}",
            file=sys.stderr,
        )
        sys.exit(1)

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

        if args.reference_mode == "frequency":
            # n_shuffles=1: the frequency reference is deterministic, so more
            # would be byte-identical copies.
            references, n_shuffles = nucleotide_frequency_references, 1
        else:
            references, n_shuffles = None, params["n_shuffles"]

        attribution_kwargs = {
            "model": model,
            "X": X,
            "hypothetical": True,
            "batch_size": params["batch_size"],
            "n_shuffles": n_shuffles,
            "random_state": params["random_state"],
            "verbose": params["verbose"],
            "additional_nonlinear_ops": additional_nonlinear_ops,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "warning_threshold": 0.01,
        }
        # Only set `references` for frequency mode; omitting it entirely is what
        # selects tangermeme's own dinucleotide shuffling.
        if references is not None:
            attribution_kwargs["references"] = references

        attributions.append(deep_lift_shap(**attribution_kwargs).numpy())

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
