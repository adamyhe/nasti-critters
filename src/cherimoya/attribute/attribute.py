"""
Calculate Cherimoya attributions on the configured PRO-cap loci.

The Cherimoya counterpart of src/bpnet/attribute/attribute.py, and deliberately
the same job: every locus of the non-ACGT filtered set, attributed with EVERY
fold's model and averaged (an ensemble attribution, not a held-out estimate),
hypothetical scores, a nucleotide-frequency reference by default, and the output
path from experiments.attribution_path(). Only the attribution engine differs.

REQUIRES cherimoya >= 0.3.1 (with tangermeme >= 1.6.0), i.e. the
cherimoya-0.3.1 image, NOT the 0.2.1 image the models were trained in.
Attribution was blocked until cherimoya 0.3 shipped `cherimoya.deep_lift_shap`,
which registers the DeepLIFT rule for its fused convolution + normalization
itself. Verified 2026-11-18 that 0.2.1 checkpoints compute a bit-identical
function under 0.3.1 (max |diff| 0.0 on both outputs), so attributing them there
is attributing the trained models -- re-check that before relying on any other
version pair.

Why the fast log-counts engine is NOT used by default: it runs only with
dinucleotide shuffles or a one-hot reference tensor, and the frequency reference
is a soft (non-one-hot) PFM, so it takes tangermeme's default path. That costs
little, because the frequency reference is deterministic and needs ONE reference
per sequence rather than 20 shuffles. `--reference-mode dinucleotide` makes the
fast path eligible for the counts head again.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
FAMILY = "cherimoya"
#: attribute type (shared with BPNet, and part of the output path) -> the
#: `output` cherimoya.deep_lift_shap attributes.
OUTPUTS = {"counts": "logcounts", "profile": "profile"}
#: Shuffle count for --reference-mode dinucleotide when cherimoya_params.json
#: sets none: bpnet_params.json's value, so the two families match.
DEFAULT_N_SHUFFLES = 20

sys.path.insert(0, str(REPO_ROOT / "src"))
from experiments import (  # noqa: E402
    ATTR_DIR,
    Experiment,
    attribution_path,
    filtered_loci_path,
    IGNORE,
    loci_stem,
    load_params,
    staged,
)


def _bpnet_attribute():
    """BPNet's attribute.py as a module, for its reference and BED reader.

    Loaded by path rather than copied, so the frequency reference -- the one
    DeepLIFT baseline both families must share for their attributions to be
    comparable -- has exactly one definition. Importing it pulls in only
    numpy/pandas/yaml; its torch imports sit inside main().
    """
    path = REPO_ROOT / "src" / "bpnet" / "attribute" / "attribute.py"
    spec = importlib.util.spec_from_file_location("bpnet_attribute", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e", "--experiment", type=str, required=True,
        help="experiment ID as it appears in config/experiment_config.yaml",
    )
    parser.add_argument(
        "--attribute-type", choices=sorted(OUTPUTS), default="profile",
        help="counts -> cherimoya's log-counts output; profile -> its profile "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--loci", type=str, default=None, metavar="BED",
        help="attribute over this BED instead of the non-ACGT filtered set, "
             "which is the default and is mandatory for the same reason as in "
             "src/bpnet/attribute/attribute.py; a custom set's stem goes into "
             "the output name",
    )
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument(
        "--reference-mode", choices=("frequency", "dinucleotide"),
        default="frequency",
        help="DeepLIFT reference (default: %(default)s, as for BPNet). See "
             "nucleotide_frequency_references() in the BPNet script",
    )
    parser.add_argument(
        "--n-shuffles", type=int, default=None,
        help="dinucleotide shuffles; only used with --reference-mode "
             f"dinucleotide (default: {DEFAULT_N_SHUFFLES}). Frequency mode "
             "forces 1",
    )
    parser.add_argument(
        "--no-fast", dest="fast", action="store_false",
        help="never use cherimoya's fast log-counts engine. It only ever applies "
             "to --attribute-type counts with --reference-mode dinucleotide, and "
             "checks itself against the default path on every call",
    )
    parser.add_argument("--output-fname", type=str, default=None)
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="suppress the progress bars (on by default, stderr)")
    parser.add_argument("-v", "--verbose", action="store_true")
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

    default_loci = filtered_loci_path(exp.id)
    loci_path = Path(args.loci).resolve() if args.loci else default_loci
    if not loci_path.exists():
        print(
            f"Error: loci not found: {loci_path}\n"
            + ("" if args.loci else
               f"  The non-ACGT filtered set is required before attribution. "
               f"Build it with:\n"
               f"    python src/bpnet/attribute/launch_filter.py -e {exp.id}"),
            file=sys.stderr,
        )
        sys.exit(1)

    params = load_params(FAMILY, {})
    progress = args.progress or args.verbose

    folds = exp.all_folds(FAMILY, models_dir=args.models_dir)
    absent = [f for f in folds if not f["model"].exists()]
    if absent:
        for f in absent:
            print(f"Error: model for fold {f['fold']} not found: {f['model']}",
                  file=sys.stderr)
        sys.exit(1)
    # No chromosome restriction, exactly as in the BPNet script: attribution
    # covers every locus with every fold's model. None under peak-level splits.
    chroms = [c for f in folds for c in (f["test_chroms"] or [])] or None

    if args.output_fname:
        output_fname = Path(REPO_ROOT / args.output_fname)
    elif loci_path != default_loci:
        output_fname = (ATTR_DIR / FAMILY / f"{exp.id}_{loci_stem(loci_path)}"
                        f"_attr_{args.attribute_type}_{args.reference_mode}.npz")
    else:
        output_fname = attribution_path(FAMILY, exp.id, args.attribute_type,
                                        args.reference_mode)

    import torch
    from tqdm import tqdm
    from tangermeme.io import extract_loci
    from tangermeme_compat import patch_numeric_chroms
    try:
        from cherimoya import Cherimoya
        from cherimoya.deep_lift_shap import deep_lift_shap
    except ImportError as err:
        # 0.2.x has no cherimoya.deep_lift_shap: the likeliest cause is running
        # in the 0.2.1 image or a 0.2.1 venv.
        import importlib.metadata as md
        print(f"Error: {err}\n  cherimoya {md.version('cherimoya')} is installed; "
              "attribution needs >= 0.3.1. Use the cherimoya-0.3.1 image "
              "(APPTAINER_IMAGE).", file=sys.stderr)
        sys.exit(1)

    patch_numeric_chroms(verbose=args.verbose)
    bpnet = _bpnet_attribute()

    X = extract_loci(
        loci=bpnet.load_bed(loci_path),
        sequences=str(exp.sequences),
        in_window=params["in_window"],
        out_window=params["out_window"],
        chroms=chroms,
        max_jitter=0,
        verbose=progress,
        min_counts=None,
        max_counts=None,
        ignore=IGNORE,
    ).to(torch.float32)

    # Same guard and message as the BPNet script: deep_lift_shap refuses an
    # unknown base, and ignore=IGNORE is what creates one.
    blank = (X.sum(dim=1) == 0).any(dim=-1)
    if blank.any():
        print(f"Error: {int(blank.sum()):,} of {len(X):,} loci contain a non-ACGT "
              f"base. Run python src/bpnet/attribute/launch_filter.py -e {exp.id}",
              file=sys.stderr)
        sys.exit(1)

    if args.reference_mode == "frequency":
        reference_kwargs = {"references": bpnet.nucleotide_frequency_references,
                            "n_shuffles": 1}
    else:
        reference_kwargs = {"n_shuffles": args.n_shuffles or params.get("n_shuffles")
                            or DEFAULT_N_SHUFFLES}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    attributions = []
    for f in tqdm(folds, desc=f"{exp.id} {args.attribute_type} folds",
                  unit="fold", disable=not progress):
        # compile=False is REQUIRED: attribution hooks the module tree, which a
        # compiled model does not expose. Cherimoya.load() defaults to True.
        model = Cherimoya.load(str(f["model"]), device=device, compile=False).eval()
        attributions.append(deep_lift_shap(
            model, X,
            output=OUTPUTS[args.attribute_type],
            group=0,
            batch_size=params["batch_size"],
            hypothetical=True,
            # The BPNet script's threshold, so convergence warnings mean the
            # same thing for both families.
            warning_threshold=0.01,
            random_state=params.get("random_state"),
            verbose=progress,
            device=device,
            fast=args.fast,
            **reference_kwargs,
        ).numpy())
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_fname.parent.mkdir(parents=True, exist_ok=True)
    with staged(output_fname) as tmp:
        np.savez_compressed(
            tmp,
            np.stack(attributions).mean(axis=0) if len(attributions) > 1
            else attributions[0],
        )


if __name__ == "__main__":
    main()
