"""
Fit a Cherimoya model for one experiment/fold.

Experiment paths, species and folds resolve through src/experiments.py, the same
way fit_bpnet.py does; hyperparameters come from config/cherimoya_params.json.
The background is restricted to that experiment's GC-matched negatives.

Usage:
    python src/cherimoya/fit/fit_cherimoya.py -e D.melanogaster-S2_PROcap -f 0
    python src/cherimoya/fit/fit_cherimoya.py -e S.cerevisiae-Ino80ctl_PROcap -f 0
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

sys.path.insert(0, str(REPO_ROOT / "src"))
from experiments import (  # noqa: E402
    IGNORE,
    Experiment,
    load_params,
    model_dir,
    model_name,
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True,
        help="experiment ID as it appears in config/experiment_config.yaml",
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        required=True,
        help="fold to hold out for testing (validation = (fold+1) %% n_folds)",
    )
    parser.add_argument("-o", "--output-dir", type=str, default=None)
    parser.add_argument("--n-filters", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--warmup-epochs", type=int, default=None,
        help="linear LR warmup length (default 5, from "
             "config/cherimoya_params.json). Upstream's historical "
             "max_epochs=20 config used 2",
    )
    parser.add_argument(
        "--decay-epochs", type=int, default=None,
        help="decouple the cosine decay's length from --max-epochs (default: "
             "decay across the whole run past warmup, so it finishes exactly "
             "when training ends). Setting it shorter anneals the LR to its "
             "floor early and then holds flat -- e.g. --max-epochs 50 "
             "--decay-epochs 18 decays as fast as upstream's historical "
             "max_epochs=20 config while still training 50 epochs",
    )
    parser.add_argument(
        "--early-stopping", type=int, default=None,
        help="stop after this many consecutive epochs without a new best "
             "valid_count_corr (default: None, training the full --max-epochs "
             "budget). Deliberately OFF: procap-atlas re-enabled it at 5 "
             "against both decay_epochs=None and decay_epochs=15 and both "
             "underperformed on every benchmark metric, profile and count "
             "alike. Note fit() selects checkpoints on valid_count_corr too, "
             "so stopping on the same metric compounds validation overfitting",
    )
    parser.add_argument("--max-jitter", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--negative-ratio", type=float, default=None)
    parser.add_argument(
        "--no-ratio-cap", action="store_true",
        help="do not cap negatives_ratio at the available pool. Same flag and "
             "same meaning as fit_bpnet.py's. The two paths must not diverge",
    )
    parser.add_argument("--muon-lr", type=float, default=None)
    parser.add_argument("--muon-wd", type=float, default=None)
    parser.add_argument("--adam-lr", type=float, default=None)
    parser.add_argument("--adam-wd", type=float, default=None)
    parser.add_argument("--lw-lr", type=float, default=None)
    parser.add_argument("--lw-wd", type=float, default=None)
    parser.add_argument("--lw-momentum", type=float, default=None)
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
    try:
        split = exp.fold_split(args.fold)
    except (KeyError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir(
        "cherimoya", args.experiment
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    params = load_params("cherimoya", {
        "n_filters": args.n_filters,
        "n_layers": args.n_layers,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "warmup_epochs": args.warmup_epochs,
        "decay_epochs": args.decay_epochs,
        "early_stopping": args.early_stopping,
        "max_jitter": args.max_jitter,
        "random_state": args.random_state,
        "negatives_ratio": args.negative_ratio,
        "muon_lr": args.muon_lr,
        "muon_wd": args.muon_wd,
        "adam_lr": args.adam_lr,
        "adam_wd": args.adam_wd,
        "lw_lr": args.lw_lr,
        "lw_wd": args.lw_wd,
        "lw_momentum": args.lw_momentum,
        "verbose": True if args.verbose else None,
    })
    params.update({
        "name": (
            str(Path(args.output_dir) / f"{args.experiment}.fold{args.fold}")
            if args.output_dir
            else model_name("cherimoya", args.experiment, args.fold)
        ),
        "sequences": str(exp.sequences),
        "signals": [str(x) for x in exp.signals],
        "controls": [str(x) for x in exp.controls] if exp.controls else None,
        "loci": str(exp.peaks),
        "negatives": str(exp.negatives),
        "blacklist": exp.blacklist,
        "training_chroms": split["train_chroms"],
        "validation_chroms": split["valid_chroms"],
        "test_chroms": split["test_chroms"],
    })
    train_chroms = split["train_chroms"]
    valid_chroms = split["valid_chroms"]
    test_chroms = split["test_chroms"]

    peaks = load_bed(params["loci"])
    negatives = load_bed(params["negatives"])

    # Peak-level splits (species with too few chromosomes, e.g. S. pombe) filter
    # the peak table instead of holding out chromosomes; fold_loci() returns the
    # same shape either way.
    loci = exp.fold_loci(peaks, args.fold)
    params["training_chroms"] = train_chroms = loci["train_chroms"]
    params["validation_chroms"] = valid_chroms = loci["valid_chroms"]

    print(f"Experiment: {exp.id} ({exp.entry['biosample']}, {exp.species})")
    print(f"Fold {args.fold}: test={test_chroms}, valid={valid_chroms}")
    print(f"Training chroms: {train_chroms}")
    # Same cap as fit_bpnet.py -- see the long note there. negative_ratio is
    # negatives per peak, so a pool smaller than peaks * ratio recycles. This is
    # NOT inert: an earlier version of this comment claimed so on the grounds
    # that the script ran D. melanogaster only, which was never true of the
    # script itself. Fly sits at 0.73-1.00 per peak against the configured 1/7
    # and is unaffected, but the dense yeast experiments are 0.013-0.28, so the
    # cap engages hard there -- Spt5IAA4h would recycle each negative 10.3x.
    configured_ratio = params["negatives_ratio"]
    available_ratio = len(negatives) / max(len(peaks), 1)
    if args.no_ratio_cap:
        available_ratio = configured_ratio
    if available_ratio < configured_ratio:
        params["negatives_ratio"] = available_ratio
        print(
            f"negative_ratio capped {configured_ratio:.4f} -> {available_ratio:.4f}: "
            f"{len(negatives):,} negatives for {len(peaks):,} peaks"
        )
    print(
        "GC-matched negatives: "
        f"{len(negatives):,} loci, ratio={params['negatives_ratio']}"
    )

    import torch
    from cherimoya import Cherimoya
    from data_loader import PeakGenerator
    from tangermeme.io import extract_loci
    from torch.optim import SGD, AdamW
    from torch.optim.lr_scheduler import (
        ConstantLR,
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )

    try:
        from torch.optim import Muon
    except ImportError as exc:
        raise ImportError(
            "torch.optim.Muon requires torch >= 2.10 (pinned in pyproject.toml). "
            "Do not install the PyPI `muon` package as a fallback -- it is an "
            "unrelated multi-omics framework and does not provide this optimizer."
        ) from exc

    train_data_loader = PeakGenerator(
        peaks=loci["train_loci"],
        negatives=negatives,
        sequences=params["sequences"],
        # Nested so cherimoya's normalize_signal_groups treats this as one
        # stranded 2-channel group (correct RC channel-swap behavior) instead of
        # two independent unstranded groups -- the latter is what a flat
        # 2-element list means as of cherimoya's signal-groups refactor.
        # params["signals"] itself stays flat: extract_loci (used directly for
        # validation below) and the model's signal_groups=[len(signals)] both
        # need the flat form.
        signals=[params["signals"]],
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
        loci=loci["valid_loci"],
        sequences=params["sequences"],
        signals=params["signals"],
        in_signals=params["controls"],
        chroms=params["validation_chroms"],
        in_window=params["in_window"],
        out_window=params["out_window"],
        max_jitter=0,
        ignore=IGNORE,
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
    # torch.compile has no Python 3.14 support below torch 2.10; the limit is
    # Dynamo, not Triton itself. torch.__version__ is a TorchVersion, which
    # supports PEP 440-aware comparison against a plain string.
    compile_supported = sys.version_info < (3, 14) or torch.__version__ >= "2.10"
    model = Cherimoya(
        name=params["name"],
        n_filters=params["n_filters"],
        # cherimoya >= 0.2 replaced n_outputs with signal_groups; a single
        # 2-element group is one stranded (pl, mn) pair.
        signal_groups=[len(params["signals"])],
        n_control_tracks=n_control_tracks,
        n_layers=params["n_layers"],
        trimming=(params["in_window"] - params["out_window"]) // 2,
        verbose=params["verbose"],
        compile=compile_supported,
    )
    model = model.to("cuda")

    # Separate parameters for Muon (2D projection weights), AdamW (everything
    # else, including the 2D depth-wise conv_weight), and SGD (the lw0/lw1
    # Kendall uncertainty loss weights, which cherimoya >= 0.2 optimizes with a
    # dedicated optimizer passed to fit()).
    muon_params, adam_params, lw_params = [], [], []
    for name, parameter in model.named_parameters():
        if name in ("lw0", "lw1"):
            lw_params.append(parameter)
        elif (
            parameter.ndim == 2
            and "weight" in name
            and name != "linear.weight"
            and "conv_weight" not in name
        ):
            muon_params.append(parameter)
        else:
            adam_params.append(parameter)

    muon_optimizer = Muon(
        muon_params, lr=params["muon_lr"], weight_decay=params["muon_wd"]
    )
    adam_optimizer = AdamW(
        adam_params, lr=params["adam_lr"], weight_decay=params["adam_wd"]
    )
    lw_optimizer = SGD(
        lw_params,
        lr=params["lw_lr"],
        weight_decay=params["lw_wd"],
        momentum=params["lw_momentum"],
    )

    # Warmup + cosine decay, ported from procap-atlas. `decay_epochs` decouples
    # the cosine decay's length from max_epochs (default None, i.e. decay across
    # the whole run past warmup, which is what this script did when warmup was
    # hard-coded at 5 and there was no decay knob -- so defaults are a no-op).
    # When decay_epochs is shorter, a third ConstantLR stage holds the LR flat
    # at eta_min for the remaining epochs: CosineAnnealingLR is PERIODIC, so
    # without that stage the LR would start rising again past T_max instead of
    # staying at its floor.
    num_warmup_epochs = params["warmup_epochs"]
    max_epochs = params["max_epochs"]
    eta_min = 1e-5
    decay_epochs = params["decay_epochs"] or max(1, max_epochs - num_warmup_epochs)
    hold_epochs = max(0, max_epochs - num_warmup_epochs - decay_epochs)
    num_warmup_iters = len(train_data_loader) * num_warmup_epochs
    num_decay_iters = len(train_data_loader) * decay_epochs
    num_hold_iters = len(train_data_loader) * hold_epochs

    def build_lr_scheduler(optimizer, base_lr):
        schedulers = [
            LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_iters),
            CosineAnnealingLR(optimizer, T_max=num_decay_iters, eta_min=eta_min),
        ]
        milestones = [num_warmup_iters]
        if num_hold_iters > 0:
            # ConstantLR reverts to the optimizer's base_lr once total_iters is
            # reached -- it is designed for warmup, not an indefinite hold -- so
            # total_iters must comfortably exceed num_hold_iters. Setting it to
            # exactly num_hold_iters would snap the LR back up to base_lr on the
            # last step of training.
            schedulers.append(
                ConstantLR(optimizer, factor=eta_min / base_lr,
                           total_iters=num_hold_iters * 10 + 10**6)
            )
            milestones.append(num_warmup_iters + num_decay_iters)
        return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

    muon_scheduler = build_lr_scheduler(muon_optimizer, params["muon_lr"])
    adam_scheduler = build_lr_scheduler(adam_optimizer, params["adam_lr"])
    # Linear warmup then flat (no cosine decay) for the Kendall loss weights.
    lw_scheduler = SequentialLR(
        lw_optimizer,
        schedulers=[
            LinearLR(lw_optimizer, start_factor=0.01, total_iters=num_warmup_iters),
            ConstantLR(lw_optimizer, factor=1.0, total_iters=1),
        ],
        milestones=[num_warmup_iters],
    )

    model.fit(
        training_data=train_data_loader,
        muon_optimizer=muon_optimizer,
        adam_optimizer=adam_optimizer,
        lw_optimizer=lw_optimizer,
        muon_scheduler=muon_scheduler,
        adam_scheduler=adam_scheduler,
        lw_scheduler=lw_scheduler,
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
