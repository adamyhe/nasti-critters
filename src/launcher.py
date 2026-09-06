#!/usr/bin/env python3
"""Shared training-job launcher for both model families.

One selection rule, two entry points: `src/bpnet/fit/launch.py` and
`src/cherimoya/fit/launch.py` are thin wrappers over `run()`. That is
deliberate. Everything a launcher decides -- which (experiment, fold) pairs
exist, which already finished, which are missing inputs -- is family-agnostic
because `src/experiments.py` is already keyed by family (`model_path(family,
...)`, `all_folds(family)`, `load_params(family)`). The only differences are the
fit script's path and whether the family accepts `--controls`.

Two copies would drift. This repo has been bitten by that repeatedly and the
survivors carry warnings about it -- `adapter_arg()`, `decoy_accessions()` and
`AWK_DEINTERLEAVE` all exist in two drivers with "keep the two in step" notes.
A launcher is local code, unlike `data_loader.py`, which is duplicated per
family precisely because it is byte-identical to procap-atlas's and must not be
forked. There is no such reason here.

Three emission modes over the one selection rule:

    --print-commands   one bare shell command per job on stdout. No SBATCH
                       directives, no env setup, nothing submitted. The
                       non-SLURM path.
    --dry-run          the full sbatch script per job, printed not submitted.
    (neither)          submit via sbatch.
"""

import argparse
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: SLURM partitions, and the GPU constraint, are SITE-SPECIFIC and were
#: deliberately absent from this repo until 2026-09-05 -- launchers emitted no
#: `#SBATCH --partition`/`-C` unless asked. That rule was relaxed on request,
#: because typing them on every submission is its own error source. Everything
#: here stays overridable with --partition / --constraint, so a different site
#: needs a flag rather than a patch.
GPU_PARTITION = "akundaje,owners"
CPU_PARTITION = "normal,akundaje,owners"

#: GPU SKUs this code is known to run on, copied from procap-atlas's cherimoya
#: launchers. `|` is SLURM's OR for a constraint list: any ONE of these will do.
#: Without it a job can land on a card too old for the pinned torch, or too
#: small for a 2114 bp window on a multi-gigabase genome.
GPU_CONSTRAINT = "|".join([
    "GPU_SKU:A100_PCIE",
    "GPU_SKU:A100_SXM4",
    "GPU_SKU:A40",
    "GPU_SKU:H100_SXM5",
    "GPU_SKU:H200_SXM5",
    "GPU_SKU:L40S",
    "GPU_SKU:RTX_3090",
])

sys.path.insert(0, str(REPO_ROOT / "src"))
from experiments import (
    Experiment,
    attribution_path,
    experiment_ids,
    filtered_loci_path,
    load_params,
    model_path,
    modisco_h5_path,
    modisco_report_dir,
    motif_db_path,
    ohe_path,
)


def build_parser(
    family: str, fit_script: Path, *, supports_controls: bool
) -> argparse.ArgumentParser:
    rel = fit_script.relative_to(REPO_ROOT)
    launcher = f"src/{family}/fit/launch.py"
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f"""\
            Enumerate {family} training jobs for all experiments and folds.

            Reads experiment IDs from config/experiment_config.yaml and produces
            one job per (experiment, fold) pair, running {rel}. Experiments whose
            model already finished, whose inputs are missing, or whose species
            has no fold assignment are skipped, and each skip says which.

            Usage:
                python {launcher}                    # submit all experiments x folds
                python {launcher} --dry-run          # print sbatch scripts
                python {launcher} --print-commands   # bare commands, for any box
                python {launcher} --time 12:00:00 --mem 32G --partition gpu

            On a single non-SLURM box, run them serially -- these are GPU jobs
            and one box almost certainly holds one at a time:

                python {launcher} --print-commands | bash

            To run N at once, having checked N models fit in VRAM:

                python {launcher} --print-commands | xargs -P N -I{{}} bash -c '{{}}'

            Skips and the summary go to stderr in that mode, so stdout stays
            pipeable.
            """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(parser, launcher, gpu=True)
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="+",
        default=None,
        metavar="EXP",
        help="limit to these experiment IDs (default: every one in the config)",
    )
    parser.add_argument(
        "--requeue",
        action="store_true",
        help="submit with --requeue so pre-empted jobs are resubmitted by SLURM. "
        "Each job retrains its fold from epoch 0 (neither bpnet-lite's nor "
        "cherimoya's fit() has resume), but re-checks for a completed model "
        "first.",
    )
    if supports_controls:
        parser.add_argument(
            "--controls",
            action="store_true",
            help=f"train with bias-control tracks (passes --controls to "
            f"{rel.name} and checks for pl_control/mn_control paths in the "
            f"experiment config)",
        )
    parser.add_argument(
        "--fit-args",
        type=str,
        default="",
        help=f"extra arguments forwarded to {rel.name} (e.g. '--max-epochs 100')",
    )
    return parser


def _add_common_args(
    parser: argparse.ArgumentParser,
    launcher: str,
    *,
    gpu: bool,
    default_cpus: int = 4,
    default_mem: str = "32G",
    default_time: str = "6:00:00",
) -> None:
    """Emission mode + SLURM resources. Shared by every launcher kind."""
    emit = parser.add_mutually_exclusive_group()
    emit.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts without submitting",
    )
    emit.add_argument(
        "--print-commands",
        action="store_true",
        help="print one bare shell command per job to stdout and submit nothing. "
        "For a non-SLURM box. The env setup block is NOT included -- "
        "activate the mamba env and uv venv yourself first, exactly as "
        "--setup-file would inside a job. Skip messages and the summary go "
        "to stderr so stdout can be piped to bash or xargs",
    )
    # SLURM resource flags. Site-specific by nature: no default, and the
    # corresponding #SBATCH line is omitted unless a value is given.
    parser.add_argument(
        "--constraint",
        "--gpus",
        dest="constraint",
        type=str,
        default=GPU_CONSTRAINT if gpu else None,
        help="value for #SBATCH -C. Defaults to the GPU SKUs this code is known "
             "to run on, `|`-joined so any one of them satisfies it; CPU-only "
             "launchers default to none, since a GPU SKU constraint there would "
             "restrict scheduling for nothing. Pass an empty string to drop it",
    )
    parser.add_argument(
        "--partition", type=str,
        default=GPU_PARTITION if gpu else CPU_PARTITION,
        help="#SBATCH --partition (default: %(default)s)",
    )
    parser.add_argument(
        "--setup-file",
        type=str,
        default=None,
        help="shell snippet sourced before training: module loads, env "
        "activation, etc. Defaults to activating the repo's mamba env "
        "and uv venv.",
    )
    # Defaults are per launcher: a training fold and a tfmodisco run want very
    # different walls, and a shared default that suits neither is how a 40-hour
    # job gets killed at 6.
    parser.add_argument("--cpus-per-task", type=int, default=default_cpus)
    parser.add_argument("--mem", type=str, default=default_mem)
    parser.add_argument(
        "--time", type=str, default=default_time,
        help="#SBATCH --time (default: %(default)s). Note the `owners` "
             "partition, which every default here includes, caps jobs at "
             "48:00:00 -- ask for more and the job simply will not schedule "
             "there. modisco motifs sits exactly on that cap",
    )


def _setup_block(args) -> str:
    """Environment setup injected into each generated job.

    Module names differ per site, so the default only activates this repo's own
    environment; --setup-file replaces it wholesale.
    """
    if args.setup_file:
        return Path(args.setup_file).read_text().rstrip()
    return textwrap.dedent(f"""\
        # Default setup: this repo's mamba env + uv venv (venv last so its
        # interpreter wins). Use --setup-file for site-specific module loads.
        if command -v mamba >/dev/null; then
            eval "$(mamba shell hook --shell bash)"
            mamba activate nasti-critters || true
        fi
        [ -f {REPO_ROOT}/.venv/bin/activate ] && . {REPO_ROOT}/.venv/bin/activate""").rstrip()


def _select_experiments(parser, args) -> list[str]:
    experiments = experiment_ids()
    if args.experiments is not None:
        unknown = [e for e in args.experiments if e not in experiments]
        if unknown:
            parser.error(f"unknown experiment(s): {', '.join(unknown)}")
        return list(args.experiments)
    return experiments


def _emit(
    args,
    log_dir: Path,
    setup: str,
    job_name: str,
    command: str,
    *,
    gpus: int = 1,
    env: dict[str, object] | None = None,
) -> bool:
    """Emit one job in whichever of the three modes is active.

    Returns True if a job was emitted. Shared by the fit and attribution
    launchers so the sbatch directives, the setup block and the three modes
    cannot drift between them.
    """
    # Environment goes on the command as a `VAR=value cmd` prefix rather than as
    # an `export` line in the sbatch body, so the SAME string carries it in all
    # three emission modes. An export would silently vanish under
    # --print-commands, which is the mode most likely to run on a box where the
    # variable matters.
    if env:
        command = (
            " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
            + " "
            + command
        )

    if args.print_commands:
        print(command)
        return True

    directives = [
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --ntasks=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --nodes=1",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --output={log_dir}/{job_name}.out",
        f"#SBATCH --error={log_dir}/{job_name}.err",
    ]
    # gpus=0 for CPU-only work. Requesting a GPU for it would queue behind the
    # GPU partition and hold an idle card for the duration.
    if gpus:
        directives.insert(4, f"#SBATCH --gpus={gpus}")
    if getattr(args, "requeue", False):
        directives.append("#SBATCH --requeue")
    # -C only on GPU jobs: the constraint is a GPU SKU list, so on a CPU job it
    # would narrow scheduling to GPU nodes for no reason. Gated on `gpus` rather
    # than on the value being unset, so a global --constraint cannot leak onto
    # the CPU launchers.
    if gpus and args.constraint:
        directives.insert(4, f"#SBATCH -C {args.constraint}")
    if args.partition:
        directives.insert(4, f"#SBATCH --partition={args.partition}")

    body = ["#!/bin/bash -l", *directives, "", setup, ""]
    if gpus:
        body.append("command -v nvidia-smi >/dev/null && nvidia-smi -L || true")
    sbatch_script = "\n".join([*body, command, ""])

    if args.dry_run:
        print(f"--- {job_name} ---")
        print(sbatch_script)
        return True

    result = subprocess.run(
        ["sbatch"], input=sbatch_script, capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"{job_name}: {result.stdout.strip()}")
        return True
    print(f"ERROR submitting {job_name}: {result.stderr.strip()}", file=sys.stderr)
    return False


def _action(args) -> str:
    if args.print_commands:
        return "Printed"
    return "Would submit" if args.dry_run else "Submitted"


def run(family: str, fit_script: Path, *, supports_controls: bool = False) -> None:
    """Enumerate and emit every (experiment, fold) job for one family."""
    parser = build_parser(family, fit_script, supports_controls=supports_controls)
    args = parser.parse_args()
    # Only bpnet takes bias controls, so the flag is absent from cherimoya's
    # parser rather than accepted and ignored -- a flag that silently does
    # nothing is worse than one that does not exist.
    use_controls = getattr(args, "controls", False)

    experiments = _select_experiments(parser, args)

    log_dir = REPO_ROOT / "logs" / f"{family}_fit"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup = _setup_block(args)

    submitted = 0
    skipped_trained = 0
    skipped_missing = 0
    skipped_experiments = 0
    total = 0
    for exp_id in experiments:
        try:
            exp = Experiment.load(exp_id, use_controls=use_controls)
        except KeyError as err:
            print(f"SKIP {exp_id}: {err}", file=sys.stderr)
            skipped_experiments += 1
            continue

        # A species with no fold assignment is skipped, not fatal. This used to
        # test `exp.species == "S.pombe"`, which crashed the whole launcher the
        # moment a second fold-less species appeared -- C.griseus and
        # S.moellendorffii both arrived with the McDonald2024/Shamie2021 update.
        # n_folds() raises with the remedy, so just report it.
        try:
            n_folds = exp.n_folds()
        except KeyError as err:
            print(f"SKIP {exp_id}: {err}", file=sys.stderr)
            skipped_experiments += 1
            continue
        total += n_folds

        missing = exp.missing
        if missing:
            print(
                f"SKIP {exp_id}: missing data — {', '.join(missing)}",
                file=sys.stderr,
            )
            skipped_missing += n_folds
            continue

        for fold in range(n_folds):
            # Skip only if the fold actually finished. Both families write
            # "{name}.torch" every time validation loss improves (so it can
            # exist after one epoch) and "{name}.final.torch" exactly once, at
            # the end of fit() -- verified against bpnetlite/bpnet.py and
            # cherimoya/cherimoya.py, which call torch.save on both paths.
            # Checking the former would silently treat a partially trained fold
            # as complete.
            suffix = "_ctl" if use_controls else ""
            done = model_path(family, exp_id, fold, final=True, suffix=suffix)
            if done.exists():
                skipped_trained += 1
                continue

            job_name = f"{family}_{exp_id}{'_ctl' if use_controls else ''}_f{fold}"
            fit_cmd = (
                f"python {shlex.quote(str(fit_script))} "
                f"-e {shlex.quote(exp_id)} --fold {fold} -v"
            )
            if use_controls:
                fit_cmd += " --controls"
            if args.fit_args:
                fit_cmd += f" {args.fit_args}"

            submitted += _emit(args, log_dir, setup, job_name, fit_cmd)

    action = _action(args)
    # `total` is accumulated in the loop above. It used to be a second pass that
    # reloaded every experiment and carried `entry.species == "S.pombe"` as a
    # special case -- dead code twice over, since a species with no fold
    # assignment raises from n_folds() and is already excluded, and the rest of
    # this launcher stopped naming pombe when C.griseus and S.moellendorffii
    # arrived. Do not reintroduce a species-name test here.
    print(
        f"\n{action} {submitted} {family} jobs of {total} (experiment x fold); "
        f"skipped {skipped_trained} already trained, {skipped_missing} missing data"
        + (
            f", {skipped_experiments} experiments unusable"
            if skipped_experiments
            else ""
        ),
        # stdout must stay pipeable when it carries commands.
        file=sys.stderr if args.print_commands else sys.stdout,
    )


def build_attribute_parser(family: str, script: Path) -> argparse.ArgumentParser:
    launcher = f"src/{family}/attribute/launch.py"
    return _finish_attribute_parser(
        argparse.ArgumentParser(
            description=textwrap.dedent(f"""\
            Enumerate {family} attribution jobs.

            The job unit here is (experiment x attribute type), NOT
            (experiment x fold) as it is for training: {script.name} loops every
            fold internally and averages their attributions, so one job covers
            all folds of one experiment. That is why this is a separate
            enumeration rather than a flag on the fit launcher; the emission
            machinery is shared through src/launcher.py either way.

            Skips an experiment whose inputs are missing or whose folds are not
            all trained -- {script.name} exits 1 on both -- and skips a job
            whose output npz already exists.

            Usage:
                python {launcher} --dry-run
                python {launcher} --attribute-type profile --attribute-type counts
                python {launcher} --print-commands | bash
            """),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        ),
        launcher,
        script,
    )


def _finish_attribute_parser(parser, launcher: str, script: Path):
    _add_common_args(parser, launcher, gpu=True)
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="+",
        default=None,
        metavar="EXP",
        help="limit to these experiment IDs (default: every one in the config)",
    )
    parser.add_argument(
        "--attribute-type",
        dest="attribute_types",
        action="append",
        choices=("profile", "counts"),
        default=None,
        metavar="TYPE",
        help="repeatable; one job per type per experiment (default: profile)",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("frequency", "dinucleotide"),
        default="frequency",
        help="passed through to %(default)s-mode attribution, and part of the "
        "output filename, so the already-done check follows it",
    )
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument(
        "--attr-args",
        type=str,
        default="",
        help=f"extra arguments forwarded to {script.name}",
    )
    return parser


def run_attribute(family: str, script: Path) -> None:
    """Enumerate and emit every (experiment, attribute type) attribution job."""
    parser = build_attribute_parser(family, script)
    args = parser.parse_args()
    types = args.attribute_types or ["profile"]

    experiments = _select_experiments(parser, args)
    log_dir = REPO_ROOT / "logs" / f"{family}_attribute"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup = _setup_block(args)

    submitted = skipped_done = skipped_missing = skipped_untrained = 0
    skipped_experiments = 0
    total = 0
    for exp_id in experiments:
        try:
            exp = Experiment.load(exp_id)
        except KeyError as err:
            print(f"SKIP {exp_id}: {err}", file=sys.stderr)
            skipped_experiments += 1
            continue
        try:
            exp.n_folds()
        except KeyError as err:
            print(f"SKIP {exp_id}: {err}", file=sys.stderr)
            skipped_experiments += 1
            continue
        total += len(types)

        if exp.missing:
            print(
                f"SKIP {exp_id}: missing data — {', '.join(exp.missing)}",
                file=sys.stderr,
            )
            skipped_missing += len(types)
            continue

        # attribute.py averages over EVERY fold, so a partly trained experiment
        # is not a partly usable one -- it exits 1. Report how far along it is
        # rather than just refusing, since "3 of 5 trained" is the actionable
        # form of that message.
        folds = exp.all_folds(family, models_dir=args.models_dir)
        absent = [f for f in folds if not f["model"].exists()]
        if absent:
            print(
                f"SKIP {exp_id}: {len(folds) - len(absent)}/{len(folds)} folds "
                f"trained; attribution needs all of them",
                file=sys.stderr,
            )
            skipped_untrained += len(types)
            continue

        # The non-ACGT filtered set is attribute.py's default and is MANDATORY,
        # not an option: deep_lift_shap refuses a sequence containing an unknown
        # base and extract_loci(ignore=...) creates one for any window holding
        # an N. So an experiment that has not been filtered cannot be attributed
        # at all, and this skips it rather than emitting a job that will exit 1.
        if not filtered_loci_path(exp_id).exists():
            print(
                f"SKIP {exp_id}: not filtered yet; run launch_filter.py "
                f"-e {exp_id} first",
                file=sys.stderr,
            )
            skipped_missing += len(types)
            continue

        for attribute_type in types:
            out = attribution_path(family, exp_id, attribute_type,
                                   args.reference_mode)
            if out.exists():
                skipped_done += 1
                continue

            job_name = f"{family}_attr_{exp_id}_{attribute_type}_{args.reference_mode}"
            cmd = (
                f"python {shlex.quote(str(script))} "
                f"-e {shlex.quote(exp_id)} "
                f"--attribute-type {attribute_type} "
                f"--reference-mode {args.reference_mode}"
            )
            if args.models_dir:
                cmd += f" --models-dir {shlex.quote(args.models_dir)}"
            if args.attr_args:
                cmd += f" {args.attr_args}"

            submitted += _emit(args, log_dir, setup, job_name, cmd)

    print(
        f"\n{_action(args)} {submitted} {family} attribution jobs of {total} "
        f"(experiment x type); skipped {skipped_done} already done, "
        f"{skipped_missing} missing data, {skipped_untrained} not fully trained"
        + (
            f", {skipped_experiments} experiments unusable"
            if skipped_experiments
            else ""
        ),
        file=sys.stderr if args.print_commands else sys.stdout,
    )


def build_filter_parser(family: str, script: Path) -> argparse.ArgumentParser:
    launcher = f"src/{family}/attribute/launch_filter.py"
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f"""\
            Enumerate non-ACGT locus-filtering jobs, one per experiment.

            Runs {script.name} over each experiment's peaks, writing a filtered
            BED and its one-hot encoding. This step is OPTIONAL and nothing
            downstream requires it: attribute.py reads the experiment's peaks
            unless pointed at the filtered BED with --loci. It matters where a
            blanked position is unacceptable -- `extract_loci(ignore=...)` keeps
            a locus containing an N and zeroes that column, where this drops the
            locus outright.

            CPU-only, so these jobs request NO GPU. That is the reason this is a
            separate launcher rather than another --attribute-type: sending
            filtering to the GPU partition would queue it behind training and
            hold an idle card while it reads a FASTA.

            Usage:
                python {launcher} --dry-run
                python {launcher} --print-commands | bash
            """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(parser, launcher, gpu=False)
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="+",
        default=None,
        metavar="EXP",
        help="limit to these experiment IDs (default: every one in the config)",
    )
    parser.add_argument(
        "--in-window",
        type=int,
        default=None,
        help="window checked for non-ACGT bases (default: in_window from "
        "config/bpnet_params.json, 2114). Must match what attribute.py "
        "will use, or the filter tests a different span than the model sees",
    )
    parser.add_argument(
        "--no-ohe",
        dest="save_ohe",
        action="store_false",
        help="write only the filtered BED, skipping the one-hot encoding",
    )
    parser.add_argument(
        "--filter-args",
        type=str,
        default="",
        help=f"extra arguments forwarded to {script.name}",
    )
    return parser


def run_filter(family: str, script: Path) -> None:
    """Enumerate and emit one locus-filtering job per experiment."""
    parser = build_filter_parser(family, script)
    args = parser.parse_args()
    in_window = args.in_window or load_params("bpnet")["in_window"]

    experiments = _select_experiments(parser, args)
    log_dir = REPO_ROOT / "logs" / f"{family}_filter"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup = _setup_block(args)

    submitted = skipped_done = skipped_missing = skipped_experiments = 0
    total = 0
    for exp_id in experiments:
        try:
            exp = Experiment.load(exp_id)
        except KeyError as err:
            print(f"SKIP {exp_id}: {err}", file=sys.stderr)
            skipped_experiments += 1
            continue
        total += 1

        # Only peaks and sequences are read, so gate on those rather than on
        # exp.missing, which also demands negatives and trained-model inputs
        # this step has nothing to do with.
        absent = exp.missing_paths(kinds=("peaks", "sequences"))
        if absent:
            print(f"SKIP {exp_id}: missing data — {', '.join(absent)}", file=sys.stderr)
            skipped_missing += 1
            continue

        out_bed = filtered_loci_path(exp_id)
        out_ohe = ohe_path(exp_id)
        if out_bed.exists() and (not args.save_ohe or out_ohe.exists()):
            skipped_done += 1
            continue

        cmd = (
            f"python {shlex.quote(str(script))} "
            f"-b {shlex.quote(str(exp.peaks))} "
            f"-f {shlex.quote(str(exp.sequences))} "
            f"-o {shlex.quote(str(out_bed))} "
            f"-w {in_window} -v"
        )
        if args.save_ohe:
            cmd += f" --save-ohe {shlex.quote(str(out_ohe))}"
        if args.filter_args:
            cmd += f" {args.filter_args}"

        submitted += _emit(
            args, log_dir, setup, f"{family}_filter_{exp_id}", cmd, gpus=0
        )

    print(
        f"\n{_action(args)} {submitted} {family} filter jobs of {total} "
        f"(one per experiment); skipped {skipped_done} already done, "
        f"{skipped_missing} missing data"
        + (
            f", {skipped_experiments} experiments unusable"
            if skipped_experiments
            else ""
        ),
        file=sys.stderr if args.print_commands else sys.stdout,
    )


def numba_env(args) -> dict[str, int]:
    """Pin numba to the cores the job actually asked for.

    numba sets `NUMBA_NUM_THREADS` from every core it can SEE, which on a shared
    node is the whole machine rather than the slice SLURM granted. A job holding
    32 CPUs on a 128-core node then spawns 128 threads, oversubscribes its own
    cgroup and can run slower than if it had asked for less -- while degrading
    whatever else is on the node. tfmodisco-lite is numba-heavy throughout, so
    this matters here more than anywhere else in the repo.
    """
    return {"NUMBA_NUM_THREADS": args.cpus_per_task}


def _add_modisco_args(
    parser, launcher: str, *, default_cpus: int, default_mem: str,
    default_time: str,
) -> None:
    """Flags shared by the motifs and report launchers.

    Resources are NOT shared and have no default here, deliberately. The two
    commands differ by more than an order of magnitude in every dimension:
    `modisco motifs` is numba-parallel and runs for many hours, `modisco report`
    is single-threaded and finishes inside two. Defaulting them together meant
    every report job reserved 32 idle cores for 48 hours, which queues badly and
    wastes allocation. Making these required keyword arguments is what stops the
    next caller inheriting the wrong set by omission.
    """
    _add_common_args(parser, launcher, gpu=False, default_cpus=default_cpus,
                     default_mem=default_mem, default_time=default_time)
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="+",
        default=None,
        metavar="EXP",
        help="limit to these experiment IDs (default: every one in the config)",
    )
    parser.add_argument(
        "--attribute-type",
        dest="attribute_types",
        action="append",
        choices=("profile", "counts"),
        default=None,
        metavar="TYPE",
        help="repeatable; one job per type per experiment (default: profile)",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("frequency", "dinucleotide"),
        default="frequency",
        help="which attribution run to consume; part of the input and output "
        "names, so the already-done check follows it",
    )


def run_modisco(family: str) -> None:
    """`modisco motifs`, one job per (experiment, attribute type).

    Consumes the attribution npz and the one-hot npz -- modisco needs both, and
    the OHE is why filter_nonACGT_regions.py writes one. Parameters follow
    procap-atlas's -n 1000000 seqlets and -w 1000 window, but NOT its -l 50:
    that argument is the number of Leiden CLUSTERINGS (random restarts), whose
    library default is 2, and upstream's help text mislabels it as a count of
    clusters.
    """
    launcher = f"src/{family}/modisco/launch.py"
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f"""\
            Enumerate `modisco motifs` jobs for {family} attributions.

            One job per (experiment x attribute type), consuming that run's
            attribution npz plus the experiment's one-hot npz. Both come from
            the attribution stage, so the order is
            launch_filter.py -> attribute/launch.py -> here.

            CPU-only: tfmodisco-lite does not use a GPU, so these jobs request
            none. They are long, though -- upstream allows two days -- so raise
            --time rather than assuming the fit default fits.

            Usage:
                python {launcher} --dry-run
                python {launcher} --attribute-type profile --attribute-type counts
                python {launcher} --print-commands | bash
            """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_modisco_args(parser, launcher, default_cpus=32,
                      default_mem="64G", default_time="48:00:00")
    parser.add_argument(
        "-n",
        "--n-seqlets",
        type=int,
        default=1_000_000,
        help="max seqlets (default: %(default)s)",
    )
    parser.add_argument(
        "-l",
        "--leiden",
        type=int,
        default=2,
        help="number of Leiden CLUSTERINGS to run, each with a different random "
             "seed -- restarts, NOT clusters (default: %(default)s, which is "
             "modisco-lite's own default). procap-atlas passes 50 and calls it "
             "'leiden clusters'; that description is wrong per modisco-lite's "
             "own --n_leiden help, and 50 restarts is 25x the library default "
             "in compute",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=1000,
        help="seqlet window (default: %(default)s)",
    )
    parser.add_argument(
        "--modisco-args",
        type=str,
        default="",
        help="extra arguments forwarded to `modisco motifs`",
    )
    args = parser.parse_args()
    types = args.attribute_types or ["profile"]

    experiments = _select_experiments(parser, args)
    log_dir = REPO_ROOT / "logs" / f"{family}_modisco"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup = _setup_block(args)

    submitted = skipped_done = skipped_missing = 0
    total = 0
    for exp_id in experiments:
        ohe = ohe_path(exp_id)
        for attribute_type in types:
            total += 1
            attr = attribution_path(family, exp_id, attribute_type,
                                    args.reference_mode)
            out = modisco_h5_path(family, exp_id, attribute_type,
                                  args.reference_mode)
            if out.exists():
                skipped_done += 1
                continue
            absent = [str(p) for p in (ohe, attr) if not p.exists()]
            if absent:
                print(
                    f"SKIP {exp_id} {attribute_type}: missing {', '.join(absent)}",
                    file=sys.stderr,
                )
                skipped_missing += 1
                continue

            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = (
                f"modisco motifs -s {shlex.quote(str(ohe))} "
                f"-a {shlex.quote(str(attr))} -o {shlex.quote(str(out))} "
                f"-n {args.n_seqlets} -l {args.leiden} -w {args.window} -v"
            )
            if args.modisco_args:
                cmd += f" {args.modisco_args}"
            submitted += _emit(
                args,
                log_dir,
                setup,
                f"{family}_modisco_{exp_id}_{attribute_type}",
                cmd,
                gpus=0,
                env=numba_env(args),
            )

    print(
        f"\n{_action(args)} {submitted} {family} modisco jobs of {total} "
        f"(experiment x type); skipped {skipped_done} already done, "
        f"{skipped_missing} missing inputs",
        file=sys.stderr if args.print_commands else sys.stdout,
    )


def run_modisco_report(family: str) -> None:
    """`modisco report`, one job per completed .h5.

    Run after run_modisco. The MEME database is chosen PER SPECIES rather than
    hardcoded -- see motif_db_path() for why a vertebrate database is wrong for
    ten of these twelve species.
    """
    launcher = f"src/{family}/modisco/launch_report.py"
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f"""\
            Enumerate `modisco report` jobs for completed {family} modisco runs.

            Skips an experiment whose .h5 is missing (run launch.py first) or
            whose report directory already exists. The MEME motif database is
            resolved per species from config/genomes.yaml's jaspar_collection;
            nothing fetches those files, so download them into data/motifs/.

            Usage:
                python {launcher} --dry-run
                python {launcher} --print-commands | bash
            """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Short: reads one .h5, matches its motifs against a MEME database and
    # writes logos. NOT single-threaded, though -- it calls memelite.tomtom,
    # which is @njit(parallel=True) and is handed no n_jobs, so it takes every
    # numba thread it can see. 4 cores rather than 1 so that parallel section
    # has something to use, and rather than 32 because the tomtom call is small
    # and the wall is dominated by logo rendering. numba_env pins
    # NUMBA_NUM_THREADS to whatever is requested, so the threads match the
    # allocation instead of the node.
    _add_modisco_args(parser, launcher, default_cpus=4,
                      default_mem="16G", default_time="2:00:00")
    parser.add_argument(
        "--motif-db",
        type=str,
        default=None,
        metavar="MEME",
        help="override the per-species MEME database with one file for every "
        "experiment. Rarely right in this repo -- see motif_db_path()",
    )
    parser.add_argument(
        "--report-args",
        type=str,
        default="",
        help="extra arguments forwarded to `modisco report`",
    )
    args = parser.parse_args()
    types = args.attribute_types or ["profile"]

    experiments = _select_experiments(parser, args)
    log_dir = REPO_ROOT / "logs" / f"{family}_modisco_report"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup = _setup_block(args)

    submitted = skipped_done = skipped_missing = skipped_experiments = 0
    total = 0
    for exp_id in experiments:
        try:
            species = Experiment.load(exp_id).species
        except KeyError as err:
            print(f"SKIP {exp_id}: {err}", file=sys.stderr)
            skipped_experiments += 1
            continue
        try:
            db = Path(args.motif_db) if args.motif_db else motif_db_path(species)
        except KeyError as err:
            print(f"SKIP {exp_id}: {err}", file=sys.stderr)
            skipped_experiments += 1
            continue

        for attribute_type in types:
            total += 1
            h5 = modisco_h5_path(family, exp_id, attribute_type,
                                 args.reference_mode)
            out = modisco_report_dir(family, exp_id, attribute_type,
                                     args.reference_mode)
            if out.exists():
                skipped_done += 1
                continue
            absent = [str(p) for p in (h5, db) if not p.exists()]
            if absent:
                print(
                    f"SKIP {exp_id} {attribute_type}: missing {', '.join(absent)}",
                    file=sys.stderr,
                )
                skipped_missing += 1
                continue

            cmd = (
                f"modisco report -i {shlex.quote(str(h5))} "
                f"-o {shlex.quote(str(out))} -m {shlex.quote(str(db))} --lite"
            )
            if args.report_args:
                cmd += f" {args.report_args}"
            submitted += _emit(
                args,
                log_dir,
                setup,
                f"{family}_modisco_report_{exp_id}_{attribute_type}",
                cmd,
                gpus=0,
                env=numba_env(args),
            )

    print(
        f"\n{_action(args)} {submitted} {family} modisco report jobs of {total} "
        f"(experiment x type); skipped {skipped_done} already done, "
        f"{skipped_missing} missing inputs"
        + (
            f", {skipped_experiments} experiments unusable"
            if skipped_experiments
            else ""
        ),
        file=sys.stderr if args.print_commands else sys.stdout,
    )
