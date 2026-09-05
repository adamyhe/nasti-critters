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

sys.path.insert(0, str(REPO_ROOT / "src"))
from experiments import (  # noqa: E402
    Experiment,
    attribution_path,
    experiment_ids,
    model_path,
)


def build_parser(family: str, fit_script: Path, *,
                 supports_controls: bool) -> argparse.ArgumentParser:
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
    _add_common_args(parser, launcher)
    parser.add_argument(
        "-e", "--experiments", nargs="+", default=None, metavar="EXP",
        help="limit to these experiment IDs (default: every one in the config)",
    )
    parser.add_argument(
        "--requeue", action="store_true",
        help="submit with --requeue so pre-empted jobs are resubmitted by SLURM. "
             "Each job retrains its fold from epoch 0 (neither bpnet-lite's nor "
             "cherimoya's fit() has resume), but re-checks for a completed model "
             "first.",
    )
    if supports_controls:
        parser.add_argument(
            "--controls", action="store_true",
            help=f"train with bias-control tracks (passes --controls to "
                 f"{rel.name} and checks for pl_control/mn_control paths in the "
                 f"experiment config)",
        )
    parser.add_argument(
        "--fit-args", type=str, default="",
        help=f"extra arguments forwarded to {rel.name} "
             f"(e.g. '--max-epochs 100')",
    )
    return parser


def _add_common_args(parser: argparse.ArgumentParser, launcher: str) -> None:
    """Emission mode + SLURM resources. Shared by every launcher kind."""
    emit = parser.add_mutually_exclusive_group()
    emit.add_argument(
        "--dry-run", action="store_true",
        help="print sbatch scripts without submitting",
    )
    emit.add_argument(
        "--print-commands", action="store_true",
        help="print one bare shell command per job to stdout and submit nothing. "
             "For a non-SLURM box. The env setup block is NOT included -- "
             "activate the mamba env and uv venv yourself first, exactly as "
             "--setup-file would inside a job. Skip messages and the summary go "
             "to stderr so stdout can be piped to bash or xargs",
    )
    # SLURM resource flags. Site-specific by nature: no default, and the
    # corresponding #SBATCH line is omitted unless a value is given.
    parser.add_argument(
        "--constraint", "--gpus", dest="constraint", type=str, default=None,
        help="value for #SBATCH -C (e.g. a GPU SKU/generation selector)",
    )
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument(
        "--setup-file", type=str, default=None,
        help="shell snippet sourced before training: module loads, env "
             "activation, etc. Defaults to activating the repo's mamba env "
             "and uv venv.",
    )
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument("--time", type=str, default="6:00:00")


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


def _emit(args, log_dir: Path, setup: str, job_name: str, command: str) -> bool:
    """Emit one job in whichever of the three modes is active.

    Returns True if a job was emitted. Shared by the fit and attribution
    launchers so the sbatch directives, the setup block and the three modes
    cannot drift between them.
    """
    if args.print_commands:
        print(command)
        return True

    directives = [
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --ntasks=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --nodes=1",
        "#SBATCH --gpus=1",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --output={log_dir}/{job_name}.out",
        f"#SBATCH --error={log_dir}/{job_name}.err",
    ]
    if getattr(args, "requeue", False):
        directives.append("#SBATCH --requeue")
    if args.constraint:
        directives.insert(5, f"#SBATCH -C {args.constraint}")
    if args.partition:
        directives.insert(5, f"#SBATCH --partition={args.partition}")

    sbatch_script = "\n".join(
        ["#!/bin/bash -l", *directives, "", setup, "",
         "command -v nvidia-smi >/dev/null && nvidia-smi -L || true",
         command, ""]
    )

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
        + (f", {skipped_experiments} experiments unusable" if skipped_experiments else ""),
        # stdout must stay pipeable when it carries commands.
        file=sys.stderr if args.print_commands else sys.stdout,
    )


def build_attribute_parser(family: str, script: Path) -> argparse.ArgumentParser:
    launcher = f"src/{family}/attribute/launch.py"
    return _finish_attribute_parser(argparse.ArgumentParser(
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
    ), launcher, script)


def _finish_attribute_parser(parser, launcher: str, script: Path):
    _add_common_args(parser, launcher)
    parser.add_argument(
        "-e", "--experiments", nargs="+", default=None, metavar="EXP",
        help="limit to these experiment IDs (default: every one in the config)",
    )
    parser.add_argument(
        "--attribute-type", dest="attribute_types", action="append",
        choices=("profile", "counts"), default=None, metavar="TYPE",
        help="repeatable; one job per type per experiment (default: profile)",
    )
    parser.add_argument(
        "--reference-mode", choices=("frequency", "dinucleotide"),
        default="frequency",
        help="passed through to %(default)s-mode attribution, and part of the "
             "output filename, so the already-done check follows it",
    )
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument(
        "--attr-args", type=str, default="",
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
            print(f"SKIP {exp_id}: missing data — {', '.join(exp.missing)}",
                  file=sys.stderr)
            skipped_missing += len(types)
            continue

        # attribute.py averages over EVERY fold, so a partly trained experiment
        # is not a partly usable one -- it exits 1. Report how far along it is
        # rather than just refusing, since "3 of 5 trained" is the actionable
        # form of that message.
        folds = exp.all_folds(family, models_dir=args.models_dir)
        absent = [f for f in folds if not f["model"].exists()]
        if absent:
            print(f"SKIP {exp_id}: {len(folds) - len(absent)}/{len(folds)} folds "
                  f"trained; attribution needs all of them", file=sys.stderr)
            skipped_untrained += len(types)
            continue

        for attribute_type in types:
            out = attribution_path(exp_id, attribute_type, args.reference_mode)
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
        + (f", {skipped_experiments} experiments unusable" if skipped_experiments else ""),
        file=sys.stderr if args.print_commands else sys.stdout,
    )
