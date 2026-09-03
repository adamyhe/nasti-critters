#!/usr/bin/env python3
"""Enumerate BPNet training jobs for all experiments and folds.

Reads experiment IDs from config/experiment_config.yaml and produces one job per
(experiment, fold) pair, running fit_bpnet.py. Experiments whose model already
finished, whose inputs are missing, or whose species has no fold assignment are
skipped, and each skip says which.

Three output modes, none of which needs SLURM to *decide* anything -- the
selection logic is identical and only the emission differs:

    --print-commands   one bare shell command per job on stdout. No SBATCH
                       directives, no env setup, nothing submitted. This is the
                       non-SLURM path.
    --dry-run          the full sbatch script per job, printed not submitted.
    (neither)          submit via sbatch.

Usage:
    python src/bpnet/fit/launch.py                     # submit all experiments x folds
    python src/bpnet/fit/launch.py --dry-run           # print sbatch scripts
    python src/bpnet/fit/launch.py --print-commands     # bare commands, for any box
    python src/bpnet/fit/launch.py --time 12:00:00 --mem 32G --partition gpu

On a single non-SLURM box, run them serially -- these are GPU jobs and one box
almost certainly holds one at a time:

    python src/bpnet/fit/launch.py --print-commands | bash

To run N at once, having checked N models fit in VRAM:

    python src/bpnet/fit/launch.py --print-commands | xargs -P N -I{} bash -c '{}'

Skips and the summary go to stderr in this mode, so stdout stays pipeable.
"""

import argparse
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIT_SCRIPT = REPO_ROOT / "src" / "bpnet" / "fit" / "fit_bpnet.py"

sys.path.insert(0, str(REPO_ROOT / "src"))
from experiments import Experiment, experiment_ids, model_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    emit = parser.add_mutually_exclusive_group()
    emit.add_argument(
        "--dry-run", action="store_true", help="print sbatch scripts without submitting"
    )
    emit.add_argument(
        "--print-commands", action="store_true",
        help="print one bare shell command per job to stdout and submit nothing. "
             "For a non-SLURM box. The env setup block is NOT included -- "
             "activate the mamba env and uv venv yourself first, exactly as "
             "--setup-file would inside a job. Skip messages and the summary go "
             "to stderr so stdout can be piped to bash or xargs",
    )
    # SLURM resource flags
    # Site-specific by nature: no default, and the corresponding #SBATCH line
    # is omitted unless a value is given.
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
    parser.add_argument(
        "--requeue", action="store_true",
        help="submit with --requeue so pre-empted jobs are resubmitted by SLURM. "
             "Each job retrains its fold from epoch 0 (bpnet-lite fit() has no "
             "resume), but re-checks for a completed model first.",
    )
    parser.add_argument(
        "--controls",
        action="store_true",
        help="train with bias-control tracks (passes --controls to fit_bpnet.py "
        "and checks for pl_control/mn_control paths in experiment config)",
    )
    # Extra args forwarded to fit.py
    parser.add_argument(
        "--fit-args",
        type=str,
        default="",
        help="extra arguments forwarded to fit_bpnet.py (e.g. '--max-epochs 100')",
    )
    args = parser.parse_args()

    experiments = experiment_ids()

    log_dir = REPO_ROOT / "logs" / "bpnet_fit"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Environment setup for the generated jobs. Module names differ per site, so
    # the default only activates this repo's own environment.
    if args.setup_file:
        setup = Path(args.setup_file).read_text().rstrip()
    else:
        setup = textwrap.dedent(f"""\
            # Default setup: this repo's mamba env + uv venv (venv last so its
            # interpreter wins). Use --setup-file for site-specific module loads.
            if command -v mamba >/dev/null; then
                eval "$(mamba shell hook --shell bash)"
                mamba activate nasti-critters || true
            fi
            [ -f {REPO_ROOT}/.venv/bin/activate ] && . {REPO_ROOT}/.venv/bin/activate""").rstrip()

    submitted = 0
    skipped_trained = 0
    skipped_missing = 0
    skipped_experiments = 0
    total = 0
    for exp_id in experiments:
        try:
            exp = Experiment.load(exp_id, use_controls=args.controls)
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
            # Skip only if the fold actually finished. bpnet-lite writes
            # "{name}.torch" every time validation loss improves (so it can
            # exist after one epoch) and "{name}.final.torch" exactly once, at
            # the end of fit(). Checking the former would silently treat a
            # partially trained fold as complete -- verified against
            # bpnetlite/bpnet.py, which calls torch.save on both paths.
            suffix = "_ctl" if args.controls else ""
            done = model_path("bpnet", exp_id, fold, final=True, suffix=suffix)
            if done.exists():
                skipped_trained += 1
                continue

            job_name = f"bpnet_{exp_id}{'_ctl' if args.controls else ''}_f{fold}"
            fit_cmd = (
                f"python {shlex.quote(str(FIT_SCRIPT))} "
                f"-e {shlex.quote(exp_id)} --fold {fold} -v"
            )
            if args.controls:
                fit_cmd += " --controls"
            if args.fit_args:
                fit_cmd += f" {args.fit_args}"

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
            if args.requeue:
                directives.append("#SBATCH --requeue")
            if args.constraint:
                directives.insert(5, f"#SBATCH -C {args.constraint}")
            if args.partition:
                directives.insert(5, f"#SBATCH --partition={args.partition}")

            sbatch_script = "\n".join(
                ["#!/bin/bash -l", *directives, "", setup, "",
                 "command -v nvidia-smi >/dev/null && nvidia-smi -L || true",
                 fit_cmd, ""]
            )

            if args.print_commands:
                print(fit_cmd)
                submitted += 1
                continue

            if args.dry_run:
                print(f"--- {job_name} ---")
                print(sbatch_script)
                submitted += 1
                continue

            result = subprocess.run(
                ["sbatch"], input=sbatch_script, capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"{job_name}: {result.stdout.strip()}")
                submitted += 1
            else:
                print(
                    f"ERROR submitting {job_name}: {result.stderr.strip()}",
                    file=sys.stderr,
                )

    if args.print_commands:
        action = "Printed"
    elif args.dry_run:
        action = "Would submit"
    else:
        action = "Submitted"
    # `total` is accumulated in the loop above. It used to be a second pass that
    # reloaded every experiment and carried `entry.species == "S.pombe"` as a
    # special case -- dead code twice over, since a species with no fold
    # assignment raises from n_folds() and is already excluded, and the rest of
    # this launcher stopped naming pombe when C.griseus and S.moellendorffii
    # arrived. Do not reintroduce a species-name test here.
    print(
        f"\n{action} {submitted} jobs of {total} (experiment x fold); "
        f"skipped {skipped_trained} already trained, {skipped_missing} missing data"
        + (f", {skipped_experiments} experiments unusable" if skipped_experiments else ""),
        # stdout must stay pipeable when it carries commands.
        file=sys.stderr if args.print_commands else sys.stdout,
    )


if __name__ == "__main__":
    main()
