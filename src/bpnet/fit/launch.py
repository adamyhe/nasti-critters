#!/usr/bin/env python3
"""Submit SLURM jobs to train BPNet models for all experiments and folds.

Reads experiment IDs from config/experiment_config.yaml and submits one
sbatch job per (experiment, fold) pair via fit.py.

Experiments with an already-trained model file are skipped automatically.

Usage:
    python src/bpnet/fit/launch.py                    # submit all experiments x folds
    python src/bpnet/fit/launch.py --dry-run           # print sbatch scripts without submitting
    python src/bpnet/fit/launch.py --time 12:00:00 --mem 32G --partition gpu
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
    parser.add_argument(
        "--dry-run", action="store_true", help="print sbatch scripts without submitting"
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

    action = "Would submit" if args.dry_run else "Submitted"
    total = 0
    for e in experiments:
        try:
            entry = Experiment.load(e)
            total += 0 if (entry.species == "S.pombe"
                           and not entry.uses_peak_level_splits) else entry.n_folds()
        except KeyError:
            pass
    print(
        f"\n{action} {submitted} jobs of {total} (experiment x fold); "
        f"skipped {skipped_trained} already trained, {skipped_missing} missing data"
        + (f", {skipped_experiments} experiments unusable" if skipped_experiments else "")
    )


if __name__ == "__main__":
    main()
