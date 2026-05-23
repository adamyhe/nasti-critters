#!/usr/bin/env python3
"""Submit SLURM jobs to train BPNet (CLIPNET) models for all experiments and folds.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per (experiment, fold) pair via fit.py.

Experiments with an already-trained model file are skipped automatically.

Usage:
    python src/bpnet/fit/launch.py                    # submit all experiments x folds
    python src/bpnet/fit/launch.py --dry-run           # print sbatch scripts without submitting
    python src/bpnet/fit/launch.py --time 12:00:00 --mem 32G --partition gpu
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
RANDOM_SPLITS_DIR = REPO_ROOT / "configs" / "splits"
FIT_SCRIPT = REPO_ROOT / "src" / "bpnet" / "fit" / "fit_bpnet.py"


def get_n_folds(species: str, chrom_splits: dict) -> int:
    random_csv = RANDOM_SPLITS_DIR / f"{species}_random_fold_assignments.csv"
    if random_csv.exists():
        return pd.read_csv(random_csv, usecols=["fold"])["fold"].nunique()
    if species == "S.pombe":
        return 0
    return len(chrom_splits[species])


def missing_data_paths(processed: dict, use_controls: bool) -> list[str]:
    """Return a list of required data path labels that are missing on disk."""
    required = ["peaks", "pl_bigwig", "mn_bigwig", "gc_negatives", "sequences"]
    if use_controls:
        required += ["pl_control", "mn_control"]
    missing = []
    for key in required:
        if key not in processed:
            missing.append(f"{key} (not in config)")
        elif not (REPO_ROOT / processed[key]).exists():
            missing.append(f"{key}: {processed[key]}")
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="print sbatch scripts without submitting"
    )
    # SLURM resource flags
    parser.add_argument(
        "--gpus", type=str, default="GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR"
    )
    parser.add_argument("--partition", type=str, default="akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument("--time", type=str, default="6:00:00")
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

    # Load experiment list
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    # Load chrom splits (keyed by species) for per-experiment fold counts
    with open(CHROM_SPLITS_PATH) as f:
        chrom_splits = yaml.safe_load(f)

    log_dir = REPO_ROOT / "logs" / "bpnet_fit"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_trained = 0
    skipped_missing = 0
    for exp_id in experiments:
        exp = config["experiments"][exp_id]
        species = exp["species"]
        processed = exp.get("processed", {})
        n_folds = get_n_folds(species, chrom_splits)
        if n_folds == 0:
            print(
                f"SKIP {exp_id}: S. pombe requires "
                f"{RANDOM_SPLITS_DIR / f'{species}_random_fold_assignments.csv'}; "
                "run src/data_preprocessing/make_pombe_random_splits.py",
                file=sys.stderr,
            )
            skipped_missing += 1
            continue

        missing = missing_data_paths(processed, args.controls)
        if missing:
            print(
                f"SKIP {exp_id}: missing data — {', '.join(missing)}",
                file=sys.stderr,
            )
            skipped_missing += n_folds
            continue

        for fold in range(n_folds):
            # Skip if model already trained
            dir_name = f"{exp_id}_ctl" if args.controls else exp_id
            model_dir = REPO_ROOT / "models" / "bpnet" / dir_name
            model_path = model_dir / f"{exp_id}.fold{fold}.torch"
            if model_path.exists():
                skipped_trained += 1
                continue

            job_name = f"bpnet_{exp_id}{'_ctl' if args.controls else ''}_f{fold}"
            fit_cmd = f"python {FIT_SCRIPT} -e {exp_id} --fold {fold} -v"
            if args.controls:
                fit_cmd += " --controls"
            if args.fit_args:
                fit_cmd += f" {args.fit_args}"

            sbatch_script = textwrap.dedent(f"""\
                #!/bin/bash -l
                #SBATCH --job-name={job_name}
                #SBATCH --ntasks=1
                #SBATCH --ntasks-per-node=1
                #SBATCH --nodes=1
                #SBATCH --gpus=1
                #SBATCH -C {args.gpus}
                #SBATCH --cpus-per-task={args.cpus_per_task}
                #SBATCH --mem={args.mem}
                #SBATCH --partition={args.partition}
                #SBATCH --time={args.time}
                #SBATCH --output={log_dir}/{job_name}.out
                #SBATCH --error={log_dir}/{job_name}.err

                ml openblas/0.3.28
                ml xsimd/8.1.0
                ml xz/5.8.1
                ml hdf5/1.14.4
                ml arrow/22.0.0
                ml load py-pyarrow/18.1.0_py312
                ml lz4/1.8.0
                ml biology
                ml htslib
                ml ucsc-utils
                
                mamba activate torch
                nvidia-smi -L
                {fit_cmd}
            """)

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
    total = sum(
        get_n_folds(config["experiments"][e]["species"], chrom_splits)
        for e in experiments
    )
    print(
        f"\n{action} {submitted} jobs, skipped {skipped_trained} already trained, "
        f"{skipped_missing} missing data ({total} total)"
    )


if __name__ == "__main__":
    main()
