#!/bin/bash -l
#SBATCH --job-name=s2_fit
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# No --partition or -C here: both are site-specific. Pass at submit time, e.g.
#   sbatch --partition=gpu -C "GPU_SKU:A100_PCIE" src/bpnet/fit/slurm.sh
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Site-specific module loads go here, if your cluster needs any.
# The repo's own environment (mamba + uv) normally suffices:
if command -v mamba >/dev/null; then
    eval "$(mamba shell hook --shell bash)"
    mamba activate nasti-critters || true
fi
[ -f .venv/bin/activate ] && . .venv/bin/activate   # venv last

command -v nvidia-smi >/dev/null && nvidia-smi -L || true
time python fit_bpnet.py -f ${SLURM_ARRAY_TASK_ID} -v
