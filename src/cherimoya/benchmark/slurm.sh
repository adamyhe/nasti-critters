#!/usr/bin/env bash
#SBATCH --job-name=cherimoya_benchmark
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cherimoya_benchmark_%j.out
#SBATCH --error=logs/cherimoya_benchmark_%j.err
#
# Deliberately no --partition or -C constraint: those are site-specific. Pass
# them at submit time, e.g.
#   sbatch --partition=gpu src/cherimoya/benchmark/slurm.sh
#   sbatch --partition=gpu -C "GPU_SKU:A100_PCIE" src/cherimoya/benchmark/slurm.sh
#
# Environment (all optional):
#   APPTAINER_IMAGE  run in a container instead of natively (images live at
#                    https://github.com/adamyhe/sherlock)
#   APPTAINER_BIND   paths to bind into the container
#   CONDA_ENV        mamba env to activate for a native run (default nasti-critters)
#   VENV             uv venv to activate for a native run (default .venv)

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

if [[ ! -d "${REPO_ROOT}/models/cherimoya" ]]; then
    echo "Could not find ${REPO_ROOT}/models/cherimoya." >&2
    echo "Submit from the nasti-critters repo root, or set SLURM_SUBMIT_DIR." >&2
    exit 1
fi

cd "${REPO_ROOT}"
mkdir -p logs

# Native runs need the repo environment; container runs already have it.
if [[ -z "${APPTAINER_IMAGE:-}" ]]; then
    CONDA_ENV="${CONDA_ENV:-nasti-critters}"
    VENV="${VENV:-${REPO_ROOT}/.venv}"
    if command -v mamba >/dev/null; then
        # shellcheck disable=SC1091
        eval "$(mamba shell hook --shell bash)" && mamba activate "${CONDA_ENV}" || \
            echo "warning: could not activate ${CONDA_ENV}" >&2
    fi
    # venv last so its interpreter wins
    [[ -f "${VENV}/bin/activate" ]] && source "${VENV}/bin/activate"
fi

exec bash "${REPO_ROOT}/src/cherimoya/benchmark/cmd.sh" "$@"
