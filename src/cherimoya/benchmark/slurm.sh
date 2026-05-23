#!/usr/bin/env bash
#SBATCH --job-name=cherimoya_benchmark
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:A40|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5|GPU_SKU:L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu,akundaje,owners
#SBATCH --time=12:00:00
#SBATCH --output=logs/cherimoya_benchmark_%j.out
#SBATCH --error=logs/cherimoya_benchmark_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

APPTAINER_IMAGE="${APPTAINER_IMAGE:-/scratch/users/ayhe/cherimoya/cherimoya.sif}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/scratch/users/${USER}/numba_cache}"
FORCE="${FORCE:-0}"

if [[ ! -d "${REPO_ROOT}/models/cherimoya" ]]; then
    echo "Could not find ${REPO_ROOT}/models/cherimoya." >&2
    echo "Submit this script from the dm-procap-models repo root, or set SLURM_SUBMIT_DIR to the repo root." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export NUMBA_CACHE_DIR
export APPTAINERENV_NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR}"

mkdir -p "${NUMBA_CACHE_DIR}"

BIND_ARGS=(
    --bind /oak/stanford/groups/akundaje/ayhe
    --bind /scratch/users/ayhe
)

cd "${REPO_ROOT}"
nvidia-smi -L

metrics_path="${REPO_ROOT}/performance_metrics/cherimoya/D.melanogaster-S2_PROcap.json"
if [[ "${FORCE}" != "1" && -f "${metrics_path}" ]]; then
    echo "Skipping benchmark: metrics already exist at ${metrics_path}"
    exit 0
fi

echo "Benchmarking Cherimoya models on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
apptainer exec --nv "${BIND_ARGS[@]}" "${APPTAINER_IMAGE}" \
    python "${REPO_ROOT}/src/cherimoya/benchmark/benchmark_cherimoya.py" \
        "$@" \
        -v
