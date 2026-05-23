#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

APPTAINER_IMAGE="${APPTAINER_IMAGE:-/scratch/users/ayhe/cherimoya/cherimoya.sif}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/scratch/users/${USER}/numba_cache}"
FORCE="${FORCE:-0}"

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
