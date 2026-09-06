#!/usr/bin/env bash
set -euo pipefail
# Benchmark Cherimoya models. Runs natively by default; set APPTAINER_IMAGE to
# run inside a container instead.
#
# Cluster-specific settings are environment variables, not hard-coded paths:
#   APPTAINER_IMAGE   path to a .sif. If unset, runs natively. Images are
#                     maintained at https://github.com/adamyhe/sherlock --
#                     this repo does not define one.
#   APPTAINER_BIND    colon/space-separated paths to bind (e.g. "/data /scratch")
#   CUDA_VISIBLE_DEVICES, NUMBA_CACHE_DIR, FORCE
#
# The experiment is REQUIRED and is passed through to benchmark_cherimoya.py:
#   bash src/cherimoya/benchmark/cmd.sh -e D.melanogaster-S2_PROcap
#
# Native use on a cluster with the repo's own environment:
#   mamba activate nasti-critters && source .venv/bin/activate \
#       && bash src/cherimoya/benchmark/cmd.sh -e <experiment>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

APPTAINER_IMAGE="${APPTAINER_IMAGE:-}"
APPTAINER_BIND="${APPTAINER_BIND:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/numba_cache}"
FORCE="${FORCE:-0}"

export CUDA_VISIBLE_DEVICES NUMBA_CACHE_DIR
mkdir -p "${NUMBA_CACHE_DIR}"

cd "${REPO_ROOT}"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "no nvidia-smi on PATH"

# The already-done check has to follow the experiment being benchmarked. It was
# hard-coded to D.melanogaster-S2_PROcap.json while "$@" was passed through
# verbatim, so benchmarking any other experiment consulted the fly metrics file:
# once fly had been benchmarked, every other experiment reported "Skipping" and
# exited 0 without running. Same class of leftover as the fit slurm.sh scripts
# invoking without -e -- a single-experiment assumption outliving the config
# unification.
EXPERIMENT=""
prev=""
for arg in "$@"; do
    case "${prev}" in
        -e|--experiment) EXPERIMENT="${arg}" ;;
    esac
    case "${arg}" in
        -e=*|--experiment=*) EXPERIMENT="${arg#*=}" ;;
    esac
    prev="${arg}"
done
if [[ -z "${EXPERIMENT}" ]]; then
    echo "No experiment given. benchmark_cherimoya.py requires -e:" >&2
    echo "  bash src/cherimoya/benchmark/cmd.sh -e D.melanogaster-S2_PROcap" >&2
    exit 2
fi

metrics_path="${REPO_ROOT}/performance_metrics/cherimoya/${EXPERIMENT}.json"
if [[ "${FORCE}" != "1" && -f "${metrics_path}" ]]; then
    echo "Skipping benchmark: metrics already exist at ${metrics_path}"
    exit 0
fi

script="${REPO_ROOT}/src/cherimoya/benchmark/benchmark_cherimoya.py"

if [[ -n "${APPTAINER_IMAGE}" && -f "${APPTAINER_IMAGE}" ]]; then
    echo "Benchmarking in ${APPTAINER_IMAGE} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    export APPTAINERENV_NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR}"
    BIND_ARGS=()
    for p in ${APPTAINER_BIND//:/ }; do BIND_ARGS+=(--bind "$p"); done
    apptainer exec --nv "${BIND_ARGS[@]}" "${APPTAINER_IMAGE}" \
        python "${script}" "$@" -v
else
    if [[ -n "${APPTAINER_IMAGE}" ]]; then
        echo "APPTAINER_IMAGE set but not found: ${APPTAINER_IMAGE}" >&2
        exit 1
    fi
    echo "Benchmarking natively on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    python "${script}" "$@" -v
fi
