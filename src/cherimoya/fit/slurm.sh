#!/bin/bash -l
#SBATCH --job-name=cherimoya_fit
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# One array task per fold, for ONE experiment. The experiment is REQUIRED and
# comes from the first argument or $EXPERIMENT:
#
#   sbatch --partition=gpu src/cherimoya/fit/slurm.sh D.melanogaster-S2_PROcap
#
# It used to run `fit_cherimoya.py -f $SLURM_ARRAY_TASK_ID` with no -e, which
# could not work at all: -e is required, so every array task exited 2. That
# invocation predates the unified config -- both fit scripts take -e and resolve
# everything through src/experiments.py now.
#
# --array=0-4 is FIVE folds and is a static directive, so it is wrong for any
# species with a different count -- C. elegans has six. Override at submit time
# (`sbatch --array=0-5 ...`), or use the launcher, which reads n_folds() per
# species, skips finished folds and covers every experiment in one command:
#
#   python src/cherimoya/fit/launch.py --partition gpu
#   python src/cherimoya/fit/launch.py --print-commands | bash    # no SLURM
#
# No --partition or -C above: both are site-specific. Pass them at submit time:
#   sbatch --partition=gpu src/cherimoya/fit/slurm.sh EXP
#   sbatch --partition=gpu -C "GPU_SKU:A100_PCIE" src/cherimoya/fit/slurm.sh EXP
#
# Environment (all optional):
#   APPTAINER_IMAGE  run inside this .sif instead of natively (images live at
#                    https://github.com/adamyhe/sherlock)
#   APPTAINER_BIND   colon/space-separated paths to bind into the container
#   CONDA_ENV        mamba env for a native run (default nasti-critters)
#   VENV             uv venv for a native run (default <repo>/.venv)

set -euo pipefail

EXPERIMENT="${1:-${EXPERIMENT:-}}"
if [[ -z "${EXPERIMENT}" ]]; then
    echo "No experiment given. Pass it as the first argument or set \$EXPERIMENT:" >&2
    echo "  sbatch --partition=gpu src/cherimoya/fit/slurm.sh D.melanogaster-S2_PROcap" >&2
    echo "For all experiments x folds, use src/cherimoya/fit/launch.py instead." >&2
    exit 2
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

FIT_SCRIPT="${REPO_ROOT}/src/cherimoya/fit/fit_cherimoya.py"
if [[ ! -f "${FIT_SCRIPT}" ]]; then
    echo "Could not find ${FIT_SCRIPT}." >&2
    echo "Submit from the nasti-critters repo root, or set SLURM_SUBMIT_DIR." >&2
    exit 1
fi

cd "${REPO_ROOT}"
mkdir -p logs

FOLD="${SLURM_ARRAY_TASK_ID:-0}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/numba_cache}"
export NUMBA_CACHE_DIR
mkdir -p "${NUMBA_CACHE_DIR}"

command -v nvidia-smi >/dev/null && nvidia-smi -L || true

if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
    if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
        echo "APPTAINER_IMAGE set but not found: ${APPTAINER_IMAGE}" >&2
        exit 1
    fi
    export APPTAINERENV_NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR}"
    BIND_ARGS=()
    for p in ${APPTAINER_BIND//:/ }; do BIND_ARGS+=(--bind "$p"); done
    exec apptainer exec --nv "${BIND_ARGS[@]}" "${APPTAINER_IMAGE}" \
        python "${FIT_SCRIPT}" -e "${EXPERIMENT}" -f "${FOLD}" -v
fi

CONDA_ENV="${CONDA_ENV:-nasti-critters}"
VENV="${VENV:-${REPO_ROOT}/.venv}"
if command -v mamba >/dev/null; then
    eval "$(mamba shell hook --shell bash)"
    mamba activate "${CONDA_ENV}" || echo "warning: could not activate ${CONDA_ENV}" >&2
fi
# venv last so its interpreter wins
[[ -f "${VENV}/bin/activate" ]] && source "${VENV}/bin/activate"

exec python "${FIT_SCRIPT}" -e "${EXPERIMENT}" -f "${FOLD}" -v
