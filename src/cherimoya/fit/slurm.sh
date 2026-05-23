#!/bin/bash -l
#SBATCH --job-name=s2_fit
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:A40|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5|GPU_SKU:L40S
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu,akundaje,owners
#SBATCH --time=6:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

FIT_SCRIPT={shlex.quote(str(FIT_SCRIPT))}
APPTAINER_IMAGE={shlex.quote(str(args.apptainer_image))}

nvidia-smi -L
apptainer_cmd = (
apptainer exec --nv \
    --bind /oak/stanford/groups/akundaje/ayhe \
    --bind /scratch/users/ayhe \
    /scratch/users/ayhe/apptainer/cherimoya.sif \
    python /scratch/users/ayhe/dm-procap-models/src/cherimoya/fit/fit_cherimoya.py -f ${SLURM_ARRAY_TASK_ID} -v