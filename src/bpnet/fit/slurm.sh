#!/bin/bash -l
#SBATCH --job-name=s2_fit
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu,akundaje,owners
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

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
time python fit_bpnet.py -f ${SLURM_ARRAY_TASK_ID} -v
