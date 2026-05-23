# Cherimoya PRO-cap models

Cherimoya training mirrors the BPNet workflow for this repository: one model per chromosome fold, using Cherimoya parameters from `configs/cherimoya_params.json`, data paths from `configs/data_paths.json`, and fold assignments from `configs/D.melanogaster_data_fold_assignments.csv`.

## Status

Cherimoya training and benchmarking are complete, but Cherimoya models are not ready for deployment yet. Treat the current scripts as the training and evaluation workflow, not as a public deployment pipeline.

The production entry point is:

```bash
python src/cherimoya/fit/fit_cherimoya.py --fold 0
python src/cherimoya/fit/fit_cherimoya.py --fold 0 --negative-ratio 0.1
python src/cherimoya/fit/fit_cherimoya.py --fold 0 --n-filters 196 --batch-size 32
```

The script uses Muon for 2D weight matrices and AdamW for the remaining parameters, with warmup plus cosine learning-rate schedules.

Cluster launchers use Apptainer by default:

```bash
bash src/cherimoya/benchmark/cmd.sh
sbatch src/cherimoya/benchmark/slurm.sh
```

These launcher defaults are for the Sherlock HPC environment. Apptainer is a Sherlock-specific workaround for its older compiler and OS stack, not a general Cherimoya requirement. Review partitions, bind paths, module setup, image paths, and resource requests before using these launchers on another cluster.

Benchmark and consolidation:

```bash
python src/cherimoya/benchmark/benchmark_cherimoya.py
python src/cherimoya/benchmark/benchmark_cherimoya.py --save-output
```

Historical note: the first Cherimoya models were trained while `cherimoya` was in early development, using commit `69f16dc7ff48ad094aafd4b93433972181c65d50`. Check out that commit only if you need to reproduce that initial model set.
