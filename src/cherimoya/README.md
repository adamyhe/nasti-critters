# Cherimoya PRO-cap models

Cherimoya training mirrors the BPNet workflow exactly: one model per chromosome fold, selected by
experiment ID. Data, species and folds resolve through `src/experiments.py` from
`config/experiment_config.yaml` and `config/chrom_splits.yaml`; hyperparameters come from
`config/cherimoya_params.json`. Models are written to
`models/cherimoya/{experiment}/{experiment}.fold{f}.torch`.

## Status

Cherimoya training and benchmarking are complete, but Cherimoya models are not ready for deployment yet. Treat the current scripts as the training and evaluation workflow, not as a public deployment pipeline.

The production entry point is:

```bash
python src/cherimoya/fit/fit_cherimoya.py -e D.melanogaster-S2_PROcap -f 0
python src/cherimoya/fit/fit_cherimoya.py -e S.cerevisiae-Ino80ctl_PROcap -f 0 --negative-ratio 0.1
python src/cherimoya/fit/fit_cherimoya.py -e D.melanogaster-S2_PROcap -f 0 --n-filters 196
```

The script uses Muon for 2D weight matrices and AdamW for the remaining parameters, with warmup plus cosine learning-rate schedules.

Cluster launchers run **natively** by default and take no site-specific values;
pass partition and GPU constraints at submit time:

```bash
bash src/cherimoya/benchmark/cmd.sh
sbatch --partition=gpu src/cherimoya/benchmark/slurm.sh
sbatch --partition=gpu src/cherimoya/fit/slurm.sh          # array: one task per fold
```

To run in a container instead, point them at an image; this repo does not define
one, and should not — the authoritative images are maintained at
[adamyhe/sherlock](https://github.com/adamyhe/sherlock):

```bash
export APPTAINER_IMAGE=/path/to/cherimoya.sif
export APPTAINER_BIND="/path/to/data:/path/to/scratch"
```

Benchmark and consolidation:

```bash
python src/cherimoya/benchmark/benchmark_cherimoya.py -e D.melanogaster-S2_PROcap
python src/cherimoya/benchmark/benchmark_cherimoya.py -e D.melanogaster-S2_PROcap --save-output
```

## Cherimoya version and API

`pyproject.toml` pins `cherimoya` to commit `8e4283fe` (version 0.2.1) via
`[tool.uv.sources]`, matching the [adamyhe/sherlock](https://github.com/adamyhe/sherlock)
image. 0.2.1 was never published to PyPI, whose latest is 0.2.0.

Installing both and diffing shows they are functionally identical: `cheri.py`,
`cherimoya.py`, `io.py`, `losses.py`, `performance.py` and `wrappers.py` are
byte-identical, and the sole change is four lines adding `__version__` to
`__init__.py`. The EMA / `valid_count_corr` rewrite sometimes attributed to this
commit is present identically in **0.1.0**, 0.2.0 and 0.2.1 — `class EMA`
(shadow weights, decay 0.999, called unconditionally from `fit()`) and
`if valid_count_corr > best_corr` checkpoint selection. The pin buys byte
agreement with the image and immunity to tag movement (upstream's `v0.2.0` tag
has been force-moved once), not different training results.

### API port

`cherimoya >= 0.2` broke the API these scripts were originally written against
(the historical `69f16dc` commit below). `fit_cherimoya.py` and
`fit/data_loader.py` were ported from [kundajelab/procap-atlas](https://github.com/kundajelab/procap-atlas), which
had already resolved it:

| change | why |
| --- | --- |
| `signal_groups=[len(signals)]` replaces `n_outputs` | `n_outputs` no longer exists; one 2-element group is one stranded (pl, mn) pair |
| `lw_optimizer` + `lw_scheduler` now required by `fit()` | third optimizer for the `lw0`/`lw1` Kendall uncertainty loss weights; `SGD` + warmup then flat `ConstantLR` |
| Muon split also excludes `conv_weight` | the 2-D depth-wise conv belongs on AdamW |
| `PeakGenerator(signals=[params["signals"]])` — nested | a flat 2-element list now means two *independent unstranded* groups, breaking reverse-complement channel swapping |
| `data_loader.py` is now a thin wrapper over `cherimoya.io` | it was a ~440-line frozen fork of the pre-refactor `PeakGenerator`, incompatible with the pinned version |

`Cherimoya.load(path, device=...)` is unchanged and still compatible, so
`benchmark_cherimoya.py` needed no changes. But `load()` reconstructs the model
via `cls(**payload['config'])`, so a checkpoint written by a pre-0.2 cherimoya
whose stored config contains `n_outputs` will not load under the pinned version.

Historical note: the first Cherimoya models were trained while `cherimoya` was in
early development, using commit `69f16dc7ff48ad094aafd4b93433972181c65d50`. Check
out that commit only if you need to reproduce that initial model set — the
current scripts will not run against it.
