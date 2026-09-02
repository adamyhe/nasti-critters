# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**nasti-critters** — NAScent Transcription Initiation Critters. Renamed from `dm-procap-models`, which
had become misleading: the `dm-` prefix dates from when this was one *D. melanogaster* dataset, and the
repo now covers 42 experiments across 12 species and three assay families.

A research repo of training/evaluation scripts (no installable package, no test suite) for base-resolution
sequence-to-function models of PRO-cap transcription-initiation profiles across non-human species. 26
experiments over 10 species (*D. melanogaster*, *M. musculus*, *C. elegans*, *S. cerevisiae*, *S. pombe*,
*A. thaliana*, and — added with the 2026-08-30 manifest update — *C. reinhardtii*, *P. patens*,
*S. moellendorffii*, *C. griseus*) are defined in `config/experiment_config.yaml`, generated from the
manifest in `planning/`.

**One experiment == one species x one biological condition == one model.** Perturbations (Ino80 depletion,
Spt5 IAA, LacZ knockdown) are separate experiments with their own models; this project deliberately does not
multi-task conditions or assay families into shared heads.

Two model families are trained on the same data:

- **BPNet** (`bpnetlite.BPNet`) — the main, multi-species path.
- **Cherimoya** — newer architecture, *D. melanogaster* only. Training and benchmarking work, but per
  `src/cherimoya/README.md` the models are **not deployment-ready**.

All data lives in `data/`; models in `models/{bpnet,cherimoya}/`. Neither was gitignored before — the
repo's `.gitignore` was a stock Python one with no `data/` rule, which only looked harmless because `data/`
did not exist yet. Both are ignored now (along with `logs/`, `predictions/`, `performance_metrics/`,
`attr/`); the full FASTQ set alone is ~120 GiB, so check `git status` before any bulk `git add`.
Every script resolves config/data paths relative to `REPO_ROOT`, computed from `__file__`, so scripts can be
invoked from anywhere — but the launcher shell scripts assume the repo root as CWD.

## Environment

Training and benchmarking require a CUDA GPU (`fit_bpnet.py` hard-codes `.to("cuda")`). Local development is
fine for reading/editing config-parsing logic; anything that imports torch and loads loci needs the cluster.

- **Two environments, split by job — not by language.** `environment.yml` (+ committed
  `conda-lock.yml`, 198 packages, linux-64) owns the **whole preprocessing DAG**, fetch through peaks:
  the external binaries,
  `snakemake`, and the only three libraries those scripts import (`pyyaml`, `pandas`, `numpy`). Nothing in
  `src/data_preprocessing/`, `src/experiments.py` or `config/write_split_csvs.py` imports `torch`,
  `bpnet-lite`, `cherimoya` or `tangermeme` — verified by loading each entry point and inspecting
  `sys.modules` — so label generation needs no GPU stack and no venv, and runs on a CPU or transfer node.
  `pyproject.toml` (+ `uv.lock`, 85 packages) owns **model work**: training, benchmarking, attribution.
  `snakemake` is deliberately NOT a uv dependency; do not add preprocessing-only deps there.
  `numpy`/`pandas`/`pyyaml` appear in both on purpose, because scripts on both sides import them.
- **GC-matched negatives are model prep, not preprocessing, and are deliberately outside the DAG.**
  They are a training input rather than a label, and `src/make_negatives.py` is the only thing here that
  shells out to `bpnet negatives` from **bpnet-lite, which is PyPI-only** — not on bioconda or
  conda-forge, so it can never move into `environment.yml` the way `umi_tools` and `pypints` did.
  Keeping it out of `workflow/Snakefile` is what makes the claim above true: the entire pipeline runs
  from the mamba env with no venv and no GPU stack. Run it separately, from the venv, before training —
  `uv run python src/make_negatives.py`. Do not add it back as a rule.
- **When auditing that boundary, grep for `subprocess`, `run([` and bare command names in `shell:`
  blocks — not just imports.** An earlier version of this file claimed preprocessing needed no uv,
  verified by loading each script and checking `sys.modules`. That covers what a script *imports* and
  misses what it *shells out to*, which is exactly what `make_negatives.py` does. `conda-lock.yml` is `linux-64` only
  and that is deliberate — `star 2.7.11a` has no `osx-arm64` build, so adding that platform fails to
  solve, and preprocessing needs the cluster anyway. `environment.yml` stays the source of truth;
  regenerate with
  `uvx --from conda-lock conda-lock lock -f environment.yml -p linux-64 --lockfile conda-lock.yml`
  and install with `conda-lock install --name nasti-critters conda-lock.yml`. `requirements.txt` is gone. Activate the venv last so its interpreter wins:
  `mamba env create -f environment.yml && mamba activate nasti-critters && uv sync && source .venv/bin/activate`.
- Version pins trace the ENCODE spec: `fastp=0.23.4`, `star=2.7.11a`, `samtools=1.18` (conda);
  `pyPINTS==1.1.10`, `umi-tools==1.1.5` (uv). `torch>=2.10` is required for `torch.optim.Muon`.
- Two package-name traps, both already handled — do not "fix" them back: the peak caller is **`pyPINTS`**
  (PyPI `pints` is unrelated time-series inference), and PyPI **`muon` is a multi-omics framework**, not the
  optimizer. `fit_cherimoya.py` imports `torch.optim.Muon` and raises a pointed error rather than falling
  back to it.
- Sherlock: `mamba activate torch` also works; install the Python side with
  `uv pip install --python "$CONDA_PREFIX/bin/python" -r pyproject.toml`. Module loads are in
  `src/bpnet/fit/slurm.sh`.
- Preprocessing shells out to: fastp, STAR, samtools, bedtools, GNU coreutils `sort`, `bgzip`,
  `bedGraphToBigWig`/`bigWigToBedGraph`/`bigWigMerge`, `pints_caller`, `umi_tools`, and the `bpnet` CLI
  (`bpnet negatives`). The pipeline pulls FASTQs straight from ENA over HTTPS, so SRA Toolkit is not
  needed; homerTools, `fasterq-dump` and `proseq2.0` are no longer used anywhere.
- **No site-specific values are hard-coded any more.** Launchers and `launch.py` take `--partition`/`-C`
  at submit time and emit no such directive by default; container use is opt-in via `APPTAINER_IMAGE` /
  `APPTAINER_BIND`; job environment setup defaults to activating this repo's mamba env + uv venv, with
  `launch.py --setup-file` for site-specific module loads. Do not reintroduce cluster paths or partition
  names into tracked files.
- **Do not add an Apptainer definition to this repo.** Container images are maintained externally and
  authoritatively at [adamyhe/sherlock](https://github.com/adamyhe/sherlock) (`cherimoya/cherimoya.def`, base
  `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime`). A project-local `src/cherimoya/apptainer/` existed and
  was deleted deliberately. Containers are optional and opt-in via `APPTAINER_IMAGE`/`APPTAINER_BIND`; the
  launchers run natively when unset.
- **`cherimoya` is pinned by git commit**, not version, in `[tool.uv.sources]` — commit `8e4283fe` = version
  0.2.1, which was never published to PyPI (PyPI stops at 0.2.0). Verified by installing both: the only
  difference is four lines adding `__version__` to `__init__.py`. `cheri.py`, `cherimoya.py`, `io.py`,
  `losses.py`, `performance.py`, `wrappers.py` are byte-identical.
  **The EMA / `valid_count_corr` rewrite is already in 0.1.0** — `class EMA` (shadow weights, decay 0.999,
  used unconditionally in `fit()` at lines 395/445/458/520/526) and `valid_count_corr` checkpoint selection
  (`if valid_count_corr > best_corr`) appear identically in 0.1.0, 0.2.0 and 0.2.1. The comment in the
  upstream `cherimoya.def` attributing that rewrite to this commit is wrong by two releases; do not repeat
  it. The pin buys byte agreement with the image and immunity to tag movement (upstream's `v0.2.0` tag has
  been force-moved), not different training behaviour. `uv.lock` resolves `torch==2.13.0`.
- **`macs3` is blocked from installing, deliberately — do not "restore" it.** It is a transitive dep of
  both `bpnet-lite` and `cherimoya`, has no wheels at any version, and nothing here imports it, so it was
  pure sdist build cost (plus `cykhash`, also sdist-only, via macs3). `[tool.uv] override-dependencies`
  gates `macs3` and `cykhash` behind `sys_platform == 'nonexistent'`, an unsatisfiable marker — that is
  how uv drops a transitive dependency, since overrides replace a requirement rather than delete it. The
  packages still appear in `uv.lock` (it is a universal lock) but the dependency edges carry the false
  marker, so they never resolve on any platform; verified with `uv sync --dry-run`. The sherlock image
  drops it the same way via `pip install --no-deps`.
  Blocking it also strands `cykhash` and `hmmlearn`, which nothing else wanted.
- **`umi_tools` and `pypints` are conda deps, not uv ones — do not move them back.** Neither is ever
  imported: the repo shells out to `umi_tools dedup` and `pints_caller` and probes with `shutil.which`,
  so both are pipeline binaries like fastp/STAR/samtools. bioconda has `umi_tools` 1.1.5
  (linux-64/aarch64, osx-64/arm64) and `pypints` 1.1.10 (`noarch`). `umi_tools` also had no wheel at any
  version, so it was the last sdist build in the tree. Moving `pypints` out drops its exclusive tail too
  (`biopython`, `pybedtools`, `statsmodels` → `formulaic`/`patsy`/`interface-meta`); `pysam` and
  `pybigwig` stay, since `bam2bw` and `biodatatools` want them. Version pins in `environment.yml` are the
  ENCODE spec's, but the parameters that actually determine the peaks are in `config/procap_pipeline.yaml`
  — that is what to check for reproducibility, not the package version.
- **Therefore `environment.yml` carries no `c-compiler`/`cxx-compiler`.** Nothing reachable in the
  resolution compiles: `triton` has manylinux wheels, and the only source installs left are `cherimoya`
  (git pin) and `connection-pool`, both pure Python. If you add a Python dep that needs building, put the
  compilers back in `environment.yml`, not `pyproject.toml`.

## Common commands

```bash
# --- ENCODE PRO-cap pipeline (the current path for producing labels) ---
python src/data_preprocessing/resolve_runs.py                  # manifest -> run accessions via ENA
python src/data_preprocessing/build_experiment_config.py       # manifest -> experiment_config.yaml
python src/data_preprocessing/fetch_fastqs.py --dry-run        # ~200 GiB for all 42 experiments
python src/data_preprocessing/fetch_fastqs.py --tier include -j 4
python src/data_preprocessing/fetch_fastqs.py --verify-only    # md5 audit of what is on disk
python src/data_preprocessing/run_procap_pipeline.py --list    # what is defined and what is ready
python src/data_preprocessing/run_procap_pipeline.py --index-only --species S.cerevisiae
python src/data_preprocessing/run_procap_pipeline.py -e S.cerevisiae-Ino80ctl_PROcap --dry-run
python src/data_preprocessing/run_procap_pipeline.py --tier include -t 16

# Mapped reads + peak counts per experiment, for exclude/merge decisions
snakemake stats -c8 --config tier=include,conditional   # target BEFORE --config
python src/qc/experiment_stats.py --all \
    -o qc/stats/experiment_stats.tsv --markdown qc/stats/experiment_stats.md

# GC-matched negatives (run from repo root; driven by experiment_config.yaml)
uv run python src/make_negatives.py --dry-run
uv run python src/make_negatives.py -e S.cerevisiae_PROcap --force -j 3

# S. pombe peak-level folds (required before pombe training; see "Splits" below)
python src/data_preprocessing/make_random_splits.py

# Train BPNet, one experiment/fold
python src/bpnet/fit/fit_bpnet.py -e D.melanogaster-S2_PROcap -f 0 -v

# Submit all (experiment x fold) BPNet jobs; skips already-trained and missing-data combos
python src/bpnet/fit/launch.py --dry-run
python src/bpnet/fit/launch.py --time 12:00:00 --mem 32G

# Train Cherimoya (D. melanogaster / dm3 config set only)
python src/cherimoya/fit/fit_cherimoya.py -f 0

# Evaluate / attribute
python src/cherimoya/benchmark/benchmark_cherimoya.py --save-output
python src/bpnet/benchmark/benchmark_predictions.py
python src/bpnet/attribute/attribute.py --attribute-type profile
```

There is no linter config, no formatter config, and no tests. Verification means running a script — use
`--dry-run` (preprocessing/launcher) or a single `-f 0` fold as the cheap smoke test.

## Architecture: one convention, via src/experiments.py

Everything resolves through **`src/experiments.py`**. Scripts add `REPO_ROOT/src` to
`sys.path` and import from it; do not reintroduce per-script path constants.

| Concern | Single source |
| --- | --- |
| What data | `config/experiment_config.yaml`, selected with `-e EXPERIMENT` |
| Which folds | `config/chrom_splits.yaml` by species (S. pombe absent by design), or peak-level `config/splits/{species}_random_fold_assignments.csv` |
| Hyperparameters | `config/{bpnet,cherimoya}_params.json`, one file per model family |
| Model paths | `models/{family}/{experiment}/{experiment}.fold{f}.torch` |
| Non-ACGT bases | `experiments.IGNORE` |

`Experiment.load(id)` returns resolved absolute paths plus `species`, `blacklist`, and a
`missing` list of absent inputs. `.fold_split(f)` applies the one fold rule (test = `f`,
validation = `(f+1) % n`, train = the rest) and transparently switches to peak-level splits
where they exist. `.all_folds(family)` gives every fold with its test chromosomes and
checkpoint path — what the benchmark and attribution scripts iterate over.

**This was two incompatible regimes until recently, and the history explains the shape.**
`eefd528` "Reviving this project" (2026-05-23) added a flat `config/data_paths.json` for a
single dm3 Drosophila dataset. `c9b1311` added cherimoya onto that same scaffolding one
commit later. `a806e0d` "Added yeast genomes" then needed multiple species, introduced
`experiment_config.yaml` + `chrom_splits.yaml`, and converted **only** `fit_bpnet.py` and
`launch.py`. The five downstream scripts kept reading `data_paths.json`, so the repo ended
up with two genome builds of the same Drosophila data, two model filename conventions, and
duplicated hyperparameters that had silently diverged (`count_loss_weight` 100 vs 50,
`max_epochs` 100 vs 200). All of that is now unified; `data_paths.json` is deleted. If you
need the old dm3 dataset, add it as an experiment rather than resurrecting a parallel path.

**Completion is `.final.torch`, not `.torch`.** Both bpnet-lite and cherimoya write
`{name}.torch` every time validation loss improves — so it can exist after one epoch — and
`{name}.final.torch` exactly once at the end of `fit()`. `model_path(..., final=True)` is
the only safe test of completion.

**Heavy imports are deferred.** `torch`, `bpnetlite`, `cherimoya`, `tangermeme` and
`data_loader` are imported *inside* `main()`, after argparse and path validation, so
`--help` and missing-data errors stay instant on a login node and are testable without a
GPU stack installed. `fit_bpnet.py` was the last holdout and now follows suit.

## Data conventions to preserve

- **Strand sign.** Minus-strand bigwigs may store signal as negative values (UCSC convention) or positive
  (direct). The codebase normalizes with `torch.abs()` on every signal/control tensor after `extract_loci`,
  and `make_negatives.py` abs-values the minus bigwig before merging strands. `src/bpnet/fit/data_loader.py`
  exists *only* to add these `abs()` calls around `bpnetlite`'s `PeakGenerator`. Any new code that reads
  signal must do the same.
- **Non-ACGT.** Every `extract_loci` call passes `ignore=list("QWERYUIOPSDFHJKLZXVBNM")`.
- **Outlier peaks: no signal-based filter is applied, deliberately.** Both fit scripts pass
  `max_counts=None`, matching procap-atlas, which does the same in its `fit_bpnet.py` and
  `fit_cherimoya.py`. The pre-unification `fit_bpnet.py` (at `a806e0d`) did drop peaks above
  `quantile(total_signal, 0.99) * 1.2`; the port to `data_loader.PeakGenerator` removed it, which brought
  this repo in line with upstream. Do not reinstate it casually: the threshold is data-dependent, so every
  species and library gets a different effective cutoff, which is corrosive in a repo whose point is
  cross-species comparison — and the top of a PRO-cap signal distribution is real biology (snRNA, histone,
  ribosomal-protein promoters), i.e. the most informative loci for an initiation model. Upstream uses
  signal quantiles only in *diagnostics* (`locus_diagnostics`, `generate_warning_flags.py`), never to drop
  training data. Artifact removal is the exclusion lists' job, and those are canonical published lists —
  **do not hand-curate regions into them**, or folds and preprocessing stop being comparable with
  plant-design, csRNANet and procap-atlas.
- **The real gap this leaves:** S. cerevisiae and S. pombe have no published exclusion list *and* no
  outlier filter, so their only guards are non-ACGT filtering and the fold structure. Their in-assembly
  rDNA arrays (`XII:451786-489469`; `III:1-23130` and `III:2440994-2452883`) will be among the
  highest-signal PINTS calls. Those are real Pol I loci rather than mismapping artifacts, and `log1p` on
  the count head compresses their influence, so this is a known and probably tolerable exposure rather
  than a bug — but check it before publishing yeast numbers.
- **Negatives live in `main_chromosomes`, from `config/genomes.yaml`** — the same allow-list the bigWig
  and PINTS steps use, so negatives are drawn from exactly the space the peaks occupy.
  `src/make_negatives.py` applies it in three places: the peak set, both bedgraphs, and the chrom.sizes
  it feeds `bedGraphToBigWig`. All three are needed — restricting only chrom.sizes makes
  `bedGraphToBigWig` abort on the first contig it no longer lists.
  This replaced a hand-written per-*experiment* `CHROM_EXCLUDE` regex map, which was wrong two ways.
  It **under-covered**: 3 of 38 experiments had an entry, so 35 filtered nothing while chrom.sizes came
  from the whole FASTA `.fai` — fine for dm6, but S. moellendorffii has 757 scaffolds and C. griseus 637.
  And it was **actively incorrect**: the patterns were substring regexes over the entire BED line, so
  dm6's `_` dropped any peak whose *name* contained an underscore, not just the scaffolds it targeted
  (verified — a `chr2L` peak named `peak_with_underscore` was being discarded).
  The new filter matches column 0 exactly, `str()`s the YAML names (bare-numeric chromosomes parse as
  ints and would match nothing), and raises if `main_chromosomes` and the FASTA disagree rather than
  silently emitting a short chrom.sizes. `ALPHA` stays per-experiment: it is a tuning parameter, not a
  property of the genome.
- **Negatives ratio.** GC-matched negatives are sampled at a low ratio (1/7 in `fit_bpnet.py`, 0.1 in the
  JSON configs) rather than a balanced mix.
- Windows are `in_window=2114` / `out_window=1000` throughout; `trimming` is always
  `(in_window - out_window) // 2`.

## Code patterns

- **Deferred heavy imports.** Scripts import `torch`, `bpnetlite`, `cherimoya`, and `tangermeme` *inside*
  `main()`, after argparse and after all config/path validation. This keeps `--help` and missing-file errors
  near-instant on a login node. Keep new imports in the same place.
- **Fail fast on paths.** Each script builds a `[(label, path), ...]` list and exits nonzero listing every
  missing file before doing any work.
- `load_config()` uses `yaml.safe_load` for both YAML and JSON configs (YAML is a JSON superset), so config
  files can be either format.
- `load_bed()` reads only columns 0–2 with `dtype={"chrom": str}` — chromosome names must stay strings
  (S. cerevisiae uses roman numerals, dm has `4`/`X`).
- Cherimoya optimization splits parameters: **Muon** for 2-D weight matrices except `linear.weight`, **AdamW**
  for everything else, each with linear warmup (5 epochs) → cosine decay, trained in `bfloat16`.

## Scope note

Human K562 PRO-cap configs from the csRNANet workspace are deliberately excluded from this repo.
`planning/nonhuman_capped_runon_manifest.xlsx` tracks candidate non-human datasets not yet wired into
`config/`.
