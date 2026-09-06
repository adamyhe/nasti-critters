# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**nasti-critters** — NAScent Transcription Initiation Critters. Renamed from `dm-procap-models`, which
had become misleading: the `dm-` prefix dates from when this was one *D. melanogaster* dataset, and the
repo now covers 42 experiments across 12 species and three assay families.

A research repo of training/evaluation scripts (no installable package, no test suite) for base-resolution
sequence-to-function models of PRO-cap transcription-initiation profiles across non-human species. 42
experiments over 12 species (*D. melanogaster*, *M. musculus*, *C. elegans*, *S. cerevisiae*, *S. pombe*,
*A. thaliana*, *C. reinhardtii*, *P. patens*, *S. moellendorffii*, *C. griseus* and — added 2026-09-01 —
*G. arboreum*, *G. hirsutum*) are defined in `config/experiment_config.yaml`, generated from the
manifest in `planning/`.

**One experiment == one species x one biological condition == one model.** Perturbations (Ino80 depletion,
Spt5 IAA, LacZ knockdown) are separate experiments with their own models; this project deliberately does not
multi-task conditions or assay families into shared heads.

Two model families are trained on the same data:

- **BPNet** (`bpnetlite.BPNet`) — the main, multi-species path.
- **Cherimoya** — newer architecture. Training and benchmarking work, but per
  `src/cherimoya/README.md` the models are **not deployment-ready**.
  **It is NOT D. melanogaster-only, and this file said so until 2026-09-03.** `fit_cherimoya.py` takes
  `-e` and resolves paths, species and folds through `src/experiments.py` exactly as `fit_bpnet.py` does,
  and `config/cherimoya_params.json` holds no species-specific value — it has been corpus-capable since
  `f9f60cf` ported it onto the unified config. What made it *look* single-species was a set of
  hand-written `slurm.sh` / `cmd.sh` scripts left at the pre-unification interface — cherimoya's fit
  script ran `-f $SLURM_ARRAY_TASK_ID` with **no `-e`**, so every array task exited 2; its benchmark
  `cmd.sh` hard-coded `D.melanogaster-S2_PROcap.json` as its already-done check while passing `"$@"`
  through, so once fly was benchmarked every other experiment printed "Skipping" and exited 0; and
  bpnet's fit script had the same missing `-e`, plus a relative path and a `--job-name=s2_fit` from the
  dm3 era. **All four are deleted.** The eight `launch.py` launchers over `src/launcher.py` replace them
  and get per-species fold counts right, which a static `--array=0-4` cannot — C. elegans has six folds.
  Do not add hand-written SLURM scripts back; a launcher generates the same sbatch script with
  `--dry-run`.

All data lives in `data/`; models in `models/{bpnet,cherimoya}/`. Neither was gitignored before — the
repo's `.gitignore` was a stock Python one with no `data/` rule, which only looked harmless because `data/`
did not exist yet. Both are ignored now (along with `logs/`, `predictions/`, `performance_metrics/`,
`attributions/`); the full FASTQ set alone is ~202 GiB, so check `git status` before any bulk `git add`.
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
  They are a training input rather than a label, and `src/make_negatives.py` imports
  **`tangermeme`, which is PyPI-only** — checked 2026-09-02, HTTP 404 on both `bioconda/tangermeme` and
  `conda-forge/tangermeme` — so it can never move into `environment.yml` the way `umi_tools` and
  `pypints` did.
  **The package behind that argument changed and the conclusion did not.** It used to be bpnet-lite,
  because the script shelled out to `bpnet negatives`; it now calls
  `tangermeme.match.extract_matching_loci` directly and does not import bpnet-lite at all. Both are
  PyPI-only, so the DAG boundary is exactly where it was — do not read "no more bpnet-lite" as
  permission to move this into `workflow/Snakefile`.
  Keeping it out of `workflow/Snakefile` is what makes the claim above true: the entire pipeline runs
  from the mamba env with no venv and no GPU stack. Run it separately, from the venv, before training —
  `uv run python src/make_negatives.py`. Do not add it back as a rule.
- **The venv side must not shell out to a conda binary, and `make_negatives.py` did — three times.**
  It failed with `No such file or directory: 'bigWigToBedGraph'`, because that binary is an
  `environment.yml` package while this script runs from the uv venv. The other two would have been the
  next two failures: `bgzip` in `filter_peaks`, which runs unconditionally, and a `samtools faidx`
  fallback. All three are now in-process — `pybigtools` for the strand merge, the `gzip` module for the
  BED, `pyfaidx` for the index. A fourth call, `bpnet negatives` itself, went the same way for a
  different reason — see the negatives bullet under "Data conventions" — so **the script now makes no
  subprocess calls at all.** Keep it that way: when adding to it, check which environment provides what
  you are calling.
- **When auditing that boundary, grep for `subprocess`, `run([` and bare command names in `shell:`
  blocks — not just imports.** An earlier version of this file claimed preprocessing needed no uv,
  verified by loading each script and checking `sys.modules`. That covers what a script *imports* and
  misses what it *shells out to*, which is exactly what `make_negatives.py` does. `conda-lock.yml` is `linux-64` only
  and that is deliberate — `star 2.7.11a` has no `osx-arm64` build, so adding that platform fails to
  solve, and preprocessing needs the cluster anyway. `environment.yml` stays the source of truth;
  regenerate with
  `uvx --from conda-lock conda-lock lock -f environment.yml -p linux-64 --lockfile conda-lock.yml`
  and install with `conda-lock install --name nasti-critters conda-lock.yml`. `requirements.txt` is gone. Activate the venv last so its interpreter wins:
  `mamba env create -f environment.yml && mamba activate nasti-critters && uv sync --extra torch && source .venv/bin/activate` — drop `--extra torch` on a CPU-only machine such as Sherlock.
- Version pins trace the ENCODE spec: `fastp=0.23.4`, `star=2.7.11a`, `samtools=1.18` (conda);
  `pyPINTS==1.1.10`, `umi-tools==1.1.5` (uv).
- **TORCH IS NOT A BASE DEPENDENCY — the base set is torch-free and there is ONE extra.**

      uv sync                 # torch-free: tfmodisco, analysis, every launcher
      uv sync --extra torch   # adds training, benchmarking, attribution

  Verified after the split: the default resolution contains **0** torch-carrying packages and
  `--extra torch` adds **21** (torch, triton, the `nvidia-*` set, bpnet-lite, tangermeme, cherimoya).
  **The split is by DEPENDENCY, not by task**, which is the part that trips people up: `tangermeme`
  requires `torch>=2.0` and `bpnet-lite` `torch>=1.9.0`, so anything importing either is on the torch
  side even where the science does not obviously involve a model — `fit_*`, `benchmark_*`,
  `attribute.py`, `make_negatives.py`, and `filter_nonACGT_regions.py --save-ohe` (the filtering is
  torch-free; only the one-hot encoding, which goes through `extract_loci`, is not). Before adding a
  base dependency, run `pip download <pkg> --no-deps` and read its `Requires-Dist`.
- **Sherlock takes the base set and runs the CPU-demanding jobs; GPU work happens elsewhere.** Its glibc
  tier has no manylinux wheel for torch above **2.6.0**, newer versions fall back to an sdist build that
  fails, and 2.6.0 is too old for cherimoya's `torch.optim.Muon` (2.10) — so there is no torch version
  that both installs there and satisfies this repo. Rather than pin a crippled one, the base set simply
  has no torch and installs cleanly, which is all tfmodisco needs.
  This replaced a briefly-lived pair of mutually exclusive `sherlock` (torch==2.6.0) and `cherimoya`
  (torch>=2.10) extras with a `[tool.uv] conflicts` declaration, copied from procap-atlas. That works
  and is more machinery than the actual requirement: **Sherlock is not used for GPU jobs here.**
- **Sherlock cannot build ANYTHING from source with its default toolchain, so the base set must install
  from wheels alone.** Two facts, both measured rather than inferred. Its glibc is below 2.28 — from the
  torch step change, `torch-2.6.0-...-manylinux1_x86_64.whl` against
  `torch-2.7.0-...-manylinux_2_28_x86_64.whl` — so any package whose newest wheel is `manylinux_2_26` or
  later falls back to an sdist. And its default compiler is **gcc 4.8.5 with binutils 2.27**, 2015
  vintage, reported by meson as `c++ (GCC) 4.8.5` / `ld.bfd 2.27`. That toolchain has **no C++17** and
  no AVX512-VNNI, so the fallback does not merely run slowly, it fails:

      contourpy 1.3.3   ERROR: C++ Compiler does not support -std=c++17
      pybigtools 0.3.0  Error: no such instruction: `vpdpbusd %ymm12,%ymm3,%ymm4`

  **An earlier version of this note said a missing wheel means a source build, "not automatically a
  failure". On this cluster it is.** That framing survived two rounds of pinning one package at a time,
  each of which just moved the failure to the next package in the graph. The base set is now capped so
  that **zero** of its 34 packages need a build — verify with the audit below after any dependency
  change.

  A newer `gcc` module would also work and would need no caps, but then every source-built extension
  links against that module's libstdc++ and the module has to be loaded in each job too. Caps keep the
  environment self-contained.

  | pin | why |
  | --- | --- |

  | pin | why |
  | --- | --- |
  | `leidenalg==0.10.2` | 0.11.0 moved from `manylinux_2_17` to `2_26/2_28`; the sdist build fails. Arrives via **modisco**, so a tfmodisco-only install is affected |
  | `igraph<1.0` | leidenalg's own dependency, same jump — 1.0.0 is 2_28-only, 0.11.9 is the last 2_17 |
  | `pillow<12.3.0` | 12.3.0 dropped `manylinux_2_17` |
  | `extra-build-variables` `HDF5PLUGIN_NATIVE=False` | hdf5plugin has no 2_17 wheel at any version, so it always builds; its `-march=native` probe emits AVX512-VPOPCNTDQ that Sherlock's assembler cannot assemble |
  | `pybigtools` **moved out of base** | not pinned — moved into the `torch` extra, so Sherlock never builds it at all. See below |
  | `contourpy<1.3.3`, `h5py<3.15.0`, `hdf5plugin<6.0.0`, `numpy<2.3.0`, `pandas<2.3.3`, `scikit-learn<1.8.0`, `scipy<1.17.0` | the first version of each whose linux x86_64 wheels moved past `manylinux_2_17` |

  The first four are procap-atlas's, which is the right authority because it runs on the same cluster;
  the caps are derived here. They are upper bounds, not exact pins, so patch releases on the
  wheel-having line still resolve. **Do not raise one without re-running the audit** — the cap is the
  wheel boundary, not a guess.
- **`pybigtools` is in the `torch` extra, NOT in base, and that placement is load-bearing.** It is
  `manylinux_2_28`-only at every version, so it always builds from source on a pre-2.28 glibc, and
  0.3.0's build dies in `libdeflate-sys` with `no such instruction: vpdpbusd` — GCC emitting AVX512-VNNI
  that Sherlock's assembler cannot assemble, the same class of failure as hdf5plugin's `-march=native`.
  Pinning to 0.2.5 (procap-atlas's choice) was the first fix and is the weaker one, because 0.2.5 also
  has to build. **Nothing on the base side needs it**: the only venv-side importer is
  `src/make_negatives.py`, which is torch-side anyway since it imports tangermeme, and `orientation_qc.py`
  reads bigWigs with it from the MAMBA env, where it is a conda package. So Sherlock never builds it.
  Do not move it back to base to "declare what we import" — the declaration lives in the `torch` extra,
  which is where the importer lives.
- **Zero base packages now need a source build.** Re-audit after ANY dependency change: resolve the base
  set, then check each wheel's platform tags against `manylinux_2_17`. A single uncapped transitive
  dependency is enough to break `uv sync` on Sherlock, and it will surface as a compiler error deep in a
  build log rather than as a resolution failure. **No 2_17 wheel means a
  source build, not automatically a failure** — but do not lean on that the way an earlier version of
  this note did. It cited `pybigtools` as proof, reasoning that upstream installs it on Sherlock despite
  its being 2_28-only at every version. That had the example exactly backwards: upstream pins
  **0.2.5** precisely *because* newer releases fail to build there. The general point survives (a
  missing wheel is not by itself a problem); the evidence for it did not.
  So the rule is empirical, not deductive: pin when a build actually fails, and take a pin from
  procap-atlas as evidence that one does. Re-audit with:

      uv export --no-emit-project --no-hashes   # then check each wheel's tags on PyPI

- **Do NOT try to get a newer torch from conda-forge. Tried, rejected.** conda-forge ships pytorch up to
  **2.13.0** for linux-64, and conda packages carry no manylinux tag, so it looks like the obvious way
  round the wheel ceiling — a modern torch there would also let cherimoya run natively instead of
  through Apptainer. It does not work: **conda's torch build does not reliably use CUDA.** The version
  list is not the problem, so re-checking it proves nothing; this was established by experience on the
  cluster.
- **`modisco` cannot move to `environment.yml`, though by this repo's own rule it belongs there.**
  Nothing imports `modiscolite`; `src/bpnet/modisco/` only ever shells out to the `modisco` CLI, exactly
  like `umi_tools` and `pints_caller`. But it is PyPI-only — checked 2026-09-05, HTTP 404 for
  `modisco`, `modisco-lite`, `modiscolite`, `tfmodisco-lite` and `tfmodisco` on **both** bioconda and
  conda-forge, and its `memelite` dependency is 404 there too. Same situation as `tangermeme`. Do not
  re-litigate this without re-checking those names.
- Two package-name traps, both already handled — do not "fix" them back: the peak caller is **`pyPINTS`**
  (PyPI `pints` is unrelated time-series inference), and PyPI **`muon` is a multi-omics framework**, not the
  optimizer. `fit_cherimoya.py` imports `torch.optim.Muon` and raises a pointed error rather than falling
  back to it.
- Sherlock: `mamba activate torch` also works; install the Python side with
  `uv pip install --python "$CONDA_PREFIX/bin/python" -r pyproject.toml`. Site-specific module loads
  go in a `launch.py --setup-file`; no tracked script carries them.
- Preprocessing shells out to: fastp, STAR, samtools, bedtools, GNU coreutils `sort`, `bgzip`,
  `bedGraphToBigWig`, `pints_caller` and `umi_tools`. **`bigWigToBedGraph` and `bigWigMerge` are gone**,
  and so is the `bpnet` CLI from this list — see the next bullet. The pipeline pulls FASTQs straight from ENA over HTTPS, so SRA Toolkit is not
  needed; homerTools, `fasterq-dump` and `proseq2.0` are no longer used anywhere.
- **Site-specific values: PARTITIONS AND THE GPU CONSTRAINT ARE NOW DEFAULTED; everything else is not.**
  This reverses part of an earlier rule, deliberately and on request (2026-09-05) — typing
  `--partition`/`-C` on every submission is its own error source. `src/launcher.py` holds
  `GPU_PARTITION = "akundaje,owners"`, `CPU_PARTITION = "normal,akundaje,owners"` and `GPU_CONSTRAINT`,
  the `|`-joined GPU SKU list copied from procap-atlas's cherimoya launchers (`|` is SLURM's OR, so any
  one card satisfies it). Both stay overridable, so another site needs a flag rather than a patch.
  **Which set a launcher gets is derived from ONE argument**, `_add_common_args(..., gpu=)`, so the
  partition and the constraint cannot drift apart from each other or from `_emit`'s `gpus`. `-C` is
  additionally gated on `gpus` at emission, so a global `--constraint` cannot leak onto a CPU job and
  narrow it to GPU nodes for nothing.
  **The `owners` partition caps jobs at 48:00:00** and is in every default here, so a longer `--time`
  will simply not schedule there; `modisco motifs` sits exactly on that cap.
  Everything else is unchanged: container use is opt-in via `APPTAINER_IMAGE` / `APPTAINER_BIND`; job
  environment setup defaults to activating this repo's mamba env + uv venv, with `launch.py
  --setup-file` for site-specific module loads. **Do not reintroduce cluster PATHS into tracked files.**
  The four hand-written `.sh` launchers, which emitted no partition or constraint of their own, are
  **deleted**: both `fit/slurm.sh` and cherimoya's `benchmark/slurm.sh` and `benchmark/cmd.sh`. So
  `src/launcher.py` is now the only thing in the repo that writes an sbatch script — do not add another.
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

# Just the initiator logos + metaplots, for every experiment. Narrow on
# purpose: `qc` also builds the three reports that read the RAW FASTQs.
snakemake orientation -c8 --rerun-triggers mtime        # target BEFORE --config

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

# Same selection, no SLURM: bare commands on stdout, skips and summary on stderr
python src/bpnet/fit/launch.py --print-commands | bash

# Train Cherimoya, one experiment/fold -- any experiment, same interface as BPNet
python src/cherimoya/fit/fit_cherimoya.py -e D.melanogaster-S2_PROcap -f 0

# Submit all (experiment x fold) Cherimoya jobs; same launcher as BPNet
python src/cherimoya/fit/launch.py --dry-run
python src/cherimoya/fit/launch.py --print-commands | bash

# Compare the two model families on shared benchmark metrics
python src/analysis/compare_bpnet_cherimoya.py

# Evaluate / attribute
python src/cherimoya/benchmark/benchmark_cherimoya.py -e D.melanogaster-S2_PROcap --save-output
python src/bpnet/benchmark/benchmark_predictions.py -e D.melanogaster-S2_PROcap
python src/bpnet/attribute/attribute.py -e D.melanogaster-S2_PROcap --attribute-type profile

# Submit all benchmark jobs; job unit is the EXPERIMENT (all folds in one run,
# because the genome-wide block is pooled across them)
python src/bpnet/benchmark/launch.py --dry-run
python src/cherimoya/benchmark/launch.py --save-predictions
python src/bpnet/benchmark/launch.py --print-commands | bash

# Submit all attribution jobs; job unit is (experiment x type), NOT x fold
python src/bpnet/attribute/launch.py --dry-run
python src/bpnet/attribute/launch.py --attribute-type profile --attribute-type counts

# Optional, and STANDALONE -- nothing calls it. Drops loci whose window holds a
# non-ACGT base, and writes the one-hot for what survives. Reaches attribution
# only via --loci; without that, attribute.py reads exp.peaks and ignores both.
# REQUIRED before attribution: deep_lift_shap rejects any window containing an N,
# and extract_loci(ignore=...) blanks rather than drops those. Also writes the
# one-hot array TF-MoDISco needs alongside the attributions.
python src/bpnet/attribute/launch_filter.py --dry-run     # one CPU job per experiment
python src/bpnet/attribute/launch.py --dry-run    # requires the filtered set

# TF-MoDISco: motifs then report. Both CPU-only; motifs is long (allow days).
python src/bpnet/modisco/launch.py --dry-run
python src/bpnet/modisco/launch_report.py --dry-run
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

**A bpnet-lite checkpoint IS a pickled module, not a state dict — load it with
`experiments.load_model()`.** `bpnetlite/bpnet.py` persists with `torch.save(self, ...)` for both
paths, verified in the installed source. Two things follow, and the downstream scripts got both wrong
until 2026-09-04:

- **`weights_only=False` is required.** PyTorch 2.6 flipped the default to `True`, so
  `benchmark_predictions.py` died with `UnpicklingError: ... Unsupported global: GLOBAL
  bpnetlite.bpnet.BPNet was not an allowed global by default`. Allowlisting with `add_safe_globals` is
  the wrong remedy: it would have to cover BPNet and every type it pickles, and these checkpoints are
  this repo's own training output, not untrusted input. procap-atlas passes `weights_only=False` at
  every bpnet-lite load site.
- **Do not reconstruct a `BPNet` and call `load_state_dict`.** Both scripts did, and it could never
  have worked at any `weights_only` setting — reproduced on torch 2.10, it raises
  `TypeError: Expected state_dict to be dict-like, got <class 'BPNet'>`. It also carried a quieter
  hazard: the architecture came from the *current* `config/bpnet_params.json`, so a checkpoint trained
  under a different `n_filters`/`n_layers` would be loaded into the wrong shape.

`attribute.py` was additionally unrunnable one line earlier: it read `params["n_outputs"]` and
`params["n_control_tracks"]`, which are set **nowhere** — absent from `bpnet_params.json`, from the
`params.update()` block and from any CLI flag — so it raised `KeyError` before reaching the load.
Deleting the reconstruction removed that too.

**Cherimoya does not go through `load_model()`.** It saves a dict payload and reconstructs via
`cls(**payload['config'])` inside its own `Cherimoya.load()`, which the `weights_only` change does not
affect. That is the only other model-load site in the repo; audited 2026-09-04, there are exactly three.

## Peak-level splits were ignored by every downstream script

S. pombe and S. moellendorffii assign folds per PEAK, not per chromosome, so `fold_split()` returns
`test_chroms=None` for them and the filtering has to happen on the peak table. `fit_bpnet.py` and
`fit_cherimoya.py` do that through `exp.fold_loci()`. **Nothing downstream did**, and the two failure
modes were very different:

- **`attribute.py` crashed.** `chroms = [c for f in folds for c in f["test_chroms"]]` raised
  `TypeError: 'NoneType' object is not iterable`. Loud, and the reported symptom.
- **Both benchmark scripts silently scored every fold's model on ALL loci** — its own training peaks
  included — because `chroms=None` means "no chromosome filter" rather than "no loci". Metrics inflated,
  no error. This is the worse one, and it was only found by chasing the crash.

Fixes, and note they are deliberately different because the two scripts want different things:

- `fold_loci()` gained **`test_loci`**. It already returned `train_loci`/`valid_loci` but only a
  `n_test` COUNT, so benchmarking had nothing to filter with. Under chromosome-level splits it returns
  every peak and `test_chroms` does the work, so that path is unchanged; under peak-level it returns the
  fold's held-out peaks. Verified on a fixture: 4 of 20 loci for fold 0 of 5, zero train/test overlap,
  and all 20 returned under chromosome-level.
- Both benchmarks now extract `exp.fold_loci(loci, fold)["test_loci"]`.
- `attribute.py` takes `[... for c in (f["test_chroms"] or [])] or None`, i.e. **no chromosome filter**
  for peak-level species. That is correct rather than a workaround: their peaks are all in the fold
  table, and attribution does not hold out anyway — see below.

**Attribution deliberately does NOT hold out, and that is inherited, not accidental.** It extracts every
locus once and attributes it with EVERY fold's model, then averages. Upstream's `attribute_bpnet.py`
does the same, with `chroms=all_chrom`. So each locus is attributed by four models that saw it in
training plus the one that did not: it is an **ensemble attribution, not a held-out estimate**, and
reading it as evidence of generalisation would be wrong. `benchmark_predictions.py` is the per-fold
held-out path and is where generalisation numbers come from.

## attributions/ and modisco/ are split by family; the filtered BED and OHE are not

    attributions/{experiment}_filtered.bed          shared
    attributions/{experiment}_ohe.npz               shared
    attributions/{family}/{experiment}_attr_{type}_{mode}.npz
    modisco/{family}/{experiment}_{type}_{mode}.modisco.h5
    modisco/{family}/{experiment}_{type}_{mode}.modisco/

**The split follows what produces each file.** Attributions come out of a MODEL, so bpnet's and
cherimoya's are different files and belong under `attributions/{family}/` beside the `models/{family}/`
and `performance_metrics/{family}/` convention. The filtered BED and its one-hot encoding depend only on
(loci, sequences, in_window), so every family attributing the same experiment shares them — putting them
under a family directory would imply they need producing twice, and would mean `launch_filter.py` had to
know which family it was filtering for, which it does not.

`attribution_path()` therefore takes `family` first, matching `model_path()`; `filtered_loci_path()` and
`ohe_path()` do not take one at all. `attribute.py` is BPNet-only and pins `FAMILY = "bpnet"` at module
level rather than accepting a flag, because cherimoya attribution is blocked on rescale rules, not on
plumbing.

`modisco/` is split the same way — `modisco/{family}/{experiment}_{type}_{mode}.modisco.h5` and the
matching `.modisco/` report directory — since motifs are discovered FROM a model's attributions and
inherit their provenance. Only `bpnet/` exists today, because cherimoya attribution is blocked, but the
level is there so the second family does not need a migration.

## `filter_nonACGT_regions.py` is REQUIRED for attribution, and produces modisco's other input

An earlier version of this section called it optional and said a blank column is usually tolerable.
**Both were wrong.**

**`deep_lift_shap` refuses a sequence containing an unknown base.** Verified directly against tangermeme
1.4.1 on a two-sequence fixture: a single all-zero column gives
`ValueError: X must be one-hot encoded. and cannot have unknown characters.` — and it fails in **BOTH**
reference modes, so the frequency default does not rescue it. The check is inside `deep_lift_shap`
itself, not in the dinucleotide shuffle.

**And `extract_loci(ignore=IGNORE)` is exactly what creates that column.** `ignore` KEEPS a locus
containing an N and zeroes the column rather than dropping the locus (verified: one N gives 1 row with
exactly 1 blank column). So the setting every `extract_loci` call in this repo passes is what makes
attribution fail, and `filter_nonACGT_regions.py` — which DROPS such loci — is the remedy. That is the
whole reason it exists; the `snp_bed` variable name is a leftover from where it was first used.

**A PINTS peak file is RAGGED, and the filter script has to read it line by line because of that.**
`combine_peaks` concatenates the unidirectional and bidirectional calls, which carry different numbers
of columns, so `pd.read_csv(peaks, sep="\t", header=None)` dies with
`Expected 6 fields in line 2, saw 9`. **`load_bed()` is unaffected** — its `usecols=[0, 1, 2]` with three
matching `names` reads a ragged file fine, which is why training, benchmarking and attribution never hit
this. Adding `usecols` to the filter script would fix the read and be wrong: its output is written back
out as a BED, and upstream deliberately keeps PINTS' strand/confidence/class/summit columns rather than
cutting to BED3 — the summit column is what makes `extract_loci(summits=True)` possible. So it keeps the
raw line, which preserves every column whatever their number, and returns `(kept_lines, coords)`.
It also has to handle **bgzipped** input, since that is what `combine_peaks` writes.

`attribute.py` now catches this before the library does, because the library's message names neither the
loci nor the remedy: it counts the offending rows, prints the first few, and prints the two commands
that fix it.

**The filtered set is the DEFAULT locus set, not an opt-in.** It was briefly a `--use-filtered` flag on
the launcher, which had the default backwards: if `deep_lift_shap` cannot accept an N and
`extract_loci(ignore=...)` always produces one, then attributing raw peaks is the broken path and must
not be what happens when you pass nothing. So `attribute.py --loci` now defaults to
`filtered_loci_path(exp.id)` and exits 1 with the `launch_filter.py` command if it is absent, and the
launcher skips an unfiltered experiment rather than emitting a job that would fail. `--loci` survives
only for a genuinely different locus set, and only then does its stem enter the output filename — the
default run keeps the plain `attribution_path()` name.

**The `--save-ohe` array is not a convenience either — TF-MoDISco requires it.** modisco takes
one-hot sequences alongside contribution scores, so the OHE is a second required input rather than a
debugging aid. That is the strongest argument for it living in the filter step: it must describe exactly
the loci that were attributed, and the filter is what decides which those are.

**`attribute.py` saves HYPOTHETICAL attributions**, which is also what modisco wants
(`hypothetical_contribs`): `hypothetical=True` is passed unconditionally, and the stored array is the
mean over folds of those. Observed/actual contributions are `hypothetical * one_hot`, derivable from the
two files, so the pair is complete for modisco and nothing else needs saving.

**Passing `--loci` changes the default output name** (`attributions/bpnet/{exp}_{stem}_attr_{type}_{mode}.npz`), for
the same reason the reference mode is in there: different loci give different numbers, and nothing else
on disk would record which set produced the file. The launcher mirrors that naming, so its already-done
check follows.

## Launchers: eight of them, one emission path

`src/launcher.py` holds the selection rule and the emission machinery; the eight
`launch.py` files are thin wrappers. All of them take the same emission modes —
`--print-commands` (bare commands on stdout, skips and summary on stderr), `--dry-run` (full sbatch
scripts) and the default (submit) — and the same SLURM flags, because `_add_common_args` and `_emit` are
shared. Do not add another copy of that block.

| launcher | job unit | jobs | GPU | skips |
| --- | --- | --- | --- | --- |
| `src/bpnet/fit/launch.py` | experiment x **fold** | 214 | yes | `.final.torch` exists, missing inputs, no fold assignment |
| `src/cherimoya/fit/launch.py` | experiment x **fold** | 214 | yes | same |
| `src/bpnet/benchmark/launch.py` | **experiment** | 42 | yes | metrics JSON exists, missing inputs, **folds not all trained** |
| `src/cherimoya/benchmark/launch.py` | **experiment** | 42 | yes | same |
| `src/bpnet/attribute/launch.py` | experiment x **attribute type** | 42 x types | yes | output npz exists, missing inputs, **folds not all trained** |
| `src/bpnet/attribute/launch_filter.py` | **experiment** | 42 | **no** | filtered BED + OHE exist, peaks/sequences missing |
| `src/bpnet/modisco/launch.py` | experiment x **attribute type** | 42 x types | **no** | .h5 exists, attribution or OHE npz missing |
| `src/bpnet/modisco/launch_report.py` | experiment x **attribute type** | 42 x types | **no** | report dir exists, .h5 or MEME db missing |

Run order is filter -> attribute: `launch_filter.py`, then `launch.py`. The second **skips any experiment the first has not covered**, because attribution of unfiltered peaks cannot work.

**The benchmark launchers' job unit is the EXPERIMENT, not the fold, and unlike attribution that is
forced rather than merely convenient.** Both benchmark scripts score every fold and then report a
`genome_wide` block pooled over all of them — one correlation across every fold's predictions
concatenated — which cannot be computed if the folds are split across jobs. So one job per experiment is
the only shape that produces the output the scripts already define. Two launchers rather than one
because the script paths and families differ, exactly as for fit; the selection rule is one function.

**Their already-done check goes through `experiments.metrics_path()`, which both scripts now write
through.** `--metrics-dir` was a bare string default in each script (`performance_metrics/bpnet`,
`performance_metrics/cherimoya`) and a launcher predicting that path would have been a third copy. Same
rule as `attribution_path()`. The deleted `src/cherimoya/benchmark/cmd.sh` is what happens without it:
it hard-coded `D.melanogaster-S2_PROcap.json` as its already-done check while forwarding `"$@"`, so once
fly was benchmarked every other experiment printed "Skipping" and exited 0.

**Attribution's job unit is not the fold**, which is why it needed its own enumeration rather than a flag
on the fit launcher: `attribute.py` loops every fold internally and averages their attributions, so one
job covers all folds of one experiment. It follows that a partly trained experiment is not partly
attributable — `attribute.py` exits 1 — so the launcher skips it and reports `3/5 folds trained` rather
than just refusing.

**`launch_filter.py` requests NO GPU, which is why it is a fourth launcher rather than a flag on the
third.** `filter_nonACGT_regions.py` reads a FASTA and one-hot encodes; sending that to the GPU
partition would queue it behind training and then hold an idle card. `_emit(..., gpus=0)` omits the
`#SBATCH --gpus` directive and the `nvidia-smi` line; every other launcher passes the default 1.
It also gates on `missing_paths(kinds=("peaks", "sequences"))` rather than on `exp.missing`, which would
additionally demand negatives and trained models that this step has nothing to do with.

**The already-done check must predict the output path**, and the reference mode is part of that path, so
`experiments.attribution_path()` is the single definition shared by `attribute.py` and the launcher. Two
copies of that format string is exactly how a launcher starts re-running finished work.

## Comparing the two model families

`src/analysis/compare_bpnet_cherimoya.py` collates
`performance_metrics/{bpnet,cherimoya}/{experiment}.json`, inner-joins on experiment, writes
`plots/bpnet_vs_cherimoya/collated.tsv` and one figure per metric: a scatter with a y=x line plus a
histogram of per-experiment deltas, with a Wilcoxon signed-rank test. Ported from procap-atlas's
`src/analysis/compare_bpnet_cherimoya.py`, which is why it looks the way it does.

It only works because **both benchmark scripts now write the same schema** — that was the point of
giving `benchmark_predictions.py` a metrics JSON at all. The four shared metrics are
`profile_pearson`, `profile_jsd`, `log_counts_pearson`, `counts_spearman`; BPNet's extra
`counts_pearson` is skipped automatically rather than half-plotted.

Two deliberate departures from upstream:

- **Points are coloured by SPECIES, not read depth.** Upstream is human-only, so depth is its only
  axis; here the question is whether one architecture wins uniformly or only on some clades, which a
  12-species corpus can actually answer. `--colour-by depth` restores the upstream view from
  `qc/stats/experiment_stats.tsv`.
- **No consolidate step.** Upstream inner-joins two pre-consolidated TSVs; reading the per-experiment
  JSONs directly removes a stage that could go stale against them.

**`--aggregate` picks how folds are reduced, and the three options give genuinely different numbers.**
The default is `fold-mean`, not the benchmark's `genome_wide` block:

| | what it is | weights equally |
| --- | --- | --- |
| `fold-mean` *(default)* | mean of the per-fold metrics | every **fold** |
| `genome-wide` | the benchmark's pooled block — one correlation over all folds' predictions concatenated | every **locus** |
| `per-fold` | one row per fold | — |

**`genome-wide` is NOT the mean of the per-fold correlations** and generally differs from it, which is
why this is a choice rather than an implementation detail. Pooling lets a large fold pull the number
harder; for C. elegans, where one fold is one chromosome, the fold sizes differ enough for that to
matter.

`fold-mean` also carries `{metric}_sd` and `n_folds` into the collated TSV and draws ±1 sd error bars on
the scatter. That is worth having: a bare point invites reading a 0.01 gap between families as real when
the folds behind it span 0.05.

`per-fold`'s Wilcoxon p is **not interpretable** — folds of one experiment share an architecture, a
library and a peak set, so 5 × 42 is not 210 independent pairs and the test is anticonservative. Use it
to see spread, not significance.

Cherimoya is not deployment-ready, so treat anything this produces as a development comparison rather
than a result.

## TF-MoDISco: the package is `modisco`, NOT `modisco-lite`

`modisco-lite` is the **deprecated name for the same project**, and both are on PyPI — `modisco` 2.5.2
against `modisco-lite` 2.4.0. They are not two packages: `modisco` installs the same top-level
`modiscolite/` package and the same `modisco` script, so anything installing both gets whichever landed
second, silently.

That is a live risk here rather than a hypothetical, because the two upstreams disagree:
**cherimoya 0.2.0 requires `modisco>=2.0.0`, bpnet-lite 1.0.0 on PyPI still requires
`modisco-lite>=2.0.0`** — so `uv sync --extra torch` would pull both. `[tool.uv] override-dependencies`
drops `modisco-lite` behind the same unsatisfiable marker used for `macs3`; verified with
`uv export`, where it appears as `modisco-lite==2.4.0 ; sys_platform == 'nonexistent'` and never
resolves, while `modisco==2.5.2` installs.

**The CLI surface is identical across the rename**, checked rather than assumed: subcommands
`motifs`/`report`/`convert`/`meme`, and `-n/--max_seqlets`, `-l/--n_leiden` (still defaulting to 2),
`-w/--window`, `-m/--meme_db`, `-l/--lite` all unchanged. So `src/bpnet/modisco/` needed no edit. The
only dependency difference is that `modisco` adds `jinja2`.

## TF-MoDISco

`src/bpnet/modisco/` follows procap-atlas's scripting: `modisco motifs` then `modisco report`, one job
per (experiment x attribute type), with their parameters — `-n 1000000` seqlets, `-w 1000` window,
and `--lite` on the report. **`-l` is 2, not procap-atlas's 50**: it is the number of Leiden
CLUSTERINGS (restarts with different seeds), not clusters, so 50 is 25x modisco's own default in
compute for a parameter upstream appears to have set under the wrong description.

    launch_filter.py -> attribute/launch.py -> modisco/launch.py -> modisco/launch_report.py

**`modisco motifs` needs BOTH npz files**: the attribution and the one-hot. That is what
`filter_nonACGT_regions.py --save-ohe` is for, and it is why the OHE lives with the filter rather than
with attribution — it must describe exactly the loci that were attributed.

**Both stages are CPU-only (`gpus=0`), and their resource defaults are NOT shared** — the two commands
differ by more than an order of magnitude in every dimension:

| | CPUs | mem | time | `NUMBA_NUM_THREADS` |
| --- | --- | --- | --- | --- |
| `modisco motifs` | 32 | 64G | 48:00:00 | 32 |
| `modisco report` | **4** | 16G | **2:00:00** | **4** |
| fit launchers (for contrast) | 4 | 32G | 6:00:00 | unset |

`motifs` is numba-parallel and runs for many hours; **`report` finishes inside two** — it reads one
`.h5`, matches its motifs against a MEME database and writes logos. Defaulting them together meant every
report job reserved 32 idle cores for 48 hours, which queues badly and wastes allocation.

**`report` is not literally single-threaded, though, and the numba pin is NOT a no-op for it.** Traced
2026-09-05 because the obvious reading is that only `motifs` touches numba: `modiscolite/report.py`
imports none, but it calls `memelite.tomtom`, which is `@njit(parallel=True, cache=True)` over a
`prange` and calls `numba.set_num_threads(n_jobs)`. `report.py` invokes it as
`tomtom(ppms, target_pwms, n_nearest=top_n_matches)` — **no `n_jobs`** — so it takes memelite's default
of `-1` and uses every numba thread available. Without the pin a 1-CPU report job would spawn one thread
per core on the node, which is precisely the oversubscription the pin exists to stop.

So report gets **4** cores: enough for that parallel section to be worth having, and far short of
motifs' 32 because the tomtom call is small — tens of query motifs against a few hundred JASPAR targets
— and the wall is dominated by logo rendering and HTML. The pin follows `--cpus-per-task`, so raising it
is picked up by tomtom rather than ignored.

`_add_modisco_args` therefore takes `default_cpus`/`default_mem`/`default_time` as **required** keyword
arguments with no fallback, so the next caller cannot inherit the wrong set by omission;
`_add_common_args` takes them the same way with the fit values as its defaults.

**`NUMBA_NUM_THREADS` is pinned to `--cpus-per-task` on every modisco job.** numba otherwise sets it
from every core it can SEE, which on a shared node is the whole machine and not the slice SLURM granted
— a job holding 32 CPUs on a 128-core node spawns 128 threads, oversubscribes its own cgroup and can run
slower than if it had asked for less, while degrading everything else on the node. tfmodisco is
numba-heavy throughout, which is why this is set here and nowhere else.

It rides on the command as a `VAR=value cmd` prefix rather than an `export` line in the sbatch body, so
one string carries it through all three emission modes. An `export` would silently vanish under
`--print-commands` — the mode most likely to be run on a box where the variable matters.

**The MEME database is chosen PER SPECIES, and this is the one place the port could not follow upstream.**
procap-atlas hardcodes JASPAR CORE **vertebrates**, which it can afford to because it is human-only;
reporting a yeast or plant motif against a vertebrate database yields matches that mean nothing. So
`config/genomes.yaml` carries `jaspar_collection` per species and `experiments.motif_db_path()` resolves
it:

| collection | species |
| --- | --- |
| vertebrates | M. musculus, C. griseus |
| insects | D. melanogaster |
| nematodes | C. elegans |
| fungi | S. cerevisiae, S. pombe |
| plants | A. thaliana, C. reinhardtii, P. patens, S. moellendorffii, G. arboreum, G. hirsutum |

**Nothing fetches those files** — download them from JASPAR into `data/motifs/` as
`JASPAR2026_CORE_{collection}_non-redundant_pfms_meme.txt`. They are deliberately not a pipeline step:
the report is a convenience layer and the database has no effect on which motifs modisco discovers.
`--motif-db` overrides with a single file for every experiment, which is rarely right here.

`modisco` is declared directly in `pyproject.toml` even though bpnet-lite already pulls it
transitively, because these scripts invoke its `modisco` CLI — same reasoning as `pybigtools`. Note it
ships the entry point as an old-style `data/scripts/modisco`, not a `console_script`.

**Not ported: upstream's `hitcall/` tree** (Fi-NeMo hit calling, `compute_trim_floor.py`,
`link_hits_to_compendium.py`) and `modisco/relaunch_timeout.py`. The first depends on `finemo`, which is
Linux-only and a further scope step; the second exists to resubmit jobs that hit a wall clock, which is
a site policy rather than a pipeline stage.

## Where benchmark output goes

Both benchmark scripts now write the same three things; `performance_metrics/`, `predictions/` and
`logs/` are all gitignored, so nothing here is committed.

| | BPNet | Cherimoya |
| --- | --- | --- |
| metrics JSON | `performance_metrics/bpnet/{experiment}.json` | `performance_metrics/cherimoya/{experiment}.json` |
| override | `--metrics-dir` | `--metrics-dir` |
| launcher | `src/bpnet/benchmark/launch.py` | `src/cherimoya/benchmark/launch.py` |
| raw predictions | `--output-fname` (joblib, opt-in) | `--save-output` -> `predictions/cherimoya/` (npz) |
| printed | per-fold **and** genome-wide | per-fold **and** genome-wide |

**`benchmark_predictions.py` saved NOTHING until 2026-09-04 — it only printed.** So every BPNet
benchmark run before then left no artifact, while `benchmark_cherimoya.py` had always written a JSON.
The JSON now carries the same shape as cherimoya's (`run_name`, `model_paths`, `per_fold`,
`genome_wide`) so the two families are directly comparable, plus `counts_pearson`, which this script
already computed and cherimoya's does not. It also gained the genome-wide block it was missing.

Both paths are now defined by `experiments.metrics_path(family, experiment, metrics_dir=None)` rather
than by a string default in each script, so the launchers' already-done check cannot drift from where
the scripts write.

**Genome-wide is POOLED across folds, not averaged over them** — `pearson_corr` over the
concatenation, so each locus counts once regardless of how large its fold was. Averaging per-fold
correlations would weight a small fold equally with a large one, and for C. elegans, where one fold is
one chromosome, the fold sizes differ enough to matter. Same construction in both scripts; keep them in
step.

Note upstream's `benchmark_bpnet.py` also reports `orientation_index_pearson`, which neither script
here computes. Not an oversight to fix silently — adding it means defining the orientation index the
same way upstream does.

**Progress bars are ON by default in the benchmarks and in `attribute.py`, via `--no-progress` to
suppress.** They are tangermeme's
`verbose` argument to `extract_loci` and `predict`, which is *only* the tqdm bar, so it is wired to
`--no-progress` rather than to `-v`: a long benchmark should show progress without turning on every
other message. Bars go to stderr, so stdout stays clean for the printed metrics and can be piped.
`benchmark_predictions.py` and `attribute.py` both had **no `-v` flag at all**, so `params["verbose"]`
was permanently `false` from `config/bpnet_params.json` and no bar could ever appear; both have one now.

`attribute.py` also gets an **outer bar over folds**, which is the one that matters there: each fold is a
whole `deep_lift_shap` pass over every locus, so without it the only feedback for minutes at a time is
tangermeme's inner bar restarting from zero with no indication of how many more times it will do so.

**Heavy imports are deferred.** `torch`, `bpnetlite`, `cherimoya`, `tangermeme` and
`data_loader` are imported *inside* `main()`, after argparse and path validation, so
`--help` and missing-data errors stay instant on a login node and are testable without a
GPU stack installed. `fit_bpnet.py` was the last holdout and now follows suit.

## Label generation: the ENCODE PRO-cap pipeline

Labels are produced by the ENCODE PRO-cap pipeline, transcribed verbatim from
`planning/20240501_PRO-cap_Computational_Pipeline.pdf` into `config/procap_pipeline.yaml` (tool versions
and exact parameter strings live there; deviations for non-human species are marked `DEVIATION`):

    fastp 0.23.4 --overlap_len_require 18 --length_required 18
      -> STAR 2.7.11a --alignMatesGapMax 1000 --outFilterMultimapNmax 10
                      --outFilterMismatchNmax 1 --outFilterMultimapScoreRange 0 --outSAMattributes All
      -> samtools 1.18 -q 255 (STAR marks unique reads MAPQ 255)
      -> umi_tools 1.1.5 dedup      (UMI libraries only)
      -> 5' stranded bigWigs, merged across replicates
      -> PINTS 1.1.10 --min-lengths-opposite-peaks 5

Two drivers, same steps:

- **`workflow/Snakefile` — preferred.** Proper DAG: parallel, resumable, atomic outputs, per-rule
  resources. `snakemake -c16 --config tier=include`. **Verified by local dry-run 2026-08-30**
  (`uvx --from snakemake --with pyyaml snakemake -s workflow/Snakefile -n -c1 --config tier=...`):
  and re-measured 2026-09-02 after cotton: **~286 jobs for `tier=include`, ~908 for all tiers** over 42
  experiments and 12 genomes. **Treat the totals as approximate**: `fetch_fastq` is one job per FASTQ
  *not already on disk*, so the number moves with local state — the same tree measured 906 with two more
  files present. The experiment and genome counts are the stable part. Note the **target name must come
  before `--config`** --
  `snakemake -n --config tier=include qc` makes Snakemake read `qc` as a config entry and die with
  "Config entries have to be defined as name=value pairs"; `snakemake qc -n --config tier=include`
  is correct.

  That dry-run was the first ever run against this Snakefile and it immediately found a **fatal
  pre-existing bug**: `fetch_annotation` declared `{species}.{ext}.gz` as output but only
  `{species}.log` as its log, and Snakemake requires output, log and benchmark to carry the same
  wildcards. It aborted at DAG construction with "Not all output, log and benchmark files of rule
  fetch_annotation contain the same wildcards", which took down the **whole** workflow, not just QC.
  Fixed. Run the dry-run after touching the Snakefile -- it is cheap and needs no cluster.
  There is a `fetch_only` target: transfers are capped at 4 by the `downloads` resource and are
  one thread each, so a big `-c` is wasted during fetching — run `snakemake fetch_only -c4` on a
  transfer node, then the compute phase with a large `-c`. Also pass `--resources mem_mb=N` when
  raising `-c`: `align` declares 40 GB for mouse but declared resources are inert without a limit,
  so `-c64` would start 5 mouse alignments and ask for 200 GB.
  `-c N` is a global thread budget; `threads:` per rule is both the scheduler's cost and the value
  interpolated into the tool's own flag, so the two cannot drift. `samtools -@` takes *additional*
  threads, hence `-@ $(({threads} - 1))`. `pints` sets `threads` from `N_CHROMS` (derived from
  `genomes.yaml`'s `n_chromosomes`) because PINTS parallelises across chromosomes — 8 threads on
  S. pombe's 3 chromosomes reserves 5 idle slots. That count comes from `genomes.yaml` (genome
  structure), not `chrom_splits.yaml` (fold assignment): the two answer different questions, and
  S. pombe has no entry in the latter. Concurrent ENA transfers are limited by a custom `downloads`
  resource, defaulted to 4 in the Snakefile via `workflow.global_resources.setdefault` and
  overridable with `--resources downloads=N`; a custom resource with no limit anywhere is
  unconstrained, which is why the default is set rather than merely declared.
  Run-level intermediates are keyed by *run*, not experiment, so a run shared by two experiments is
  mapped once. Scope is fetch -> negatives; `resolve_runs.py`/`build_experiment_config.py` stay outside
  (metadata, not DAG work) and training stays on `launch.py` — one per family, both thin wrappers over
  `src/launcher.py`, which also drives `src/bpnet/attribute/launch.py`.
- **`src/data_preprocessing/run_procap_pipeline.py`** — single-experiment path, serial, caches on output
  existence. Useful for `-e <one>` debugging and `--fetch-genomes`/`--index-only`.

ENCODE orchestrates these steps with [rmsp](https://github.com/aldenleung/rmsp), a DAG/caching layer;
Snakemake fills the same role here.

**ArrayExpress submissions need their ENA counterpart filled in by hand.** `resolve_runs.py` queries ENA
with `bioproject`/`sra_study` from the manifest and skips a project that has neither — which is what
`Embryo_PROcap_dm: no bioproject/sra_study in manifest; skipping query` means. `E-MTAB-6154` is an
ArrayExpress accession, and ENA's portal does not resolve it. The sequencing data *is* in ENA, brokered
under a different accession; find it via the BioStudies API
(`https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-6154`), whose record links out to the ENA study.
For this one that is **ERP106976 / PRJEB25091**, now recorded per sample in the manifest, and all 44
manifest rows resolve. If another ArrayExpress row appears, do the same lookup rather than teaching
`resolve_runs.py` a second archive.

**Fetching is separate from mapping.** `fetch_fastqs.py` bulk-downloads every FASTQ the config references
(~202 GiB, 82 files over 64 runs — measured from ENA `fastq_bytes`) with md5 verification, so transfers
can run on a login/transfer node instead of burning
GPU allocation and a failed mapping run never re-downloads. Transfers stage through
`data/fastq/.incoming/` and are only moved to the final path once the md5 matches — a file at the final path
always means complete and verified. `run_procap_pipeline.py --fastq-dir` picks those up and only falls back
to downloading on demand. Set `PROCAP_FASTQ_DIR` or symlink `data/fastq` to scratch on a cluster.

Fetch only what the config references, not whole studies: whole-study downloads are **far larger**,
because several deposits bundle unrelated assays and other organisms (PRJNA834081 is 8/11 human Ramos
libraries; SRP131922 is 294 runs of which one is wanted; PRJNA1105209 is 506 GiB of which 26 GiB is wanted).
Bulk-fetching also does not substitute for the sample->run crosswalk: the crosswalk is what *labels* each
file, and for the TAP+/TAP- studies those labels are the entire point.

Things that will bite you:

- **Paired-end libraries must contribute ONE MATE ONLY to the signal.** `bedtools genomecov -5` reports
  the 5' end of *every* alignment record, so on a paired BAM it counts R1's 5' end (the initiation site)
  and R2's 5' end (the RNA 3' end, on the opposite strand) alike — roughly half of each track becomes
  strand-flipped 3'-end signal. `final_bam` therefore applies `samtools view -f 64`, selected by
  `steps.signal.five_prime_mate`. It happens **after dedup** because `umi_tools --paired` needs both
  mates. Only 8 of 42 experiments are paired for processing (5 S. cerevisiae, both cottons, and the
  interleaved GCB run), which is why this
  went unnoticed. **R1 is a convention here, not a documented fact** — the manifest records read
  orientation only for `Liver_ChROcap_mm`, which ENA reports as single-end anyway. Validate it like
  `reverse_strand`: confirm signal piles up at annotated TSSs rather than 3' ends.
- **The Spt5 UMI is at the START OF READ 1, and `--umi_loc` comes from the manifest — not from the
  layout.** It was `read2 if paired else read1`, which was backwards for the only three experiments it
  applied to. Established by indexing the S. cerevisiae genome and locating the first genome-matching
  24-mer inside each read: the modal offset is **10 in read 1** (51-58% of reads, with the expected 1/4
  and 1/16 chance-match tail at 9 and 8) and **0 in read 2**. The 10 bases are random — no sequence
  recurs more than 3 times in 400 reads — so they are a UMI, not a fixed barcode. Note the two candidate
  models are **indistinguishable from read structure alone** if you only compare adapter offsets, because
  a UMI ligated between insert and 3' adapter appears at the end of R1 and the start of R2 exactly like
  insert sequence would; the genomic test is what separates them.
  `steps.dedup.umi_locations` maps the manifest's prose (`5' adaptor` -> `read1`) to fastp's value, with
  two guards that both fire: an unmapped prose value fails DAG construction and the serial driver, and a
  `3' adaptor` UMI on a *single-end* run is rejected outright, because there it sits at the read's 3' end
  where `--umi_loc` cannot reach it.
  What having it backwards cost the three Spt5 experiments: fastp cut 10 real genomic bases off read 2
  (harmless — read 2 is dropped by the one-mate filter); the actual UMI stayed on read 1 and reached STAR,
  where `alignEndsType Local` soft-clips it, so **5' coordinates were preserved** but aligned length was
  spent against `--outFilterMatchNminOverLread 0.66`; and umi_tools deduplicated on read 2's first 10
  *genomic* bases — a tag determined by read 2's own mapping position, so it added nothing beyond the
  fragment's 3' coordinate while dropping 43-57% of reads in the only 3 of 42 experiments that are
  deduped at all.
- **How a library is known to have a UMI: the manifest's `umi_len`/`umi_loc` columns, and nothing else.**
  `build_experiment_config.py` derives `raw.umi` from them (it used to be a hardcoded dict transcribed by
  hand from free text in `library_layout`). **The archives cannot corroborate this** — ENA's
  `library_construction_protocol` was queried for all 45 runs then resolved and mentions a UMI for
  *none* of them,
  including the three Spt5 experiments that demonstrably have one, so the manifest is authoritative and
  must be curated from the paper or GEO record. The direct evidence for Spt5 is the extracted read name,
  `SRR29037352.25948720:ACTAGATAGC` — a 10-base tag, matching `umi_len: 10`.
  The scheme is **default-deny**: a library whose UMI was never noted is treated as having none and keeps
  its PCR duplicates. That is the safer error, because deduplicating a non-UMI PRO-cap library destroys
  real stacked 5' ends — but it does mean a missed UMI is silent.
- **umi_tools does nothing to non-UMI libraries, by design.** `final_bam` takes `unique.bam` directly
  when `has_umi(run)` is false, so the `dedup` rule never runs for them — 39 of 42 experiments. This is
  deliberate: PRO-cap legitimately stacks many reads on one initiation base, so deduplicating a non-UMI
  library destroys real signal. Only the three Spt5 experiments carry a UMI.

- **fastp and umi_tools disagree about the UMI separator by default.** fastp appends the extracted UMI
  to the read name with `:`; umi_tools' `--umi-separator` defaults to `_`. An SRA-style read name
  (`SRR29037352.1234567`) contains no `_`, so umi_tools takes the *entire read name* as the UMI and
  aborts with `AssertionError: not all umis are the same length(!): 30 - 31` — the two lengths differing
  only because the read number gained a digit. Both drivers now pass
  `--umi-separator` from `steps.dedup.umi_separator`. If you change fastp's UMI handling, change this
  with it.
- **`--paired` and `--method` are orthogonal in umi_tools** — the first describes the library layout, the
  second picks the clustering algorithm. The dedup rule used to return one *or* the other, so paired runs
  silently got the default method while single-end runs were forced onto `unique`. Both now come from
  `steps.dedup` (`method: directional`, umi_tools' own default) with `--paired` added only for paired-end
  runs. Only the three `Spt5_PROcap_sc` experiments have a UMI at all, so this path is easy to leave
  broken unnoticed.

- **STAR's `--limitBAMsortRAM` defaults to the size of the GENOME INDEX**, not to anything about the
  library, so the sort budget is smallest exactly where libraries are deepest relative to the genome.
  A yeast run died with `not enough memory for BAM sorting: SOLUTION: re-run STAR with at least
  1134315395` while mouse was fine — counter-intuitively, the compact genomes break first. Both drivers
  now pass it explicitly: the Snakefile derives it from the rule's own `mem_mb` minus an index reserve
  (13 GB for the compact genomes, 10 GB for mouse), and `run_procap_pipeline.py` takes
  `--star-sort-ram GB` (default 10). Note this is separate from Snakemake's `mem_mb`, which STAR knows
  nothing about — raising `--resources mem_mb` alone does not fix it.

- **bigWigs and PINTS are restricted to `main_chromosomes` (in `config/genomes.yaml`), not the whole
  assembly.** Two reasons, both learned the hard way. `bedGraphToBigWig` records only contigs that appear
  in its input, so on dm6's 1,862 scaffolds one strand has reads where the other has none and the two
  bigWigs end up with different contig sets — PINTS then aborts with
  `bw_pl and bw_mn should have the same chromosomes`. And scaffolds cannot enter a fold anyway, since
  `chrom_splits.yaml` lists only main chromosomes. Both the `bedgraph`/`chrom_sizes` rules and
  `run_procap_pipeline.py` filter to the same list. The rDNA sink is excluded here too: it belongs in the
  STAR index, not the peak-calling space.
- **PINTS' `--chromosome-start-with` defaults to `chr`, which silently matches nothing for the
  Ensembl-named species.** S. cerevisiae (`I`..`XVI`), S. pombe (`I`..`III`) and A. thaliana (`1`..`5`)
  would have produced **zero peaks with no error**. Both drivers now pass the prefix from `chrom_style`:
  `chr` for the UCSC assemblies, empty for the Ensembl ones. This is a silent-wrong-answer bug, not a
  crash — check the PINTS log reports a sane chromosome count before trusting a new species.

- **The peak set is PINTS `unidirectional` + `bidirectional` calls, CONCATENATED and sorted — not
  interval-merged.** This follows kundajelab/ProCapNet's `_merge_uni_bi_peaks.py` (via procap-atlas), which
  does `sorted(uni + bi)` and nothing else. Do not add `bedtools merge`: collapsing overlapping intervals
  destroys the one-row-per-called-peak structure and folds unidirectional calls into overlapping
  bidirectional ones. PINTS `divergent` calls are excluded — they are a subset of the bidirectional set, so
  including them duplicates loci. Upstream also keeps strand/confidence/class/summit columns rather than
  cutting to BED3; summits allow `extract_loci(summits=True)`. Many non-human species have a large fraction
  of unidirectional TSSs, so both classes are needed; this union is the locus set for train, validation
  *and* test.
- **Replicates are merged at the BAM level, and that already IS summing.** `merge_runs` pools the
  per-run BAMs with `samtools merge`, then `genomecov -5` runs once. That is numerically identical to
  summing per-run count tracks, because 5'-end counting is additive over reads — pooling reads then
  counting equals counting per pool then summing. So `biodatatools` is not needed for merging either.
  procap-atlas sums at the bigWig level (`bigWigMerge`, in `src/preprocess/merge_bigwigs.py`) because it
  starts from GEO-provided bigWigs and has no BAMs to pool; starting from FASTQ, merging earlier is one
  `genomecov` pass instead of N passes plus a merge plus a re-conversion. It also avoids a real trap:
  **`bigWigMerge` defaults `-threshold` to 0 and drops values at or below it**, so a minus-strand track
  stored as negative values merges to nothing. The deleted legacy fly script needed
  `-threshold=-1000000` for precisely this. Do not "modernise" this into a bigWig-level merge.
  **Nothing in this repo calls `bigWigMerge` any more** — `src/make_negatives.py` was the last user and
  now sums the two strands in-process with `pybigtools`. It still abs-values the minus track first, and
  that is still load-bearing: without it the strands cancel instead of summing. The threshold trap went
  with the tool; the reason it existed is worth keeping.
- **We do NOT use `biodatatools`, and the unrecorded subcommand does not matter.** ENCODE produces its
  per-replicate bigWigs with `biodatatools` 0.0.7 and the source document omits the subcommand. That was
  logged as a blocker; it is not one, because the quantity is pinned from both ends. procap-atlas consumes
  ENCODE's per-replicate bigWigs **directly by accession** (`ENCFF*.bigWig` as `pl_bigwigs`/`mn_bigwigs`
  in its config) and sums them with `bigWigMerge` — and summing replicates is only meaningful for raw,
  unnormalised per-base counts, so that is what those files are. We produce the same quantity from the
  same read-level spec and merge at the BAM level, which is arithmetically the same sum. The subcommand
  would only buy byte-identity, which was never the goal.
  What actually runs is recorded as `steps.signal.tool`; ENCODE's choice sits beside it as
  `encode_tool` / `encode_subcommand: unrecorded`. There is deliberately **no `backend` switch** — nothing
  imports biodatatools, no code reads such a key, and the `encode-exact` extra was removed rather than
  left installing an unused package.
  **The distinction that does matter is raw vs normalised.** Summing normalised tracks is meaningless,
  which is exactly why the GEO-provided `R1Normed` yeast bigWigs had to be re-mapped rather than reused.
- **rDNA decoy.** ENCODE aligns to genome + rDNA (U13369.1 for human). No per-species rDNA accession has been
  verified — every `rdna_accession` in `config/genomes.yaml` is `null` with a note. This matters most for
  mouse (rDNA largely absent from the primary assembly) and least for yeast (rDNA inside chrXII).
- **`Liver_ChROcap_mm` is single-end — the paired-end claim was a curation error.** All 8 runs report
  `library_layout=SINGLE` with exactly one FASTQ, no `nominal_length`, and a uniform ~74.8 bp read. The
  manifest had said "paired-end; RNA 5' end in R1 and RNA 3' end in R2", which almost certainly came from
  misreading the sample names: they carry `R1`/`R2` as the **biological replicate**
  (`Old_Female_R1_PROCAP` / `Old_Female_R2_PROCAP`), pairing with the age×sex strata and with the
  manifest's own `biological_replicate` column — not read 1 / read 2. Since `library_layout` in
  `experiment_config.yaml` is resolved from ENA rather than from the manifest, the pipeline was already
  treating these correctly; regenerating the config after the fix is a no-op. `five_prime_mate` does not
  apply to them. ChRO-cap is cap-selected, so the single read's 5' end is the initiation site — the same
  thing proseq2.0 expresses as `-G/--SE_READ=RNA_5prime` ("like GRO-seq") rather than `-P` ("like
  PRO-seq"). Confirm with the initiator logo from `snakemake qc`.
- **Strand orientation.** `--reverse-strand` swaps which read strand becomes the plus track. R1/R2 conventions
  vary across these deposits; validate at known unidirectional promoters before trusting a new dataset.
- **No UMI means no dedup.** PRO-cap legitimately stacks many reads on a single initiation base, so
  deduplicating a non-UMI library destroys signal. The driver skips `umi_tools` and says so. Only
  `Spt5_PROcap_sc` has a UMI (10 nt, 3' adaptor).
- **Booth2016 was NOT TAP-bundled — that was a curation error, now resolved.** `S.cerevisiae_PROcap`
  and `S.pombe_PROcap` were skipped on the belief that TAP+ and TAP− runs shared one GEO sample. The
  archive says otherwise: each sample's two runs sit under a **single SRA experiment**
  (`SRX1490952`/`SRX1490953`), and in SRA one experiment is one library, so two runs means one library
  sequenced twice. TAP+ and TAP− are different libraries and would carry different SRX. All **10** samples
  in `PRJNA306424` follow the same 2-runs-per-SRX pattern — including mRNA-seq, where TAP is meaningless —
  and **no sample in the study is labelled TAP−/minus/control**. No TAP− data was deposited. Both runs are
  technical replicates and merging them is correct; both experiments now run, taking targets from 24 to 26.
  Residual risk, which metadata cannot exclude: TAP+ and TAP− reads pooled *inside* one library without
  demultiplexing. The initiator logo from `snakemake qc` would be diluted if so — check it before
  publishing yeast numbers.
  **The `--allow-bundled-tap` guard stays**, because it is right for a genuinely bundled deposit; it is
  just no longer triggered. Other projects here do have real TAP− controls as separate samples
  (`LacZ_PROcap_dm`, `Kruesi2013_ce_GROcap`), which is why the assumption was reasonable.
- **TAP− rows are controls, not targets.** Where they exist they appear as `raw.protocol_controls` and are
  for background and specificity checks only.
- **Spike-ins.** Ino80/Spt5 (S. pombe), LacZ (mouse MEFs), and Bcell (Drosophila) carry spike-ins. They must
  never contribute to target labels; the driver warns, but splitting them off is not yet implemented.


## Editing a QC script is INVISIBLE to Snakemake unless the script is an input

Snakemake hashes a rule's own code and params — never the contents of a file those
params merely name. So every rule that ran `python {params.script}` was immune to edits of
that script, and `snakemake qc` reported **"Nothing to be done"** after the initiator measure
changed and `rdna_regions` was added. Nothing was stale on disk by mistake; the DAG simply had no way
to know.

All six report scripts are now declared as `input:` as well as `params:` — `orientation_qc`,
`umi_report`, `read_structure_qc`, `rrna_content`, `experiment_stats`, `stats_table`. Do the same for
any new one, and remember it cuts both ways: editing a script now re-runs every rule that uses it.

**That change also had to remove `merged.bam` from `experiment_stats`'s inputs.** It is `temp()` and
normally already deleted, so invalidating the rule — which now happens on every script edit — made
Snakemake rebuild all 38 via `merge_runs`. `peaks` already guarantees align/merge/pints ran, so
merged.bam bought ordering that was guaranteed anyway. Same argument the rule already made for
`Log.final.out`: an input whose only effect is to force expensive recreation is worse than a blank
column.

**And a local dry-run cannot tell you what the cluster will do.** `data/` is empty on a dev machine, so
every job shows as pending and the totals are meaningless for judging what an edit invalidates. Reason
about rerun-triggers from the rule definitions, or run the dry-run where the data is.

**Editing `config/genomes.yaml` is invisible the same way, and worse.** It is read at parse time
(`GENOMES = load_yaml(...)`) and is a declared input of nothing, so a rule only notices a change if the
changed value happens to be interpolated into its own `params`. For rules whose script reads the config
itself that is never true: `rrna_content`'s params are the script path and the FASTQ dir, so adding
`rdna_regions` for C. reinhardtii changes **nothing** Snakemake can see — not under `mtime`, not under
default triggers. Force it: `snakemake --forcerun rrna_content --config experiments=<exp>`.

Declaring `genomes.yaml` as an input of every rule that reads it would fix the class, at the cost of
re-running the whole corpus for any one species' edit. Until that trade is taken, treat a `genomes.yaml`
change as needing an explicit `--forcerun` and work out which rules consume the field you touched.

## Editing the `trim` rule re-runs everything — know this before you do it

`trim` has two PERSISTENT outputs (`fastp.json`, `fastp.html`) alongside its `temp()` FASTQs. So unlike
the deeper rules, whose temp outputs are long deleted, `trim` still has metadata to compare against — and
Snakemake's default rerun-triggers include `params` and `code`. **Any edit to the trim rule therefore
re-runs trim for all 59 runs and cascades through align, peaks and QC**, whether or not the edit affects a
given library. Adding `--adapter_fasta` also gives `trim` a new input file, which forces it the same way.

That is a full uniform re-map, which is defensible — it is what this repo is for — but it is not cheap
(59 STAR alignments including hamster at 2.4 Gb) and it should be a decision, not a surprise. To scope it
down instead, run with `--rerun-triggers mtime` and delete only the affected experiments' outputs.

The reverse is also worth knowing: a change *below* trim usually will NOT be picked up automatically,
because the intermediate chain is `temp()` and already deleted, so the params/code comparison has no
output file to compare and the persistent files downstream look up to date. Force those explicitly.

## One timestamp on a `.fai` re-fetches 74 FASTQs — use the standalone scripts

Observed 2026-09-06 on the real tree: `snakemake orientation -n --rerun-triggers mtime`, asking for
nothing but 42 plots, planned **701 jobs** — 74 `fetch_fastq`, 56 `trim`/`align`/`filter_unique`/
`final_bam`, 10 `star_index`, 80 `bedgraph`/`bigwig`, 40 `pints`/`combine_peaks`. `snakemake stats` does
the same. Nothing was wrong with the tree; the script edit was not the cause.

**The chain is short and worth memorising, because any invalidation near the top of it rebuilds from
FASTQ.** `chrom_sizes`' output is a **declared input of `bigwig`** (`sizes=`). So one `.fai` newer than
one `.chrom.sizes` re-runs `chrom_sizes`, whose output is then newer than every bigWig ->
`bigwig` -> needs `{strand}.bg`, `temp()` and deleted -> `bedgraph` -> needs `merged.bam`, `temp()` and
deleted -> `merge_runs` -> `final_bam` -> `align` -> `trim` -> the FASTQs, deleted after mapping ->
`fetch_fastq`. **The `temp()` design that makes the pipeline cheap to store is exactly what makes a
top-of-chain mtime expensive to satisfy**, and on a tree whose STAR indices have been cleaned up it
rebuilds those too. Read the `Reasons:` block to find the roots: "updated input files" lists the rules
mtime actually triggered, and anything else is downstream of them.

**The escape hatch is that every QC script runs standalone against what is already on disk**, which is
why they take `-e`/`--all` at all. None of these touch the DAG:

    python src/qc/experiment_stats.py --all -o qc/stats/experiment_stats.tsv \
        --markdown qc/stats/experiment_stats.md
    python src/qc/rrna_content.py -e <exp> --fastq-dir data/fastq -o qc/rrna/<exp>.tsv
    xargs -P 8 -I{} python src/qc/orientation_qc.py -e {} --outdir qc/orientation \
        --tsv qc/orientation/{}.tsv < experiments.txt

Three traps in doing it that way. Run `rrna_content` **before** `experiment_stats --all`, which reads
`qc/rrna/` opportunistically, or the adjusted mapping rate is computed from a stale rRNA figure. Loop
`orientation_qc` per experiment rather than using `--all`: `--tsv` is a single path reopened per
experiment, so `--all --tsv` leaves one file describing only the last one. And check
`ls data/annotation/` before parallelising it — without `--annotation` the script fetches its own, and
concurrent jobs on one species race for the same file, which is the whole reason the rule passes the
path.

**To make `snakemake` usable again afterwards, `--touch`** — but confirm first that the `.fai` are merely
newer rather than different, and run it **after** regenerating by hand, since `--touch` marks the stale
QC outputs current along with everything else:

    snakemake orientation --touch --rerun-triggers mtime
    snakemake orientation -n --rerun-triggers mtime      # expect ~nothing

`--touch` asserts the existing outputs are correct. If a `.fai` differs in *content* from what the
bigWigs were built against, that assertion is false and a real rebuild is owed — which is the one case
where paying for the 701 jobs is the right answer.

## `--rerun-triggers mtime` CANNOT see a new input, so it cannot see a new decoy

Measured on a fixture 2026-09-06, after
`snakemake --rerun-triggers mtime --config experiments=...` reported **"Nothing to be done"** for a
re-map that had just gained two organelle decoys.

`mtime` compares the timestamps of a job's **existing** inputs against its outputs. A decoy that has
never been fetched has no timestamp, and Snakemake does not schedule a missing input's rule when the
downstream output already exists -- the same behaviour that makes a deleted `temp()` chain fail to
retrigger. So adding an `organelle_accessions` or `rdna_accession` entry is **invisible** under `mtime`:
no `fetch_decoy`, no `star_index`, no `align`. The `input` trigger, which is ON by default, is the one
that notices a rule's input *set* changed.

| invocation | result on the fixture |
| --- | --- |
| `--rerun-triggers mtime` | **Nothing to be done** |
| default triggers | `fetch_decoy` -> `index` -> `align` |
| `--rerun-triggers mtime --forcerun index` | `fetch_decoy` -> `index` -> `align` |

**So use DEFAULT triggers for a config change and scope it with `--config experiments=`**, which is what
bounds the cascade -- the reason to reach for `mtime` in the first place was avoiding a corpus-wide
re-run, and a subset achieves that without disabling the trigger you need. It is also the more correct
choice for a `genomes.yaml` edit generally, since those values reach `bedgraph` and `pints` as `params`,
which `mtime` equally cannot see. Keep `mtime` for an edited SCRIPT, where the trigger is an existing
input with a fresh timestamp.

## Data provenance and the re-mapping transition

`processed:` paths in `config/experiment_config.yaml` now point at ENCODE-pipeline output under
`data/procap/`, which **does not exist until the pipeline has run**. `launch.py` and `make_negatives.py`
already skip experiments with missing inputs, so until then every experiment reports `SKIP: missing data` —
including the three that previously trained. That is the intended state of an in-progress uniform re-map, not
a regression.

The three original experiments keep their old paths under `legacy_processed:` for provenance and
before/after comparison. Point `processed:` back at them (in the generator's `LEGACY` dict) to reproduce the
existing checkpoints.

**The six legacy `src/data_preprocessing/*.sh` download scripts are deleted** — `download_genomes.sh`,
`download_{drosophila_s2_procap,scer,spombe}.sh`, `download_S2_PROcap_dm3.sh` and `utils.sh`. Everything they
did is now in `workflow/Snakefile` (`fetch_genome`/`faidx`/`chrom_sizes`/`star_index` cover all twelve species;
the old script covered three) or was part of the deleted dm3 regime. `utils.sh` was sourced by nothing and
hard-coded `/programs/...` paths. Read them at `a806e0d` if you need the exact legacy provenance; do not
restore them.

Why the re-map was necessary: the *S. cerevisiae* and *S. pombe* labels were GEO-provided **normalized**
bigWigs (`R1Normed`), only re-wrapped by hand (chr stripped, mito removed), while Drosophila labels came
from raw reads via `proseq2.0`. BPNet's count head consumes
`log1p(total signal)` with `count_loss_weight`, and the training outlier filter is a quantile of total
signal — both are meaningless when one species' labels are arbitrarily scaled and another's are read counts.

## Upstream reference: procap-atlas

[kundajelab/procap-atlas](https://github.com/kundajelab/procap-atlas) is the mature sibling of this repo (ENCODE PRO-cap atlas, human K562).
It is the source of truth for **io, sampling, and training standards**; this repo tracks it. It lives at
`~/github/procap-atlas` locally.

**Copy its standards, not its assumptions.** procap-atlas is human-only and hardcodes `data/hg38.fa`,
`data/hg38.blacklist.bed.gz`, `data/GRCh38-cCREs.bed.gz`, and a 7-fold `% 7` split. This repo is
multi-species: per-species FASTA from `experiment_config.yaml`, species-keyed `chrom_splits.yaml`, and
peak-level random folds for S. pombe. **Never** replace that logic with a wholesale copy of theirs.

**Training hyperparameters were NOT synced until 2026-09-03, and both families diverged.** Asked
directly ("does early stopping match upstream?") and the answer was no, on more than early stopping.
Upstream's defaults live in each fit script's `params` dict; ours in `config/{family}_params.json`. What
differed, now aligned:

| | upstream | was here | note |
| --- | --- | --- | --- |
| bpnet `max_epochs` | 50 | 100 | |
| bpnet `early_stopping` | **None** | 20 | |
| cherimoya `max_epochs` | 50 | 100 | |
| cherimoya `early_stopping` | **None** | 15 | |
| cherimoya `max_jitter` | 500 | **50** | a 10x augmentation difference |
| cherimoya `muon_wd` | 0.03 | 0.01 | |
| cherimoya `adam_lr` | 0.001 | 0.004 | |
| cherimoya `adam_wd` | 0.0 | **0.2** | |
| ~~cherimoya `negatives_ratio`~~ | ~~1/7~~ | 1/4 | **reverted 2026-09-04** — procap-atlas overrides it, but 1/4 is `cherimoya.io.PeakGenerator`'s own default and is what this repo follows |
| cherimoya `warmup_epochs` | 5, `--warmup-epochs` | hard-coded 5 | now configurable |
| cherimoya `decay_epochs` | None, `--decay-epochs` | absent | now present |

Everything else already matched: bpnet's `max_jitter` 200, `n_filters` 512, `n_layers` 8,
`count_loss_weight` 100, `learning_rate` 0.0005, `batch_size` 64, `negatives_ratio` 1/7, `n_shuffles` 20;
cherimoya's `n_filters` 128, `n_layers` 9, `batch_size` 64, `muon_lr` 0.025 and all three `lw_*`.

**`early_stopping: null` is a decision with evidence behind it, so do not "restore" a value.** Upstream
swept it — `performance_metrics/cherimoya/{20_5_2,100_None_5,50_None_5,50_None_5_15decay}` — and settled
on `50_None_5` (50 epochs, no early stopping, 5 warmup). Re-enabling it at 5, against both
`decay_epochs=None` and `decay_epochs=15`, **underperformed on every benchmark metric, profile and count
alike** — not a profile/count tradeoff. The proposed mechanism is architecture-independent and applies to
bpnet-lite identically: `fit()` checkpoints whenever `valid_count_corr > best_corr`, a bare validation
count-correlation comparison, so more epochs give that rule more chances to overfit the validation set —
and stopping on the *same* metric compounds it.

**Upstream states the caveat itself and it should travel with the number**: none of those comparisons
control for random initialization. `--random-state` only makes negative sampling and data-loader order
reproducible, and no `torch.manual_seed` is set anywhere in either script, so run-to-run noise is not
separated from the hyperparameter effect. Treat it as upstream's considered default, weakly evidenced —
not a settled result. A seed-controlled repeat would be needed to do better.

Aligning cost nothing here: no model in this repo has trained yet, so there were no checkpoints to stay
comparable with. Had there been, this would have been a re-train.

Already synced:

- `src/bpnet/fit/data_loader.py` — byte-identical to theirs; do not fork it.
- `src/cherimoya/fit/data_loader.py` — now a ~46-line thin wrapper over `cherimoya.io.PeakGenerator`,
  ported from theirs. It was a ~440-line frozen fork of the *pre-refactor* PeakGenerator, incompatible with
  the pinned cherimoya. The only thing layered on top is `torch.abs()` on returned signal, because upstream
  cherimoya no longer un-negates minus-strand values in `__getitem__`.
- `src/bpnet/fit/fit_bpnet.py` — builds its training loader via `data_loader.PeakGenerator` instead of
  reimplementing `extract_loci` + outlier filter + `PeakNegativeSampler` inline (~55 lines removed). Also
  `from bpnetlite.bpnet import BPNet`, optional `blacklist`/`exclusion_lists`, `dtype=torch.float`, and
  `alpha` renamed to `count_loss_weight` (`--alpha` kept as an alias).
- `src/cherimoya/fit/fit_cherimoya.py` — ported to the cherimoya >= 0.2 API (see below).
- `src/bpnet/attribute/attribute.py` — **the DeepLIFT reference**, synced 2026-09-04. It was using
  tangermeme's default dinucleotide shuffling (`n_shuffles=20`) where upstream defaults to a
  **nucleotide-frequency reference**: one soft PFM per input sequence carrying that sequence's own
  A/C/G/T frequencies at every position. `--reference-mode {frequency,dinucleotide}` selects, default
  `frequency`, and `--n-shuffles` now has a CLI override (it was JSON-only).
  **The reason is that a dinucleotide shuffle is not reliably NEUTRAL.** Upstream's locus diagnostics
  found shuffles that produce cryptic promoter-like signal — for some loci as active as, or more active
  than, the genomic input — which makes the baseline reference-sensitive, the one thing a DeepLIFT
  reference must not be. **That argument is stronger here than upstream**, because several of these
  genomes are far denser than human: S. cerevisiae carries 1.2-4.1 peaks per 2114 bp window, so nearly
  every window contains a promoter and a composition-preserving shuffle is correspondingly more likely
  to reassemble something initiation-competent. Same reasoning that moved the initiator PWM to relative
  entropy against *local* composition.
  Two implementation details that matter: the reference is passed as a **callable**, so it is built per
  batch and never reaches tangermeme's tensor-reference one-hot validator, which would reject a soft
  tensor; and frequency mode forces **`n_shuffles=1`**, since that reference is deterministic and
  further copies are byte-identical (verified). Verified numerically: shape `(N, n, 4, L)`, sums to 1
  at every position, per-sequence composition matches the input exactly, positionally flat, genuinely
  soft, and `n=0` rejected.
  **The default output path now carries the mode** (`attributions/bpnet/{exp}_attr_{type}_{mode}.npz`), a deliberate
  divergence from upstream's mode-less name: the two references give different numbers, and without it
  a frequency run silently overwrites a dinucleotide one with nothing on disk recording which is which.

Also not copied from their attribution tree: `--head orientation`, which attributes the profile
orientation index `max(sum(plus), sum(minus)) / (sum(plus) + sum(minus))` through a DeepLIFT-compatible
ReLU form of the binary maximum. Same metric as the `orientation_index_pearson` our benchmarks do not
report, so those two are one piece of work. (`attribute/launch.py` was the other gap here and is now
closed.)

**Cherimoya attribution is BLOCKED, not merely absent — do not start it.** There is no
`src/cherimoya/attribute/` and it should stay that way for now: it needs DeepLIFT **rescale rules that
are still in development and are not in tangermeme yet**. This is not a wrapper-writing exercise, and
the missing piece is upstream of this repo entirely.

What the code shows, for whoever picks it up when the rules land. The ordinary nonlinearities are
already covered — tangermeme 1.4.1 ships rules for `GELU` and `Softmax`, and those are the only
activations `cherimoya/cheri.py` and `cherimoya.py` use — so the gap is at the WRAPPER level, exactly
where bpnet-lite needs `{_ProfileLogitScaling: _nonlinear}`. `cherimoya/wrappers.py` defines
`ControlWrapper`, `_ProfileLogitScaling`, `ProfileWrapper`, `LogCountWrapper` and
**`ExpectedCountsWrapper`**. The first four mirror bpnet-lite's, but note cherimoya's
`_ProfileLogitScaling` is its OWN class, so a rule keyed on bpnetlite's would not match it.
`ExpectedCountsWrapper` is the one with no counterpart: it composes `torch.expm1` with a per-group
`softmax` over `cat`/`split` tensors, and `expm1` is absent from tangermeme's rule table. Treat that as
the visible candidate rather than the confirmed blocker — the authority here is that the rules are in
development, not this inspection.

Deliberately not copied: `--background NAME:RATIO` multi-source negatives (its `ccre` source is
GRCh38-only), `--min-reads` (needs their `config/n_reads.txt`), and their hitcall/modisco/predict/
motifcompendium/model_upload trees (out of scope here).

## Cherimoya API compatibility

`cherimoya >= 0.2` broke the API this repo was written against (the historical `69f16dc` commit). Both
breaks are fixed, ported from procap-atlas:

- `Cherimoya.__init__` has **no `n_outputs`**; use `signal_groups=[len(signals)]` — one 2-element group is
  one stranded (pl, mn) pair.
- `fit()` requires **`lw_optimizer` and `lw_scheduler`** (no defaults) — a third optimizer for the `lw0`/`lw1`
  Kendall uncertainty loss weights, stepped separately in the training loop. Built as
  `SGD(lw_params, lr=lw_lr, weight_decay=lw_wd, momentum=lw_momentum)` with linear warmup then a flat
  `ConstantLR` (no cosine decay). Defaults `lw_lr=0.001, lw_wd=0.0, lw_momentum=0.9` live in
  `config/cherimoya_params.json`.
- The Muon/AdamW split must also exclude `conv_weight` (2-D depth-wise conv belongs on AdamW), and `lw0`/`lw1`
  go to neither.
- `PeakGenerator(signals=[params["signals"]])` — **nested**. A flat 2-element list now means two independent
  unstranded groups, which breaks reverse-complement channel swapping. `params["signals"]` itself stays flat
  for `extract_loci` and for `signal_groups`.
- `Cherimoya.load(path, device=...)` is unchanged and still compatible — but **it defaults to
  `compile=True`**, so `benchmark_cherimoya.py` was compiling unconditionally until 2026-09-04. Both
  scripts now take an explicit flag, with **deliberately opposite defaults**, because the warmup
  economics differ: `benchmark_cherimoya.py --compile` is **opt-in** (one inference pass over the test
  set does not amortise compilation), while `fit_cherimoya.py --no-compile` is **opt-out** (50 epochs
  do). Both stay gated on `sys.version_info < (3, 14) or torch.__version__ >= "2.10"` — `torch.compile`
  raises unconditionally on Python 3.14+ below torch 2.10, and the limit is Dynamo, not Triton. The
  benchmark warns when `--compile` is asked for and cannot be honoured, rather than silently ignoring
  it. Not applicable to this repo's own lock (torch 2.13 on Python 3.11); it matters where a site
  interpreter differs.

Note `load()` reconstructs via `cls(**payload['config'])`, so a checkpoint saved by a pre-0.2 cherimoya
whose stored config contains `n_outputs` will fail to load under the pinned version.

## The 2026-09-01 cotton addition: GRO-cap in diploid and tetraploid cotton

Wen et al. 2026, *Genome Biology* 27(1):12, doi
[10.1186/s13059-025-03907-w](https://doi.org/10.1186/s13059-025-03907-w) (PMID 41484656) — 4 GRO-cap
libraries, 2 species x 2 biological replicates, taking the repo to **42 experiments and 12 species**.

| experiment | runs | species | pairs |
| --- | --- | --- | --- |
| `G.arboreum-ovule_GROcap` | `DRR991139`, `DRR991140` | *G. arboreum* cv. Shixiya1, **diploid AA** | 498.6 M |
| `G.hirsutum-ovule_GROcap` | `DRR991137`, `DRR991138` | *G. hirsutum* cv. Xuzhou 142, **tetraploid AD** | 577.0 M |

150 bp paired-end, `smallRNA_RA3` (the paper trims with TrimGalore `--small_rna`), no UMI.
**82.5 GiB**, which is a ~70% increase on the existing corpus, and `DRR991138` at 375.6 M pairs is the
deepest library here by a wide margin.

**The data is in ENA, but only as a mirror.** It was deposited at CNCB as `PRJCA024429` and reaches ENA
through DDBJ as `PRJDB41007` with `DRR` run accessions. Consequences worth knowing:

- **`resolve_runs.py` needed a fix to match it.** DDBJ reports `sample_title` as
  `AA_GROcap_Rep1 (SAMC4531003)` — the library name with the *source archive's* accession appended — and
  ENA's own `sample_accession` is the DDBJ one (`SAMD01888145`), not the CNCB one. So neither the
  accession path nor the whole-string name match worked, and all 4 rows came back `UNRESOLVED`.
  `index_runs()` now also indexes the title with a trailing parenthetical stripped. It only ever *adds*
  keys, so any ambiguity it creates is caught by the existing bijectivity check; verified that the other
  60 resolved rows are byte-identical.
- `library_strategy` is mislabelled **`miRNA-Seq`** in the archive. Cosmetic, but do not filter on it.
- The paper deduplicates with Picard despite there being no UMI. **This pipeline deliberately does not** —
  see the no-UMI-no-dedup rule; deduplicating a non-UMI run-on library destroys real stacked 5' ends.
  That is a knowing divergence from the paper's numbers, not an oversight.

**These are the repo's first NCBI-sourced genomes.** Neither species is in Ensembl Plants, which carries
only *G. raimondii*. FASTA and annotation therefore both come from the same NCBI RefSeq release, which is
what keeps their sequence names consistent — the requirement `chrom_style` exists to record. The names are
**RefSeq accessions** (`NC_069070.1`, `NC_053424.1`) rather than the community `Chr01` / `A01`–`D13`
labels, so `main_chromosomes` is accessions too; `extract_loci` matches literally, and a readable-but-wrong
name yields zero loci in silence. The mapping is in each assembly's `*_assembly_report.txt`.

| species | assembly | accession | size | chromosomes |
| --- | --- | --- | --- | --- |
| G. arboreum | ASM2569848v2 | `GCF_025698485.1` | 1.62 Gb | 13 (`NC_069070.1`–`NC_069082.1`) |
| G. hirsutum | Gossypium_hirsutum_v2.1 | `GCF_007990345.1` | 2.31 Gb | 26 (`NC_053424.1`–`NC_053449.1`) |

Both include MT and Pltd, so **no organelle decoys are needed** — unlike C. reinhardtii and P. patens.
They are recorded in `organelle_contigs` so `rrna_content.py` can measure organellar content, and are
outside `main_chromosomes` so they never reach a fold. Both are 14 `star_sa_index_nbases`, hence
`LARGE_GENOME`, hence the 40 GB align/index reservation automatically.

**`chrom_style: ncbi_refseq` is a third naming style, and the PINTS prefix is now an explicit map.** It was
`"chr" if chrom_style == "ucsc" else ""` in both drivers; a new style would have fallen through to `""`
without anyone choosing that. `PINTS_CHROM_PREFIX` maps each style and **raises** on an unknown one. `""`
is correct for RefSeq names because every name starts with the empty string and the bigWigs PINTS reads
are already restricted to `main_chromosomes` by `bedgraph` — but defaulting to PINTS' own `"chr"` would
match nothing, which is the silent-zero-peaks bug that map exists to prevent.

**Tetraploid AD1 is the caveat to watch, and it is measurable rather than arguable.** *G. hirsutum* has 13
A-subgenome and 13 D-subgenome chromosomes, homoeologous at ~96%+, so the ENCODE MAPQ 255 unique filter
discards any read that cannot be assigned to one subgenome. That bites hardest on the **short GRO-cap
inserts**: a 30 bp window may contain no distinguishing difference at all, even where the full 150 bp read
would. The paper reports 124 M unique of 577 M pairs using relaxed STAR settings
(`--outFilterMatchNminOverLread 0.33`); ours are stricter, so expect a low `pct_unique` for *structural*
reasons rather than library-quality ones. Read the measured value before concluding anything — and note
`pct_unique_adj` corrects for rRNA, not for homoeology, so it will not rescue this one.

**Now measured, and the homoeology cost is much smaller than that framing implies — about 4.5 points.**
Both cottons map low, and the *diploid* is barely better, so most of the shortfall is not subgenome
ambiguity at all. Numbers below are post-correction, i.e. trimmed with the right adapter:

| experiment | archive | unique | `pct_unique` | `pct_rrna` | `pct_unique_adj` | peaks |
| --- | --- | --- | --- | --- | --- | --- |
| `G.arboreum-ovule_GROcap` (AA) | 498.6 M | 142.9 M | 30.6% | 28.7% | **42.9%** | 95,714 |
| `G.hirsutum-ovule_GROcap` (AD) | 577.0 M | 143.5 M | 29.8% | 22.4% | **38.4%** | 171,640 |

So: rRNA accounts for ~22-29 points, homoeology for the **4.5-point AA-to-AD gap**, and a common ~50-point
residual for the rest — the same unexplained plant residual A. thaliana, P. patens and C. reinhardtii all
show, not a cotton-specific problem. **We also beat the paper's own number** (143.5 M vs 124 M unique on
the identical run set) *despite* the stricter match fraction, for the adapter reason below.

Both are heavily unidirectional — 94% and 96% of calls — which is in line with the other plants and is
why `G.hirsutum` has the largest peak set in the corpus. Its initiator logo is correspondingly the
weakest (0.11 bits, correctly placed at +0, with the dilution note), so treat that peak set as
low-confidence-heavy rather than as a 5' problem.

**The adapter in the manifest was wrong, and both cottons carried a `FAIL:adapter_mismatch` when the
check for it existed.** It was `smallRNA_RA3`, transcribed from the paper's TrimGalore `--small_rna`;
`read_structure_qc.py` finds **`truseq_universal` in 97.2-97.7% of R1 across all four runs**, and the two
candidates share no 12-mer prefix, so the detection is unambiguous. The manifest is corrected to
`truseq_universal` — the reads override the paper, the same precedent as `Liver_ChROcap_mm`'s layout and
the Spt5 `umi_loc`.

**It cost no LABELS, and the reason is worth knowing: fastp's paired-end overlap analysis is
sequence-independent.** These are 150 bp mates over a 34-50 bp insert, so the pair fully overlaps and
fastp trimmed *most* of it correctly without ever matching the configured sequence. Re-trimmed with
`truseq_universal`, against the same runs:

| | `input_reads` | `unique_reads` | `signal_reads` | `peaks` | `pct_unique` |
| --- | --- | --- | --- | --- | --- |
| AA, wrong adapter | 496.1 M | 142,945,904 | 141.1 M | 95,793 | 28.8% |
| AA, corrected | 467.7 M (**-5.7%**) | 142,869,387 | 141.0 M | 95,714 | **30.6%** |
| AD, wrong adapter | 573.8 M | 143,593,080 | 140.1 M | 171,659 | 25.0% |
| AD, corrected | 481.8 M (**-16.0%**) | 143,499,436 | 140.1 M | 171,640 | **29.8%** |

**It is a pure denominator correction** — unique, signal and peaks all move by under 0.1%, while
`input_reads` drops 28.5 M and 92.0 M. Those were adapter-only reads that overlap analysis could not
rescue and STAR was discarding anyway; naming the adapter drops them at `--length_required 18` instead.
Same shape as the truncated-TruSeq case: **an honest mapping rate, not recovered depth.** Note the two
libraries carried very different amounts of that junk (16.0% vs 5.7%), which is why the *apparent*
homoeology gap shrank from 8.2 points to 4.5 once both were trimmed correctly — comparing mapping rates
across libraries trimmed with different adapters compares the adapters.

It is also the likeliest explanation for beating the paper: their single-adapter TrimGalore `--small_rna`
pass had no overlap fallback. **Do not generalise the reprieve** — the same error on a single-end library
is the 45%-of-the-library failure the truncated-TruSeq work already documented.

### Do NOT adopt the paper's looser STAR settings for the tetraploid

Asked and answered 2026-09-02. Wen et al. run `--outFilterMatchNminOverLread 0.33` against ENCODE's
default 0.66, and the tempting inference is that a tetraploid needs the slack. Two independent reasons it
is the wrong lever, one mechanical and one measured.

**Mechanically, that flag cannot touch homoeology.** It thresholds *matched bases / read length* and its
failures are booked `unmapped: too short`. Homoeologous reads are not too short — they align fine, to two
places, and are discarded as **multimappers** by `--outFilterMultimapNmax 10` and the MAPQ 255 filter.
The only flags that would change that outcome are the multimapper ones, i.e. the ENCODE unique filter
itself, which is not relaxable without breaking comparability with every other experiment here. So 0.33
buys nothing for the problem it looks like it addresses.

**What 0.33 actually compensates for is an untrimmed adapter — which is a problem we fixed instead.**
Their `--small_rna` TrimGalore pass targeted the small-RNA RA3 adapter on a `truseq_universal` library
(see above), so their reads reached STAR still carrying adapter and their aligned fraction was low. Note
0.33 was probably not even enough: for a 150+150 pair over a ~35 bp insert the matched fraction is around
0.23. We trimmed correctly and got **143.5 M unique against their 124 M on the identical runs, with the
stricter filter**. Loosening ours now would import their workaround for a problem we no longer have, and
on a 65-70% repetitive genome a 0.33 threshold invites partial and spurious alignments.

**And measured: homoeology is not depleting labels.** The test is a diploid control — *G. arboreum* is
AA, so AD1's own A subgenome can be compared against it directly. Lengths from each assembly report,
peaks from our own PINTS calls:

| | length | peaks | peaks/Mb |
| --- | --- | --- | --- |
| *G. arboreum*, diploid AA | 1,593.8 Mb | 95,714 | **60.1** |
| *G. hirsutum*, A subgenome | 1,445.3 Mb | 89,661 | **62.0** |
| *G. hirsutum*, D subgenome | 836.3 Mb | 81,979 | 98.0 |

**AD1's A subgenome is 62.0 peaks/Mb against the diploid's 60.1 — a 3% difference, in the wrong direction
for a depletion story.** If the unique filter were eating homoeologous promoters, the tetraploid's A
subgenome would be systematically thinner than the diploid's; it is not. Per-pair A/D ratios are tight
(0.86-1.39, median 1.10) with no pair collapsed, which is the same conclusion chromosome by chromosome.
D's higher density is ordinary biology, not an artifact: it carries a comparable gene set in 58% of the
sequence, and 98.0/62.0 = 1.58 against a 1.73 size ratio, so the two subgenomes hold near-equal peak
counts per homoeolog.

**The full STAR accounting, measured 2026-09-02 — and it does not point at the match fraction either.**
The prediction above was that `too short` would be small after the adapter fix. **It is not**, for the
tetraploid: 27%. The accounting closes to 100%, so nothing here is a rounding artifact.

| run | unique | 2-10 loci | >10 loci | too short | other |
| --- | --- | --- | --- | --- | --- |
| DRR991137 AD | 29.76% | 17.75% | 7.12% | **27.55%** | **17.81%** |
| DRR991138 AD | 29.80% | 17.80% | 7.08% | **27.01%** | **18.31%** |
| DRR991139 AA | 27.38% | 10.54% | **46.38%** | 11.56% | 4.15% |

Four things this settles:

- **The homoeology signature is real, visible and modest.** The 2-10 loci bucket is 17.8% in AD against
  10.5% in AA — about +7 points, exactly what a duplicated genome should do, since a homoeologous pair
  gives two alignments. It agrees with the 4.5-point `pct_unique` gap and with the peaks/Mb result above.
- **The diploid's dominant loss is the opposite one**: 46.4% of `DRR991139` goes to `>10 loci`, against
  7.1% for AD. Repeat and rDNA content, not ploidy. Whatever limits cotton, it is not primarily subgenome
  ambiguity.
- **`too short` tracks INSERT LENGTH, not ploidy** — AD's median insert is 34-35 bp against AA's 46-50 —
  which is the final argument against `0.33`. On a 35 bp insert that threshold accepts a ~12 bp match, and
  a 12-mer occurs by chance roughly 137 times in a 2.3 Gb genome. ENCODE's 0.66, about 23 bp, is already
  near the floor of what can be placed uniquely at all.
- **`unmapped: other` is 18% of the tetraploid and is NOT any ENCODE filter.** `too many mismatches` is
  **0.00% in all four runs**, which refutes the hypothesis that `--outFilterMismatchNmax 1` was being
  double-charged by cotton's fully overlapping mates — a plausible story, since cotton is the only large
  fully-overlapping PE library here, and simply wrong. `other` is also not the match fraction (that is
  `too short`) and not multimapping (separate counters). So **no parameter ENCODE specifies accounts for
  it**, and loosening any of them cannot recover it.

The remaining hypothesis, explicitly untested: STAR's `--winAnchorMultimapNmax` (default 50) discards a
read whose seeds anchor in more than 50 windows and books it as `other` rather than letting it reach the
multimapping counters. That fits shorter inserts in a large repetitive genome, and fits AD at 18% against
AA at 4%. **Treat it as a guess** — the previous guess in this section was refuted by one grep.

Whether it is worth chasing is a separate question and the likely answer is no: those reads are repetitive
by construction, so raising the limit should mostly convert `other` into `>10 loci`, which the MAPQ 255
filter discards anyway. The test is cheap and has the same shape as the adapter isolation test — run STAR
twice on a 2 M-read subsample of `DRR991137`, default against `--winAnchorMultimapNmax 200`, and compare
**`Uniquely mapped reads number`**. Only a rise there is a gain; `other` becoming `>10 loci` is not.

Re-derive the table with:

    grep -HE "Uniquely mapped reads %|multiple loci|too many loci|too short|other|mismatches" \
        data/procap_work/runs/DRR9911{37,38,39,40}/Log.final.out

Related: cotton is 77-86% poly-G in **every** file, the highest in the corpus by a wide margin (GCB is
65%), which is what a 34-50 bp insert in a 150 bp two-colour read looks like. `--trim_poly_g` is doing
real work here.

**Both are assigned now, chromosome-level, from peak counts (2026-09-02).** Both entries **originate
here** — there was no prior cotton assignment — so carry them across before using
cotton beyond this repo, or a locus in test here becomes train there. Reproduce with
`python config/write_split_csvs.py --peak-counts -e G.{arboreum,hirsutum}-ovule_GROcap`.

*G. arboreum* is unremarkable: 13 diploid chromosomes, 95,714 peaks, exhaustively optimal at spread
**2,694 (14.1% of target)**, in family with C. griseus (17.5%) and C. elegans (12%).

**`G. hirsutum` pairs its homoeologs into one fold, and that is the whole story of its balance.** A0i and
D0i are ~96%+ identical, so splitting a pair across train/test puts recognisably the same sequence on both
sides — about 85 differences per 2114 bp window — and would inflate exactly the tetraploid side of the
diploid-vs-tetraploid comparison these two libraries exist to support. Same rule as the C. griseus
chromosome-1 arms: judge the sequence by what it is, not by the label the assembly gives it.

**The pairing was read from the assembly report, not inferred from the accession order.** For
`GCF_007990345.1`, `NC_053424.1`-`NC_053436.1` are A01-A13 and `NC_053437.1`-`NC_053449.1` are D01-D13, so
the homoeolog of index *i* is *i*+13. Worth checking rather than assuming, since the constraint is
worthless if the pairing is wrong.

The cost is real and is stated rather than hidden: **spread 6,509 (19.0% of target), the worst in
`chrom_splits.yaml`**, against 950 (2.8%) if the 26 chromosomes were free. Both are optimal for their
constraint — exhaustive over every partition of the 13 pair-units into 5 non-empty blocks, and extensive
local search respectively. Most of the excess is structural rather than a bad assignment: `A05+D05`
(18,113) and `A11+D11` (16,700) are half a fold each against a 34,328 target. **Do not "rebalance" this by
separating a pair.**

One measured aside that did *not* change the decision: at **4** folds the same paired constraint gives
spread 3,684 (8.6%), because a larger target absorbs those two pairs. Five folds was kept for convention;
the 4-fold option is there if 19% ever proves intolerable.

**`--peak-counts` used to refuse a species with no `chrom_splits.yaml` entry** — exiting 1 with
`Error: <species> not in chrom_splits.yaml`, i.e. refusing precisely when it is most useful, which is why
the C. griseus counts were gathered by hand. It now falls back to a per-CHROMOSOME report over
`main_chromosomes` (the same allow-list `bedgraph` and PINTS applied, so a name outside it could not be a
fold member anyway), and `--by-chrom` forces that view for an assigned species — e.g. to retune from a
deeper library. Anything outside the allow-list still prints under "NOT in any fold", which is the
chrom-naming-mismatch tell.

## Assemblies and exclusion lists

**Remap to the ENCODE/modENCODE standard assembly wherever one exists, and take chromosome naming from it.**
Per ENCODE's reference-sequences page: mouse **mm10**, *D. melanogaster* **dm6**, *C. elegans* **ce11**
(human GRCh38). All are UCSC-derived, so those three use **chr-prefixed** names. Yeast and Arabidopsis are
not ENCODE/modENCODE organisms and keep the Ensembl bare names their FASTAs use.

`extract_loci` compares chromosome names literally, so FASTA, peaks, bigWigs, `chrom_splits.yaml` and the
exclusion list must all agree. `config/genomes.yaml` records `chrom_style` per species; a mismatch fails
silently (zero loci, or an exclusion list that excludes nothing).

Two consequences already applied:

- **Mouse is mm10, not mm39** — this overrides the manifest's `recommended_realign_assembly` column. mm10
  also matches the lab's other mouse work, is what the peak-matched folds were tuned on, and is the
  only mouse assembly with a published exclusion list. Chromosome names are identical between the two
  assemblies, so the fold assignment transferred unchanged.
- **dm6 names are chr-prefixed here** (`chr2L`), where the earlier lab file writes `2L`.
  The fold *assignment* is identical; only naming differs, so models stay comparable. This resolves a
  long-standing unverified flag and is what makes the dm6 exclusion list usable at all.

| species | assembly | exclusion list | regions |
| --- | --- | --- | --- |
| M. musculus | mm10 | Boyle-Lab v2 (also ENCODE `ENCSR064IDX`) | 3,435 |
| D. melanogaster | dm6 | Boyle-Lab v2 | 182 |
| C. elegans | ce11 | Boyle-Lab v2 | 97 |
| A. thaliana | TAIR10 | Klasfeld 20-inputs, **Boyle-Lab software**, versioned in-repo | 83 |
| S. cerevisiae | R64-1-1 | none published | — |
| S. pombe | ASM294v2 | none published | — |
| C. reinhardtii | ASM4749649v1 (was Chlamydomonas_reinhardtii_v5.5) | none published | — |
| P. patens | Physcomitrium_patens_V7 (was Phypa_V3) | none published | — |
| S. moellendorffii | v1.0 | none published | — |
| C. griseus | CriGri-PICRH-1.0 | none published | — |

Boyle-Lab lists arrive **gzipped**, matching their `.bed.gz` names — `file` reports
`gzip compressed data, was "ce11-blacklist.v2.bed"` despite the URL being raw.githubusercontent, so
nothing gunzips them. An earlier version of this line called them plain BEDs; it was wrong. Naming
verified chr-prefixed for all three (`chr2L`, `chrI`, `chr1`).

**They were in NEITHER driver's DAG until 2026-09-03, and that silently cost 114 of 214 training jobs.**
Only `run_procap_pipeline.py --fetch-genomes` fetched them, so a run driven by `workflow/Snakefile` — the
preferred path — left all three absent, and `launch.py` then skipped every mouse, fly and worm experiment
with `missing data — blacklist[0]: ...`. Three files under 60 KB gating 65 + 25 + 24 jobs. Same class of
asymmetry as `rdna_accession`, which this driver used to ignore while the serial one read it.
`rule fetch_blacklist` now covers it, and it is in `all`, in `fetch_only` and in a standalone
`blacklists` target (`snakemake blacklists -c1`, the cheapest way to unblock an existing tree).

**An exclusion list is a TRAINING input and is in the DAG anyway** — worth being precise about, because
GC-matched negatives are also a training input and are deliberately *out*. The boundary that keeps the
Snakefile runnable from the mamba env with no venv and no GPU stack is a **dependency** one, not a
labels-versus-training one: fetching a list needs wget, where negatives need PyPI-only `tangermeme`. So
this costs the boundary nothing. `BLACKLIST_FILES` is built only from species with a `blacklist_url`,
which excludes A. thaliana's in-repo list and the eight species with none, and a `blacklist_url` pointing
anywhere but `data/` **raises at DAG construction** rather than downloading into a path nothing reads
(verified by pointing C. elegans at `elsewhere/`).

Arabidopsis is the exception: excluderanges ships **only as R `.rds`**, so it is converted to BED and
versioned at `config/blacklists/TAIR10.Klasfeld.Excludable.bed.gz` rather than fetched — no R dependency at
runtime. The set chosen is the one generated by the *same Boyle-Lab/Blacklist software* as the other three
species (over 20 Arabidopsis inputs), **not** the peakPass set: peakPass predicts regions with an ML
classifier whose input features include gene annotation, which is a circularity hazard for a model of
transcription initiation. Source is Zenodo [10.5281/zenodo.21480958](https://doi.org/10.5281/zenodo.21480958), not BEDbase:
`bedbase.org/api/*` serves the SPA catch-all (returns HTML with HTTP 200, so status codes alone are
misleading), and the real API at `api.bedbase.org/v1/` returned nothing for these digests. **The published
set is `chr1`-`chr5` and was stripped to bare `1`-`5`** to match the Ensembl Plants FASTA — unstripped it
would have excluded nothing. See `config/blacklists/README.md` for the regeneration recipe (base R only;
`GenomicRanges` is not needed, the GRanges slots deserialise directly).

The lab convention was exclusion lists for human and mouse only; this repo extends that to fly, worm and
Arabidopsis because published lists exist.

There is **no mm39 exclusion list** in Boyle-Lab or ENCODE (excluderanges has one, `mm39.excluderanges`,
3,147 regions) — another reason mm10 is the better target.

**mm10 vs mm39, re-checked 2026-08-30.** GRCm39/mm39 is the newer assembly, but both original reasons
for mm10 still hold, so **do not migrate**:

- ENCODE has not moved. Its portal serves **97,794 mm10 files vs 63 GRCm39 files** — mm10 is still the
  mouse assembly in practice, not just by legacy.
- Boyle-Lab still ships no mm39 list. The `lists/` directory is exactly ce10, ce11, dm3, dm6, hg19,
  hg38, mm10 — unchanged.

Add the date and the two counts if you re-check, so the next person can see whether the situation moved
rather than re-deriving it.

## Alignment analysis set (no_alt) — audited, no action needed

ENCODE aligns to no_alt references: a locus present twice (primary + alt) turns its reads into
multi-mappers, which `--outFilterMultimapNmax 10` then MAPQ 255 discard, leaving a hole exactly where
the duplication is. **All twelve references here were checked against their real contig lists and all
twelve are already alt-free.** Do not add a contig-filtering step; there is nothing to filter.

| reference | contigs | alt |
| --- | --- | --- |
| dm6 | 1,870 = 8 primary + 645 `*_random` + 1,217 `chrUn_*` | 0 |
| mm10 | 66 = 22 primary + 22 `*_random` + 22 `chrUn_*` | 0 |
| ce11 | 7 = 6 chromosomes + chrM | 0 |
| R64-1-1 | 17 = 16 chromosomes + Mito | 0 |
| TAIR10 | 7 = 5 chromosomes + Mt + Pt | 0 |
| ASM294v2 | 6 = I, II, III, MT, MTR, AB325691 | 0 |
| ASM4749649v1 (C. reinhardtii) | **17 = 17 chromosomes, nothing else** — no scaffolds, **NO organelles** | 0 |
| Physcomitrium_patens_V7 | **26 = 26 chromosomes, nothing else** — no scaffolds, **NO organelles** | 0 |
| v1.0 (S. moellendorffii) | 759 = 0 chromosomes + Pt + 757 scaffolds | 0 |
| CriGri-PICRH-1.0 | 647 = 10 chromosomes + 637 unplaced | 0 |

**The two swapped plant references were re-audited 2026-09-06** and are trivially alt-free, from their
assembly reports rather than by contig-list diff: ASM4749649v1 is 17 assembled molecules and *nothing
else*, and V7 is 26 chromosomes / 26 component sequences (via its NCBI mirror `GCA_059467195.1`). Both
therefore also lost the unplaced scaffolds their predecessors carried — 36 and 330 — which is a real
change to the alignment space, not just naming, and it is why `main_chromosomes` for both is now exactly
the whole assembly. The rows they replace were v5.5 (53 contigs) and Phypa_V3 (357, including a
**spurious 27th chromosome** V7 resolves away).

**mm10 already *is* ENCODE's analysis set** — contig names and lengths diff clean against ENCODE's own
`mm10_no_alt.chrom.sizes` (file `mm10_no_alt_analysis_set_ENCODE`).

**S. pombe's two extra contigs look like alts and are not.** Both are additional *unique* sequence the
chromosome assembly lacks — the opposite of an alternate representation:

- **`MTR`** = FP565355, 20,128 bp, "S. pombe chromosome mating type region": the silent mat2-P/mat3-M
  cassettes. A distinct locus, not a second copy of mat1 (which is on II). Excluding it would push
  mat2/mat3 reads onto mat1 through the shared H1/H2/H3 homology boxes and *manufacture* signal there.
- **`AB325691`** = 20,000 bp of gap-filling sequence between SPBPB21E7.09 and SPBPB10D8.01 on the left
  arm of chromosome II (Sasaki et al. 2008, *Yeast* 25(9):673-679, PMID 18727152) — sequence **missing**
  from II. Excluding it discards real sequence.

Neither is in NCBI's ASM294v2 (`GCA_000002945.2` has only I, II, III, MT); both are PomBase/Ensembl
additions, and Ensembl assigns both `coord_system: chromosome`, not scaffold or alt. The mat cassettes
do multimap against each other, but that is ordinary biological repetition — like rDNA, transposons or
gene families — and is never fixed by deleting a copy from the reference.

**Ensembl `dna.toplevel` is correct, not an oversight.** Ensembl only emits `dna.primary_assembly` when
toplevel contains haplotypes or patches; no such file exists for any of the three Ensembl species here.

**The two plant references have NO organelle contigs, and that costs ~10% of those libraries.** The
k-mer test below was run on v5.5 and Phypa_V3; **the conclusion carries to their 2026-09 replacements and
was re-verified for them the cheap way**, from assembly reports showing 17 and 26 sequences with no
organelle among them. The decoys are therefore still required — see the note on both being dropped and
restored. An earlier
version of this table claimed Chlamydomonas carried `MT/cpDNA` and Phypa_V3 carried "organelles"; both were
wrong. Checked from the FASTAs themselves: 15 random 30-mers from each organelle genome fetched from ENA,
searched against the assembly on both strands —

| reference | organelle | length | 30-mers found in the assembly |
| --- | --- | --- | --- |
| Chlamydomonas_reinhardtii_v5.5 | chloroplast `BK000554` | 203,828 bp | **0 / 15** |
| Chlamydomonas_reinhardtii_v5.5 | mitochondrion `U03843` | 15,758 bp | 1 / 15 (NUMT) |
| Phypa_V3 | chloroplast `AP005672` | 122,890 bp | **0 / 15** |
| Phypa_V3 | mitochondrion `AB251495` | 105,340 bp | 1 / 15 (NUMT) |

The single mito hit in each is what a nuclear insertion of organellar sequence looks like, not a present
genome. **Do not read "a contig of about the right length exists" as evidence** — `KZ454947` is within 5%
of the chloroplast's length and `scaffold_51` within 5% of the mitochondrion's, and neither contains any
of that sequence; length coincidence is why this needed a sequence check rather than a size scan.

Consequence: **12.2% of `C.reinhardtii-liquidculture_5GRO` and 8.6% of `P.patens-plateculture_5GRO` reads
are organellar with nowhere to map** (chloroplast 12.1% / 8.1%, mitochondrion 0.1% / 0.5%; measured by
20-mer match against the ENA sequences over 5,000 adapter-trimmed inserts). Those reads are pure
`unmapped: too short`. Adding the organelles as decoys is the fix and it is consistent with the rule
below; it needs a new STAR index, so **do it before a re-map, not after**.

**Organelles and unplaced scaffolds stay.** Organelles are decoys — they belong *in* the index, absorbing
reads that would otherwise mismap onto nuclear NUMT/NUPT copies. dm6's 1,862 scaffolds are not
duplicates. Neither can reach a fold, since `chrom_splits.yaml` lists only main chromosomes.

## rDNA sink

ENCODE aligns to genome + rDNA. **The decoy is a sink, not a multimapping trap** — an earlier version of
this file had the mechanism backwards, so be precise about it: the decoy is the *true, full-length*
rDNA sequence, so rDNA-derived reads align to it with a **better score** than to the degenerate
rDNA-like fragments scattered through the nuclear genome, and STAR takes the best alignment. Reads
whose real source is a nuclear locus still map to that locus and are kept. Multimapper-dropping happens
only where a nuclear copy is genuinely identical; it is a side effect, not the point.

This matters more for PRO-cap than for most assays: a nuclear run-on captures Pol I as well as Pol II,
so these libraries carry abundant rDNA reads.

**The rule: sink only where the array is MISSING from the assembly. Do not mask rRNA that lives on a real
chromosome.** Reads landing on a genuine in-assembly rDNA locus are mapping to their true source — that
is correct behaviour, not an artifact, and masking it would throw away real signal. The only problem
worth solving is *spurious* alignment at other loci, which is exactly what a sink prevents and what a
mask cannot.

Checked per species — is the 45S/35S array actually in the assembly?

| species | array in assembly | action |
| --- | --- | --- |
| M. musculus | **no** — only `Rn18s-rs5`, a dispersed 18S copy at `chr17:40157244-40159092` | **sink `BK000964.3`** |
| D. melanogaster | yes — `chrUn_CP007120v1` (76,973 bp), an unplaced scaffold | none, it is a built-in sink |
| C. elegans | yes — `I:15062083-15071033` (18S, 26S, 18S) | none |
| S. cerevisiae | yes — `XII:451786-489469` (RDN37-1/2 plus RDN5-1..6) | none |
| S. pombe | yes — `III:1-23130` and `III:2440994-2452883`, both ends | none |
| A. thaliana | yes — `2:3706-5945` and `3:14197677-14199916` | none |
| C. reinhardtii | yes — **re-derived on ASM4749649v1 2026-09-06**: subtelomeric arrays on `CM104926.1` (chr_08) and `CM104932.1` (chr_14), each running to its terminus | none |
| P. patens | **unclear** — 80 rRNA genes but every SSU call is partial | none available |
| S. moellendorffii | yes — 426 rRNA genes incl. a 4,408 bp LSU on `GL377567` | none |
| C. griseus | **unchecked** — but no reference sequence exists to use | none available |

**Mouse is the only species needing a sink.** `BK000964.3` is TPA, 45,306 bp, "Mus musculus ribosomal
DNA, complete repeating unit" — the exact analogue of the human `U13369.1` ENCODE uses. mm10's largest
unplaced scaffolds near that size (`GL456366`, `GL456367`, `GL456239`) are generic unplaced/unlocalized
contigs, not rDNA; and `BK000964` being a *Third Party Annotation* record is itself a sign the sequence
was never deposited as primary assembly data.

**Do not test for rDNA by grepping contig names or filtering on `rRNA_gene`.** Both fail, and both
failed here first:

- **UCSC renames scaffolds to accession-based names.** dm6's rDNA is `chrUn_CP007120v1` — `CP007120` is
  "Drosophila melanogaster chromosome X; Y rDNA sequence", the same 76,973 bp scaffold Ensembl BDGP6
  exposes as `rDNA`. Grepping for "rDNA"/"rRNA" in `dm6.chrom.sizes` returns zero and makes it look
  absent. Match on *length* against the Ensembl region list instead, which is how it was found.
- **Ensembl GFF3 types these as `rRNA` under `ncRNA_gene`, not `rRNA_gene`.** Filtering on `rRNA_gene`
  returns nothing for S. pombe chromosome III and makes those arrays look absent too.

`M21017.1` (12,026 bp, clone pDm238) is a real single repeat unit — 18S, ITS1, 5.8S, 2S, ITS2, 28S, IGS,
running into the next unit's 18S — so the earlier note calling it "not a full repeat unit" was wrong. It
is simply unnecessary: `chrUn_CP007120v1` is 6x larger and already in the assembly.

Decoys are fetched from ENA (`https://www.ebi.ac.uk/ena/browser/api/fasta/{accession}`) into
`data/decoy/` — renamed from `data/rdna/` now that organelles share the mechanism.

**Decoys go in the STAR INDEX ONLY, and are deliberately absent from `chrom.sizes`.** An earlier version
of this file said the opposite — that the sink *had* to be listed in chrom.sizes or `bedGraphToBigWig`
would abort on an unlisted contig. The reasoning is right but the premise is not: `bedgraph` greps to
`main_chromosomes` *before* writing, so decoy rows never reach `bedGraphToBigWig`. The `chrom_sizes` rule
also declared the decoy `.fai` as an input its shell never used, which in turn was the only thing keeping
a `faidx_rdna` rule reachable; both are gone. If you ever drop the filter from `bedgraph`, this changes
with it.

This is the same treatment **mm10's `chrM` already gets** — no organelle appears in `main_chromosomes`
for any of the 12 species — so a decoy can never reach a fold, a peak or a bigWig.

**`organelle_accessions` covers a reference that omits its own organelles.** Unlike the rDNA sink, which
prevents *mis*mapping, this fixes reads that cannot map at all:

| species | organelles in reference | decoy added | share of the library |
| --- | --- | --- | --- |
| C. reinhardtii | none | `BK000554` cp, `U03843` mt | **12.2%** (12.1 / 0.1) |
| P. patens | none | `AP005672` cp, `AB251495` mt | **8.6%** (8.1 / 0.5) |
| C. griseus | none | `DQ390542` mt | 0.3% |
| S. moellendorffii | `Pt` present, mt absent | **none exists** — no complete *Selaginella* mitochondrial genome is deposited in ENA | — |
| dm6 / mm10 / ce11 | `chrM` present | — | — |
| TAIR10 / R64-1-1 / ASM294v2 | `Mt`+`Pt` / `Mito` / `MT` present | — | — |

All five sequences are complete genomes with 0–1 ambiguous bases. C. griseus is included for the decoy
rationale — NUMTs would otherwise attract those reads — not for yield; 0.3% is small enough that dropping
it is reasonable if a re-index is unwelcome.

**Adding a decoy raises `pct_unique` without adding usable signal.** Those reads now map (uniquely, since
there is one copy) and are then filtered out at the `bedgraph` step, so `signal_reads` barely moves. The
gain is correctness — they stop landing on nuclear NUPT/NUMT copies and manufacturing initiation signal
inside `main_chromosomes` — not depth. Do not read the improved percentage as recovered training data.

**One trap when adding an accession:** the `accession` wildcard constraint was `[A-Z]{2}[0-9]{6}`, and
`U03843` is 1 letter + 5 digits. It failed to match, and surfaced as a `MissingInputException` on
`star_index` that said nothing about wildcards. Now `[A-Z]{1,4}[0-9]{5,8}`.

**No other dataset needs an organelle decoy — verified against the assemblies actually in use**, not
against Ensembl's current default for each species, which is a different assembly in several cases:

| reference | checked how | organelles |
| --- | --- | --- |
| dm6 / mm10 / ce11 | UCSC `chrom.sizes` | `chrM` present (19,524 / 16,299 / 13,794 bp) |
| R64-1-1 | contig list of the FASTA in use | `Mito` present (85,779 bp) |
| ASM294v2 | contig list of the FASTA in use | `MT` present |
| TAIR10 | Ensembl toplevel | `Mt` 366,924 + `Pt` 154,478 present |
| v1.0 (S. moellendorffii) | Ensembl toplevel | `Pt` 143,775 present; **mitochondrion absent** |

**S. moellendorffii's mitochondrion is the one real remaining gap, and it is not fixable**: no complete
*Selaginella* mitochondrial genome is deposited in ENA (the largest non-scaffold record for the taxon is a
40 kb BAC clone). Leave `organelle_accessions` empty rather than substituting a related species.

**`organelle_contigs` records the organelles that are ALREADY in each assembly**, and exists purely so
`src/qc/rrna_content.py` can measure organellar content for them. Without it the tool scores only against
`data/decoy/` and reports 0% for the 9 of 12 species whose organelles are in the reference — a false
negative, not a missing measurement. Verified with a positive control: a k-mer taken from R64-1-1's `Mito`
is present in the organellar index and absent from the rRNA one, and Booth S. cerevisiae then genuinely
measures ~0% mitochondrial.

**`signal_reads` counts only `main_chromosomes`, and used not to.** `samtools idxstats` sums every contig,
but `bedgraph` greps to `main_chromosomes` afterwards, so reads on `chrM`, `Pt` or any decoy never enter a
bigWig or a peak. Counting them made the column documented as "the number that matters" overstate usable
depth for every organelle-containing assembly — and the error grows with each decoy added, since 12.2% of
the C. reinhardtii library lands on the plastid decoy alone. This is also why adding a decoy raises
`pct_unique` while leaving `signal_reads` flat: the reads map, then get filtered.

`run_procap_pipeline.py` previously read `rdna_accession` while `workflow/Snakefile` ignored it
entirely, and nothing fetched the sequence. Setting any accession would have given the two drivers
different STAR indices. Keep both implementations in step — the shared entry point is
`decoy_accessions()`, which exists in both.

**Do not test for an organelle by contig length.** `KZ454947` is within 5% of the Chlamydomonas
chloroplast's length and `scaffold_51` within 5% of the P. patens mitochondrion's, and neither holds any
of that sequence. Match on sequence: 15 random 30-mers from the organelle against the assembly, both
strands. 0/15 means absent; 1/15 is what a NUMT looks like and is not evidence of presence.

## SRR19034544: interleaved mates deposited as a single-end run

**`M.musculus-GCB_PROcap` was modelling 3' ends on the antisense strand**, and the orientation metaplot is
what caught it. Worth reading as a case study, because every layer of metadata was wrong and only the
reads settled it.

The symptom: sense signal peaked sharply at the annotated TSS (correct), but antisense peaked **just
downstream at ~+36 instead of divergent upstream**, with `antisense/sense` of exactly **1.00** where every
other mouse experiment sits at 0.52-0.70. Annotation was not the cause -- GCB uses 23,798 mm10 TSSs, the
most of any mouse experiment and essentially the same as `liver-old`'s 23,496, which produces the cleanest
plot in the set.

That geometry is the signature of an unfiltered R2. `genomecov -5` takes the 5' end of every record, and
R2's 5' end is the RNA **3' end** -- opposite strand, displaced downstream by the fragment length. It is
the exact artifact `final_bam`'s `samtools view -f 64` exists to prevent, and the guard never fired
because the run is recorded as single-end.

**ENA and SRA both report SINGLE, and both are wrong.** Proven from the reads alone:

- every instrument name occurs exactly twice (2 M names over 4 M reads);
- consecutive reads share `flowcell:lane:tile:x:y` (`A00700:262:HW3FYDRXX:1:2101:15365:1016`), i.e. one
  cluster, i.e. one fragment;
- mate2's insert is the **reverse complement** of mate1's -- 35/36 identity, the mismatch being an `N` --
  so both mates read one ~36 bp fragment from opposite ends;
- still interleaved 200 M reads deep, so it is not a header-only artifact.

`read_count` is 404,607,360 = 2 x 202,303,680, which also explains why this run looked like a 7-10x depth
outlier against its 41-52 M siblings, and why it is 65% poly-G (both mates read ~100 dark cycles past a
36 bp fragment).

**What hid it:** SRA labels every read `/1` when it dumps a run as single-end, and gives the two mates
different accession names (`SRR19034544.1`, `.2`). So the mate flag says `/1` on both, and the names look
unique. `library_layout` is resolved from ENA precisely because the manifest was wrong twice
(`Liver_ChROcap_mm`, `Tome2018_mm_CoPRO` both claimed paired and are single) -- this is the same class of
error in the opposite direction, with the archive as the source.

**The mechanism.** `deposited_interleaved` in `planning/manifest_samples.tsv` -> `raw.interleaved` in
`experiment_config.yaml` -> a `deinterleave` rule in `workflow/Snakefile` and the matching branch in
`run_procap_pipeline.py`. Note:

- `library_layout` stays **SINGLE**. That is what the archive says and it remains recorded; `interleaved`
  is the measured correction layered on top.
- `is_paired()` is now the PROCESSING decision (paired deposit **or** interleaved) and is deliberately
  distinct from `RUN_LAYOUT`. It is what makes `-f 64` fire.
- `run_fastqs()` is keyed on `RUN_LAYOUT`, not `is_paired()` -- it describes what is ON DISK, and using
  `is_paired()` would make the DAG look for two files that do not exist.
- Read names are rewritten to the instrument name with `/1` stripped, so both mates share a name. Safe
  here because this run has no UMI, so fastp's `:`-appended UMI convention and umi_tools'
  `--umi-separator` are not in play.
- `--overlap_len_require 18` finally does something for this run: the mates fully overlap a 36 bp
  fragment, so fastp can do overlap-based correction.

**Two traps when editing the awk**, both hit while writing it:

1. It must be ONE source line. Written across lines, `\n` inside the awk string becomes a literal newline
   and the program is an unterminated string literal. **A Snakemake dry-run will not catch this** -- it
   does not parse shell content, so the DAG builds and the rule fails at runtime.
2. The awk braces must be doubled (`{{`/`}}`). Snakemake formats the shell string, so a single `{` is read
   as a substitution field.

Verified by running the exact command Snakemake emits against a fixture built to mimic the real file:
5/5 records to each mate, odd records to mate 1 and even to mate 2, names matching between mates, `/1` and
the SRA accession stripped. `AWK_DEINTERLEAVE` in `run_procap_pipeline.py` is the same program; keep them
identical.

## Read-structure QC: the untrimmed adapter

**Nothing in this pipeline configured a sequencing adapter until now, and it cost most of the reads in
14 of the 38 experiments that existed when the survey was run.** `src/qc/read_structure_qc.py` (rule `read_structure_qc`, in `all` and `qc`,
output `qc/reads/{exp}.tsv`) surveys the raw FASTQs so this is visible before alignment burns the compute.

The failure chain, established from `Log.final.out` plus the reads themselves:

1. PRO-seq/ChRO-cap/CoPRO libraries carry the **Illumina small-RNA 3' adapter**,
   `TGGAATTCTCGGGTGCCAAGG` — which is also proseq2.0's `ADAPT1` default. The repo cross-checked mate
   handling against proseq2.0 and adopted `--RNA5`/`--map5`/`--opposite-strand` but never picked up its
   adapter.
2. Inserts are 20-45 bp inside a 74 bp read, so the adapter sits **mid-read at a different offset in
   every read**. fastp's auto-detection looks for an overrepresented tail and therefore missed it:
   `No adapter detected for read1`, `reads with adapter trimmed: 0`, and read length unchanged by
   filtering (74.44 -> 74.45 bp).
3. STAR aligns the short insert, soft-clips the adapter, and fails
   `--outFilterMatchNminOverLread` (default **0.66 of read length**), which on a 74 bp read demands
   >= 49 bp aligned. `M.musculus-liver-*` therefore reported **95-99.85% `unmapped: too short`** with
   only 0.25-0.92% multimapping — so it was never rRNA, never the mismatch filter, and never read length.

`C.griseus-CHO_GROcap` is the control that proves the parameters are not at fault: **76 bp reads, 5.21%
too short**, under identical settings.

Two further traps found in the same reads, both of which survive adapter trimming:

- **A dead sequencing cycle.** Every read in one library had `N` at position 6, the preceding 5 bases
  varying randomly. This matters out of proportion because **STAR counts an N as a mismatch** and
  `--outFilterMismatchNmax 1` allows exactly one, so every read enters the aligner with its budget
  already spent and any real variant is fatal. `dead_cycles` reports it. The fix is a length-scaled cap
  (`--outFilterMismatchNoverLmax`) rather than `Nmax 1`, marked `DEVIATION` in
  `config/procap_pipeline.yaml`.
- **`--overlap_len_require 18` is paired-end only** and therefore inert for 34 of 42 experiments. The
  ENCODE trim string is doing less than it looks.

**Read `pct_short_untrimmed`, not `pct_adapter`.** Every library carries a few percent of adapter
**dimers** (insert length 0) which cost nothing to leave in — `SRR826225/6` are 5.4%/5.0% adapter and
need no action. `pct_short_untrimmed` is the share of reads STAR will actually discard, i.e. exactly what
trimming recovers, and it is validated against STAR's own accounting: predicted 5.4%/4.9% against
observed 4.72%/5.30% per run, and 92.9% on a synthetic library built to mimic the 95.29% real case.

Survey everything at once, without Snakemake:

    python src/qc/read_structure_qc.py --tsv qc/reads/read_structure.tsv

### The fix, and what the survey decided

**Two adapters, not one, assigned per experiment.** The survey over all 69 FASTQs found 37 needing
trimming, split 27 `smallRNA_RA3` / 10 `truseq_universal`, and the split follows assay lineage exactly:

| adapter | sequence | studies |
| --- | --- | --- |
| `smallRNA_RA3` | `TGGAATTCTCGGGTGCCAAGG` | PRO-cap / ChRO-cap / CoPRO (= proseq2.0's `ADAPT1`) |
| `truseq_universal` | `AGATCGGAAGAGC` | 5'GRO / GRO-cap |

Mechanism, mirroring how UMIs work — **measured evidence in the manifest, not a lookup table in code**:

1. `planning/manifest_samples.tsv` gains an **`adapter`** column holding the NAME, set from the survey
   and consistent within every project (which is why project granularity is right).
2. `build_experiment_config.py::adapter_from_rows` propagates it to `raw.adapter`, raising on a
   conflicting value within one project.
3. `steps.trim.adapters` in `config/procap_pipeline.yaml` maps name -> sequence.
4. Both drivers resolve it: `adapter_arg()` in `workflow/Snakefile` and in `run_procap_pipeline.py`.
   Keep them in step.

**Three guards, all verified to fire**: a conflicting adapter within one project fails the generator; an
unknown adapter name fails DAG construction (`WorkflowError`) *and* fails the serial driver. An empty
value means none was detected and fastp auto-detects — that is the honest default, and it is what every
library silently got before this column existed.

**A fourth guard, added after the cotton case slipped past the other three: the manifest's curated name
is now checked against the reads.** All three guards above are *internal consistency* checks — they catch
a name that conflicts with a project-mate, or a name with no sequence behind it. None of them asks whether
the name is **true**, so `G.{arboreum,hirsutum}-ovule_GROcap` sat at `smallRNA_RA3` while every one of
their runs is 97%+ `truseq_universal`, and both mapped and were reported clean.
`experiment_stats.read_survey_flags` now compares `raw.adapter` against the survey's `best_adapter` and
emits `FAIL:adapter_mismatch(config=X,reads=Y)` when a *different* adapter is detected in at least
`ADAPTER_CONFLICT_PCT` (50%) of R1. Verified as a screen, not just on the known case: it fires on exactly
the two cotton experiments and on none of the other 40, whose configured and detected names agree or
whose configured value is deliberately blank.

**One subtlety: the flag compares the config to the reads, so correcting the config clears it while the
mapped output on disk is still stale.** Snakemake catches the staleness itself — the adapter reaches
`trim` through both `params` and the `adapter_fasta` input, so changing it re-runs trim and everything
below for the affected runs — but the table will look clean before that happens. A blank pipeline column
means unmapped; a *populated* one does not mean populated by the current config.

**A survey TSV that predates a column disables the check reading it, silently — so `report_flags` now
verifies the schema.** `SURVEY_REQUIRED` lists the columns the read-level checks need
(`best_adapter`, `interleave_suspect`, `declared_interleaved`) and every file under `qc/reads` is checked
against it. This is the same class of failure as a missing survey, which was already reported, and it
bites hardest on the corpus-wide `read_structure.tsv`: `read_survey_flags` falls back to that file when a
per-experiment one is absent, so a stale copy there **satisfies the fallback and answers for every
experiment** while quietly running fewer checks. Verified on the real tree — it names the stale
`read_structure.tsv` and the two orphaned pre-sex-split liver files, and passes all 42 current ones.

**`steps.trim.adapters` maps a name to a LIST, and both drivers pass `--adapter_fasta`, not
`--adapter_sequence`.** A dimer can be deposited TRUNCATED, and fastp matches an adapter by looking for
its *beginning* somewhere in the read — so a read that starts five bases into the adapter never matches,
survives trimming as a full-length run of pure adapter, and STAR discards it as `unmapped: too short`.
Measured per read on the raw FASTQs:

| run | starts with the 5-truncated TruSeq form | pct_unique |
| --- | --- | --- |
| `SRR12774945` C. griseus KLA | **43.4%** | 28% |
| `SRR6660402` Link2018 BMDM | **18.7%** | 38% |
| `SRR12513906` mouse liver (control) | 0.0% | 91% |
| `SRR639142` C. elegans (control) | 0.0% | 82% |

Only the truseq lineage shows it; the small-RNA RA3 libraries deposit dimers with a variable 0-2 bp
insert, which fastp already handles because the adapter's own start is still present.

**Confirmed by isolation, and watch the right counter.** On a 2 M-read subsample of `SRR12774945`,
counting `too_short_reads`:

| config | `too_short` |
| --- | --- |
| `--adapter_sequence AGATCGGAAGAGC` (the old config) | 23,864 |
| `--adapter_fasta`, full sequence only | 27,930 |
| nothing — fastp auto-detects a 33-mer | 27,136 |
| **`--adapter_fasta`, full + truncated** | **904,438** (45.2%) |
| **both flags together** | **903,891** |

Only the configs carrying the truncated entry move, so that entry is doing the work and auto-detection is
irrelevant. `too_short` is the counter to watch, not `reads with adapter trimmed`: a read that is
*entirely* adapter trims to length 0 and is booked as too short, so the adapter counter barely moves
(10.27 M -> 10.43 M on the full run) even as 12.07 M reads are removed.

**Do NOT verify this from fastp's `read1_adapter_counts`.** It reports **zero** for
`GGAAGAGCACACGTCTGAAC` while that sequence is discarding 45% of the library, because it cannot attribute
a trim that leaves nothing behind. That counter was read as proof of a no-op once already; the isolation
test above is what settles it.

**Both drivers pass `--adapter_sequence` (the first entry) as well as `--adapter_fasta`.** The fasta
catches the variants; pinning the sequence suppresses fastp's auto-detection, so trimming is a pure
function of the config rather than of the data. The last two rows above are the measurement showing that
pinning costs nothing — worth keeping, because suppressing auto-detection is exactly the kind of change
that could have silently undone the fix.

**And the fix removes reads rather than recovering them.** For the small-RNA RA3 libraries trimming
recovers real inserts (the 449 M -> ~1,300 M estimate). A truncated TruSeq dimer has no insert behind it,
so for C. griseus and Link2018 `input_reads` roughly halves, `unique_reads` holds, and `pct_unique` rises
from ~28% toward ~50% because the denominator stops counting unmappable reads. The gain is an honest
mapping rate and about half the STAR compute, not depth.

**The fix has to be a second sequence, not a shortened first one** — configuring `GGAAGAGCACACGTCTGAAC` alone would catch the
truncated dimers but leave 5 bp of adapter on every *real* read, since fastp would then trim at
`insert + 5`. fastp takes only one `--adapter_sequence`, so multiple sequences require a file: the
`adapter_fasta` rule writes `{WORK}/adapters/{name}.fa` from the config (a rule, not an inline write, so
many trim jobs sharing one adapter cannot race), and `run_procap_pipeline.py` writes the same content
next to the run. A scalar config value is still accepted and wrapped, and a single-entry list behaves
exactly as `--adapter_sequence` did.

`--trim_poly_g --trim_poly_x` (`steps.trim.poly_params`) go to every library. fastp only trims a real
poly-X tail so they are inert where there is none, but `SRR19034544`
(`M.musculus-GCB_PROcap`, 150 bp, 200 M reads) is **65% poly-G** against under 1% everywhere else.

**STAR was deliberately NOT relaxed.** A dead-cycle concern was raised from a hand-picked read sample and
the survey refuted it: `deadCyc` is empty for all 69 files and N content never exceeds 1.1%. So no
systematic N is consuming the mismatch budget and `--outFilterMismatchNmax 1` stays. Check `qc/reads/*.tsv`
before revisiting.

Expected recovery, from `pct_short_untrimmed`: roughly **449 M -> ~1,300 M mapped reads** over the affected
experiments, with liver going 8.2 M -> ~190 M and CoPRO 3.5 M -> ~50 M. **That estimate is optimistic for
dimer-heavy runs**: several Spt5/Ino80 R1 files have a median insert of only 4-8 bp, which trimming
DISCARDS via `--length_required 18` rather than recovering. The solid gains are where median insert is
comfortably above 18 — liver (29-32), CoPRO (28-30), Kim2018 (25-31), Booth2016 (21-29), hamster (37-52),
Chlamydomonas (43-47), P. patens (50), GCB (51).

Two things the survey explained that need no action: **C. elegans is 0% adapter at 30 bp** (the read is
shorter than the insert, so the adapter never enters it — which is why worm maps at 75-83%), and
**A. thaliana has 17% adapter at median insert 57**, above the ~49 bp threshold, so only 4.5% is lost.
A. thaliana's 44% mapping rate therefore remains **unexplained** and is the open question.

## Orientation QC — the two silent assumptions

`src/qc/orientation_qc.py` exists because two things in this pipeline are conventions rather than
documented facts, and **both are silent when wrong**: which mate of a paired library carries the RNA
5' end (`steps.signal.five_prime_mate`), and which read strand becomes the plus track
(`reverse_strand`). Get either backwards and the pipeline still runs, still calls peaks, and still
trains — it just models 3' ends, or the wrong strand.

**Readers are `pybigtools` and `pyfaidx`, matching tangermeme — not pyBigWig/pyfastx.** tangermeme's
`extract_loci` is what reads sequence and signal for training, and it uses these two, so the QC sees the
data through the same path the model does. pyBigWig and pyfastx are present only as transitive deps
(`bam2bw`, `biodatatools`, and `pypints` needs pyBigWig itself), which is a bad reason to read data with
them. They are also less portable: pyBigWig has no macOS arm64 wheel, so QC written against it cannot be
developed or tested off-cluster, while pybigtools (Rust) and pyfaidx (pure Python) both install anywhere.

**A stale conda env is the likeliest QC failure, and it surfaces late.** `pybigtools`, `pyfaidx`,
`logomaker` and `matplotlib-base` are conda deps added after the env was first created, and the QC rules
run at the END of the DAG -- so an environment predating them fails after ~600 jobs of real work with a
bare `ModuleNotFoundError` from deep inside `peak_maxima`. `orientation_qc.py` now calls `require_deps()`
right after argparse, which names the missing packages and the three ways to fix them. Nothing computed is
lost when this happens (peaks and bigWigs persist), but update the env before a long run:
`mamba env update -f environment.yml -n nasti-critters`.

**Both QC steps are rules in the DAG**, so a normal `snakemake` run produces them; `snakemake qc -c8`
runs only the QC against existing signal. Outputs land in `qc/orientation/` (PNG + a one-line verdict
TSV) and `qc/umi/`. They stay inside the DAG only because `pybigwig`, `pyfastx`, `logomaker` and
`matplotlib-base` are all available for linux-64 on conda — if any of them ever has to move to
`pyproject.toml`, the QC rules must leave the DAG with it, exactly as `negatives` did.

- **Initiator PWM + logo around in-peak signal maxima.** Real initiation carries a Py at −1 and a Pu at
  0. A *flat* logo means the maxima are not initiation sites — the 5'/mate assignment is wrong.
  **Information is RELATIVE ENTROPY against local base composition, not against a uniform background.**
  The null is the composition of the same windows' own flanks (`|offset| >= 5`, outside the initiator but
  inside the same promoter context), because the question is whether a base differs from a random base
  *near a peak* — promoters are compositionally unlike genome average even in an AT-rich genome, so
  genome-wide frequencies would credit that context as signal at every position and inflate the maximum
  with it.
  **The old uniform-background measure inflated skewed genomes rather than penalising them**, which is the
  opposite of the intuitive guess and worth stating plainly: a position matching a 64% GC background
  exactly still scored `2 - H(bg)` = 0.057 bits of pure artifact, visible as the 0.02–0.03 bit flanking
  letters in the C. reinhardtii logo where mouse's flanks sit near 0.005. Verified on synthetic PWMs —
  a completely flat PWM scores 0.057 (GC-rich) and 0.042 (AT-rich) under the old measure and **exactly
  0.000 under the new one**, while a real initiator scores by how unexpected it is against its own
  context. So `C.reinhardtii-liquidculture_5GRO` will move DOWN from 0.13 bits, not up; its flat logo is
  about a thin, noisy peak set (8,789 peaks at 720 reads/peak, 90.6% unidirectional), not about GC.
  Two consequences: the value is **no longer capped at 2 bits** (relative entropy is unbounded above), and
  **the 0.15 FLAT threshold is provisional** — it was calibrated on the old measure, every value shifts
  down by roughly `2 - H(background)`, and it should be recalibrated from a full run's output.
  `information_content()` is shared by the plot and the printed verdict so the two cannot disagree; they
  were separate copies of the same expression before.
  **The FLAT flag needs LOW AMPLITUDE *and* a MISPLACED maximum — amplitude alone is not a fault.**
  The first run using relative entropy flagged `D.melanogaster-S2_5GROcap` at 0.13 bits, and it was a
  false positive: that logo is a textbook C at −1 and A at 0 with a sharp metaplot, so the motif is
  present and correctly placed, merely diluted across 42,853 peak maxima of which many come from
  low-confidence peaks whose maxima are random; `C.reinhardtii` at 0.23 bits is the same pattern.
  **Amplitude measures peak-set quality; POSITION is what says whether the 5' assignment is right.** The
  verdict now prints the offset of the maximum and flags only when it falls outside {−1, 0}; where it is
  correctly placed and low, it prints an explicit dilution note instead. Verified on synthetic PWMs across
  four regimes: flat → flagged, strong Inr at 0 → clean, *weak* Inr at 0 → clean with the note, weak
  signal displaced to +7 → flagged. This is still a position check, not a score — do not grow it into one.
- **Stranded metaplot around annotated TSSs. THE PLOT IS THE CHECK — look at it.** The purpose is to
  see, by eye, whether 5' signal sits where it should relative to the annotated TSS. It is not a
  measurement of antisense or divergent transcription, and it is deliberately not a scoring system.
  **Each site is normalised by its own window total before averaging**, so every TSS carries equal
  weight and the y-axis is "mean fraction of site signal". Summing raw profiles let one locus dominate:
  in a numerical check, 1 outlier site out of 1001 (0.1% of the data) contributed **78.8%** of the raw
  profile and inverted its argmax; normalised it contributes 1.9%. Sense and antisense share ONE
  denominator per site — normalising them separately would equalise them and destroy the strand-swap
  comparison. Adding more sites does not fix outlier dominance; only the statistic does.
  `qc/orientation/{exp}.orientation_qc.png` has the dashed line at 0 and "expect sense peak just
  downstream of 0" in the panel title; that is what you are verifying.
  **Do not trust the printed notes as pass/fail.** The `sense downstream/upstream` ratio is descriptive
  only, and is demonstrably not diagnostic: on synthetic profiles it scores **27.37 for a wrong-placement
  case (signal displaced to +400) against 6.31 for the correct one** — i.e. better for the broken input,
  because a ratio of two window halves says nothing about *where within* a half the signal sits. It is
  kept because it is occasionally informative, not because it decides anything.
  A numeric TSS-enrichment score was added here and then removed: deciding placement by eye is the
  actual requirement, and a score invites trusting the number over the picture. **Do not reintroduce
  one.**
  **Divergent upstream antisense is NOT an expectation at all.** C. elegans promoters are predominantly
  unidirectional — the same fact that makes PINTS' `unidirectional` class essential — so absent antisense
  there is the expected result, and nothing flags it.
  **Metaplot flags are gated on `annotation_tss_anchored` in `config/genomes.yaml`.** Where the annotated
  gene start is not the TSS the metaplot measures the annotation, not the pipeline, so its lines are
  printed as `(advisory: annotation is not TSS-anchored)` and raise no flag. Currently false for
  **S.cerevisiae** (Ensembl `gene` start is the ATG), **C.elegans** (post-trans-splicing 5' end; a
  biological limit, not fixable by changing annotation source) and **S.moellendorffii** (predicted CDS
  models on a 2011 draft). The value is set from annotation PROVENANCE, not by fitting to observed
  ratios. The PWM check is never gated — it is annotation-free.
- **Neither panel is capped any more.** `--max-peaks`/`--max-tss` default to all. The old defaults
  (20,000 / 5,000) took a **prefix of a sorted file, not a sample**: `combine_peaks` writes peaks
  `sort -k1,1 -k2,2n`, so the logo for the 16 of the 38 experiments then defined that hit the cap was
  estimated from a
  genomic prefix — `C.griseus-CHO_GROcap` used 30% of its peaks, roughly one third of the genome in
  lexicographic chromosome order. The PWM's standard error at n=20,000 was already negligible, so this
  buys freedom from that bias rather than precision. The TSS cap bound for **all 38** experiments
  (`n_tss` was 5,000 everywhere); annotation files happen to interleave chromosomes so that subset was
  less skewed, but it was still arbitrary and unstable against file reordering.

The two are complementary, verified against synthetic data with known answers: correct orientation gives
1.00 bits at both −1 and 0 and no flags; swapping the strands is caught by the metaplot (the PWM still
looks fine, because it is strand-corrected using each maximum's own strand); moving signal to 3' ends is
caught by the PWM going flat.

Annotation (`annotation_url` in `config/genomes.yaml`, UCSC GTF for the chr-prefixed assemblies and
Ensembl GFF3 for the bare-named ones, so naming always matches the FASTA) is used **for QC only** —
never for training, peak calling or fold assignment, so no circularity reaches the model.

**C. reinhardtii has NO annotation under ASM4749649v1, and a null `annotation_url` is now a
SUPPORTED state rather than a DAG failure.** `annotation_path()` interpolated
`annotation_format` into the filename, so a null pair produced `data/annotation/C.reinhardtii.None.gz`
— a path no rule can produce. Both consumers declare it as an input (`orientation_qc` and
`rrna_content`), so `snakemake qc` died at **DAG construction** for that experiment instead of skipping
a panel, and the exclusion was invisible from the totals: a whole-corpus dry-run simply reported 40
`orientation_qc` jobs where 42 were expected. It now returns **no dependency**, `orientation_qc` omits
`--annotation` (an empty value would be read as the next flag), and `load_annotation()` returns no TSSs.
Degrading is right rather than convenient — annotation is QC-only, never labels/folds/peaks;
`rrna_content` already reports a missing annotation as NOT MEASURED (blank `pct_rrna`, distinct from 0);
and the two read-outs that carry the orientation verdict, the initiator logo and the **summit-anchored
metaplot**, are annotation-free. Only the TSS-anchored panel is lost — which the `notes` field of that
`genomes.yaml` entry already recorded as a known cost of the move, taken for a near-gapless assembly
(contig N50 6.29 Mb and 17 gaps against v5.5's 215 kb and 1,512 gaps).

**Do not go looking for that annotation again — checked 2026-09-06 and it does not exist.** Four
independent confirmations for `GCA_047496495.1` (strain **CC-1690**, University of Georgia, 2025-02-06):
the NCBI FTP directory carries no `*_genomic.gff.gz` at all; `feature_count.txt` reports
**0 unique ids and 0 placements** for `gene protein_coding`; the features/locations hashes in
`annotation_hashes.txt` are `d41d8cd98f00b204e9800998ecf8427e`, the MD5 of the **empty string**; and
`RefSeq-Accn` is `na` for every chromosome in the assembly report, so there is no paired RefSeq record
to annotate it. Across all 19 *C. reinhardtii* assemblies at NCBI the only annotated chromosome-level
one is **v5.5** itself (`GCA_000002595.3` / `GCF_000002595.2`, JGI) — the assembly this entry moved away
from — plus a contig-level CCAP 11-32A; even `GCA_026108075.1`, the reference-guided assembly this one
was built against, carries none there.

**And borrowing v5.5's (or v6.1's) annotation is not the fallback it looks like.** Different assembly,
different strain, and different chromosome naming — v5.5 is `1`-`17` where this FASTA is CM accessions
(`CM104919.1`…). `extract_loci` and the QC loader match literally, so it would yield **zero TSSs in
silence**, which is exactly the failure `chrom_style` exists to prevent; making it real would take a
liftover, not a URL. Leave `annotation_url` null.

**And `experiment_config.yaml` must be REGENERATED after a genome swap — `genomes.yaml` alone is not
enough.** `processed.sequences` is baked into the config by `build_experiment_config.py`, so updating
P. patens to V7 and C. reinhardtii to ASM4749649v1 left both experiments naming the *old* FASTA, which
no `fetch_genome` produces: same silent-exclusion symptom as above, two more missing jobs. Run
`python src/data_preprocessing/build_experiment_config.py` and commit the result with the
`genomes.yaml` change.

**GTF exists for all 12 species, but the source split is not free to change.** Ensembl ships a parallel
`gtf/` tree at the same release for all seven Ensembl species (probed 2026-08-30, all HTTP 200). Switching
the three UCSC species to Ensembl GTF is nevertheless **wrong**: the annotation source is chosen so
chromosome names match the FASTA, and Ensembl's fly GTF says `2L` where dm6 says `chr2L`. Literal name
matching would then yield **zero TSSs, silently** — the same failure mode as a `chrom_style` mismatch.

**The two format branches therefore select different things, and it is a known, deliberate asymmetry.**
`orientation_qc.py` keeps `feat == "transcript"` for GTF and `feat == "gene"` for GFF3, and Ensembl types
non-coding genes as `ncRNA_gene` rather than `gene`. So:

| species | source | selection | distinct TSSs |
| --- | --- | --- | --- |
| D. melanogaster | UCSC GTF | per-transcript, all biotypes (31,515 NM_ + 4,779 NR_) | 22,610 |
| M. musculus | UCSC GTF | per-transcript, all biotypes (37,727 NM_ + 6,719 NR_) | 30,757 |
| C. elegans | UCSC GTF | per-transcript, all biotypes (28,944 NM_ + **25,200 NR_**) | 49,486 |
| S. pombe | Ensembl GFF3 | per-gene, protein-coding only | 5,144 (7,030 if `ncRNA_gene` were added) |

Left as-is on purpose: the metaplot is a coarse orientation check, both sets are dominated by real Pol II
TSSs, and no label depends on it. But **do not read `n_tss` as comparable across species**, and in
particular:

**C. elegans is ~47% non-coding in refGene** (25,200 NR_ of 54,144 transcripts), much the worst of the
three. Many NR_ entries are snoRNA/snRNA/misc_RNA — Pol III or intron-processed — so they dilute the worm
metaplot for reasons unrelated to orientation. If this ever needs fixing, the fix is to filter the GTF
branch to protein-coding (`NM_`/`gene_biotype "protein_coding"`), not to change the annotation source.

**For worm this dilution flattens the metaplot without anything being wrong**, because ~half the
annotated "TSSs" are snoRNA/snRNA/misc_RNA starts with no Pol II initiation there at all. C. elegans is
also unidirectional, so its plot will look sparser than fly or mouse on both counts. If the worm plot
looks unconvincing, look at fly or mouse before concluding anything about `reverse_strand` — those are
the cleaner tests — and weigh the initiator logo more heavily than the metaplot for worm.

**Mate handling was checked against [Danko-Lab/proseq2.0](https://github.com/Danko-Lab/proseq2.0) and agrees on every point:** its
`--RNA5=R1_5prime` default matches `five_prime_mate: R1`; `--map5=TRUE` matches `genomecov -5`; its docs
state that exactly one mate is reported, which is what makes the both-mates bug a bug and not a
preference; `--opposite-strand` is our `reverse_strand`; and `UMI1=0/UMI2=0 -> no dedup` matches our
default-deny. Note proseq2.0's Example 3 uses `--RNA3=R1_5prime`, the opposite assignment — so it is
genuinely library-dependent and R1 stays a convention to validate.

## The 2026-08-30 manifest update: 3 projects, 12 experiments, 4 species

`planning/nonhuman_capped_runon_manifest.xlsx` gained 3 projects / 14 sample rows, taking the repo from
26 to **38 experiments** and 6 to **10 species**. The shared rows are byte-identical to the previous
workbook, and the only Projects-sheet change is the `COUNTIFS` range moving `$A$49` -> `$A$63`.

**The workbook does NOT carry the corrections that live in the TSVs.** `planning/manifest_samples.tsv`
and `manifest_projects.tsv` are the pipeline's actual inputs and hold curation the xlsx never received
(Liver single-end, Booth2016 `cap_status`, the `umi_len`/`umi_loc` columns, the resolved embryo runs). So
the update was applied by **appending the 14 new rows**, not by regenerating the TSVs from the workbook.
Regenerating would silently revert all of it. Do it the same way next time.

| project | species | experiments | notes |
| --- | --- | --- | --- |
| `Tome2018_mm_CoPRO` | M. musculus | 2 | CoPRO capped fraction, MEF ± heat shock. Nothing new needed. |
| `McDonald2024_plant_5GRO` | C. reinhardtii, P. patens, S. moellendorffii | 3 | 5'GRO-seq from GSE233927. |
| `Shamie2021_cg_5GRO` | C. griseus | 7 | Chinese hamster GRO-cap atlas; CHO-K1 has 2 reps, 6 tissues n=1. |

All 14 runs were resolved against ENA and all report `SINGLE`. Findings worth keeping:

- **GSE233927 holds csRNA-seq AND 5'GRO-seq for the same tissues, and this repo pulls the 5'GRO-seq —
  verified per sample 2026-09-06.** The series is five assays deep (csRNA-seq, 5'GRO-seq, plain GRO-seq,
  sRNA-seq "input", total RNA-seq), so picking the wrong sample would silently substitute a *steady-state*
  capped short-RNA library for a nascent run-on one. All four rows are correct: `GSM7439223`/`GSM7439224`,
  `GSM7439241` and `GSM7439248` carry per-sample descriptions
  `5'GRO-seq; nascent TSS mapping in {species} cells; Experiment SD102/SD103/SD150/SD180`,
  `Library strategy: 5'GRO-seq` in `data_processing`, and supplementary files named
  `*_5GRO-seq_r*.bed.gz`. The csRNA-seq counterparts (`GSM7439225`-`7439227`, `GSM7439243`,
  `GSM7439249`/`GSM7439250`) are **not** in the manifest.
  **Two traps in that metadata, both of which mislead if read alone:**
  GEO's `library_strategy` is **`OTHER` for csRNA-seq and 5'GRO-seq alike**, so it cannot separate them —
  only the title/description can (same class as cotton's `miRNA-Seq` mislabel). And
  `!Sample_extract_protocol_ch1` is a **series-wide concatenated blob** describing csRNA-seq, sRNA-seq and
  total RNA-seq, attached to every sample *including* the 5'GRO ones — read it on its own and these look
  like csRNA-seq libraries. Use `!Sample_description`.
  **Only three species in that series have 5'GRO-seq at all**: C. reinhardtii, P. patens,
  S. moellendorffii. `A.thaliana-seedling_5GRO` is a different project — `Hetzel2016_at_5GRO`,
  `GSM2193123`, "5'GRO-seq in 6 day seedlings" — because GSE233927's Arabidopsis samples are csRNA-seq
  only. **So extending this project to its other species (papaya, barley, maize, fly S2) means taking
  csRNA-seq, which is a DIFFERENT RNA POPULATION** — capped short RNAs from total RNA, no nuclear run-on —
  and belongs in its own assay family with its own models, not folded into `GRO-cap-equivalent`. The
  plain GRO-seq samples (`GSM7439228`, `GSM7439251`) are the paper's peak-calling input/control and are
  correctly excluded as targets by the Field Guide rule.
- **`Shamie2021_cg_5GRO` really is all Chinese hamster.** `BMDM…KLA`, `Brain`, `Kidney`, `Liver`, `Lung`
  look like Glass-lab *mouse* sample names (and Lam2013/Link2018 in this same manifest *are* Glass-lab
  mouse BMDM), so this was checked rather than assumed: ENA reports **72/72 runs as
  *Cricetulus griseus*** for PRJNA667472, and the study is a Chinese hamster TSS atlas. The manifest is
  right — hamster BMDMs were run with KLA, mirroring the mouse assay.
- **`Tome2018_mm_CoPRO` is single-end, not paired.** The manifest says "paired-end; RNA 5' initiation +
  Pol II active-site 3' ends", which describes CoPRO *as an assay*; the **deposit** is 1 FASTQ, no
  `nominal_length`, uniform 76.0 bp for both runs. Same signature, and same class of error, as the
  `Liver_ChROcap_mm` case. `five_prime_mate` does not apply. `library_layout` is resolved from ENA, so
  the pipeline was already correct.
- **`GSM7439223` had a blank `sra_experiment`**; it is `SRX20570354` / `SRR24798072`.
- **Matched non-cap libraries are excluded as targets, per the workbook's new Field Guide rule**
  ("Ordinary GRO, TAP−/noTAP, uncapped … never train them as positive TSS labels"). That means the
  hamster `*_GRO1` samples (`GSM4818105`-`4818112`) and McDonald's GRO-seq/csRNA-seq/sRNA-seq/total-RNA
  rows. `PRJNA978596` holds **68 runs over 8 species**; only 4 are wanted, so the "fetch what the config
  references" rule matters here as much as anywhere.
- **`Shamie2021` `Liver_GROCap1` is shallow at 5.4 M reads** — an order of magnitude below its siblings
  (23-40 M). Treat its model with suspicion.
- Download total at that update was **~120 GiB / 74 files / 64 runs**, measured from ENA `fastq_bytes`
  (the new projects added only 12.5 GiB; the old "~78 GiB" figure was already stale at 107.3 GiB).
  **Superseded by cotton — it is ~202 GiB / 82 files now.**

### Two things this update broke, both fixed

- **STAR memory was keyed on the string `"M.musculus"`.** `C.griseus` is 2.37 Gb — the same order as
  mouse — so under `RUN_SPECIES[w.run] == "M.musculus"` it would have been handed 16 GB and a 3 GB index
  reserve, reproducing the exact `not enough memory for BAM sorting` failure that `--limitBAMsortRAM` was
  added to prevent. `workflow/Snakefile` now derives `LARGE_GENOME` from `star_sa_index_nbases >= 14`
  (currently `{C.griseus, M.musculus}`). **Do not reintroduce a species-name test for a resource.**
- **`launch.py` crashed instead of skipping** when a species had no fold assignment. It special-cased
  `exp.species == "S.pombe"`, which was fine while pombe was the only fold-less species; `C.griseus` and
  `S.moellendorffii` made it a hard `KeyError` that killed the whole launcher. It now catches
  `n_folds()` generically, and the message from `experiments.py` is no longer pombe-specific.

`build_experiment_config.py` also lost its local `FASTA` and `ASSEMBLY` dicts, which were a second copy
of three fields already in `config/genomes.yaml` and had to be edited in lockstep on every species
addition. It reads `genomes.yaml` at runtime now.

### Still open on the new data

- **`C.griseus` and `S.moellendorffii` have no folds yet** — see "Cross-validation splits" below.
- **`S.moellendorffii` is the species most likely to hit the PINTS `bw_pl and bw_mn should have the same
  chromosomes` failure**, because `main_chromosomes` is 189 scaffolds and sparse ones can have reads on
  one strand only. Remedy is in `config/genomes.yaml`: raise the length cutoff (>= 500 kb keeps 118
  scaffolds and 87.7% of the assembly) rather than dropping the species.
- **`P.patens` and `C.griseus` rDNA are unresolved** (see the rDNA table). Neither has a reference
  sequence available to use as a sink, so both are `null`; C. griseus is the one to watch, being a rodent
  like mouse.

## Are any experiments replicates of each other? Should any be merged?

Asked and answered 2026-08-30 over all 38 experiments then, and revised once on 2026-09-01.
**Conclusion: merge nothing further, and UNMERGE the mouse liver pair by sex** — done, taking the repo to
40 experiments. Two different questions hide in "replicate", and they have different answers.

**Within an experiment, replicates are already pooled, and correctly.** `merge_runs` pools the BAMs of
every n>1 experiment before `genomecov -5`, which is arithmetically identical to summing per-replicate
count tracks. Checked against the manifest's own `replicate_group` column: **every experiment now maps to
exactly one `replicate_group`**, and no group is split across two experiments.

**The mouse liver experiments were the one exception, and were split by sex on 2026-09-01.** They had
pooled `liver old female` + `liver old male`, and the young pair — collapsing sex within age, which
overrode the manifest's own grouping. Now four experiments:

| experiment | runs | archive reads |
| --- | --- | --- |
| `M.musculus-liver-old-female_ChROcap` | `SRR12513902`, `SRR12513903` | 96.9 M |
| `M.musculus-liver-old-male_ChROcap` | `SRR12513904`, `SRR12513905` | 102.6 M |
| `M.musculus-liver-young-female_ChROcap` | `SRR12513906`, `SRR12513907` | 115.5 M |
| `M.musculus-liver-young-male_ChROcap` | `SRR12513908`, `SRR12513909` | 105.3 M |

Three reasons it was the right way round. The manifest records sex as a distinct `replicate_group`, so
pooling was overriding curated evidence. Mouse liver is one of the most strongly **sex-dimorphic**
transcriptional programs known — driven by growth-hormone pulsatility — so pooling averaged over a bimodal
program rather than over noise. And depth never justified it: each sex carries ~97–116 M archive reads and
~84 M signal, deeper than every experiment in the corpus except GCB.

**The split cost no re-alignment**, which is the general point about this pipeline's shape: run-level
intermediates are keyed by RUN, so only `merge_runs` and downstream change. `align` stayed at 59 jobs.

The mechanism is `SEX_SPLIT_PROJECTS` in `build_experiment_config.py`: a project listed there gets `sex`
appended as a fifth field of the `experiment_key()` tuple. The key stays a 4-tuple elsewhere rather than
carrying an empty fifth field for the 36 experiments where sex is meaningless (cell lines, pooled
embryos) — an empty field only invites someone to fill it in. Adding another sex-split project means
adding it to that set and giving its `EXPERIMENT_IDS` entries 5-tuples.

**The numbers to argue from live in `qc/stats/experiment_stats.tsv`.** Archive read counts alone cannot
support an exclude/merge decision -- a library can arrive deep and map badly -- so `src/qc/experiment_stats.py`
reports what the pipeline actually produced, per experiment:

| column | meaning |
| --- | --- |
| `archive_reads` | ENA `read_count`, summed over runs. What was deposited. |
| `input_reads` | STAR "Number of input reads". POST-trim, so below `archive_reads`. |
| `unique_reads` | STAR "Uniquely mapped reads number" -- exactly what the MAPQ 255 filter keeps. |
| `pct_unique` | `unique/input`. Low means wrong assembly, contamination, or unsplit spike-in. |
| `signal_reads` | reads in the merged BAM, i.e. what `genomecov -5` counts. **The number that matters.** |
| `peaks_total` | PINTS uni + bi, matching the training locus set (`divergent` excluded). |
| `reads_per_peak` | signal density, and **the basis of the `thin_coverage` flag** — the cross-species-comparable depth measure. See the note below on why absolute depth is not. |
| `pct_rrna` | rRNA + organellar share of the raw reads, from `src/qc/rrna_content.py`. Blank means NOT MEASURED, which is not the same as 0. |
| `pct_unique_adj` | `unique / non-rRNA input`. **This is the mapping-quality number**; `pct_unique` is not. |
| `qc_flags` | comma-joined `FAIL:`/`WARN:` findings. **Advisory only — nothing is excluded on them; see below.** Note the `low_mapping(N%,adj)` flags contain a comma themselves, so naive splitting on `,` breaks them; new flags should avoid commas. |

**`pct_unique` is not a quality metric, and reading it as one produced three wrong verdicts.** Where the
rDNA array sits in the assembly in 2+ near-identical copies, every rRNA read is a multimapper, gets
MAPQ ~3, and is correctly dropped by the `-q 255` filter — so the ceiling on `pct_unique` is set by the
organism, not the library. Measured with `src/qc/rrna_content.py`:

| experiment | rRNA + organellar | ceiling | observed `pct_unique` | `pct_unique_adj` | residual |
| --- | --- | --- | --- | --- | --- |
| `S.cerevisiae_PROcap` | **70.3%** | 29.7% | 16.3% | **54.8%** | +13 pts |
| `S.pombe_PROcap` | 46.1% | 53.9% | 42.2% | **78.2%** | +12 pts |
| `C.reinhardtii-liquidculture_5GRO` | 39.1% | 60.9% | 12.6% | 20.7% | **+48 pts** |
| `P.patens-plateculture_5GRO` | 47.4% | 52.6% | 14.6% | 27.8% | **+38 pts** |

On the raw number the first three were `FAIL:very_low_mapping`, which said little more than "this organism
has rDNA in its assembly". On the adjusted number the two yeasts raise no mapping flag at all, and the two
plants still do — correctly, because their residual is far too large for rRNA to explain. Every mapping
flag carries a `,adj` or `,raw` suffix recording which basis it used.

**These are the pipeline's own per-experiment numbers, and they replace an earlier version of this table
whose rows were single-run probes labelled as libraries.** It mattered for one row.
`C.reinhardtii-liquidculture_5GRO` was recorded at 67.4% rRNA and 33.1% adjusted, both measured on
`SRR24798065` alone; the experiment pools two runs differing ~5x in depth, and at experiment level it is
39.1% rRNA and **20.7% adjusted — below `VERY_LOW_MAPPING_PCT`, so it is now the corpus's one
`FAIL:very_low_mapping`.** Its residual is 48 points, not the 22 the old row implied. The other three rows
moved by under 2.5 points. **Quote `qc/stats/experiment_stats.tsv`, not a one-run spot check**, whenever
an experiment has more than one run of unequal depth.

Two measurement details that matter, both learned the hard way. rRNA features are type **`rRNA`**, not
`rRNA_gene` — Ensembl files them under `ncRNA_gene`, and filtering on `rRNA_gene` returns nothing for
S. pombe chromosome III. And the 35S components are **merged with a 5 kb gap tolerance** so the span
covers the whole unit *including ITS1/ITS2*: a nascent assay reads the precursor, and the commonest read
in the C. reinhardtii library spans the 5.8S/ITS2 junction. Cross-checked against an independent
coordinate-based count on S. cerevisiae — 70.4% at k=20 with ITS, 63.6% at k=24 without.

**`rdna_regions` in `config/genomes.yaml` makes the UCSC species measurable.** refGene GTF types
everything exon/CDS/transcript and carries no `rRNA` feature, so `D. melanogaster` and `C. elegans` had no
rRNA source at all — and once `organelle_contigs` was added they scored 0.88% and 0.54%, **organellar
only, with nuclear rRNA silently unmeasured but looking like a real number**. Their arrays are in-assembly
and their coordinates are already in the rDNA table above, so they are now configured directly:
`chrUn_CP007120v1` for dm6 and `chrI:15062083-15071033` for ce11. M. musculus needs none — its sink
already serves — and the seven Ensembl species are covered by their GFF3.

**C. reinhardtii's rRNA is measured from `rdna_regions` now, not from annotation, and the coordinates
were derived on the new assembly.** Its `pct_rrna` came from v5.5's **GFF3 `rRNA` features**, and
ASM4749649v1 has no annotation at all — so with `rdna_regions: []` the only thing left to score against
was `data/decoy/`, i.e. the two organelles, giving an **organellar-only ~12%** where the previous figure
was **39.1% rRNA + organellar**. That is exactly the false negative `rdna_regions` exists to prevent for
dm6 and ce11: a number that looks measured while nuclear rRNA is silently absent, which `rrna_indexed`
cannot catch because the entry *is* indexed. And it is load-bearing here, because `pct_unique_adj`
**20.7%** — the corpus's one `FAIL:very_low_mapping` — was computed from that 39.1%.

**The array is LOCALIZED TO CHROMOSOMES in ASM4749649v1, so no sink is needed** (the rule is sink only
where the array is missing) — measured 2026-09-06 with the repo's usual k-mer method: 29 30-mers tiled
across an 18S sequence (`JN903984.1`) plus a 5.8S sequence (`PX737393.1`), both strands, against the
assembly.

| contig | chromosome | array | hits |
| --- | --- | --- | --- |
| `CM104932.1` | chr_14 | 4,123,744-4,155,420, to the terminus at 4,156,167 | 79 18S + 84 5.8S |
| `CM104926.1` | chr_08 | 4,583,746-4,602,203, to the terminus at 4,602,485 | 45 18S + 50 5.8S |

Both subtelomeric, reproducing v5.5's picture. `rdna_regions` is set to those two spans extended to each
contig end. Two dispersed partial copies are deliberately excluded — `CM104926.1:~2,937,548` and
`CM104930.1:4,130,938-4,131,477` (539 bp) — as degenerate fragments whose flanks are not rRNA; they are
also the reason an in-assembly array matters, since reads from them score better against the true array
than against the fragment.

**Note `chrI`, not `I`.** The rDNA table above writes C. elegans' array as `I:15062083-15071033`, which is
the WormBase name; ce11 is chr-prefixed, so the bare form would have indexed **nothing** and reported 0%
rRNA as though measured. `rrna_content.py` therefore **raises** on a configured region whose contig is
absent from the FASTA, listing the contigs it did find — a config typo should fail, not degrade.

**`rrna_indexed` distinguishes "measured 0%" from "never measured".** Adding `organelle_contigs` made
`n_scored > 0` for the species with no rRNA source, which populated `pct_rrna` with an organellar-only
figure and destroyed the blank-means-unmeasured signal the column depends on — so `pct_unique_adj` was
computed from it. `experiment_stats.rrna_pct()` now returns `(entry_found, value)` rather than a bare
value, because "the rrna output says there is no rRNA source" has to be distinguishable from "there is no
rrna output"; conflating them let a stale organellar number resurrect through the `--combine` fallback.

**`qc/rrna` and `qc/reads` ARE declared `stats_table` inputs, after the opportunistic version failed in
an instructive way.** The argument for leaving them out was that declaring them forces the cheap
`snakemake stats` target to depend on the ~202 GiB of raw FASTQ, with the `,raw`/`,adj` suffix as the
safeguard. The suffix held, but the outcome was still wrong: nothing ordered the two, so `stats_table` ran
before `rrna_content` finished for **one of 38** experiments, and `C.griseus-BMDM_GROcap` alone was judged
on the raw rate while the other 37 were adjusted. A wholly unadjusted table would have been obvious; one
row silently on a different basis is not. **Per-row races are worse than a coarse dependency.**
`experiment_stats.py --combine` still reads both directories opportunistically, so a table can be rebuilt
by hand from whatever survives.

**But the fix that ordered them also broke the table, because the rule ran `--combine {input}`.**
Naming the two new input groups was only half the job: `{input}` expands to *every* input, so
`--combine` was handed the rrna TSVs, the reads TSVs and **this rule's own script** as well as the 42
per-experiment TSVs. `csv.DictReader` absorbed all three silently — it takes whatever the first line of a
file offers as `fieldnames` — so the committed table was **685 rows over 42 experiments** with every
pipeline column blank: one duplicate row per rrna file carrying only `pct_rrna`, one row per *run* from
the reads files, and **one blank row per line of Python** from `experiment_stats.py`. The markdown sorts
by species, and a row with no species sorts first, so the file opened on ~500 empty rows.

Two changes, because either alone would have left the trap:

- The rule's inputs are **named** (`stats`/`rrna`/`reads`/`script`) and the shell passes an explicit
  per-experiment glob, with the two directories passed as `params` instead. **Never `--combine
  {input}`.**
- `--combine` now **rejects any file whose header is not exactly `COLUMNS`**, naming the file and both
  column counts. A too-broad glob has to fail, not average out — the whole failure was that a
  plausible-looking table hid it.

**That header check is now scoped by NAME, because on its own it fires on the wrong things — and
misses one.** `--combine`'s row set is `experiment_config.yaml`, not the directory it globs:

- A TSV **naming no configured experiment is skipped with a warning.** Once an experiment is renamed or
  split, no rule has its wildcards, so nothing can ever rewrite its file and it is frozen at whatever
  schema it had. Both failure modes were observed from the *same* pre-sex-split pair:
  `M.musculus-liver-old_ChROcap.tsv` froze at **18 columns** and hard-failed `stats_table` during a
  two-experiment `--config experiments=C.reinhardtii…,P.patens…` re-map that had nothing to do with
  mouse liver, while `liver-young` froze at the current 21, passed the check, and **silently produced a
  44-row table over 42 experiments** — the worse of the two, since it reads as complete. Delete such
  files; the warning says so.
- A **configured** experiment with a wrong header still **exits nonzero**, and now says it is stale
  output with the command to rebuild it. That is also what still catches `--combine {input}`: the rrna
  and reads TSVs *are* named by experiment, so they hit this branch rather than the skip.
- **Zero rows exits nonzero too.** Skipping by name means a `--combine` pointed somewhere entirely wrong
  no longer fails on a header, and an empty global table reads exactly as complete as a 40-row-short one.

Note this exposure is a consequence of `aac86f6` globbing the whole directory rather than taking
`TARGETS` — right for the row set, but it puts every stale file in the corpus in scope of every subset
run. `report_flags`'s survey-schema check is scoped the same way, to the files
`read_survey_flags` can actually consult (`qc/reads/{exp}.tsv` for a configured experiment, plus the
corpus-wide `read_structure.tsv` fallback); orphaned survey files were emitting warnings for
experiments that no longer exist, which is noise in the one place whose value is that a warning means
something.

Also `--combine`'s sort key is `str()`-wrapped now, matching `--all`. It was the only reason the garbage
rows did not crash on a `None`-vs-`str` comparison, i.e. the one thing that made the corruption survivable
enough to be committed.

**`stats_table` combines the DIRECTORY, not `TARGETS` — its scope is deliberately decoupled from its
inputs.** The rule's inputs are scoped to `TARGETS`, but its output path,
`qc/stats/experiment_stats.tsv`, is global and unscoped. While it also passed `--combine {input.stats}`,
those were the same set, so `--config experiments=A,B` rebuilt the one global table from two rows and
dropped the other 40 — found while re-running just the two updated genomes (P. patens V7,
C. reinhardtii ASM4749649v1), where a 50-job DAG quietly included `stats_table`. **A subsetting flag must
not narrow a global output.** The inputs still answer "when must this rerun, and after what" (the
`rrna_content` race above); the glob `'{params.per_experiment}/*.tsv'` answers "what belongs in the
table". It is **single-quoted so Python globs it, not bash** — `experiment_stats.py` globs any `--combine`
argument containing `*`, and letting the shell expand it would resolve against whatever the subset left on
disk. The residual cost is the reverse staleness gap: under a subset, a per-experiment TSV *outside*
`TARGETS` that changes will not retrigger the rule. That trade is deliberate — a table one row stale is
repairable with `--forcerun stats_table`, whereas a table 40 rows short reads as complete.

**`umi_report.py` was checking the WRONG MATE, and it was the last place the old 3'-adaptor assumption
survived.** It hardcoded `expect = declared if (not paired or mate_i == 2) else 0`, with a comment
asserting "for a 3' adaptor UMI that is R2, which is what fastp is told (--umi_loc read2)" — the exact
claim the Spt5 investigation overturned. The pipeline was corrected to `5' adaptor -> read1`; this report
was not, so for all six Spt5 runs it expected 10 nt on R2, found none, and printed
`MISMATCH (manifest says 10)` on R2 while R1 read `no UMI signature`. **Both lines were artifacts of the
report, not findings about the data.** It now resolves the mate through the same
`steps.dedup.umi_locations` table both drivers hand to fastp, and raises on an unmapped prose value rather
than defaulting. Verified: `5' adaptor -> mate 1`, `3' adaptor -> mate 2`, and `declared` now sits on R1
for Spt5.

**But the `detected` column is BLIND for most of this corpus, so do not re-fetch FASTQs to populate it.**
`candidate_len` counts the leading run of positions whose per-base entropy is at or above
`UNIFORM_BITS = 1.95` and needs `MIN_RUN = 4`. A uniform random UMI is 2.000 bits, and random genomic
sequence at *f* GC is `H(f)` — so the signal to resolve is `2.000 - H(f)`, which is tiny wherever base
composition is near-even:

| species | %GC | genomic H | vs 1.95 | can the screen see a UMI? |
| --- | --- | --- | --- | --- |
| P. patens | 33.4 | 1.9190 | −0.031 | yes |
| G. arboreum / G. hirsutum | 33.5 / 34.5 | 1.920 / 1.930 | −0.030 / −0.021 | yes |
| C. elegans, S. pombe, A. thaliana, C. reinhardtii | 35-36 / 64 | 1.938-1.943 | −0.012 to −0.007 | marginal |
| S. moellendorffii | 37.5 | 1.9544 | **+0.004** | **no** |
| **S. cerevisiae** | 38.2 | 1.9594 | **+0.009** | **no** |
| C. griseus, M. musculus, D. melanogaster | 41.5-41.8 | 1.979-1.981 | **+0.029 to +0.031** | **no** |

For S. cerevisiae the target signal is **0.041 bits** against observed position-to-position scatter of
**~0.13 bits**, and genomic entropy (1.959) is *above* the 1.95 threshold — so genomic sequence itself
counts as "random" and whether a position passes is sampling noise. That is exactly what happened: the
Spt5 R1 profile reads `1.98 1.93 1.90 1.95 …`, the run breaks at position 1, and the verdict is
`no UMI signature` for a library that demonstrably has a 10-nt UMI.
**So a `-` or a `0` in `detected` is not evidence against a declared UMI here.** The real evidence for the
Spt5 UMI is the genomic k-mer offset test recorded above, not this screen. And the screen's *primary*
purpose — catching an UNDECLARED UMI — is unavailable for 6 of 12 species including all three of mouse,
fly and hamster, which leaves the default-deny policy resting entirely on manifest curation with no
working automated backstop. Fixing it properly needs a different statistic (the k-mer offset test, or
per-position base *composition* against the genome's own rather than against uniform), not a threshold
tweak.

**Fixing the mate requires no re-mapping, but it does re-run against the raw FASTQs.** `umi_report.py` is pure
reporting — its only output is `qc/umi/{exp}.tsv`, consumed by nothing but the `all` and `qc` targets, and
the pipeline's UMI handling never came from it. No BAM, bigWig, peak or negative changes. But the script
is a declared `input:` of the `umi_report` rule, so editing it re-runs that rule for all 42 experiments,
and the rule's other inputs are the **raw FASTQs**. Where those have been deleted, `snakemake qc` will
try to re-fetch them. Run `python src/data_preprocessing/umi_report.py -e <exp>` standalone instead if
the FASTQs are gone and only the table is wanted.

**A handled interleaved deposit is not a defect, and `declared_interleaved` in the survey TSV is what says
so.** `M.musculus-GCB_PROcap` was reported `FAIL:interleave_suspect` after the re-map even though its
interleaving is declared and deinterleaved — the per-file printout said "(already declared)" but the
column did not, so the flag had no way to know. The screen still fires on it, which is useful as a
self-test that the check works; only undeclared suspicions become flags.

**`long_for_project` cannot fire under the DAG when a project spans two experiments.** The rule runs
`read_structure_qc.py -e {exp}`, so project-mates in another experiment are never in the same invocation —
which is precisely GCB, whose only project-mate is priB. There it is caught by `polyg` alone, which is why
having two independent tells mattered. Run the standalone sweep for the full screen:
`python src/qc/read_structure_qc.py --tsv qc/reads/read_structure.tsv`.

**`signal_reads` reports NOT MEASURED (blank) rather than 0 when no contig matches, and its fallback is
SAMPLE-keyed.** Two defects that combined to print `signal_reads 0` for
`C.reinhardtii-liquidculture_5GRO` and `P.patens-plateculture_5GRO` beside 9,216 and 9,833 called peaks —
which cannot both be true, and which reads as a dead library rather than as a measurement failure:

- `mapped_reads()` skipped every contig outside `main_chromosomes` and returned the running total, so a
  BAM from a *different naming regime* — i.e. an older assembly still on disk — was indistinguishable
  from a library with no usable reads. It now returns `None` and names the file. Same rule `pct_rrna`
  already follows: blank is NOT 0. A library genuinely carrying zero reads on its main chromosomes would
  also have no peaks, so blank is the honest cell either way.
- The `merged.bam` fallback still probed the **pre-refactor `runs/{run}/final.bam`** after alignment moved
  to one BAM per sample. That is worse than finding nothing, because those files are **not `temp()`** and
  survive a re-map, so on a re-mapped tree it read BAMs aligned to the PREVIOUS assembly. It now tries
  `samples/{sample}/final.bam` first and falls back to the run-keyed path only for a tree that predates
  the refactor. `star_logs()` was taught both layouts for exactly this reason; this was the counterpart it
  missed.

`signal_reads` sits below `unique_reads` by dedup (UMI libraries only, 3 of 40) and by the one-mate filter
(paired libraries only, 5 of 40). For paired libraries it counts one mate per fragment, since that is what
`final_bam` keeps.

Built by two rules -- `experiment_stats` per experiment and `stats_table` to aggregate -- and included in
both `all` and `qc`, plus a standalone `stats` target. Counting is essentially free: `samtools idxstats`
reads the BAM index, not the reads. `experiment_stats` takes `merged.bam` as an explicit input so the count
happens while that `temp()` file still exists, and falls back to summing the per-run `final.bam` files if it
has already been cleaned up. `align` now declares STAR's `Log.final.out` as an output for the same reason:
the BAM it sits beside is `temp()`, and the mapping rates have to outlive it.

`qc/` is gitignored, tables included, and that is deliberate: `qc/stats/experiment_stats.{tsv,md}` are
outputs of `stats_table`, so a committed copy makes Snakemake report **"Nothing to be done"** and never
rebuild them. A tracked table that looks authoritative but is stale is worse than none. (This was tried
briefly and reverted -- the committed copy had every pipeline column blank.) Read the tables where the
pipeline ran, or refresh with `--all`.

Blank pipeline columns mean that experiment has not been mapped yet.

**Across projects, several experiments are the same biology — and they should stay separate.** The
clusters, from `(species, biological_material)` with conditions inspected:

| cluster | experiments | why not merge |
| --- | --- | --- |
| *D. melanogaster* S2 | `S2_PROcap` (Kwak2013, 47.8 M), `S2_5GROcap` (Duttke2017, 61.1 M), `S2-LacZKD_PROcap` (LacZ, 27.3 M) | three labs, three assay chemistries, and LacZ-KD is a knockdown control rather than untreated |
| *M. musculus* BMDM | `BMDM_PROcap` (Kim2018, 26.0 M), `BMDM_GROcap` (Link2018, 25.5 M), `BMDM_5GRO-ctl` (Lam2013, 28.8 M) | PRO-cap vs GRO-cap vs 5'GRO, three labs; Lam2013's "control" is a biotin-tag control |
| *S. cerevisiae* unperturbed | `S.cerevisiae_PROcap` (Booth2016 W303a, 42.3 M), `Ino80ctl_PROcap` (62.9 M), `Spt5EtOH_PROcap` (66.5 M) | different strain backgrounds (W303a vs Ino80-AID vs Spt5-AID) and different vehicle treatments |

Three reasons to leave them alone:

1. **Depth does not force it.** Every one of these is 25 M+ reads on its own, so merging buys little.
   (The libraries that *are* thin are single ones with no partner to merge with: hamster
   `Liver_GROCap1` at 5.4 M, C. reinhardtii 5'GRO r2 at 9.2 M, Lam2013 RevErb at 10.9 M. At the other
   extreme `GCB_B18hi_cap` is a single 404.6 M-read run — verified 1:1, not a crosswalk error.)
2. **Merging pools batch with biology.** Different cap-selection chemistry means different 5'-end
   capture bias; different labs mean different batch effects. Uniform re-mapping fixes the *assembly*
   differences, not the assay ones.
3. **Separate is more useful.** These clusters are the only cross-assay, cross-lab held-out sets in the
   repo — train on Kim2018 BMDM PRO-cap, evaluate on Link2018 BMDM GRO-cap. Merging destroys the one
   honest generalisation test available and buys nothing measurable.

This also matches the project's stated design (one experiment == one species x one condition == one
model, no multi-tasking) and the workbook's own Field Guide rule for `replicate_group`: *"Assess
replicate concordance before pooling; do not combine distinct conditions as replicates."*

## Assay provenance: swept all 64 runs, no non-cap assay is a target

Checked 2026-09-06, after the GSE233927 csRNA-seq/5'GRO-seq question above, because that series proves a
deposit can hold both and nothing structural stops the wrong sample being picked. Method: every GEO series
(16 of them) pulled as a `targ=gsm` dump and each manifest row's `sample_accession` matched against its
own `!Sample_title`/`!Sample_description`, plus ENA `experiment_title` for the 9 non-GEO rows (4
ArrayExpress, 4 CNCB cotton, 1 run-only).

**Result: every one of the 55 GEO samples and 9 non-GEO samples names a cap-selected initiation assay** --
PRO-cap, GRO-cap, 5'GRO-seq, ChRO-cap, CoPRO, or Spt5's `CAP_*`. No csRNA-seq, sRNA-seq, total RNA-seq,
PRO-seq or non-cap GRO-seq sample appears as a target. The 5 TAP-/noTAP rows are all
`target_use=control`. Two names that look wrong and are not: the fly embryo rows are
`PROseq_PROcap34h1`-style submitter names, but ENA's `experiment_title` is "PRO-cap in Drosophila
melanogaster embryo" with **`library_selection: CAGE`**, and PRJEB25091 holds only those 4 runs, so there
is no PRO-seq there to leak.

**The load-bearing finding is that NO ARCHIVE FIELD can verify cap selection for this corpus, so the
manifest's `assay_label`/`cap_status` is the only record** -- the same default-deny situation as
`umi_len`/`umi_loc`. Measured across all 64 runs:

| field | values |
| --- | --- |
| `library_strategy` | **`OTHER` for 51/64**; wrongly `RNA-Seq` for Lam2013's 3 (GEO agrees, titles say 5'GRO-seq); `miRNA-Seq` for cotton's 4 |
| `library_selection` | **`other` for 51/64**; `CAGE` for the 4 fly embryo runs; `cDNA` for Lam2013's 3 |

So `library_strategy` is `OTHER` for csRNA-seq and 5'GRO-seq alike (see the GSE233927 note) *and* for
almost everything else here. Re-run the sweep by title, never by strategy, and treat a new dataset's
assay as curated-until-proven rather than archive-verified.

## Nothing is excluded: every dataset is analysed and modelled

**Standing decision, 2026-09-03, and it generalises every "should we drop X?" question below.** Model
every experiment, including the problematic ones, and QC at the end. The reason is structural rather than
optimistic: **one experiment == one species x one condition == one model**, with no multi-tasking and no
shared heads, so a weak dataset cannot contaminate a strong one. There is nothing to protect by excluding
it in advance, and a trained model is *better* evidence about a library than a pre-hoc read count is.

So `tier` gates preprocessing only, `launch.py` deliberately does not filter on it or on `qc_flags`, and
proposals to add an `exclude` tier have been declined. The flags exist to tell you which numbers to
distrust when reading results, not to decide what runs.

### Depth requirements scale with the nascent transcriptome, NOT with a constant

**The depth a library needs is proportional to the size of the transcribed space being sampled.** An
organism with a small genome, few distal elements and little intergenic transcription reaches the same
coverage per initiation site on far fewer reads than mouse or human. Ranking libraries by raw
`signal_reads` across species therefore penalises the compact genomes for being compact — it measures the
organism, not the library.

This was not a hypothetical: `experiment_stats.py` carried `SHALLOW_SIGNAL_READS = 10_000_000`, one
absolute threshold applied to all twelve species, and it was **measurably wrong in both directions**:

| | signal | reads/peak | old flag |
| --- | --- | --- | --- |
| `S.pombe_PROcap` | 23.1 M | **2,513** | none — and it is the best-sampled experiment in the corpus |
| `M.musculus-GCB_PROcap` | **150.4 M** | 2,325 | none |
| `S.cerevisiae_PROcap` | 4.9 M | 721 | **SHALLOW** — false positive; matches CHO's 723 on 1/10 the reads |
| `C.reinhardtii-liquidculture_5GRO` | 6.3 M | 770 | **SHALLOW** — false positive |
| `M.musculus-BMDM_5GRO-ctl` | 18.2 M | **450** | **none** — false negative |
| `D.melanogaster-S2_5GROcap` | 22.2 M | **518** | **none** — false negative |

S. pombe on 23 M reads is better sampled than mouse on 150 M. Two libraries above 18 M were covering
their much larger transcriptomes more thinly than several flagged ones and escaped silently.

Replaced with `THIN_COVERAGE_READS_PER_PEAK = 500` and a `WARN:thin_coverage(N/peak)` flag. `peaks_total`
is this pipeline's own measure of how much transcribed space exists, so dividing by it asks how deeply
each initiation site is covered. It is **not fully depth-independent** — peak calling saturates, so a
shallow library calls fewer peaks and shrinks its own denominator — but it errs conservatively, since
peaks fall more slowly than reads. The threshold is a heuristic recalibrated from this corpus with no
clean gap in the distribution, the same provisional status as the FLAT initiator threshold; do not treat
500 as principled.

Net effect on the real corpus: 10 flagged before, 8 after. `M.musculus-BMDM_5GRO-ctl` gained a flag it
should always have had; `S.cerevisiae_PROcap`, `C.reinhardtii-liquidculture_5GRO` and
`C.griseus-BMDM-KLA1h_GROcap` lost ones they never deserved.

## Should the S. cerevisiae perturbation experiments be dropped?

Asked 2026-09-03, on the grounds that Ino80/Spt5 depletion produces massive widespread transcription and
that even their controls look poor. **Answer: no — keep and train them.** The evidence points the other
way, and the framing has the yeast libraries ranked backwards.

**The extra peaks carry the strongest initiator motif in the corpus's yeast set.** Ranked with every other
S. cerevisiae experiment:

| experiment | %rRNA | %uniq_adj | Inr bits | offset | orientation flags |
| --- | --- | --- | --- | --- | --- |
| `Spt5IAA1h` | 17.8 | 88.7 | **1.18** | −1 | 0 |
| `Spt5IAA4h` | 4.5 | 79.8 | **1.18** | −1 | 0 |
| `Ino80KD` | 8.9 | 77.2 | 1.09 | −1 | 0 |
| `Ino80ctl` | 14.0 | 82.4 | 1.00 | −1 | 0 |
| `Spt5EtOH` | **49.7** | 82.4 | 0.61 | +0 | 0 |
| `S.cerevisiae_PROcap` (Booth WT) | **70.3** | **54.8** | **0.45** | −1 | 0 |

A depleted sample calling 23,642 peaks at 1.18 bits is not calling noise — that is a stronger, correctly
placed Inr than any other yeast library here. Adjusted mapping is 77-89% across all five; the raw 41.4%
for `Spt5EtOH` is entirely its rRNA. **Dropping this study would remove the best yeast data and leave the
worst**, since the independent Booth WT baseline is the weakest S. cerevisiae library in the corpus on
rRNA, depth and motif alike.

**And they are not the corpus's worst libraries — not close.** Ranked worst-first on rRNA, adjusted
mapping, depth and motif together, the six below all sit beneath `Spt5EtOH`, and the four Ino80/Spt5
depletion experiments sit in the better half:

| experiment | %rRNA | %uniq_adj | signal | Inr |
| --- | --- | --- | --- | --- |
| `P.patens-plateculture_5GRO` | 47.4 | 27.8 | 2.4 M | 0.39 |
| `C.reinhardtii-liquidculture_5GRO` | 39.1 | **20.7** | 6.3 M | 0.23 |
| `S.cerevisiae_PROcap` | **70.3** | 54.8 | 4.9 M | 0.45 |
| `S.moellendorffii-stemleaf_5GRO` | 26.1 | 36.8 | 6.9 M | 0.78 |
| `M.musculus-BMDM_GROcap` | 30.7 | 84.2 | **4.0 M** | 0.42 |
| `C.griseus-BMDM_GROcap` | 15.5 | 49.7 | 5.3 M | 0.65 |

**Do not rank on Inr bits alone.** It is diluted by peak-set size, so `G.hirsutum` (0.11 over 171,640
peaks) and `D.melanogaster-S2_5GROcap` (0.13 over 42,854) score low while being fine — both correctly
placed at +0. Bits are only interpretable against a comparable peak count.

**The real caveat is narrower than "these experiments are bad".** It is that *within-study, cross-condition*
comparison in the Spt5 series is confounded: the vehicle control is 49.7% rRNA against 4.5% at IAA 4 h, an
11-fold difference in cap-selection quality running in the same direction as the peak counts
(10,811 -> 21,591 -> 23,642). That gradient is what Spt5 loss should do biologically, and it is also what
differing library quality would do, and these data cannot separate them — note the rRNA difference runs
*opposite* to the biology, since depleting Pol II elongation should raise the rRNA fraction, not cut it by
11-fold. A depth-matched subsample would settle it.
**That confound does not touch a per-experiment model**, which is all this repo builds — one experiment,
one model, no multi-tasking. So: train them, and deprioritise any analysis that reads *across* the Spt5
conditions. The `tier` column already encodes this — controls `include`, perturbations `conditional`.
The Ino80 pair is much better matched (14.0% vs 8.9% rRNA, 35.4 vs 33.3 M signal) and shows almost no
peak-count difference, which is itself a useful negative result.

## Cross-validation splits

Convention everywhere: **test = fold `i`, validation = fold `(i+1) % n_folds`, train = the rest.**

Two split mechanisms:

- *Chromosome-level* (default): whole chromosomes held out, passed to `extract_loci(chroms=...)`.

**`config/chrom_splits.yaml` is the single source of truth.** The per-species CSVs under `config/splits/`
are *derived* from it by `config/write_split_csvs.py`; regenerate rather than hand-edit, and
`python config/write_split_csvs.py --check` fails on drift.

**`--check` also verifies fold members against the real contig names now, which it did not before.**
Comparing the YAML to the CSVs it generated could never catch the failure that actually matters:
`extract_loci` matches literally, so a readable-but-wrong name yields **zero loci in silence** — the same
class as an exclusion list that excludes nothing, and the risk `chrom_style` exists to record. Three
comparisons per species: fold members against `<fasta>.fai`; fold members against `main_chromosomes` in
both directions (a member outside the allow-list can never carry a peak, and a main chromosome in no fold
is silently untrained); and `main_chromosomes` against `{work}/genome/<species>.chrom.sizes`, which is
what `bedGraphToBigWig` and PINTS actually saw.

**A missing `.fai`/`chrom.sizes` is skipped, not failed** — this has to stay runnable where `data/` is
empty — but the summary line then says naming is **UNVERIFIED** and names the skipped species rather than
printing a clean bill. Run it where the data lives; off-cluster it checks the YAML/CSV agreement only.
Verified against fixtures on all three failure modes: a wrong name, a main chromosome in no fold, and the
clean case.

**It also names the species that have no chromosome folds at all**, which are invisible to every check
above and would otherwise let an all-green report cover 10 of 12 species. S. pombe and S. moellendorffii
are absent from `chrom_splits.yaml` *by design* — but "absent by design" and "absent and not built yet"
look identical from there, and the second means the species cannot train. So the check distinguishes them
by whether `config/splits/{species}_random_fold_assignments.csv` exists, and says
`NO FOLDS AT ALL ... cannot train` when it does not. It does **not** fail on that: building the CSV needs
peaks, so its absence is a to-do rather than drift. The CSVs are a legacy artifact of the
pre-unification split; `src/experiments.py` reads the YAML, so the CSVs now exist only for external
consumers and could be dropped once nothing outside this repo reads them.

**Fold assignments are the lab's own, and several are reused UNCHANGED from earlier projects** —
`A.thaliana`, `D.melanogaster`, `M.musculus`, `S.cerevisiae`, `C.reinhardtii` and `P.patens`. Do not
"improve" a reused entry: the only thing it buys is that a locus in test here is in test everywhere else
we train, and a divergence destroys that silently. **Folds are assigned by manually matching peak counts across folds — not sequence length.** bp totals are
not expected to match and must not be "rebalanced": a fold can be tight in bp and badly skewed in loci, and
loci are what affect training. Check any assignment against real peaks with
`python config/write_split_csvs.py --peak-counts -e <experiment>`, which also flags chromosomes present in
the peaks but absent from every fold (a naming mismatch symptom).

**ENCODE publishes no mouse folds.** chrombpnet ships only `helpers/make_chr_splits/splits.py`, a formatter
that takes `--test_chroms`/`--valid_chroms` and writes train as the remainder — no assignment logic, no
committed fold files for any genome, nothing mouse-specific.

**There is NO public or authoritative source for ANY of these fold assignments — every one of them was
made in this lab.** That matters for how to treat them. They are not standards to be looked up and they
carry no external validation; their whole value is that the same assignment is used everywhere the lab
trains, so a model here and a model in another of our repos are comparable. Read "canonical" throughout
this file as "the shared lab assignment", never as "published". Anything reused from
Anything reused is reused for consistency with our own prior work, not for authority.

**Reuse checked 2026-08-30 for the species added then**, with the result that two of the four were
already assigned in earlier lab work and two were not:

| species | assigned before? | outcome |
| --- | --- | --- |
| C. reinhardtii | yes | reused unchanged **until the 2026-09 assembly swap REPLACED it — see below** |
| P. patens | yes | reused unchanged **until the 2026-09 assembly swap REPLACED it — see below** |
| S. moellendorffii | no | peak-level folds, permanently (no chromosomes exist); **built 2026-09-02** |
| C. griseus | no | chromosome-level; **assigned here 2026-09-01** from CHO peak counts |

**THE TWO PLANT ENTRIES ARE NO LONGER THE REUSED LAB ASSIGNMENT, and nothing said so — OPEN DECISION.**
The 2026-09 assembly swap rewrote both `chrom_splits.yaml` blocks onto the new chromosome names, and in
doing so replaced the peak-matched groupings with a mechanical **round-robin over chromosome index**
(`i % 5`), while the comment above them still claimed "Reused UNCHANGED from earlier lab work … same
peak-matched assignment". Measured against the previous entries:

| species | chromosomes whose fold changed | old fold sizes | new fold sizes |
| --- | --- | --- | --- |
| C. reinhardtii | **13 of 17** | 2, 3, 3, 3, 6 | 4, 4, 3, 3, 3 |
| P. patens | **24 of 26** | 4, 5, 5, 6, 7 | 6, 5, 5, 5, 5 |

Two things are wrong with that, in the terms this file already sets out. It **breaks the one thing reuse
buys** — "a locus in test here is in test everywhere else we train, and a divergence destroys that
silently" — so a locus held out in the earlier lab work is now training data here for most chromosomes.
And a round-robin is **not this project's method**: folds are matched on PEAK COUNTS, never on index or
length.

**Restoring the reused assignment is mechanical for C. reinhardtii**, because the assembly report gives
the mapping directly: `chr_01`-`chr_17` are `CM104919.1`-`CM104935.1`, i.e. chromosome *N* is
`CM(104918+N).1`, and Chlamydomonas chromosome numbering is standard across strains. For P. patens the
same carry-over needs the V3-number -> GWH-ID correspondence confirmed from the FASTA's `OriSeqID`
headers, and old fold 0's member `27` simply disappears (V7 resolves it as spurious). The alternative is
to re-derive both from peak counts on the new assemblies, per the project's actual method, and accept the
break with earlier work deliberately. **Either way it is a decision to record, not a rename to leave
implicit.**

The two cottons were added later and are the same story: no prior assignment, both
chromosome-level, and both were **assigned here 2026-09-02** from their own peak counts. All four
locally-originated entries — plus the retuned six-fold `C.elegans` — need pushing upstream before those
species are used outside this repo.

For both reused species the two copies we hold agree exactly, so there is one assignment rather than two
candidates. Note an earlier assignment also exists for `S.pombe` with **3** folds; this repo deliberately
does not use it, because peak-level folds are S. pombe's single mechanism here.

**GSE233927 has been touched before in this lab, but for a different assay in the same series.** Only the
genome URLs and the fold assignments carry over; this repo maps the series' **5'GRO-seq** FASTQs from
scratch rather than reusing anyone's processed tracks.

**One trap inherited from an older download script: Chlamydomonas moved Ensembl divisions.** It is under
**plants** as of release-63, not **protists**, so the old `protists/release-55` URL now 404s. The
assembly (v5.5) is unchanged, so the fold assignment is unaffected — only the URL. All eight new
FASTA/GFF3 URLs in `config/genomes.yaml` were verified to return HTTP 200 with real content lengths.

**`S.moellendorffii` has no chromosomes at all** — v1.0 is a 759-scaffold JGI draft with an *empty*
Ensembl karyotype (212.6 Mb, N50 1.75 Mb, largest scaffold 6.95 Mb). This is a stronger version of the
S. pombe situation: there are no chromosomes to assign, not merely too few. It uses peak-level folds via
`make_random_splits.py`, which was already species-agnostic (`--experiment`/`--output`) and whose
docstring now says so. `src/experiments.py` picks the result up automatically, since
`random_splits_path` has always been `{species}_random_fold_assignments.csv`.

**The script was `make_pombe_random_splits.py` until S. moellendorffii arrived** — renamed because the
name was the only pombe-specific thing left about it. Its *defaults* still target S. pombe
(`DEFAULT_EXPERIMENT = "S.pombe_PROcap"`), so a bare invocation behaves as before. The rename was safe to
do without a compatibility shim because no peak-level CSV exists yet, and the `generated_by` field it
writes into the provenance header is never compared by `--check`.

**`C.griseus` is assigned, chromosome-level, from peak counts — done 2026-09-01.** It was waiting on
peaks, because this project assigns folds by matching peak counts and never by sequence length; that is
exactly how the `C.elegans` entry below went wrong.

Counts came from `C.griseus-CHO_GROcap`, the deepest of the seven hamster experiments (65,575 peaks; liver
has 7,568). One assignment serves all seven, so the thin libraries are necessarily less balanced.

| fold | peaks | share | vs target | units |
| --- | --- | --- | --- | --- |
| 0 | 13,009 | 19.8% | −106 | `2`, `9` |
| 1 | 12,796 | 19.5% | −319 | `3`, `8`, `10` |
| 2 | 12,965 | 19.8% | −150 | `4`, `5` |
| 3 | 12,256 | 18.7% | −859 | `6`, `7`, `X` |
| 4 | 14,549 | 22.2% | +1434 | `RAZU02000001.1`, `RAZU02000002.1` |

Reproduce with `python config/write_split_csvs.py --peak-counts -e C.griseus-CHO_GROcap`.

**The two scaffolds SHARE a fold, because they are the two arms of the unassembled chromosome 1 and one
chromosome belongs to one fold.** That constraint costs balance and the cost is stated rather than hidden:
spread is 2293 peaks against a 13,115 target, where splitting the arms would give 699. Both figures are
provably optimal for their constraint — exhaustive search over every set partition of the units into 5
non-empty blocks, with a unique optimum in each case.

**Most of that spread is unavoidable, not a bad assignment.** Chromosome 1 alone holds 22.2% of all peaks,
so its fold exceeds a fifth of the data however the other 10 units are arranged. The floor given the
constraint is 1793 (14,549 against a perfectly even 12,756 elsewhere); the remaining ~500 is the lumpiness
of the other units. **Do not "rebalance" this by moving an arm.**

**This entry originates here** — there was no prior C. griseus assignment. Push it
upstream before using hamster beyond this repo, or a locus in test here becomes train there, which is the
`C.elegans` mistake repeated.

The 12 units are chromosomes `2`-`10`, `X`, **plus the two unplaced scaffolds `RAZU02000001.1` (275.7 Mb)
and `RAZU02000002.1` (274.4 Mb)**. Including scaffolds as fold members breaks the usual rule on purpose:
CriGri-PICRH-1.0 does not assemble chromosome 1, and those two contigs are almost certainly its arms.
Verified from both sides — Ensembl lists 10 karyotype entries with no `1`, and NCBI `GCF_003668045.3`
lists exactly 10 assembled molecules (`NC_048595.1`-`NC_048604.1`). Each is larger than mouse chr1 and
together they are 23% of the genome, so excluding them would discard all of chromosome 1. Same reasoning
as S. pombe's MTR/AB325691: judge the sequence by what it is, not by the label the assembly gives it.

**`C.elegans` is the one species on SIX folds, retuned 2026-09-01.** It originates here — no earlier
assignment existed — and it had been assigned by chromosome count rather than by matching peak
counts, the one entry in the file that did not follow the project's method.

Retuning exposed a structural problem rather than a mis-tuned split. Worm has 6 chromosomes of near-equal
peak count (3,294–3,726 in `C.elegans-embryo_GROcap`), so **any** 5-fold split must pair two of them, and
that fold then holds 31.4% of the peaks against 16.2% for the smallest:

| assignment | spread | as % of target |
| --- | --- | --- |
| previous, by chromosome count | 3,725 | 89% |
| best possible 5-fold | 3,188 | 76% |
| **6 folds, one chromosome each** | **432** | **12%** |

76% would have remained the worst balance in the repo by a wide margin, so worm now uses **6 folds**.
`n_folds` is already per-species (`len(folds)`, with validation at `(f+1) % n`), an earlier lab
assignment uses 3 folds for `S.pombe`, and worm has no prior assignment to stay compatible with. The four
worm experiments therefore contribute 24 (experiment × fold) jobs rather than 20.

`D.melanogaster` has the same 6-units-into-5-folds shape and is fine, because dm6's `chr4` is tiny so
`[chr4, chrX]` is a natural pair. Worm has no cheap pair — that is the whole difference.

**The membership is forced and the numbering is genomic on purpose.** Six chromosomes into six folds
leaves no balancing decision, so unlike C. griseus the assignment does not depend on which library was
counted — a further argument for it. Only the numbering is free, and genomic order is stable where count
order would shuffle if reassigned from a deeper library.

Add it upstream before using worm beyond this repo.

**Balance holds for two of the four worm libraries and NOT the other two, and the fold assignment cannot
help that** — six chromosomes into six folds admits no alternative. Measured across all four:

| library | spread | % of target | worst fold |
| --- | --- | --- | --- |
| `embryo` | 432 | 12% | `chrX` 17.7% |
| `L1starved` | 662 | 13% | `chrX` 15.3% |
| `embryo-sdc2` | 2,054 | 46% | `chrX` **21.3%** |
| `L3` | 6,079 | **102%** | `chrIV` **29.7%** |

Both outliers are biology, and both are worth knowing before reading per-fold metrics.

**`embryo-sdc2`'s X excess is dosage compensation.** `sdc-2(y93); sdc-2 RNAi` removes the master regulator
that recruits the DCC to X, so X is derepressed: X share goes 17.73% → 21.30%, **+3.6 pp / +20% relative**,
against `embryo`. That is Kruesi 2013's central result reproducing in our own peak calls, which is a
useful end-to-end check on the pipeline. Fold 5 is the `chrX` fold, so for that experiment fold 5 is
exactly where the perturbation lives.

**`L3`'s chromosome IV excess is the 21U-RNA (piRNA) clusters.** chrIV is 29.7% of L3's peaks against
~16% in the other three. Scaled on the unaffected chromosomes (chrI+II+III, 1.30x — peak calling saturates,
so total depth overstates the expected gain), the excess sits in two discrete blocks:

    chrIV  5- 6 Mb    4.0x, 2.7x
    chrIV 13-17 Mb    2.6x, 8.9x, 15.5x, 9.8x, 2.8x     ~6,000 peaks = 17% of the whole L3 library

The decisive observation is an INVERSION: 14–17 Mb is the *least* dense part of chrIV in `embryo`
(144/81/112/86 peaks per Mb, the gene-poor arm) and the *most* dense in `L3` (1669/1623/1422). Extra depth
finding more ordinary promoters cannot do that. Two broad clusters in the gene-poor arms of chrIV,
independently Pol II-initiated, germline-dependent — L3 has an expanding germline, embryos and starved L1s
do not — is the 21U-RNA architecture. Multi-Mb blocks also rule out a CNV, a repeat or mismapping, which
would be sub-Mb. Identified from position, stage and architecture rather than from sequence; a Ruby-motif
check on those peaks would confirm it.

Consequences, none of which change the assignment:

- **`L3` fold 3 tests on 30% of that library's peaks**, most of them piRNA loci. Its fold-3 numbers are not
  comparable to its other folds or to the other worm experiments.
- **`L3` is a partly different task from the other three worm libraries**: ~17% of its training loci are a
  class the others barely contain. Weigh that before training on one worm stage and evaluating on another.
- Do not treat the cluster as an artifact to exclude. It is real Pol II initiation, and hand-curating
  regions into an exclusion list is forbidden here for good reason.

Chromosome naming follows the `chrom_style` in `config/genomes.yaml` and must match the actual FASTA.
Note an earlier lab config uses `chr1..chr5` for A. thaliana, but that is a dataset-specific renaming for
a different TAIR10 build (NCBI GCA_000001735.1, organelles as `chrM`/`chrC`); ours is bare `1`-`5` as in
the Ensembl Plants FASTA.
- *Peak-level* (S. pombe): only 3 chromosomes, so `make_random_splits.py` assigns individual peaks to
  5 folds, grouping peaks whose training windows could overlap (centers within `in_window + 2*max_jitter`
  = 2514 bp) so they never straddle a split — this is the leakage guard, preserve it if you touch that script.

`fit_bpnet.py` and `launch.py` auto-detect peak-level splits by the existence of
`config/splits/{species}_random_fold_assignments.csv`; when present, `training_chroms`/`validation_chroms`
are set to `None` and loci are pre-filtered by fold instead. S. pombe **errors out** if that file is missing.

**S. pombe has exactly one mechanism.** It is deliberately absent from `chrom_splits.yaml` — 3
chromosomes cannot give 5 balanced folds — so peak-level splits are the only path, and any code asking
for its folds errors with a pointer to `make_random_splits.py` rather than silently falling back.
Both model families read peak-level splits through the shared `fold_loci()`. Do not re-add an
`S.pombe:` entry: two mechanisms for one species is the bug that was just removed.

**Name-based run matching reports ambiguity, not mere name-use.** `resolve_runs.py` used to label every
`sample_name ~ sample_alias` match "(weak; verify)", which left five Spt5 rows permanently flagged. That
match is only weak if it is *ambiguous*, so the label now depends on a bijectivity check: it says
`(unique)` when one manifest sample maps to one run with no collisions, and `(AMBIGUOUS -- verify)`
otherwise. For Spt5 the ENA aliases are exactly the manifest `sample_name` with `rep` abbreviated to `r`
(`CAP_SPT5_EtOH_1h_rep1` -> `CAP_SPT5_EtOH_1h_r1`), bijective across all six runs, and every run's alias
condition agrees with the experiment it was assigned to — verified against `PRJNA1105209`. Zero rows are
ambiguous now. Editing `planning/manifest_runs_resolved.tsv` to record this would not have worked: that
file is regenerated, so the fix had to be in the code.

**Peak-level split files carry provenance and are verified, not trusted.**
`make_random_splits.py` writes a `#`-comment header (`peaks_sha256`, `n_peaks`, `n_folds`, `seed`,
`window`, `jitter`); `--check` re-derives it and exits nonzero on drift. Readers pass `comment="#"`.
The digest is over coordinates **in file order**, because the CSV is written in its input's row order and
`fold_loci()` joins the two positionally — a peak set with identical content but different row order is
not interchangeable. Misalignment raises rather than warning: the old coordinate-join fallback could
silently produce folds that no longer describe the peaks, which is a train/test leakage hazard.
Determinism itself is already sound — same input gives byte-identical output, and reordering input rows
changes only the CSV's row order, not which fold a peak lands in (numpy `default_rng` for the shuffle,
`kind="mergesort"` for the sweep sort).

## Data conventions to preserve

- **Strand sign.** Minus-strand bigwigs may store signal as negative values (UCSC convention) or positive
  (direct). The codebase normalizes with `torch.abs()` on every signal/control tensor after `extract_loci`,
  and `make_negatives.py` abs-values the minus bigwig before merging strands. `src/bpnet/fit/data_loader.py`
  exists *only* to add these `abs()` calls around `bpnetlite`'s `PeakGenerator`. Any new code that reads
  signal must do the same.
- **Non-ACGT.** Every `extract_loci` call passes `ignore=list("QWERYUIOPSDFHJKLZXVBNM")`.
- **Outlier peaks ARE dropped, at `quantile(0.99) * 1.2`. An earlier version of this file said the
  opposite and it was wrong.** The claim was that the port to `data_loader.PeakGenerator` removed the
  filter the pre-unification `fit_bpnet.py` (at `a806e0d`) had. It did not remove it — it MOVED it. The
  filter now lives inside `src/bpnet/fit/data_loader.py`:

      outlier_threshold = torch.quantile(loci_counts, 0.99) * 1.2
      outlier_idxs = loci_counts > outlier_threshold
      ...
      peak_sequences=X_peaks[0][~outlier_idxs],

  `max_counts=None` in both fit scripts is true and was the evidence for the wrong claim, but it is a
  *different* knob — `max_counts` is tangermeme's own cutoff inside `extract_loci`, and this quantile
  filter is applied afterwards, on top. `data_loader.py` is byte-identical to procap-atlas's, so upstream
  drops them too; this is inherited, not local.
  **The objection the old text raised is therefore live, not avoided.** The threshold is data-dependent,
  so every species and library gets a different effective cutoff, which is corrosive in a repo whose
  point is cross-species comparison — and the top of a PRO-cap signal distribution is real biology
  (snRNA, histone, ribosomal-protein promoters), i.e. the most informative loci for an initiation model.
  Roughly the top 1% of peaks per experiment is being discarded. Decide deliberately whether to keep it;
  do not assume it is off. Artifact removal is the exclusion lists' job, and those are canonical published lists —
  **do not hand-curate regions into them**, or folds and preprocessing stop being comparable with
  procap-atlas.
- **The real gap this leaves:** S. cerevisiae and S. pombe have no published exclusion list *and* no
  outlier filter, so their only guards are non-ACGT filtering and the fold structure. Their in-assembly
  rDNA arrays (`XII:451786-489469`; `III:1-23130` and `III:2440994-2452883`) will be among the
  highest-signal PINTS calls. Those are real Pol I loci rather than mismapping artifacts, and `log1p` on
  the count head compresses their influence, so this is a known and probably tolerable exposure rather
  than a bug — but check it before publishing yeast numbers.
- **Negatives live in `main_chromosomes`, from `config/genomes.yaml`** — the same allow-list the bigWig
  and PINTS steps use, so negatives are drawn from exactly the space the peaks occupy.
  `src/make_negatives.py` applies it to the peak set, to the merged bigWig, and — the one that was
  missing — to the **negative sampling space itself**.
  **That last one was not being applied at all, and the claim above was false until 2026-09-02.**
  The script shelled out to `bpnet negatives`, whose first line is
  `chroms = list(pyfaidx.Fasta(args.fasta).keys())` — the whole assembly. `chroms` is not merely a filter
  on the input loci: `extract_matching_loci` builds its candidate space from it
  (`chrom_sizes = {key: len(fa[key]) for key in chroms}`), so negatives were drawn from organelles and
  unplaced scaffolds no matter what the peaks and signal were restricted to. It is now a direct
  `tangermeme.match.extract_matching_loci(..., chroms=keep)` call, which is the same three lines as the
  CLI with that one argument changed. **Do not go back to the CLI** unless bpnet-lite grows a `--chroms`
  flag.
  It surfaced as a crash rather than as bad data only because the merged bigWig *is* restricted:
  `_counts_from_coords` asked it for a contig it does not contain and pybigtools raised
  `KeyError: 'No chromomsome with name \`scaffold_37\` found.'` (and `Mito` on S. cerevisiae). **A less
  restricted bigWig would have returned counts and the negatives would have been quietly wrong** — which
  is what had been happening for every species whose peaks happened to cover the whole assembly.
  Calling `extract_matching_loci` directly is fine rather than a fork: `bpnet negatives` is a thin
  wrapper around exactly that function. The alternative considered and rejected was to keep the CLI and
  hand it a **symlink to the FASTA with a `.fai` subset to `main_chromosomes`** — which does work
  (verified: pyfaidx reports only the indexed contigs and still reads sequence through the original
  offsets), but restricts the CLI by trickery rather than by saying what is meant.
- **`-j` is a PROCESS pool, and has to be.** It was a `ThreadPoolExecutor`, which was fine while the
  work happened in `bigWigToBedGraph`/`bpnet` subprocesses that release the GIL. Once the strand merge
  and the GC matching both moved in-process, threads serialised on the GIL and `-j 42` used about two
  cores. Note the cost inside `extract_matching_loci` is a `joblib.Parallel(n_jobs=...)` scan over
  chromosomes that the CLI pins to `n_jobs=1`, so a single experiment is single-threaded either way —
  `-j` buys parallelism ACROSS experiments, not within one. Each worker holds a genome, so a large `-j`
  on the multi-gigabase species is memory-hungry.
  The `.fai` fallback in `make_chrom_sizes` used to be guarded by a `threading.Lock`, which processes do
  not share; `main()` now builds every missing index serially before the pool starts.
  This replaced a hand-written per-*experiment* `CHROM_EXCLUDE` regex map, which was wrong two ways.
  It **under-covered**: 3 of the 38 experiments then defined had an entry, so 35 filtered nothing while
  chrom.sizes came
  from the whole FASTA `.fai` — fine for dm6, but S. moellendorffii has 757 scaffolds and C. griseus 637.
  And it was **actively incorrect**: the patterns were substring regexes over the entire BED line, so
  dm6's `_` dropped any peak whose *name* contained an underscore, not just the scaffolds it targeted
  (verified — a `chr2L` peak named `peak_with_underscore` was being discarded).
  The new filter matches column 0 exactly, `str()`s the YAML names (bare-numeric chromosomes parse as
  ints and would match nothing), and raises if `main_chromosomes` and the FASTA disagree rather than
  silently emitting a short chrom.sizes. `ALPHA` stays per-experiment: it is a tuning parameter, not a
  property of the genome.
- **A purely numeric chromosome column silently drops EVERY peak inside `extract_matching_loci`.** Its
  path branch is `pandas.read_csv(loci, sep='\t', usecols=[0,1,2], header=None, names=[...])` with **no
  `dtype`**, so a BED whose first column is all digits infers as `int64`. The next line is
  `numpy.isin(loci['chrom'], chroms)` against our all-string `chroms`, every comparison is False, `loci`
  becomes empty, and the run dies further down on `zero-size array to reduction operation maximum` —
  which names nothing.
  This is the exact failure `main_chromosomes()`'s `str()` exists to prevent, happening one library
  deeper. It took out **A. thaliana (1-5), C. reinhardtii (1-17) and P. patens (1-27)** — the only three
  species with purely numeric names — and spared C. griseus purely because it has an `X`, which makes the
  column `object`. Roman numerals and `chr`/`NC_` prefixes are safe for the same accidental reason.
  **bpnet-lite has the same latent bug — verified, not assumed.** `pyfaidx.Fasta(...).keys()` returns
  `str` unconditionally (checked on a FASTA with contigs `1`/`2`/`10`), so `bpnet negatives` passes string
  `chroms` against the same int64 column and keeps **0 of 3** rows. It never surfaces upstream only
  because bpnet-lite is used on human and mouse, where `chr`-prefixed names force `object` dtype. So this
  is pre-existing rather than caused by dropping the CLI, and those three species never had working
  negatives by either route.
  Curiously the `chroms=None` fallback is the one safe path: it derives `chroms` from the loci column
  itself, so `fa[numpy.int64(1)]` raises `TypeError: Record name must be a string, not int64` — loud
  instead of silent. **The real fix belongs upstream**, as `dtype={0: str}` in that one `read_csv`; it
  would fix the CLI too and is worth a tangermeme PR.
  `sample_negatives` now reads the BED itself with `dtype={0: str}` and passes the **DataFrame**, which
  skips tangermeme's read entirely. **A fixture with `chrA`/`chrI`-style names cannot catch this** — the
  regression test uses all three naming styles on purpose. (`dtype={0: str}` does work alongside
  `names=`, checked directly: both `{0: str}` and `{"chrom": str}` give `object`.)

  **The SAME bug has a second instance, in `tangermeme.io._load_exclusion_zones`, and it hit A. thaliana
  training on 2026-09-03.** `KeyError: 1` out of
  `exclusion_zones[chrom][start:end] = True` — the zones dict is keyed by the FASTA's string names while
  the exclusion BED's column came back `int64`. Same missing `dtype`, opposite failure mode: **loud**
  here, where `extract_matching_loci` was silent. A. thaliana is the only species that is both
  numerically named *and* has a published exclusion list, which is why it was the one to break; the other
  two numeric species carry `blacklist: null`, so the call never happens.

  **It could not be fixed at the call site the way the first one was.** `_load_exclusion_zones` calls
  `pandas.read_csv` on each element of `exclusion_lists` itself, so there is no pre-typed DataFrame to
  hand it, and no file-level trick makes pandas infer `object` for an all-digit column. Renaming the BED's
  contigs is worse than the bug: the published list was deliberately stripped to bare `1`-`5` to match the
  Ensembl FASTA, and a list whose names do not match **excludes nothing, silently**. The remaining choices
  were to fork `data_loader.py` — forbidden, it is byte-identical to procap-atlas's — or to patch the one
  function, so `src/tangermeme_compat.py` patches it.

  **`patch_numeric_chroms()` is SELF-RETIRING**, which is the part worth preserving. It functionally
  probes the installed tangermeme with a numeric BED and returns without patching if the probe passes, so
  the shim vanishes when tangermeme is fixed instead of shadowing a corrected implementation forever. It
  also refuses to install a patch that fails its *own* probe, rather than silently breaking exclusion
  lists for the nine species that were working. Called from all five scripts that pass a blacklist into
  `extract_loci` (both fit scripts, both benchmarks, `attribute.py`), right after the deferred tangermeme
  import.
  Verified against the genuine upstream body lifted from the 1.4.1 wheel: unpatched reproduces
  `KeyError: 1` on the real `TAIR10.Klasfeld.Excludable.bed.gz`; patched excludes 2.86 Mb across
  chromosomes `1`-`5`; `chr`-prefixed lists behave identically through the patch; and a simulated
  fixed-upstream is declined.
  **So the tangermeme PR is now worth two `dtype={0: str}` edits, not one** — `match.extract_matching_loci`
  and `io._load_exclusion_zones`.
- **The two yeasts get 1-7% of the negatives every other species gets, and it is STRUCTURAL.** Measured
  over the first full run (2026-09-03), negatives per peak. **These are the filter-ON numbers**, kept
  because they are what motivated `NO_SIGNAL_FILTER`; the yeast rows are 3-4x higher at the sparse end
  now that the filter is off for them, and the shape of the finding is unchanged:

  | species | negatives/peak | negatives as % of all candidate windows |
  | --- | --- | --- |
  | S. cerevisiae (6 experiments) | **0.01-0.07** | 4.5-7.7% |
  | S. pombe | **0.03** | 5.3% |
  | C. elegans | 0.26-0.70 | 20-36% |
  | D. melanogaster | 0.73-1.00 | 31-48% |
  | everything else | **1.00** | 0.7-16% |

  **The predictor is PEAKS PER 2114 bp WINDOW, not genome size**, and the cutoff is sharp at 1.0:

  | experiment | bp per peak | peaks per window | negatives per peak |
  | --- | --- | --- | --- |
  | `S.cerevisiae-Spt5IAA4h` | 512 | **4.13** | 0.01 |
  | `S.cerevisiae-Ino80ctl` | 784 | **2.70** | 0.02 |
  | `S.pombe_PROcap` | 1,358 | **1.56** | 0.03 |
  | `S.cerevisiae_PROcap` | 1,790 | **1.18** | 0.07 |
  | `C.elegans-L3` | 2,792 | 0.76 | 0.26 |
  | `D.melanogaster-S2_PROcap` | 5,641 | 0.37 | 1.00 |

  Everything at or above one peak per window collapses; everything below it is fine. **Small genomes are
  not the problem** — C. reinhardtii is 13.2 kb per peak and P. patens 46 kb, the two sparsest in the
  corpus. It is the two yeasts, where the spacing between peaks is at or below the training window itself,
  so no 2114 bp window can avoid one.

  Worth knowing where those yeast peak counts come from, since they drive this. Against annotated TSSs,
  `S.cerevisiae_PROcap` (Booth, 4.9 M signal) calls **6,759 peaks = 1.04 per TSS**, which is a textbook
  number. The Spt5 and Ino80 experiments call 10,811-23,642, i.e. **1.7-3.6 per TSS**, and are only 7-22%
  unidirectional against Booth's 59%. The Spt5 series rises with depletion time (10,811 EtOH -> 21,591
  IAA1h -> 23,642 IAA4h), which is the direction Spt5 loss should push cryptic initiation — **but
  `signal_reads` rises with it too** (11.3 -> 16.7 -> 18.5 M), so depth and biology are confounded here
  and neither reading is established. Either way the yeast peak sets are 2-3x denser than the annotation,
  which is what breaks negative sampling.

  `extract_matching_loci` tiles each chromosome into NON-OVERLAPPING `in_window` blocks, so the entire
  candidate pool is `genome / 2114` — about **5,700 windows for a 12 Mb yeast genome**. Windows
  overlapping a peak are masked out, and `S.cerevisiae-Spt5IAA4h_PROcap` has **23,642 peaks, four times
  more than there are windows in the whole genome**. Almost nothing survives. Every large genome instead
  hits 1.00, meaning GC matching found a partner for essentially every peak and the peak count is the
  binding constraint.

  **`NEGATIVE_WINDOW` in `make_negatives.py` is the knob, and every run now reports the cost.** It
  overrides the candidate-tiling width per species for GC matching only. Shrinking it places candidate
  midpoints more finely; it does NOT shrink the training window, because both
  `_resize_coords_generator` and the loader's `extract_loci` resize to the same midpoint. The written BED
  intervals do take that width, so nothing downstream may depend on it. `out_window` is scaled with it,
  since `extract_matching_loci` asserts `in_window >= out_window`.
  Alongside it, `sample_negatives` prints negatives per peak and **`pct_peak_overlap`** — the share whose
  *2114 bp* window overlaps a peak, which is what the model will see. On a synthetic 2.4 Mb genome with a
  peak every 780 bp (S. cerevisiae density) the trade is stark:

  | tiling | negatives | per peak | overlap a peak |
  | --- | --- | --- | --- |
  | 2114 (default) | **0** | 0.00 | — |
  | 1200 | 2 | 0.00 | 0.0% |
  | 600 | 620 | 0.20 | **99.7%** |
  | 400 | 2,466 | 0.80 | **99.8%** |

  So narrowing does not buy clean negatives at yeast density — it buys many contaminated ones. That
  synthetic is uniform and therefore worst case; real yeast peaks cluster, which is why the real run finds
  310 rather than 0 at the default. **`NEGATIVE_WINDOW` is deliberately empty**: set an entry only with a
  measured `pct_peak_overlap` in front of you, and record why.
  **`--no-signal-filter` is the other lever, and for a densely transcribed genome it is the one that
  binds.** `bigwig=None` disables the signal restriction and nothing else: in `_extract_and_filter_chrom`
  both the threshold and the `values <= signal_threshold` mask sit behind `if bigwig is not None`, while
  GC matching, `max_n_perc` and the peak-tile mask are unconditional. On a synthetic genome at
  `S.cerevisiae_PROcap` density (one peak per 1,840 bp, 1.15 per window):

  | inter-peak signal | filter ON | filter OFF |
  | --- | --- | --- |
  | none | 792 negatives | 792 negatives (filter inert) |
  | pervasive | **0 negatives** | 792 negatives, **median signal 41% of peak median** |

  So where the genome is quiet between peaks the filter costs nothing, and where it is not the filter
  alone takes the count to zero. Yeast is the second case. The bar is
  `window signal <= signal_beta x (1st percentile of peak signal)`, which in a genome where essentially
  everything is transcribed almost nothing clears.
  **The recovered negatives are peak-free but not quiet** — 0.0% overlap a called peak, yet they carry a
  large fraction of peak-level signal. Whether that is a defect depends on what negatives are for here:
  they are not labelled zero, the loader extracts their real measured signal, so a GC-matched peak-free
  window carrying ordinary background transcription is arguably a *better* sample of "non-peak yeast
  genome" than an artificially quiet subset of it. `sample_negatives` now prints
  `median signal N vs M in peaks = X%` on every run so this is measured per experiment rather than
  assumed.

  **Measured on the real `S.cerevisiae_PROcap`, and it is a clear gain:**

  | | negatives | per peak | overlap a peak | median signal |
  | --- | --- | --- | --- | --- |
  | filter on | 440 | 0.07 | — | — |
  | filter off | **1,905** | **0.28** | **0.0%** | 271 vs 961 in peaks (28.2%) |

  **Read the 271 against the genome, not against the peaks.** An average 2114 bp window in this library
  holds **853** reads (4.87 M over 12.07 Mb), so the negatives sit at **0.32x a random window** while the
  *median* peak window sits at 1.13x. 28.2% sounds like contamination and is not: it is the peak median
  that is unremarkable, because at 1.18 peaks per window essentially every window contains one and the
  informative peaks are in the tail. So the filter was discarding 4.3x more negatives that are three times
  quieter than average genome and never overlap a called peak.
  **Every run now also reports the RANDOM-GENOME baseline**, from 2,000 random windows, because the peak
  median is the misleading comparator wherever the peak set saturates the genome. Read
  `negatives are Nx genome`, not the percentage against peaks.

  **Turning the filter off is NOT a no-op on the large genomes, and an earlier version of this note said
  it was.** The claim was that where the count is already capped at one negative per peak — 9 of the 23
  measured experiments — removing the filter can change nothing. The COUNT cannot change, that part is
  right. The COMPOSITION changes completely. Measured on a synthetic at 0.10 peaks per window, capped, with
  heterogeneous background:

  | | negatives | median signal | vs random genome |
  | --- | --- | --- | --- |
  | filter on | 559 | **0** | 0.00x |
  | filter off | 559 | **24** | **0.67x** |

  Identical count, different windows. `matched_loci_bin_count = min(bg, loci)` saturates per GC bin, but
  the pool it draws from is `random_state.shuffle`d and truncated, so a larger pool means a different — and
  noisier — sample. The earlier synthetic that showed no change had *uniform* background, so the filter
  rejected nothing; it demonstrated only that the test was degenerate.

  What this means for the choice. **With the filter on, negatives are the SILENT TAIL of the genome**
  (median 0); with it off they are **representative peak-free background** (0.67x genome). Both are
  defensible and they are different things, so this is not a free switch to flip corpus-wide — it would
  change the negatives of every experiment, including the 33 that have no problem. bpnet-lite and
  procap-atlas choose the silent tail, and this repo tracks their standards.
  So the realistic options were: keep it on everywhere and accept that the dense yeast experiments train
  on a few hundred negatives; or turn it off for those specific experiments as a **deliberate, documented
  departure**, accepting that their negatives then mean something slightly different from every other
  species'.

  **The second was chosen, 2026-09-03: `NO_SIGNAL_FILTER = {"S.cerevisiae", "S.pombe"}`.** It is not a
  comparability-free option — it makes the cost explicit and per-species rather than silent, and anything
  derived from yeast negatives is no longer strictly comparable with the other ten species. The switch is
  three-state now, because populating the set has to leave a way back to upstream behaviour: `auto`
  (respect the set, the default), `--no-signal-filter` (off for every species), and
  **`--force-signal-filter`** (on for every species, overriding the set — which is how the `filter on`
  column below was measured and how it can be re-measured). `resolve_signal_filter()` returns the reason
  alongside the decision, so every run prints which of the four states it is in rather than leaving the
  reader to infer it from a species name.

  **All seven yeast experiments measured (2026-09-03), and the gain decays monotonically with density:**

  | experiment | peaks/window | filter on | filter off | gain | negs/peak | median negative | median peak |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | `S.cerevisiae_PROcap` | 1.18 | 440 | 1,905 | **4.3x** | 0.28 | 271 | 961 |
  | `S.pombe_PROcap` | 1.56 | 311 | 947 | **3.0x** | 0.10 | 831 | 3,518 |
  | `Spt5EtOH` | 1.89 | 257 | 738 | **2.9x** | 0.07 | 316 | 1,873 |
  | `Ino80ctl` | 2.70 | 310 | 579 | 1.9x | 0.04 | 90 | 4,701 |
  | `Ino80KD` | 2.95 | 313 | 520 | 1.7x | 0.03 | 95 | 4,975 |
  | `Spt5IAA1h` | 3.78 | 348 | 419 | 1.2x | 0.02 | 41 | 2,956 |
  | `Spt5IAA4h` | 4.14 | 301 | 327 | **1.1x** | 0.01 | 16 | 3,264 |

  Two conclusions, and they point different ways.

  **Turning the filter off is safe.** Zero peak overlap everywhere, and the recovered negatives are not
  merely acceptable but very quiet — median 16 to 831 reads per 2114 bp against peak medians of
  1,873-4,975. The densest experiment's negatives sit at 16 against a peak median of 3,264. There is no
  contamination to trade against, so the earlier worry was unfounded at every density measured.

  **But it does not rescue the dense experiments, and the crossover is near 2 peaks per window.** Below
  it the signal threshold is what binds and removing it gains 3-4x; above it the supply of peak-free tiles
  binds and removing it gains 10-20%. `Spt5IAA4h` goes 301 -> 327, still 0.014 negatives per peak, using
  5.7% of all tiles in the genome — which is simply the peak-free fraction at 4.14 peaks per window.
  **This corrects an earlier note here that `NEGATIVE_WINDOW` was probably unnecessary.** That was true
  for `S.cerevisiae_PROcap`, where the filter bound; it is false for the four densest, where only finer
  tiling could help and finer tiling costs the peak contamination measured above. For those four there is
  no good option — at the configured 1/7 and 23,642 peaks, `Spt5IAA4h` would draw 3,377 negatives an
  epoch from a pool of 327, recycling each one 10.3x.
  **This is not a bug and `--force` will not change it.** Two honest readings, and this one is genuinely
  undecided: non-peak sequence space in a 12 Mb, densely transcribed genome is *genuinely* tiny, so ~330
  windows may be a fair sample of what exists; or the pool is too small to teach anything and yeast needs
  a different background scheme (a strided rather than tiled candidate set would give many more, and would
  need an upstream change). What the repo does instead is refuse to recycle: the ratio cap in
  `fit_bpnet.py` lowers `negatives_ratio` to the pool, so the pool size becomes visible in the training
  log rather than hidden in a resampling loop. **Read yeast negatives-derived metrics with this in mind.**
- **Cherimoya has THREE sources of defaults and they disagree — checked 2026-09-04 against the 0.2.0
  wheel.** Worth having in one place, because "cherimoya's default" is ambiguous:
  `cherimoya_cli.defaults.default_fit_parameters` (its CLI), the Python API's own signatures
  (`Cherimoya.__init__`, `fit()`, `io.PeakGenerator`), and procap-atlas.

  | | CLI | API | procap-atlas | here |
  | --- | --- | --- | --- | --- |
  | `negative_ratio` | 0.25 | 0.25 | 1/7 | **0.25** |
  | `max_jitter` | 500 | 500 | 500 | **500** |
  | `n_filters` / `n_layers` | 128 / 9 | 128 / 9 | 128 / 9 | 128 / 9 |
  | `expansion` / `residual_scale` | 2 / 0.15 | 2 / 0.15 | unset | unset -> 2 / 0.15 |
  | `muon_lr`/`wd`, `adam_lr`/`wd`, `lw_*` | 0.025/0.03, 0.001/0.0, … | — | same | same |
  | `max_epochs` | **20** | **50** | 50 | 50 |
  | `early_stopping` | **5** | **None** | None | None |
  | warmup epochs | **2** | — | 5 | 5 |
  | `dtype` | float32 | float32 | float32 | **bfloat16** |

  Three things fall out.

  **The CLI's schedule is exactly the `20_5_2` config procap-atlas swept and rejected** — max_epochs 20,
  early_stopping 5, warmup 2. So upstream's comparison was, in effect, testing cherimoya's own CLI
  default and finding `50_None_5` better on every benchmark metric.

  **On that schedule the API and the CLI disagree with each other**, and we follow the API:
  `Cherimoya.fit()` is declared `max_epochs=50, early_stopping=None`, which is also procap-atlas's
  choice and ours. So `max_epochs: 50, early_stopping: null` here is not a departure from cherimoya at
  all — it matches the library's *function* default, and only the CLI wrapper differs.

  **`expansion` and `residual_scale` are not passed by `fit_cherimoya.py` and do not need to be**: the
  class defaults (2, 0.15) are identical to the CLI's, so the architecture is the same either way.

  **`dtype=torch.bfloat16` is the one place this repo diverges from BOTH sources**, and it was never
  recorded as a decision — cherimoya's CLI, its `fit()` signature and procap-atlas all use `float32`
  (upstream passes `dtype=torch.float32` explicitly). `fit()` applies it through
  `torch.autocast(device_type=device, dtype=dtype)`, so this is autocast precision for the whole
  training loop, not a storage detail. Faster on Ampere and later, and usually harmless, but it is an
  unflagged numerical divergence in a repo that otherwise tracks upstream — **decide it deliberately
  rather than inheriting it.**

- **Negatives ratio is 1/7 for BPNet and 1/4 for Cherimoya** — negatives are 1/8 and 1/5 of a batch.
  Each family follows ITS OWN library's `PeakGenerator` default, which is the thing to remember, because
  the two libraries disagree and the number has been wrong here in three different ways.
  **Verified from the installed sources**: `bpnetlite`'s `PeakGenerator` defaults to `negative_ratio=0.1`
  and `cherimoya.io.PeakGenerator` to `negative_ratio=0.25` (its `PeakNegativeSampler` uses 0.1, which is
  the easy one to misread). `config/bpnet_params.json` sets 0.142857… — 1/7, which is procap-atlas's
  choice rather than bpnet-lite's 0.1 — and `config/cherimoya_params.json` sets 0.25.
  The history, since this line keeps attracting corrections: it first read "1/7 in `fit_bpnet.py`, 0.1 in
  the JSON configs", which was backwards, since the JSON is where 1/7 lives and **0.1 never applies at
  all** (both fit scripts pass `params["negatives_ratio"]` explicitly). It was then briefly set to 1/7
  for both families on 2026-09-03, on the reasoning that procap-atlas's `--background` defaults to
  `gc:0.1429` for both — that is true of procap-atlas but overrides cherimoya's own default, and
  **1/4 was restored on 2026-09-04**. Note `max_jitter=500`, aligned in the same pass, IS cherimoya's
  library default as well as upstream's, so only the ratio moved back.
  The ratio is negatives per peak, so 1/4 means one negative for every four peaks.
  It matters for the yeasts, where it sets how hard the small pool is recycled. Draws per epoch against
  the pool available with the signal filter off:

  | experiment | peaks | draws/epoch at 1/7 | pool | reuse |
  | --- | --- | --- | --- | --- |
  | `Spt5IAA4h` | 23,642 | 3,377 | 327 | **10.3x** |
  | `Spt5IAA1h` | 21,591 | 3,084 | 419 | **7.4x** |
  | `Ino80KD` | 16,865 | 2,409 | 520 | 4.6x |
  | `Ino80ctl` | 15,431 | 2,204 | 579 | 3.8x |
  | `Spt5EtOH` | 10,811 | 1,544 | 738 | 2.1x |
  | `S.pombe_PROcap` | 9,208 | 1,315 | 947 | 1.4x |
  | `S.cerevisiae_PROcap` | 6,759 | 966 | 1,905 | **0.5x** |

  So `S.cerevisiae_PROcap` is fine once the filter is off — it cannot even use its pool once per epoch —
  and the two Spt5 depletions recycle roughly ten and seven times over.
- **`negative_ratio` is therefore CAPPED at the available pool**, in both `fit_bpnet.py` and
  `fit_cherimoya.py`: `min(configured, len(negatives) / len(peaks))`, so no negative is drawn more than
  once per epoch and the cap is printed when it engages. **It is on by default and complements
  `NO_SIGNAL_FILTER` rather than substituting for it** — an earlier version of this line said the cap was
  chosen *over* loosening the filter, which is no longer the arrangement. The filter is what decides how
  many distinct negatives exist; the cap is what stops whatever number that is from being recycled. Both
  are needed for the dense experiments, where the filter buys only 10-20% and the pool stays a few
  hundred.
  Computed from **whole-genome** counts rather than the fold's: `PeakGenerator` filters peaks and
  negatives by the same `chroms`, so `pool/peaks` is near-constant across folds, and the exact per-fold
  numbers are not knowable at the call site without duplicating `extract_loci`. An approximate cap that
  always errs in the right direction beats forking `data_loader.py`, which is byte-identical to
  procap-atlas's.
  **The cap addresses RECYCLING, not DIVERSITY, and they are orthogonal.** The pool is the same N distinct
  windows at any ratio; capping only lowers how often each is seen, and with it the negative share of a
  batch — 12.5% to ~1.3% for `Spt5IAA4h`. If the negative class's batch weight matters more than avoiding
  repeats, `--no-ratio-cap` keeps the configured composition — present on **both** fit scripts, since
  cherimoya had the cap but no way off it. Neither setting adds a single new background sequence; only
  `NO_SIGNAL_FILTER` does that.

- **Negative DIVERSITY in the yeasts has a GENOMIC ceiling, and dropping the signal filter is what
  reaches it.** S. cerevisiae holds **5,710** non-overlapping 2114 bp windows in total, and the peak-free
  share of them runs from 33% at 1.18 peaks per window down to 5.7% at 4.14. **The filter-off pools are
  essentially those peak-free windows; the filter-on pools were a subset of them**, which is why removing
  it gains 3-4x at the sparse end and 10-20% at the dense end. Poisson on the observed density predicts
  1,753 free windows for `S.cerevisiae_PROcap` and 91 for `Spt5IAA4h`: the first matches the measured
  1,905 almost exactly, and the second is exceeded (327 found) because real peaks cluster, which leaves
  more empty tiles than a uniform model allows.
  An earlier version of this bullet read "no pipeline setting can raise it" and quoted the 257-440
  filter-on pools as the peak-free total. The **ceiling** is genomic, which is the part that holds; the
  pools were not at it.

  **In RELATIVE terms the pool is not impoverished at all**, which is worth knowing before treating it as
  a defect. Yeast rows are the filter-off pools now in use; mouse keeps the filter:

  | | pool | unique background sequence | share of genome |
  | --- | --- | --- | --- |
  | `Spt5IAA4h` | 327 | 691 kb | **5.7%** |
  | `S.pombe_PROcap` | 947 | 2,002 kb | **16.0%** |
  | `S.cerevisiae_PROcap` | 1,905 | 4,027 kb | **33.4%** |
  | `M.musculus-GCB_PROcap` | 64,667 | 136,706 kb | 5.15% |

  So the densest yeast experiment samples about the same fraction of its genome as mouse does of its
  (5.7% vs 5.15%), and the sparse ones sample far more. What is small is the genome, not the sampling.

  **The asymmetry that IS real is negative vs positive unique sequence, and it now spans an order of
  magnitude within one species.** Yeast peak windows overlap heavily, so their union is roughly the whole
  genome minus the peak-free part:

  | | positive union | negative union | ratio |
  | --- | --- | --- | --- |
  | `Spt5IAA4h` | ~11.4 Mb | 0.69 Mb | **1:17** |
  | `S.cerevisiae_PROcap` | ~8.0 Mb | 4.03 Mb | **1:2** |
  | `M.musculus-GCB_PROcap` | ~137 Mb | ~137 Mb | 1:1 |

  Mouse peak windows barely overlap, hence parity. So the model trained on `Spt5IAA4h` sees seventeen
  times more distinct positive than negative sequence while the one trained on the WT library sees
  roughly a 2:1 split — and that spread is set by peak density, not by any setting. It is also the
  strongest argument for reading the two groups' negatives-derived numbers separately rather than as
  "the yeasts".

  Levers, with what each actually buys:
  - **Jitter on negatives.** `data_loader.py` passes `max_jitter=0` for the background while peaks get
    the configured 200. Enabling it would add **+19%** unique sequence (2114 -> 2514 bp per locus), which
    is augmentation rather than diversity, and it means forking a file that is byte-identical to
    procap-atlas's. Not worth it for 19%.
  - **Stricter peak calling — NOT available through `--min-mu-percent`, checked 2026-09-03.** The
    `Spt5IAA4h` PINTS log ends with *"To reduce false positives, PINTS overrided your current
    --min-mu-percent value… consider increasing (current: 0.10) to 0.15"*, which reads like an untaken
    opportunity and is not one. In `calling_engine.py` the override is **applied in place** before the
    calls are made — `if bkg_mu_threshold < 0.5 and len(all_peak_mus) > 1000: bkg_mu_threshold =
    np.quantile(all_peak_mus, suggest_val)` — and the `.mmp` file it writes exists only so
    `on_the_fly_qc` can print that message afterwards. The log shows it firing: chromosome IV reports
    `Minimum mu in local environment 0.500000`, exactly the floor. So passing `0.15` would silence the
    message and reproduce roughly the thresholds already used. (Verified against the 1.2.1 source while
    the run used the pinned 1.1.10; the message text is identical, so the logic almost certainly is, but
    that is inference.)
    **What the log does show is that saturation is a depth-versus-genome-size effect, not loose
    settings.** The per-chromosome candidate thresholds are densities of **1.92-2.82**, while the library
    averages **1.53 reads/bp** across the genome (18.5 M over 12.07 Mb) — so the bar for a candidate peak
    sits at 1.3-1.8x the genome-wide mean. Nearly everything clears it because nearly everything is
    covered.
    Also worth recording as a **negative** result: PINTS' own cap-selection check, which warns that "the
    proportion of significant short peaks is relatively high, which usually indicates the cap-selection
    process didn't work well" and suggests `--disable-small`, **did not fire**. So the library is not
    failing cap selection, despite `keep_sticks: True` and `disable_small: False`.
  - **Accept it**, and treat yeast negatives-derived numbers as weakly supported.

**Should the signal filter be loosened for the other experiments where negatives < peaks? No.** That is
the wrong threshold — what matters is `pool/peaks` against `negatives_ratio`, not against 1. Sorted, the
corpus has a clean gap with nothing in it:

| | pool/peaks | vs 1/7 |
| --- | --- | --- |
| the 7 yeast experiments | 0.013 - 0.065 | **all below** |
| `C.elegans-L3` (worst non-yeast) | **0.262** | 1.8x above |
| `C.elegans` others, `D.melanogaster` | 0.57 - 1.00 | 4-7x above |
| everything else | ~1.00 | 7x above |

`C.elegans-L3` has 9,411 negatives for 35,923 peaks, which *looks* alarming and is not: at 1/7 an epoch
draws 5,132, well inside the pool, so nothing recycles. Loosening the filter there would change what its
negatives mean — silent tail to representative background — for **no training benefit at all**. Leave
every non-yeast experiment alone.
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
  for everything else, each with linear warmup → cosine decay, trained in `bfloat16` — which is a
  divergence from cherimoya's own default and from procap-atlas, both `float32`; see the defaults table
  above. Warmup was
  hard-coded at 5 epochs and is now `warmup_epochs` in the config with a `--warmup-epochs` flag, plus
  `decay_epochs` / `--decay-epochs` to decouple the decay's length from `max_epochs` — both ported from
  upstream, both no-ops at their defaults (5 and None give exactly the previous schedule, verified: 4,500
  decay iterations either way and no third stage). Setting `decay_epochs` shorter adds a **`ConstantLR`
  hold at `eta_min`**, which is load-bearing rather than cosmetic: `CosineAnnealingLR` is *periodic*, so
  without it the LR would start climbing again past `T_max`. Its `total_iters` is deliberately far larger
  than the hold (`num_hold_iters * 10 + 10**6`) because `ConstantLR` reverts to the optimizer's base LR
  once reached — sized exactly, it would snap the LR back up on the last step of training.

## Scope note

Human K562 configs from other work in this lab are deliberately excluded from this repo.
`planning/nonhuman_capped_runon_manifest.xlsx` tracks candidate non-human datasets not yet wired into
`config/`.
