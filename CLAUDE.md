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
  228 jobs for `tier=include`, **704 for all tiers** over 38 experiments and 10 genomes;
  `fetch_only` 78, `signal_only` 542. Note the **target name must come before `--config`** --
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
  (metadata, not DAG work) and training stays on `launch.py`.
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
(~120 GiB, 74 files over 64 runs — measured from ENA `fastq_bytes`) with md5 verification, so transfers
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
  mates. Only 5 of 40 experiments are paired (all S. cerevisiae: Ino80 x2, Spt5 x3), which is why this
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
  fragment's 3' coordinate while dropping 43-57% of reads in the only 3 of 40 experiments that are
  deduped at all.
- **How a library is known to have a UMI: the manifest's `umi_len`/`umi_loc` columns, and nothing else.**
  `build_experiment_config.py` derives `raw.umi` from them (it used to be a hardcoded dict transcribed by
  hand from free text in `library_layout`). **The archives cannot corroborate this** — ENA's
  `library_construction_protocol` was queried for all 45 runs and mentions a UMI for *none* of them,
  including the three Spt5 experiments that demonstrably have one, so the manifest is authoritative and
  must be curated from the paper or GEO record. The direct evidence for Spt5 is the extracted read name,
  `SRR29037352.25948720:ACTAGATAGC` — a 10-base tag, matching `umi_len: 10`.
  The scheme is **default-deny**: a library whose UMI was never noted is treated as having none and keeps
  its PCR duplicates. That is the safer error, because deduplicating a non-UMI PRO-cap library destroys
  real stacked 5' ends — but it does mean a missed UMI is silent.
- **umi_tools does nothing to non-UMI libraries, by design.** `final_bam` takes `unique.bam` directly
  when `has_umi(run)` is false, so the `dedup` rule never runs for them — 37 of 40 experiments. This is
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
  `-threshold=-1000000` for precisely this, and `src/make_negatives.py` still abs-values the minus bigWig
  before merging strands. Do not "modernise" this into a bigWig-level merge.
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
did is now in `workflow/Snakefile` (`fetch_genome`/`faidx`/`chrom_sizes`/`star_index` cover all six species;
the old script covered three) or was part of the deleted dm3 regime. `utils.sh` was sourced by nothing and
hard-coded `/programs/...` paths. Read them at `a806e0d` if you need the exact legacy provenance; do not
restore them.

Why the re-map was necessary: the *S. cerevisiae* and *S. pombe* labels were GEO-provided **normalized**
bigWigs (`R1Normed`), only re-wrapped by hand (chr stripped, mito removed), while Drosophila labels came
from raw reads via `proseq2.0`. BPNet's count head consumes
`log1p(total signal)` with `count_loss_weight`, and the training outlier filter is a quantile of total
signal — both are meaningless when one species' labels are arbitrarily scaled and another's are read counts.

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
here** — neither csRNAnet nor plant-design has a cotton assignment — so push them upstream before using
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
  also matches csRNANet (`data/mm10.fa`), is what the canonical peak-matched folds were tuned on, and is the
  only mouse assembly with a published exclusion list. Chromosome names are identical between the two
  assemblies, so the fold assignment transferred unchanged.
- **dm6 names are chr-prefixed here** (`chr2L`), where the canonical csRNANet/plant-design file writes `2L`.
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
| C. reinhardtii | Chlamydomonas_reinhardtii_v5.5 | none published | — |
| P. patens | Phypa_V3 | none published | — |
| S. moellendorffii | v1.0 | none published | — |
| C. griseus | CriGri-PICRH-1.0 | none published | — |

Boyle-Lab lists download as plain BEDs and are fetched by
`run_procap_pipeline.py --fetch-genomes`; naming verified chr-prefixed for all three (`chr2L`, `chrI`,
`chr1`).

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

csRNANet's convention was exclusion lists for human and mouse only; this repo extends that to fly, worm and
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
the duplication is. **All six references here were checked against their real contig lists and all six
are already alt-free.** Do not add a contig-filtering step; there is nothing to filter.

| reference | contigs | alt |
| --- | --- | --- |
| dm6 | 1,870 = 8 primary + 645 `*_random` + 1,217 `chrUn_*` | 0 |
| mm10 | 66 = 22 primary + 22 `*_random` + 22 `chrUn_*` | 0 |
| ce11 | 7 = 6 chromosomes + chrM | 0 |
| R64-1-1 | 17 = 16 chromosomes + Mito | 0 |
| TAIR10 | 7 = 5 chromosomes + Mt + Pt | 0 |
| ASM294v2 | 6 = I, II, III, MT, MTR, AB325691 | 0 |
| Chlamydomonas_reinhardtii_v5.5 | 53 = 17 chromosomes + 36 scaffolds, **NO organelles** | 0 |
| Phypa_V3 | 357 = 27 chromosomes + 330 scaffolds, **NO organelles** | 0 |
| v1.0 (S. moellendorffii) | 759 = 0 chromosomes + Pt + 757 scaffolds | 0 |
| CriGri-PICRH-1.0 | 647 = 10 chromosomes + 637 unplaced | 0 |

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

**The two plant references have NO organelle contigs, and that costs ~10% of those libraries.** An earlier
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
| C. reinhardtii | yes — subtelomeric `1:~1100-19100` and `14:~4145334-4154986` | none |
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
for any of the 10 species — so a decoy can never reach a fold, a peak or a bigWig.

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
`data/decoy/` and reports 0% for the 7 of 10 species whose organelles are in the reference — a false
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
| `McDonald2024_plant_5GRO` | C. reinhardtii, P. patens, S. moellendorffii | 3 | 5'GRO-seq; same GEO series (GSE233927) csRNANet already uses. |
| `Shamie2021_cg_5GRO` | C. griseus | 7 | Chinese hamster GRO-cap atlas; CHO-K1 has 2 reps, 6 tissues n=1. |

All 14 runs were resolved against ENA and all report `SINGLE`. Findings worth keeping:

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
- Download total is now **~120 GiB / 74 files / 64 runs**, measured from ENA `fastq_bytes`. The new
  projects add only 12.5 GiB; the old "~78 GiB" figure was already stale at 107.3 GiB.

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
