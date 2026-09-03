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
- **Cherimoya** — newer architecture, *D. melanogaster* only. Training and benchmarking work, but per
  `src/cherimoya/README.md` the models are **not deployment-ready**.

All data lives in `data/`; models in `models/{bpnet,cherimoya}/`. Neither was gitignored before — the
repo's `.gitignore` was a stock Python one with no `data/` rule, which only looked harmless because `data/`
did not exist yet. Both are ignored now (along with `logs/`, `predictions/`, `performance_metrics/`,
`attr/`); the full FASTQ set alone is ~202 GiB, so check `git status` before any bulk `git add`.
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
  `bedGraphToBigWig`, `pints_caller` and `umi_tools`. **`bigWigToBedGraph` and `bigWigMerge` are gone**,
  and so is the `bpnet` CLI from this list — see the next bullet. The pipeline pulls FASTQs straight from ENA over HTTPS, so SRA Toolkit is not
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
- `Cherimoya.load(path, device=...)` is unchanged and still compatible.

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
| `reads_per_peak` | crude signal density; very low means peaks called from thin coverage. |
| `pct_rrna` | rRNA + organellar share of the raw reads, from `src/qc/rrna_content.py`. Blank means NOT MEASURED, which is not the same as 0. |
| `pct_unique_adj` | `unique / non-rRNA input`. **This is the mapping-quality number**; `pct_unique` is not. |
| `qc_flags` | comma-joined `FAIL:`/`WARN:` findings. Advisory — exclusion stays a manual `tier` decision. |

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

- The rule's inputs are **named** (`stats`/`rrna`/`reads`/`script`) and the shell passes
  `--combine {input.stats}`, with the two directories passed as `params` instead. **Never `--combine
  {input}`.**
- `--combine` now **rejects any file whose header is not exactly `COLUMNS`**, naming the file and both
  column counts. A too-broad glob has to fail, not average out — the whole failure was that a
  plausible-looking table hid it.

Also `--combine`'s sort key is `str()`-wrapped now, matching `--all`. It was the only reason the garbage
rows did not crash on a `None`-vs-`str` comparison, i.e. the one thing that made the corruption survivable
enough to be committed.

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
| C. reinhardtii | yes | **reused unchanged** |
| P. patens | yes | **reused unchanged** |
| S. moellendorffii | no | peak-level folds, permanently (no chromosomes exist); **built 2026-09-02** |
| C. griseus | no | chromosome-level; **assigned here 2026-09-01** from CHO peak counts |

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
  regression test uses all three naming styles on purpose.
- **The two yeasts get 1-7% of the negatives every other species gets, and it is STRUCTURAL.** Measured
  over the first full run (2026-09-03), negatives per peak:

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

  **Whether to do this per species is the wrong question — prefer a uniform rule.** Sorting the corpus by
  negatives per peak splits it cleanly: **9 experiments sit at the peak cap (1.00)**, where GC matching
  already found a partner for every peak, so removing the filter cannot raise the count and in a synthetic
  at that density changes neither the count nor the selection. The other 14 are constrained — fly embryo
  0.75-0.80, fly 5GROcap 0.73, worm 0.26-0.70, yeasts 0.01-0.07.
  Toggling this per species is exactly the per-species tuning this repo refuses elsewhere: it would mean
  yeast negatives are drawn from one distribution and mouse negatives from another, which is a systematic
  between-species difference in a repo whose point is between-species comparison — the same objection
  raised against the outlier filter's data-dependent threshold. Since the filter appears inert wherever
  the count is capped, **off everywhere is both the simpler rule and the more comparable one**, and on
  everywhere is the only other defensible choice. `NO_SIGNAL_FILTER` is per-species because that is the
  cheapest thing to experiment with; if the decision lands on uniform, collapse it to a single default.

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
  no good option — with `negatives ratio 0.1` and 23,642 peaks, `Spt5IAA4h` draws ~2,364 negatives an
  epoch from a pool of 327.
  **This is not a bug and `--force` will not change it** — but with `negatives ratio 0.1` a 23,642-peak
  yeast experiment draws ~2,364 negatives an epoch from a pool of 301, so the same regions recur about
  eight times over and the GC match is thin. Two honest readings, and the choice has not been made:
  non-peak sequence space in a 12 Mb, densely transcribed genome is *genuinely* tiny, so 300 windows may
  be a fair sample of what exists; or the pool is too small to teach anything and yeast needs a different
  background scheme (a strided rather than tiled candidate set would give many more, and would need an
  upstream change). **Read yeast negatives-derived metrics with this in mind.**
- **Negatives ratio is 1/7 for BPNet and 1/4 for Cherimoya, i.e. negatives are 1/8 and 1/5 of a batch.**
  An earlier version of this line had it backwards, as "1/7 in `fit_bpnet.py`, 0.1 in the JSON configs".
  It is the other way round: `config/bpnet_params.json` sets `negatives_ratio: 0.142857…` and
  `config/cherimoya_params.json` sets `0.25`, while **0.1 is only `PeakGenerator`'s default and never
  applies**, because `fit_bpnet.py` passes `params["negatives_ratio"]`. The ratio is negatives per peak,
  so 1/7 means one negative for every seven peaks.
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
  and the two Spt5 depletions recycle roughly ten and seven times over. Lowering `negatives_ratio` for
  those two is the cheapest lever, and unlike finer tiling it costs no peak contamination.
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

Human K562 configs from other work in this lab are deliberately excluded from this repo.
`planning/nonhuman_capped_runon_manifest.xlsx` tracks candidate non-human datasets not yet wired into
`config/`.
