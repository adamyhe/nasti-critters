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
