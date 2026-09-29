# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**nasti-critters** — NAScent Transcription Initiation Critters. A research repo of training/evaluation
scripts (no installable package, no test suite) for base-resolution sequence-to-function models of PRO-cap
transcription-initiation profiles across non-human species. **42 experiments over 12 species**
(*D. melanogaster*, *M. musculus*, *C. elegans*, *S. cerevisiae*, *S. pombe*, *A. thaliana*,
*C. reinhardtii*, *P. patens*, *S. moellendorffii*, *C. griseus*, *G. arboreum*, *G. hirsutum*), defined
in `config/experiment_config.yaml` and generated from the manifest in `planning/`.

**One experiment == one species x one biological condition == one model.** Perturbations (Ino80 depletion,
Spt5 IAA, LacZ knockdown) are separate experiments with their own models; this project deliberately does
not multi-task conditions or assay families into shared heads.

Two model families are trained on the same data:

- **BPNet** (`bpnetlite.BPNet`) — the main path.
- **Cherimoya** — newer architecture. It is **not** D. melanogaster-only: `fit_cherimoya.py` takes `-e`
  and resolves paths, species and folds through `src/experiments.py` exactly as `fit_bpnet.py` does.
  Training and benchmarking work, but per `src/cherimoya/README.md` the models are **not
  deployment-ready**, and attribution is blocked (see below).

`data/`, `models/`, `logs/`, `predictions/`, `performance_metrics/`, `attributions/`, `modisco/` and
`qc/` are gitignored — the FASTQ set alone is ~202 GiB, so check `git status` before any bulk `git add`.
Every script resolves config/data paths relative to `REPO_ROOT`, computed from `__file__`, so scripts can
be invoked from anywhere; the launchers assume the repo root as CWD.

## Where the detail lives

This file is the live rules. The evidence behind them — measurements, per-dataset provenance, rejected
options and refuted hypotheses — is in `docs/`, dated and with its method recorded so it is not
re-derived. **Check the relevant doc before reopening a question**; several entries exist specifically to
stop a plausible idea being re-proposed.

| doc | what is in it |
| --- | --- |
| `docs/environment.md` | the two-environment split, Sherlock's wheel ceiling and pins, package-name traps |
| `docs/snakemake-operations.md` | rerun triggers, stale/incomplete metadata, `--touch`, scoping a run |
| `docs/assemblies-and-references.md` | assembly choices, exclusion lists, alt audit, rDNA/organelle decoys |
| `docs/cross-validation-folds.md` | how each species' folds were derived; which still need pushing upstream |
| `docs/datasets-and-provenance.md` | dataset additions, curation errors, merge/exclude decisions |
| `docs/investigations.md` | G. hirsutum depth analysis, adapter survey, orientation QC, yeast negatives |
| `docs/decisions-and-history.md` | pre-unification regimes and deleted scripts, the procap-atlas hyperparameter sync, cherimoya's three sources of defaults |
| `README.md` | install and operator runbook |
| `config/procap_pipeline.yaml` | the ENCODE spec, transcribed; parameters that determine the peaks |

## Environment

Training, benchmarking and attribution need a CUDA GPU (`fit_bpnet.py` hard-codes `.to("cuda")`). Local
development is fine for config-parsing logic; anything importing torch or loading loci needs the cluster.

**Two environments, split by DEPENDENCY, not by task** — see `docs/environment.md` for the full rules.

- `environment.yml` (+ `conda-lock.yml`, linux-64 only) owns the **whole preprocessing DAG**, fetch
  through peaks: the external binaries, `snakemake`, and the only three libraries those scripts import
  (`pyyaml`, `pandas`, `numpy`). Nothing in `src/data_preprocessing/`, `src/experiments.py` or
  `config/write_split_csvs.py` imports torch, bpnet-lite, cherimoya or tangermeme, so label generation
  needs no GPU stack and no venv. Do not add preprocessing-only deps to `pyproject.toml`.
- `pyproject.toml` (+ `uv.lock`) owns **model work**. `uv sync` is torch-free (tfmodisco, analysis, every
  launcher); `uv sync --extra torch` adds training, benchmarking, attribution. Anything importing
  `tangermeme` or `bpnet-lite` is on the torch side even where the science does not obviously involve a
  model — `fit_*`, `benchmark_*`, `attribute.py`, `make_negatives.py`, and
  `filter_nonACGT_regions.py --save-ohe`. Before adding a base dependency, run
  `pip download <pkg> --no-deps` and read its `Requires-Dist`.

```bash
mamba env create -f environment.yml && mamba activate nasti-critters \
  && uv sync --extra torch && source .venv/bin/activate   # activate the venv LAST so its interpreter wins
```

Drop `--extra torch` on a CPU-only machine such as Sherlock, which runs the CPU-demanding jobs (tfmodisco)
and no GPU work. **The base set must install from wheels alone** — Sherlock cannot build anything with its
default toolchain, so several dependencies are capped at the last `manylinux_2_17` release. Re-audit with
`uv export --no-emit-project --no-hashes` after ANY dependency change, and do not raise a cap without it.

Things already settled — do not "fix" them back:

- **Do NOT try to get a newer torch from conda-forge.** Tried, rejected: its torch build does not reliably
  use CUDA.
- **Do not add an Apptainer definition here.** Images are maintained at
  [adamyhe/sherlock](https://github.com/adamyhe/sherlock); containers are opt-in via
  `APPTAINER_IMAGE`/`APPTAINER_BIND` and the launchers run natively when unset.
- **`cherimoya` is pinned by git commit** (`8e4283fe` = 0.2.1, never published to PyPI), for byte
  agreement with the image and immunity to tag movement — not for different training behaviour.
- **`macs3`/`cykhash` and `modisco-lite` are blocked** behind an unsatisfiable marker in
  `[tool.uv] override-dependencies`. `modisco-lite` is the deprecated name for the same project as
  `modisco` and installs the same `modiscolite/` package, so installing both silently keeps whichever
  landed second.
- **`umi_tools`, `pypints`, `pybigtools` placements are deliberate**; `tangermeme` and `modisco` are
  PyPI-only and cannot move to `environment.yml` however well they fit the shell-out rule.
- **Two package-name traps:** the peak caller is **`pyPINTS`** (PyPI `pints` is unrelated), and PyPI
  **`muon` is a multi-omics framework**, not the optimizer — `fit_cherimoya.py` imports
  `torch.optim.Muon` and raises a pointed error rather than falling back to it.
- **Do not reintroduce cluster paths into tracked files.** Site-specific module loads go in
  `launch.py --setup-file`.

Preprocessing shells out to: fastp, STAR, samtools, bedtools, GNU `sort`, `bgzip`, `bedGraphToBigWig`,
`pints_caller`, `umi_tools`. FASTQs come from ENA over HTTPS, so SRA Toolkit is not needed.
**The venv side must not shell out to a conda binary** — `make_negatives.py` did, three times, and now
makes no subprocess calls at all (`pybigtools`, `gzip`, `pyfaidx` in-process). When adding to it, check
which environment provides what you are calling. When auditing that boundary, grep for `subprocess`,
`run([` and bare command names in `shell:` blocks — not just imports.

**Syntax-check against the OLDEST interpreter you can find, not the newest.** The env is Python 3.11; a
dev machine may be newer, and PEP 701 (3.12) legalised multi-line f-string expressions that are a hard
`SyntaxError` on the cluster — failing at *import*, so it takes the whole run down before any work
happens. There is no CI. Sweep with macOS's `/usr/bin/python3`:

```bash
for f in $(git ls-files '*.py'); do
  /usr/bin/python3 -c "import sys,ast;ast.parse(open(sys.argv[1]).read())" "$f" || echo "FAILS: $f"
done
```

All tracked files passed under 3.9.6 as of 2026-09-06. Keep it that way: precompute a conditional into a
variable rather than inlining it across lines inside an f-string.

## Common commands

```bash
# --- ENCODE PRO-cap pipeline (the current path for producing labels) ---
python src/data_preprocessing/resolve_runs.py                  # manifest -> run accessions via ENA
python src/data_preprocessing/build_experiment_config.py        # manifest -> experiment_config.yaml
python src/data_preprocessing/fetch_fastqs.py --dry-run         # ~202 GiB for all 42 experiments
python src/data_preprocessing/fetch_fastqs.py --tier include -j 4
python src/data_preprocessing/fetch_fastqs.py --verify-only     # md5 audit of what is on disk
python src/data_preprocessing/run_procap_pipeline.py --list     # what is defined and what is ready
python src/data_preprocessing/run_procap_pipeline.py --index-only --species S.cerevisiae
python src/data_preprocessing/run_procap_pipeline.py -e S.cerevisiae-Ino80ctl_PROcap --dry-run
python src/data_preprocessing/run_procap_pipeline.py --tier include -t 16

# Snakemake is the preferred driver. TARGET BEFORE --config, always.
snakemake -c16 --config tier=include
snakemake orientation -c8 --rerun-triggers mtime        # initiator logos + metaplots only
snakemake stats -c8 --config tier=include,conditional   # mapped reads + peak counts per experiment
snakemake blacklists -c1                                # cheapest way to unblock an existing tree

# GC-matched negatives: model prep, deliberately OUTSIDE the DAG. Run from the venv.
uv run python src/make_negatives.py --dry-run
uv run python src/make_negatives.py -e S.cerevisiae_PROcap --force -j 3

# Peak-level folds (required before S. pombe / S. moellendorffii training)
python src/data_preprocessing/make_random_splits.py

# Train, one experiment/fold. Same interface for both families.
python src/bpnet/fit/fit_bpnet.py -e D.melanogaster-S2_PROcap -f 0 -v
python src/cherimoya/fit/fit_cherimoya.py -e D.melanogaster-S2_PROcap -f 0

# Evaluate / attribute
python src/bpnet/benchmark/benchmark_predictions.py -e D.melanogaster-S2_PROcap
python src/cherimoya/benchmark/benchmark_cherimoya.py -e D.melanogaster-S2_PROcap --save-output
python src/bpnet/attribute/attribute.py -e D.melanogaster-S2_PROcap --attribute-type profile
python src/analysis/compare_bpnet_cherimoya.py

# Every launcher takes the same three emission modes and the same SLURM flags.
python src/bpnet/fit/launch.py --dry-run          # full sbatch scripts, nothing submitted
python src/bpnet/fit/launch.py --print-commands | bash   # bare commands; skips + summary on stderr
python src/bpnet/fit/launch.py --time 12:00:00 --mem 32G  # submit

# Attribution -> motifs, in order. The MEME databases are fetched once (5 files, ~1.3 MB).
python src/bpnet/attribute/launch_filter.py --dry-run    # REQUIRED first; one CPU job per experiment
python src/bpnet/attribute/launch.py --dry-run           # skips any experiment the filter has not covered
python src/bpnet/modisco/fetch_motif_dbs.py
python src/bpnet/modisco/launch.py --dry-run
python src/bpnet/modisco/launch_report.py --dry-run
```

There is no linter config, no formatter config, and no tests. **Verification means running a script** —
`--dry-run` (preprocessing/launchers) or a single `-f 0` fold as the cheap smoke test.

## Architecture: one convention, via src/experiments.py

Everything resolves through **`src/experiments.py`**. Scripts add `REPO_ROOT/src` to `sys.path` and import
from it; do not reintroduce per-script path constants.

| Concern | Single source |
| --- | --- |
| What data | `config/experiment_config.yaml`, selected with `-e EXPERIMENT` |
| Which folds | `config/chrom_splits.yaml` by species, or peak-level `config/splits/{species}_random_fold_assignments.csv` |
| Hyperparameters | `config/{bpnet,cherimoya}_params.json`, one file per model family |
| Model paths | `models/{family}/{experiment}/{experiment}.fold{f}.torch` |
| Metrics paths | `experiments.metrics_path(family, experiment)` |
| Attribution paths | `experiments.attribution_path(family, ...)`, `filtered_loci_path()`, `ohe_path()` |
| MEME database | `experiments.motif_db_path()`, per species via `genomes.yaml: jaspar_collection` |
| Non-ACGT bases | `experiments.IGNORE` |

`Experiment.load(id)` returns resolved absolute paths plus `species`, `blacklist`, and a `missing` list of
absent inputs. `.fold_split(f)` applies the one fold rule (test = `f`, validation = `(f+1) % n`, train =
the rest) and transparently switches to peak-level splits where they exist. `.all_folds(family)` gives
every fold with its test chromosomes and checkpoint path. **A path format string must have exactly one
definition** — two copies is how a launcher starts re-running finished work, and how a script writes where
its launcher is not looking.

**Peak-level splits must be applied on the PEAK TABLE, and every downstream script has to do it.**
`fold_split()` returns `test_chroms=None` for S. pombe and S. moellendorffii, and `chroms=None` means "no
chromosome filter" rather than "no loci" — so before this was fixed both benchmark scripts **silently
scored every fold's model on ALL loci, its own training peaks included**, with metrics inflated and no
error. That is the worse failure mode and it was only found by chasing a crash elsewhere. `fold_loci()`
returns **`test_loci`** for this: under chromosome-level splits it returns every peak and `test_chroms`
does the work, so that path is unchanged; under peak-level it returns the fold's held-out peaks. Both
benchmarks extract `exp.fold_loci(loci, fold)["test_loci"]`. Anything new that evaluates per fold must too.

**Completion is `.final.torch`, not `.torch`.** Both libraries write `{name}.torch` every time validation
loss improves — so it can exist after one epoch — and `{name}.final.torch` exactly once at the end of
`fit()`. `model_path(..., final=True)` is the only safe test of completion.

**A bpnet-lite checkpoint IS a pickled module, not a state dict — load it with `experiments.load_model()`.**
`bpnetlite` persists with `torch.save(self, ...)`, so `weights_only=False` is required (PyTorch 2.6 flipped
the default) and reconstructing a `BPNet` to `load_state_dict` can never work — it raises
`TypeError: Expected state_dict to be dict-like`, and would silently take the architecture from the
*current* `bpnet_params.json` rather than from the checkpoint. Allowlisting with `add_safe_globals` is the
wrong remedy: these are this repo's own training output, not untrusted input.
**Cherimoya does not go through `load_model()`** — it saves a dict payload and reconstructs via
`cls(**payload['config'])` in its own `Cherimoya.load()`. Those are the only model-load sites in the repo.

**Heavy imports are deferred.** `torch`, `bpnetlite`, `cherimoya`, `tangermeme` and `data_loader` are
imported *inside* `main()`, after argparse and path validation, so `--help` and missing-data errors stay
instant on a login node and are testable without a GPU stack. Keep new imports in the same place.

**Fail fast on paths.** Each script builds a `[(label, path), ...]` list and exits nonzero listing every
missing file before doing any work.

## Cross-validation splits

Convention everywhere: **test = fold `i`, validation = fold `(i+1) % n_folds`, train = the rest.**
`n_folds` is per-species — C. elegans uses **6**, everything else 5.

Two mechanisms:

- *Chromosome-level* (default): whole chromosomes held out, passed to `extract_loci(chroms=...)`.
- *Peak-level* (**S. pombe**, **S. moellendorffii**): too few chromosomes, or none at all, so
  `make_random_splits.py` assigns individual peaks to folds, **grouping peaks whose training windows could
  overlap** (centers within `in_window + 2*max_jitter` = 2514 bp) so they never straddle a split. That
  grouping is the leakage guard — preserve it. Both species **error out** if the CSV is missing rather than
  falling back; two mechanisms for one species is a bug that was already removed once.

**`config/chrom_splits.yaml` is the single source of truth.** The per-species CSVs under `config/splits/`
are *derived* by `config/write_split_csvs.py`; regenerate rather than hand-edit, and
`python config/write_split_csvs.py --check` fails on drift. `--check` also verifies fold members against
the real contig names (`.fai`, `main_chromosomes`, `chrom.sizes`), because `extract_loci` matches
**literally** and a readable-but-wrong name yields **zero loci in silence**. Off-cluster it can only check
YAML/CSV agreement and says so — run it where the data lives.

**Folds are assigned by manually matching PEAK COUNTS across folds, never sequence length.** Check any
assignment with `python config/write_split_csvs.py --peak-counts -e <experiment>`, which also flags
chromosomes present in the peaks but absent from every fold (the naming-mismatch tell).

**There is NO public or authoritative source for ANY of these assignments — every one was made in this
lab.** Read "canonical" as "the shared lab assignment", never "published". Their whole value is that a
locus in test here is in test everywhere else we train, so **do not "improve" a reused entry**: a
divergence destroys that silently. Reused unchanged from earlier lab work: `A.thaliana`,
`D.melanogaster`, `M.musculus`, `S.cerevisiae`, `C.reinhardtii`, `P.patens`. Originating **here** and
needing pushing upstream before those species are used elsewhere: `C.griseus`, both cottons, the retuned
6-fold `C.elegans`, and S. moellendorffii's peak-level folds. Derivations, the paired-homoeolog constraint
for G. hirsutum, and the per-library balance notes are in `docs/cross-validation-folds.md`.

## Launchers: eight of them, one emission path

`src/launcher.py` holds the selection rule and the emission machinery; the eight `launch.py` files are thin
wrappers over `_add_common_args` and `_emit`. **It is the only thing in the repo that writes an sbatch
script — do not add another**, and do not add another copy of the flag block.

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

Why the job units differ, since each was a design question:

- **Benchmarking is per EXPERIMENT because it is forced.** Both scripts report a `genome_wide` block
  pooled over all folds, which cannot be computed if the folds are split across jobs.
- **Attribution is per (experiment x type), not per fold**, because `attribute.py` loops every fold
  internally and averages. It follows that a partly trained experiment is not partly attributable — the
  script exits 1, so the launcher skips it and reports `3/5 folds trained`.
- **`launch_filter.py` requests NO GPU**, which is why it is a separate launcher rather than a flag:
  filtering reads a FASTA and one-hot encodes, so sending it to the GPU partition would queue it behind
  training and then hold an idle card. It also gates on `missing_paths(kinds=("peaks", "sequences"))`
  rather than `exp.missing`, which would demand negatives and models it has nothing to do with.

**Partitions and the GPU constraint are DEFAULTED; everything else is not.** `src/launcher.py` holds
`GPU_PARTITION`, `CPU_PARTITION` and `GPU_CONSTRAINT` (the `|`-joined GPU SKU list — `|` is SLURM's OR, so
any one card satisfies it). Which set a launcher gets is derived from **one** argument,
`_add_common_args(..., gpu=)`, so the partition and the constraint cannot drift apart from each other or
from `_emit`'s `gpus`; `-C` is additionally gated on `gpus` at emission so a global `--constraint` cannot
narrow a CPU job to GPU nodes for nothing. Both stay overridable, so another site needs a flag rather than
a patch. **The `owners` partition caps jobs at 48:00:00** and is in every default here, so a longer
`--time` will simply not schedule there; `modisco motifs` sits exactly on that cap.

## Attribution, and the filter step it requires

Run order is **filter -> attribute -> modisco motifs -> modisco report**.

    attributions/{experiment}_filtered.bed          shared across families
    attributions/{experiment}_ohe.npz               shared across families
    attributions/{family}/{experiment}_attr_{type}_{mode}.npz
    modisco/{family}/{experiment}_{type}_{mode}.modisco.h5
    modisco/{family}/{experiment}_{type}_{mode}.modisco/

**The split follows what produces each file.** Attributions come out of a MODEL, so they live under
`attributions/{family}/` beside `models/{family}/` and `performance_metrics/{family}/`; motifs are
discovered FROM a model's attributions and inherit that provenance. The filtered BED and its one-hot
depend only on (loci, sequences, in_window), so every family shares them — putting them under a family
directory would imply they need producing twice and would mean `launch_filter.py` had to know which family
it was filtering for, which it does not. `attribution_path()` therefore takes `family` first, matching
`model_path()`; `filtered_loci_path()` and `ohe_path()` do not take one at all.

`attribute.py` is BPNet-only and pins `FAMILY = "bpnet"` at module level rather than taking a flag.
**Cherimoya attribution is BLOCKED, not merely absent — do not start it.** It needs DeepLIFT rescale rules
that are still in development and are not in tangermeme yet; the missing piece is upstream of this repo.
(`cherimoya/wrappers.py`'s `ExpectedCountsWrapper` composes `torch.expm1` with a per-group softmax, and
`expm1` is absent from tangermeme's rule table — treat that as the visible candidate, not the confirmed
blocker.)

### `filter_nonACGT_regions.py` is REQUIRED, and produces modisco's other input

**`deep_lift_shap` refuses a sequence containing an unknown base** — verified against tangermeme 1.4.1, in
**both** reference modes, so the frequency default does not rescue it. **And `extract_loci(ignore=IGNORE)`
is exactly what creates that column**: `ignore` KEEPS a locus containing an N and zeroes the column rather
than dropping the locus. So the setting every `extract_loci` call in this repo passes is what makes
attribution fail, and this script — which DROPS such loci — is the remedy. That is the whole reason it
exists; the `snp_bed` variable name is a leftover.

**The filtered set is the DEFAULT locus set, not an opt-in.** `attribute.py --loci` defaults to
`filtered_loci_path(exp.id)` and exits 1 with the `launch_filter.py` command if it is absent; the launcher
skips an unfiltered experiment rather than emitting a job that would fail. `--loci` survives only for a
genuinely different locus set, and only then does its stem enter the output filename.

**`--save-ohe` is not a convenience either — TF-MoDISco requires the one-hot** alongside the contribution
scores. That is the strongest argument for it living in the filter step: it must describe exactly the loci
that were attributed, and the filter is what decides which those are. `attribute.py` saves
**hypothetical** attributions (`hypothetical=True` unconditionally, mean over folds), which is also what
modisco wants; observed contributions are `hypothetical * one_hot`, derivable from the pair.

**A PINTS peak file is RAGGED, so the filter reads it line by line.** `combine_peaks` concatenates the
unidirectional and bidirectional calls, which carry different column counts, so
`pd.read_csv(peaks, sep="\t", header=None)` dies with `Expected 6 fields in line 2, saw 9`.
**`load_bed()` is unaffected** — its `usecols=[0,1,2]` reads a ragged file fine, which is why training,
benchmarking and attribution never hit this. Adding `usecols` to the filter would fix the read and be
wrong: its output is written back out as a BED, and upstream deliberately keeps PINTS'
strand/confidence/class/summit columns rather than cutting to BED3 — the summit column is what makes
`extract_loci(summits=True)` possible. So it keeps the raw line and returns `(kept_lines, coords)`. It also
handles **bgzipped** input, since that is what `combine_peaks` writes.

**THE TWO PEAK WIDTHS ARE NOT THE WAY ROUND THEY LOOK** — measured from the real files 2026-09-07.
`combine_peaks` is a plain `cat` plus a coordinate sort, so it adds **no class column**: the class is
recoverable ONLY from the field count.

| width | class | layout | example |
| --- | --- | --- | --- |
| **9** | **unidirectional** | chrom, start, end, name, **q-value**, **strand**, ?, summit, ? | `NC_053424.1 263 278 NC_053424.1-6 0.0141816 - 108 264 26` |
| **6** | **bidirectional** | chrom, start, end, **confidence label**, summit, summit | `NC_053424.1 55691 55926 Relaxed 55922 55692` |

The intuitive reading — "unidirectional is plain BED6" — is **wrong**. Unidirectional rows are the WIDE
ones (name, q-value, real strand); bidirectional rows are narrow and carry PINTS' `Relaxed` /
`Stringent(qval)` label plus **two** summits, which is what a divergent pair needs.
`src/analysis/stratify_peaks.py` checks both before splitting and aborts rather than writing misleading
subsets. **So per-peak confidence is already on disk** in two forms, and nothing consumes it yet —
`load_bed` reads columns 0-2 only, so training, benchmarking and attribution are all confidence-blind.

### Attribution deliberately does NOT hold out

It extracts every locus once and attributes it with EVERY fold's model, then averages — inherited from
upstream's `attribute_bpnet.py`, not accidental. So each locus is attributed by four models that saw it in
training plus the one that did not: it is an **ensemble attribution, not a held-out estimate**, and reading
it as evidence of generalisation would be wrong. `benchmark_predictions.py` is the per-fold held-out path
and is where generalisation numbers come from. For peak-level species `attribute.py` passes
`[... for c in (f["test_chroms"] or [])] or None`, i.e. no chromosome filter, which is correct rather than
a workaround: their peaks are all in the fold table and attribution does not hold out anyway.

**The reference is a nucleotide-FREQUENCY reference by default** (`--reference-mode {frequency,dinucleotide}`),
synced to upstream. A dinucleotide shuffle is not reliably NEUTRAL — upstream found shuffles producing
cryptic promoter-like signal, which makes the baseline reference-sensitive, the one thing a DeepLIFT
reference must not be. That argument is **stronger here**, because several of these genomes are far denser
than human: S. cerevisiae carries 1.2-4.1 peaks per 2114 bp window, so nearly every window contains a
promoter and a composition-preserving shuffle is more likely to reassemble something initiation-competent.
Two implementation details that matter: the reference is passed as a **callable**, so it is built per batch
and never reaches tangermeme's tensor-reference one-hot validator, which would reject a soft tensor; and
frequency mode forces **`n_shuffles=1`**, since that reference is deterministic.

**The reference mode and any `--loci` stem are part of the output path**, a deliberate divergence from
upstream's mode-less name: the two references give different numbers, and without it a frequency run
silently overwrites a dinucleotide one with nothing on disk recording which is which. `{stem}` comes from
`experiments.loci_stem()`, which strips **suffixes** rather than cutting at the first dot — every
experiment id in this corpus contains dots, so cutting at the first one reduced
`G.hirsutum-ovule_GROcap_uni.bed` and `G.arboreum-ovule_GROcap_bi.bed` both to `G`, and the second run
overwrote the first.

## TF-MoDISco

`src/bpnet/modisco/` shells out to the `modisco` CLI (nothing here imports `modiscolite`), one job per
(experiment x attribute type), with procap-atlas's parameters: `-n 1000000` seqlets, `-w 1000` window,
`--lite` on the report. **`-l` is 2, not procap-atlas's 50**: it is the number of Leiden CLUSTERINGS
(restarts with different seeds), not clusters, so 50 is 25x modisco's own default in compute for a
parameter upstream appears to have set under the wrong description.

**`modisco motifs` needs BOTH npz files**, the attribution and the one-hot — which is why the OHE lives
with the filter step.

**Both stages are CPU-only (`gpus=0`) and their resource defaults are NOT shared:**

| | CPUs | mem | time | `NUMBA_NUM_THREADS` |
| --- | --- | --- | --- | --- |
| `modisco motifs` | 32 | 64G | 48:00:00 | 32 |
| `modisco report` | **4** | 16G | **2:00:00** | **4** |
| fit launchers (for contrast) | 4 | 32G | 6:00:00 | unset |

Defaulting them together meant every report job reserved 32 idle cores for 48 hours. `report` is not
single-threaded though, and the numba pin is **not** a no-op for it: `modiscolite/report.py` calls
`memelite.tomtom`, which is `@njit(parallel=True)` and is invoked with **no `n_jobs`**, so it takes
memelite's default of `-1` and uses every numba thread available. Hence 4 cores — enough for that section
to be worth having, far short of motifs' 32 because the wall there is logo rendering and HTML.
`_add_modisco_args` takes `default_cpus`/`default_mem`/`default_time` as **required** keyword arguments
with no fallback, so the next caller cannot inherit the wrong set by omission.

**`NUMBA_NUM_THREADS` is pinned to `--cpus-per-task` on every modisco job.** numba otherwise sets it from
every core it can SEE, which on a shared node is the whole machine and not the slice SLURM granted — a job
holding 32 CPUs on a 128-core node spawns 128 threads and oversubscribes its own cgroup. It rides on the
command as a `VAR=value cmd` prefix rather than an `export` in the sbatch body, so one string carries it
through all three emission modes; an `export` would silently vanish under `--print-commands`, the mode most
likely to be run on a box where the variable matters.

**The MEME database is chosen PER SPECIES** — this is the one place the port could not follow upstream,
which hardcodes JASPAR CORE vertebrates because it is human-only. Reporting a yeast or plant motif against
a vertebrate database yields matches that mean nothing, so `config/genomes.yaml` carries
`jaspar_collection` per species and `experiments.motif_db_path()` resolves it:

| collection | species |
| --- | --- |
| vertebrates | M. musculus, C. griseus |
| insects | D. melanogaster |
| nematodes | C. elegans |
| fungi | S. cerevisiae, S. pombe |
| plants | A. thaliana, C. reinhardtii, P. patens, S. moellendorffii, G. arboreum, G. hirsutum |

**Fetch them with `src/bpnet/modisco/fetch_motif_dbs.py`** — standalone, and deliberately **not** a
Snakemake rule: the report is a convenience layer and the database has no effect on which motifs modisco
discovers, so it must not become a network dependency of the pipeline (same placement argument as
`make_negatives.py`, for a different reason). Twelve species collapse to 5 files, ~1.3 MB.
**WHICH files are needed comes from `motif_db_path()`, and the RELEASE is parsed back out of the filename
it returns** rather than being a flag, so a `--release` disagreeing with it cannot produce files nothing
reads; if that format string changes, the script exits saying so instead of guessing a URL.
**It checks CONTENT, not just the status code** — `bedbase.org/api/*` and `jaspar.elixir.no` both serve an
SPA catch-all as HTML with HTTP 200. A download must start with `MEME version` **and** carry at least one
`MOTIF` record, and it stages through `.incoming`, renamed only after that passes:
**a file at the final path always means complete and valid** (same convention as `fetch_fastqs.py`).
`--motif-db` overrides with a single file for every experiment, which is rarely right here.

**`modisco motifs` goes through `run_modisco.py`, because it crashes on a ONE-SIGNED track.** Upstream's
`extract_seqlets` thresholds the positive and negative sides of the contribution track
**unconditionally**, so a track with no negative windowed sum dies four frames deep in sklearn with
`ValueError: Found array with 0 sample(s)`, naming nothing about attributions. modiscolite already
supports the outcome (`neg_patterns = None`) and simply cannot reach it. `src/bpnet/modisco/run_modisco.py`
+ `src/modiscolite_compat.py` patch the threshold and hand `argv` to the real `modisco` script via
`runpy`, so every subcommand, flag and default stays whatever the installed version provides. It had to be
a CLI front end because nothing here imports modiscolite, so unlike `tangermeme_compat.py` there is no
in-process call site to patch. It is **self-retiring**: it functionally probes the installed version and
does nothing if the probe passes, and refuses to install a patch that fails its own probe.
`launcher.py` emits it for **motifs only**; `modisco report` still calls `modisco` directly.
Read the **exception type** to tell the degenerate cases apart — a `ValueError` rules out a blank or
corrupt npz, because an all-zero array fails with `IndexError` instead. Which experiments are near that
boundary, and why, is in `docs/investigations.md`; treat it as fragile for any low-variance counts head.

## Where benchmark output goes

| | BPNet | Cherimoya |
| --- | --- | --- |
| metrics JSON | `performance_metrics/bpnet/{experiment}.json` | `performance_metrics/cherimoya/{experiment}.json` |
| override | `--metrics-dir` | `--metrics-dir` |
| raw predictions | `--output-fname` (joblib, opt-in) | `--save-output` -> `predictions/cherimoya/` (npz) |
| printed | per-fold **and** genome-wide | per-fold **and** genome-wide |

Both write the same schema (`run_name`, `model_paths`, `per_fold`, `genome_wide`) so the two families are
directly comparable — that was the point of giving `benchmark_predictions.py` a metrics JSON at all; it
only printed until 2026-09-04. Both paths come from `experiments.metrics_path()` rather than a string
default in each script, so the launchers' already-done check cannot drift from where the scripts write.

**Genome-wide is POOLED across folds, not averaged over them** — `pearson_corr` over the concatenation, so
each locus counts once regardless of fold size. Averaging per-fold correlations would weight a small fold
equally with a large one, and for C. elegans, where one fold is one chromosome, sizes differ enough to
matter. Same construction in both scripts; keep them in step.

**Every metric here is a CORRELATION, so none of them can see a GLOBAL OFFSET.** A model whose output is
uniformly far too low but correctly *ranked* still scores well. Not hypothetical:
`G.hirsutum-ovule_GROcap` scores `log_counts_pearson` 0.49 while its counts head puts real peak sequence
at **e^-16 ≈ 1e-7 times** the counts of its own reference. It surfaced through a `modisco motifs` crash
survey, not through the benchmarks, and `compare_bpnet_cherimoya.py` cannot see it either. If absolute
calibration ever matters, that needs a new metric.

**`counts_pearson` is near zero for everything and is NOT a model failing** — 0.0037-0.0601 over both
cottons' ten folds while `log_counts_pearson` on the same folds is 0.46-0.60. Raw PRO-cap counts are
heavy-tailed enough that Pearson on them is dominated by a handful of loci. It is BPNet-only,
`compare_bpnet_cherimoya.py` skips it, and it should not be quoted.

**`--per-locus-tsv` writes coordinates, fold, per-locus profile Pearson and JSD, observed counts and
predicted log counts** — one row per evaluated locus, which makes depth-matching and stratified work
analyses rather than reruns. `--output-fname` cannot serve: it dumps `{preds, signals}` with **no
coordinates**. Rows are aligned with `extract_loci(return_mask=True)`, which is REQUIRED rather than tidy,
since that call drops loci falling off a contig end or inside an exclusion zone.

**THE MASK INDEXES THE INTERLEAVED LOCI, NOT THE LOCI AS PASSED.** `extract_loci` runs
`_interleave_loci(loci, chroms)` *before* its loop, so the chromosome filter is already applied by the time
the first `kept_mask` entry is appended. Indexing the passed frame is wrong by exactly the size of the
fold's chromosome subset — on the real tree, a mask of 36,732 against 171,640 loci. The frame is rebuilt
with tangermeme's own private `_interleave_loci` rather than by reproducing its filter, so the two cannot
drift and a rename upstream gives an immediate `ImportError` instead of a silent misalignment.
**A fixture without `chroms` CANNOT catch this** — with `chroms=None` the mask length equals the provided
length and naive indexing coincidentally works, so any test of this must pass a real chromosome subset.
**The mask is appended LAST**, so it must be popped before the `len(data) == 3` control-track test, or a
signals-only call plus a mask looks exactly like a call with controls and the mask is used AS a control
track.

Note upstream's `benchmark_bpnet.py` also reports `orientation_index_pearson`, which neither script here
computes. Not an oversight to fix silently — adding it means defining the orientation index the same way
upstream does, and is one piece of work with upstream's `--head orientation` attribution.

**Progress bars are ON by default** in the benchmarks and `attribute.py`, suppressed with `--no-progress`.
They are tangermeme's `verbose` argument, which is *only* the tqdm bar, so it is wired to `--no-progress`
rather than to `-v`: a long benchmark should show progress without turning on every other message. Bars go
to stderr, so stdout stays clean for the printed metrics. `attribute.py` also gets an **outer bar over
folds**, which is the one that matters there — each fold is a whole `deep_lift_shap` pass, so without it
the only feedback is tangermeme's inner bar restarting from zero.

## Comparing the two model families

`src/analysis/compare_bpnet_cherimoya.py` collates `performance_metrics/{bpnet,cherimoya}/{experiment}.json`,
inner-joins on experiment, writes `plots/bpnet_vs_cherimoya/collated.tsv` and one figure per metric
(scatter with y=x plus a histogram of per-experiment deltas and a Wilcoxon signed-rank test). The four
shared metrics are `profile_pearson`, `profile_jsd`, `log_counts_pearson`, `counts_spearman`; BPNet's extra
`counts_pearson` is skipped automatically rather than half-plotted.

Two deliberate departures from upstream: **points are coloured by SPECIES, not read depth** (upstream is
human-only, so depth is its only axis; here the question is whether one architecture wins uniformly or only
on some clades, which a 12-species corpus can answer — `--colour-by depth` restores the upstream view), and
there is **no consolidate step**, since reading the per-experiment JSONs directly removes a stage that
could go stale.

**`--aggregate` picks how folds are reduced, and the three options give genuinely different numbers:**

| | what it is | weights equally |
| --- | --- | --- |
| `fold-mean` *(default)* | mean of the per-fold metrics | every **fold** |
| `genome-wide` | the benchmark's pooled block | every **locus** |
| `per-fold` | one row per fold | — |

**`genome-wide` is NOT the mean of the per-fold correlations**, which is why this is a choice rather than an
implementation detail. `fold-mean` also carries `{metric}_sd` and `n_folds` and draws ±1 sd error bars —
worth having, because a bare point invites reading a 0.01 gap as real when the folds behind it span 0.05.
`per-fold`'s Wilcoxon p is **not interpretable**: folds of one experiment share an architecture, a library
and a peak set, so 5 × 42 is not 210 independent pairs. Use it to see spread, not significance.

Cherimoya is not deployment-ready, so treat anything this produces as a development comparison.

## Reading a `profile_pearson` — it is largely a DEPTH statistic

Measured on the cottons 2026-09-08 and the most consequential caveat in the repo. Within G. arboreum
alone, per-locus correlation runs **0.0040 in the shallowest count decile to 0.6617 in the deepest — a 165x
range**, while the largest *between-experiment* difference in the whole corpus is 4.5x. Depth-matching
accounts for **84%** of the AA-vs-AD gap. Peak-set confidence adds a second, independent gradient:
stratifying on PINTS q-value moves `profile_pearson` 3.7-4.9x from the most to the least confident
quartile, in **both** cottons.

Consequences when quoting any number:

- **Say which locus set it is on**, or match. The diploid's headline 0.2764 understates its model, which
  reaches 0.5883 on its confident quarter.
- **Only the PROFILE columns are valid across strata.** `profile_pearson` and `profile_jsd` are per-locus
  medians, so subsetting chooses which loci to median over. `log_counts_pearson` and `counts_spearman` are
  correlations ACROSS loci, so subsetting restricts the count range and depresses them mechanically —
  range restriction, not evidence.
- This is measured for the cottons only; the other 40 experiments are unchecked.

`src/analysis/matched_comparison.py` does the depth/confidence matching (validated on four fixtures with
known ground truth) and `stratify_peaks.py` the confidence split. **Only `profile_pearson` is a valid
response there**, for the reason above; the count columns serve as the covariate.

## Label generation: the ENCODE PRO-cap pipeline

Labels come from the ENCODE PRO-cap pipeline, transcribed verbatim from
`planning/20240501_PRO-cap_Computational_Pipeline.pdf` into `config/procap_pipeline.yaml` — tool versions
and exact parameter strings live there, and deviations for non-human species are marked `DEVIATION`.
**That file, not the package version, is what to check for reproducibility.**

    fastp 0.23.4 --overlap_len_require 18 --length_required 18
      -> STAR 2.7.11a --alignMatesGapMax 1000 --outFilterMultimapNmax 10
                      --outFilterMismatchNmax 1 --outFilterMultimapScoreRange 0 --outSAMattributes All
      -> samtools 1.18 -q 255 (STAR marks unique reads MAPQ 255)
      -> umi_tools 1.1.5 dedup      (UMI libraries only)
      -> 5' stranded bigWigs, merged across replicates
      -> PINTS 1.1.10 --min-lengths-opposite-peaks 5

ENCODE orchestrates these steps with [rmsp](https://github.com/aldenleung/rmsp), a DAG/caching layer;
Snakemake fills the same role here. Two drivers, same steps:

- **`workflow/Snakefile` — preferred.** Proper DAG: parallel, resumable, atomic outputs, per-rule
  resources. ~286 jobs for `tier=include`, ~908 for all tiers (approximate — `fetch_fastq` is one job per
  FASTQ *not already on disk*, so the total moves with local state). Scope is fetch -> negatives;
  `resolve_runs.py`/`build_experiment_config.py` stay outside as metadata work, and training stays on the
  launchers.
- **`src/data_preprocessing/run_procap_pipeline.py`** — single-experiment path, serial, caches on output
  existence. Useful for `-e <one>` debugging and `--fetch-genomes`/`--index-only`.

**Keep the two drivers in step.** Several values are resolved by a shared entry point for exactly this
reason (`decoy_accessions()`, `adapter_arg()`, `is_paired()`, `PINTS_CHROM_PREFIX`); where they diverged
before, one driver silently ignored a config field the other honoured.

Snakemake-specific rules:

- **The target name must come before `--config`.** `snakemake -n --config tier=include qc` makes Snakemake
  read `qc` as a config entry and die with "Config entries have to be defined as name=value pairs".
- **Run the dry-run after touching the Snakefile** — it is cheap, needs no cluster, and the first one ever
  run found a fatal pre-existing bug that took down the whole workflow at DAG construction:
  **a rule's output, log and benchmark must all carry the same wildcards**, and `fetch_annotation`
  declared `{species}.{ext}.gz` against a `{species}.log`.
- **`-c N` is a global thread budget**; `threads:` per rule is both the scheduler's cost and the value
  interpolated into the tool's own flag, so the two cannot drift. `samtools -@` takes *additional* threads,
  hence `-@ $(({threads} - 1))`. `pints` sets `threads` from `N_CHROMS` because PINTS parallelises across
  chromosomes — that count comes from `genomes.yaml` (genome structure), not `chrom_splits.yaml` (fold
  assignment), which is a different question and has no S. pombe entry.
- **Pass `--resources mem_mb=N` when raising `-c`.** Declared resources are inert without a limit, so
  `-c64` would start 5 mouse alignments and ask for 200 GB.
- **Transfers are capped at 4** by a custom `downloads` resource, defaulted via
  `workflow.global_resources.setdefault` (a custom resource with no limit anywhere is unconstrained, which
  is why the default is set rather than merely declared), overridable with `--resources downloads=N`. Use
  `snakemake fetch_only -c4` on a transfer node, then the compute phase with a large `-c`.
- **`LARGE_GENOME` is derived from `star_sa_index_nbases >= 14`, never from a species name.** A
  species-name test for a resource was a real bug: C. griseus is the same order as mouse and would have
  been handed 16 GB.
- Run-level intermediates are keyed by *run*, so a run shared by two experiments is mapped once, and a
  change to experiment grouping costs no re-alignment.

**Fetching is separate from mapping.** `fetch_fastqs.py` bulk-downloads every FASTQ the config references
(~202 GiB, 82 files over 64 runs) with md5 verification, so transfers can run on a login/transfer node and
a failed mapping run never re-downloads. Transfers stage through `data/fastq/.incoming/` and move to the
final path only once the md5 matches — **a file at the final path always means complete and verified.**
`run_procap_pipeline.py --fastq-dir` picks those up and only falls back to downloading on demand; set
`PROCAP_FASTQ_DIR` or symlink `data/fastq` to scratch on a cluster.
**Fetch only what the config references, not whole studies**: several deposits bundle unrelated assays and
organisms (PRJNA834081 is 8/11 human Ramos libraries; SRP131922 is 294 runs of which one is wanted;
PRJNA1105209 is 506 GiB of which 26 GiB is wanted). Bulk-fetching also does not substitute for the
sample->run crosswalk, which is what *labels* each file.

### Things that will bite you

- **Paired-end libraries must contribute ONE MATE ONLY to the signal.** `bedtools genomecov -5` reports
  the 5' end of *every* alignment record, so on a paired BAM it counts R1's 5' end (the initiation site)
  and R2's 5' end (the RNA 3' end, opposite strand) alike — roughly half the track becomes strand-flipped
  3'-end signal. `final_bam` applies `samtools view -f 64`, selected by `steps.signal.five_prime_mate`,
  **after dedup** because `umi_tools --paired` needs both mates. Only 8 of 42 experiments are paired, which
  is why this went unnoticed. **R1 is a convention here, not a documented fact** — validate it like
  `reverse_strand`, by confirming signal piles up at annotated TSSs rather than 3' ends.
- **Strand orientation.** `--reverse-strand` swaps which read strand becomes the plus track. R1/R2
  conventions vary across these deposits; validate at known unidirectional promoters before trusting a new
  dataset.
- **`--umi_loc` comes from the MANIFEST, not from the layout.** It was `read2 if paired else read1`, which
  was backwards for the only three experiments it applies to (the Spt5 UMI is at the **start of read 1**).
  `steps.dedup.umi_locations` maps the manifest's prose (`5' adaptor` -> `read1`) to fastp's value, with two
  guards that both fire: an unmapped prose value fails DAG construction and the serial driver, and a
  `3' adaptor` UMI on a *single-end* run is rejected outright, because there it sits at the read's 3' end
  where `--umi_loc` cannot reach it.
- **How a library is known to have a UMI: the manifest's `umi_len`/`umi_loc` columns, and nothing else.**
  **The archives cannot corroborate this** — ENA's `library_construction_protocol` mentions a UMI for
  *none* of the 45 runs, including the three that demonstrably have one. The scheme is **default-deny**: a
  library whose UMI was never noted keeps its PCR duplicates. That is the safer error, because
  deduplicating a non-UMI PRO-cap library destroys real stacked 5' ends — but a missed UMI is silent, and
  `umi_report.py`'s automated backstop is **blind for 6 of 12 species** (see `docs/investigations.md`), so
  this rests on manifest curation.
- **No UMI means no dedup**, by design: `final_bam` takes `unique.bam` directly when `has_umi(run)` is
  false, so the `dedup` rule never runs for 39 of 42 experiments.
- **fastp and umi_tools disagree about the UMI separator by default.** fastp appends the UMI with `:`;
  umi_tools' `--umi-separator` defaults to `_`. An SRA-style read name contains no `_`, so umi_tools takes
  the *entire read name* as the UMI and aborts with `AssertionError: not all umis are the same length`.
  Both drivers pass `--umi-separator` from `steps.dedup.umi_separator`; if you change fastp's UMI handling,
  change this with it.
- **`--paired` and `--method` are orthogonal in umi_tools** — layout versus clustering algorithm. The rule
  used to return one *or* the other, so paired runs silently got the default method while single-end runs
  were forced onto `unique`. Both now come from `steps.dedup`, with `--paired` added only for paired runs.
- **STAR's `--limitBAMsortRAM` defaults to the size of the GENOME INDEX**, so the sort budget is smallest
  exactly where libraries are deepest relative to the genome — counter-intuitively, the compact genomes
  break first. Both drivers pass it explicitly; the Snakefile derives it from the rule's own `mem_mb` minus
  an index reserve, and the serial driver takes `--star-sort-ram GB`. This is separate from Snakemake's
  `mem_mb`, which STAR knows nothing about.
- **bigWigs and PINTS are restricted to `main_chromosomes`, not the whole assembly.** `bedGraphToBigWig`
  records only contigs present in its input, so on dm6's 1,862 scaffolds one strand has reads where the
  other has none and PINTS aborts with `bw_pl and bw_mn should have the same chromosomes`. Scaffolds cannot
  enter a fold anyway. Decoys are excluded here too — they belong in the STAR index, not the peak-calling
  space.
- **PINTS' `--chromosome-start-with` defaults to `chr`, which silently matches nothing for the
  Ensembl-named species.** S. cerevisiae, S. pombe and A. thaliana would have produced **zero peaks with no
  error**. `PINTS_CHROM_PREFIX` maps each `chrom_style` and **raises** on an unknown one rather than
  falling through to `""` without anyone choosing that. This is a silent-wrong-answer class — check the
  PINTS log reports a sane chromosome count before trusting a new species.
- **The peak set is PINTS `unidirectional` + `bidirectional`, CONCATENATED and sorted — not
  interval-merged.** Following ProCapNet via procap-atlas: `sorted(uni + bi)` and nothing else. **Do not
  add `bedtools merge`** — collapsing overlapping intervals destroys the one-row-per-called-peak structure
  and folds unidirectional calls into overlapping bidirectional ones. `divergent` calls are excluded as a
  subset of the bidirectional set. Many non-human species have a large fraction of unidirectional TSSs, so
  both classes are needed; this union is the locus set for train, validation *and* test.
- **Replicates are merged at the BAM level, and that already IS summing.** `merge_runs` pools per-run BAMs
  with `samtools merge`, then `genomecov -5` runs once — numerically identical to summing per-run count
  tracks, because 5'-end counting is additive over reads. **Do not "modernise" this into a bigWig-level
  merge**: `bigWigMerge` defaults `-threshold` to 0 and drops values at or below it, so a minus-strand
  track stored as negative values merges to nothing. Nothing in this repo calls it any more.
- **rDNA decoy.** ENCODE aligns to genome + rDNA. Only mouse needs a sink here; every other species has its
  array in-assembly. See `docs/assemblies-and-references.md` — the rule is **sink only where the array is
  MISSING from the assembly**, and do not mask rRNA that lives on a real chromosome.
- **Spike-ins.** Ino80/Spt5 (S. pombe), LacZ (mouse MEFs) and Bcell (Drosophila) carry spike-ins. They must
  never contribute to target labels; the driver warns, but splitting them off is not yet implemented.
- **TAP− rows are controls, not targets**, and appear as `raw.protocol_controls` for background and
  specificity checks only.
- **Interleaved mates can be deposited as a single-end run.** `SRR19034544` is, and both ENA and SRA report
  SINGLE. `deposited_interleaved` in the manifest -> `raw.interleaved` -> a `deinterleave` rule and the
  matching serial branch. `library_layout` stays **SINGLE** (that is what the archive says);
  `is_paired()` is the PROCESSING decision (paired deposit **or** interleaved) and is what makes `-f 64`
  fire, while `run_fastqs()` is keyed on `RUN_LAYOUT` because it describes what is ON DISK. **Two traps in
  the awk**, both hit while writing it: it must be ONE source line (a literal newline inside the awk string
  makes it an unterminated string literal, and **a Snakemake dry-run will not catch this** — it does not
  parse shell content), and the braces must be doubled. `AWK_DEINTERLEAVE` in `run_procap_pipeline.py` is
  the same program; keep them identical.

### Adapters

**Two adapters, assigned per experiment from the manifest — not a lookup table in code.**

| adapter | sequence | studies |
| --- | --- | --- |
| `smallRNA_RA3` | `TGGAATTCTCGGGTGCCAAGG` | PRO-cap / ChRO-cap / CoPRO (= proseq2.0's `ADAPT1`) |
| `truseq_universal` | `AGATCGGAAGAGC` | 5'GRO / GRO-cap |

The chain is `planning/manifest_samples.tsv: adapter` -> `build_experiment_config.py::adapter_from_rows`
-> `raw.adapter` -> `steps.trim.adapters` (name -> sequence list) -> `adapter_arg()` in both drivers.
An empty value means none was detected and fastp auto-detects, which is the honest default.
**Four guards, all verified to fire:** a conflicting adapter within one project fails the generator; an
unknown name fails DAG construction *and* the serial driver; and `experiment_stats.read_survey_flags`
compares the manifest's curated name against the reads and emits `FAIL:adapter_mismatch` — the first three
are internal-consistency checks and none of them asks whether the name is **true**, which is how both
cottons sat on the wrong adapter while mapping and reporting clean.

**`steps.trim.adapters` maps a name to a LIST, and both drivers pass `--adapter_fasta`, not just
`--adapter_sequence`.** A dimer can be deposited TRUNCATED, and fastp matches an adapter by looking for its
*beginning*, so a read starting five bases into the adapter never matches, survives trimming as a
full-length run of pure adapter, and STAR discards it as `unmapped: too short` — 43.4% of one C. griseus
run. Configuring the shortened form *alone* would leave 5 bp on every real read, so the fix must be a
second sequence; fastp takes only one `--adapter_sequence`, hence the file. The `adapter_fasta` rule writes
it (a rule, not an inline write, so many trim jobs sharing one adapter cannot race).
**Both drivers also pass `--adapter_sequence` (the first entry)**, which suppresses auto-detection so
trimming is a pure function of the config rather than of the data; measured to cost nothing.
**Do NOT verify adapter work from fastp's `read1_adapter_counts`** — it reports **zero** for a sequence
that is discarding 45% of the library, because a read that is *entirely* adapter trims to length 0 and is
booked as `too_short`. Watch `too_short_reads`.
**Read `pct_short_untrimmed`, not `pct_adapter`**, in the read survey: every library carries a few percent
of adapter dimers that cost nothing to leave in, while `pct_short_untrimmed` is the share STAR will
actually discard. `--trim_poly_g --trim_poly_x` go to every library (inert where there is no tail; cotton
and GCB need them).
**STAR was deliberately NOT relaxed.** A dead-cycle concern was raised from a hand-picked read sample and
the survey refuted it — `deadCyc` empty for all files, N content never above 1.1% — so
`--outFilterMismatchNmax 1` stays. Check `qc/reads/*.tsv` before revisiting. Likewise **do not adopt the
cotton paper's `--outFilterMatchNminOverLread 0.33`**: it cannot touch homoeology (which is booked as
multimapping, not `too short`), and what it compensates for is an untrimmed adapter, which we fixed
instead.

## Assemblies, decoys and exclusion lists

Full record, including the rejected and reverted options, in `docs/assemblies-and-references.md`.
`config/genomes.yaml` is the single source for assembly, `chrom_style`, `main_chromosomes`,
`n_chromosomes`, decoy accessions, `rdna_regions`, `organelle_contigs` and `jaspar_collection`.

**Remap to the ENCODE/modENCODE standard assembly wherever one exists, and take chromosome naming from
it**: mouse **mm10** (not mm39 — there is no mm39 exclusion list, and ENCODE has not moved), *D.
melanogaster* **dm6**, *C. elegans* **ce11**, all chr-prefixed. Yeast, Arabidopsis and the other plants are
not ENCODE organisms and keep the bare Ensembl names their FASTAs use; the cottons use NCBI RefSeq
accessions. **`extract_loci` compares chromosome names literally**, so FASTA, peaks, bigWigs,
`chrom_splits.yaml` and the exclusion list must all agree — a mismatch fails **silently** (zero loci, or an
exclusion list that excludes nothing).

| species | assembly | exclusion list |
| --- | --- | --- |
| M. musculus | mm10 | Boyle-Lab v2, 3,435 regions |
| D. melanogaster | dm6 | Boyle-Lab v2, 182 |
| C. elegans | ce11 | Boyle-Lab v2, 97 |
| A. thaliana | TAIR10 | Klasfeld 20-inputs, **Boyle-Lab software**, versioned in-repo, 83 |
| S. cerevisiae | R64-1-1 | none published |
| S. pombe | ASM294v2 | none published |
| C. reinhardtii | Chlamydomonas_reinhardtii_v5.5 | none published |
| P. patens | Phypa_V3 | none published |
| S. moellendorffii | v1.0 | none published |
| C. griseus | CriGri-PICRH-1.0 | none published |
| G. arboreum | ASM2569848v2 | none published |
| G. hirsutum | Gossypium_hirsutum_v2.1 | none published |

**C. reinhardtii v5.5 and P. patens Phypa_V3 are the post-revert assemblies.** Both were briefly swapped
for near-gapless replacements and **reverted on 2026-09-07** — Chlamydomonas because the library's reads
match CC-503 better than the new assembly's CC-1690 strain, P. patens because Phypa_V3 is the only version
the UCSC browser supports and reverting restored its reused lab fold assignment. Do not re-swap
Chlamydomonas without a strain-matched chromosome-level assembly; none exists today. The reverts cost the
gap closure (v5.5 is 1,512 components against 17, ~5.6% of loci lost to gap-adjacent N) and returned
P. patens' spurious 27th chromosome, which sits in fold 0 — **do not silently drop `27` to fix that**,
since removing it is itself a divergence from the reused assignment.

Boyle-Lab lists arrive **gzipped** despite the raw.githubusercontent URL, so nothing gunzips them.
Arabidopsis is the exception to fetching: excluderanges ships only as R `.rds`, so it is converted and
versioned at `config/blacklists/TAIR10.Klasfeld.Excludable.bed.gz` — the set generated by the *same
Boyle-Lab software* as the other three, **not** the peakPass set, whose ML classifier uses gene annotation
as an input feature and would be a circularity hazard for a model of transcription initiation.
**An exclusion list is a training input and is in the DAG anyway** (`rule fetch_blacklist`, in `all`,
`fetch_only` and a standalone `blacklists` target) — the boundary that keeps the Snakefile venv-free is a
**dependency** one, and fetching a list needs only wget. `BLACKLIST_FILES` is built only from species with
a `blacklist_url`, and a URL pointing anywhere but `data/` **raises at DAG construction**.
Missing them is expensive and silent: while only `--fetch-genomes` fetched them, a Snakefile-driven run
left all three absent and `launch.py` skipped every mouse, fly and worm experiment — three files under
60 KB gating 114 of 214 training jobs.

**All twelve references are already alt-free** — checked against their real contig lists. Do not add a
contig-filtering step; there is nothing to filter. S. pombe's `MTR` and `AB325691` look like alts and are
not (additional *unique* sequence the chromosome assembly lacks), and Ensembl `dna.toplevel` is correct
because no `primary_assembly` file exists for any Ensembl species here.

**Decoys: sink only where the array is MISSING from the assembly, and never mask rRNA that lives on a real
chromosome** — reads landing on a genuine in-assembly rDNA locus are mapping to their true source. The
decoy is the *true, full-length* sequence, so rDNA-derived reads score better against it than against the
degenerate nuclear fragments; it is a **sink, not a multimapping trap**. Only **mouse** needs an rDNA sink
(`BK000964.3`); `organelle_accessions` additionally covers references that omit their own organelles
(C. reinhardtii, P. patens, C. griseus), which is a different problem — those reads cannot map at all.
Three traps:
**do not test for rDNA by grepping contig names** (UCSC renames dm6's rDNA scaffold to
`chrUn_CP007120v1`) **or by filtering on `rRNA_gene`** (Ensembl types these as `rRNA` under `ncRNA_gene`),
and **do not test for an organelle by contig length** — two contigs here are within 5% of an organelle's
length and contain none of its sequence. Match on sequence: 15 random 30-mers, both strands; 0/15 means
absent and 1/15 is what a NUMT looks like.
**Decoys go in the STAR INDEX ONLY** and are deliberately absent from `chrom.sizes`, because `bedgraph`
greps to `main_chromosomes` before writing — if you ever drop that filter, this changes with it. No
organelle appears in `main_chromosomes` for any species, so a decoy can never reach a fold, a peak or a
bigWig. **Adding a decoy raises `pct_unique` without adding usable signal** (those reads map, then get
filtered), so do not read the improved percentage as recovered training data. One trap when adding an
accession: the wildcard constraint is `[A-Z]{1,4}[0-9]{5,8}`, widened because `U03843` failed to match
and surfaced as a `MissingInputException` on `star_index` that said nothing about wildcards.

## QC

Both QC steps are rules in the DAG, so a normal run produces them; `snakemake qc -c8` runs only the QC
against existing signal. Outputs land in `qc/`. They stay inside the DAG only because `pybigtools`,
`pyfaidx`, `logomaker` and `matplotlib-base` are all conda-available — if any has to move to
`pyproject.toml`, the QC rules must leave the DAG with it, exactly as `negatives` did.
**Readers are `pybigtools` and `pyfaidx`, matching tangermeme** — the QC must see the data through the same
path the model does; pyBigWig/pyfastx are transitive deps only and have no macOS arm64 wheel.
**A stale conda env is the likeliest QC failure and it surfaces late**, after ~600 jobs of real work;
`require_deps()` names the missing packages. Update with
`mamba env update -f environment.yml -n nasti-critters`.

`src/qc/orientation_qc.py` exists because two things are conventions rather than documented facts, and
**both are silent when wrong**: which mate carries the RNA 5' end, and which read strand becomes the plus
track. Get either backwards and the pipeline still runs, still calls peaks and still trains — it just
models 3' ends, or the wrong strand. Three read-outs:

- **Initiator PWM + logo** around in-peak signal maxima. Annotation-free, so never gated. Information is
  **relative entropy against local base composition**, not against uniform — the null is the same windows'
  own flanks, because the question is whether a base differs from a random base *near a peak*. The
  **FLAT flag needs LOW AMPLITUDE *and* a MISPLACED maximum**: amplitude measures peak-set quality,
  POSITION is what says whether the 5' assignment is right, so the verdict prints the offset and flags only
  when it falls outside {−1, 0}. Do not rank libraries on bits alone — the value is diluted by peak-set
  size. The 0.15 threshold is **provisional**, calibrated on the superseded uniform-background measure.
- **Summit-anchored metaplot.** The statistic is a **log2 upstream/downstream ratio** over
  `|offset| ∈ [20, 300]`, not an argmax (on a channel with no peak, argmax lands wherever noise is highest
  and tripped a naive strand-swap test). Antisense owns the y-axis; the sense spike at 0 is guaranteed by
  the anchor. **An upstream peak at ~−100 bp is a TETRAPOD expectation, not a corpus-wide one** — for most
  taxa here its absence is the null, and only the DOWNSTREAM reading is ever a fault. **Never read the
  antisense *fraction* as independent evidence**: it correlates +0.939 with PINTS' bidirectional share, so
  it is close to a restatement of it. Band bound and threshold are provisional.
- **Stranded metaplot around annotated TSSs. THE PLOT IS THE CHECK — look at it.** Each site is normalised
  by its own window total before averaging, so every TSS carries equal weight; summing raw profiles let 1
  site in 1001 contribute 78.8% of the profile and invert its argmax. Sense and antisense share ONE
  denominator per site — normalising separately would equalise them and destroy the strand-swap comparison.
  **Do not trust the printed ratio as pass/fail** (it scores *better* for a displaced-signal case than for
  the correct one), and **do not reintroduce a numeric TSS-enrichment score**: deciding placement by eye is
  the requirement, and a score invites trusting the number over the picture.
  Flags here are gated on `annotation_tss_anchored` in `genomes.yaml` — where the annotated gene start is
  not the TSS the metaplot measures the annotation, not the pipeline, so its lines are advisory. Currently
  false for S. cerevisiae, C. elegans and S. moellendorffii.

Annotation is used **for QC only** — never for training, peak calling or fold assignment, so no
circularity reaches the model. A null `annotation_url` is a **supported state**, not a DAG failure:
`annotation_path()` returns no dependency, `orientation_qc` omits `--annotation` and `rrna_content` reports
NOT MEASURED. **Do not read `n_tss` as comparable across species** — the GTF branch selects transcripts of
all biotypes while the GFF3 branch selects protein-coding genes, and C. elegans refGene is ~47%
non-coding, which flattens its metaplot without anything being wrong. If the worm plot looks
unconvincing, look at fly or mouse before concluding anything about `reverse_strand`.

`src/qc/experiment_stats.py` reports what the pipeline actually produced per experiment, which is what an
exclude/merge argument needs — archive read counts cannot support one, since a library can arrive deep and
map badly. Key columns: `signal_reads` (**the number that matters** — reads in the merged BAM, counted over
`main_chromosomes` only, since `bedgraph` filters to those afterwards), `pct_unique_adj`
(**the mapping-quality number**; bare `pct_unique` is not, because where the rDNA array sits in-assembly in
near-identical copies every rRNA read is correctly dropped by `-q 255`, so the ceiling is set by the
organism), `reads_per_peak` (the cross-species-comparable depth measure and the basis of `thin_coverage`),
and `qc_flags`. **Blank means NOT MEASURED, which is not 0** — `pct_rrna` and `signal_reads` both return
blank rather than a misleading zero, and `rrna_indexed` distinguishes "measured 0%" from "never measured".
Every mapping flag carries a `,adj` or `,raw` suffix recording its basis. Flags contain commas themselves,
so naive splitting on `,` breaks them; new flags should avoid commas.

Two rules for the table, both earned by real corruption:
**never `--combine {input}`** (it swept in the rrna TSVs, the reads TSVs and the rule's own script, and
`csv.DictReader` absorbed all three silently, producing 685 rows over 42 experiments with every pipeline
column blank), and `--combine` **rejects any file whose header is not exactly `COLUMNS`** while skipping
files naming no configured experiment. `stats_table` combines the DIRECTORY rather than `TARGETS`,
deliberately: its output path is global, and **a subsetting flag must not narrow a global output**.
`qc/` is gitignored tables included — a committed copy makes Snakemake report "Nothing to be done" and
never rebuild them, and a tracked table that looks authoritative but is stale is worse than none.

## Snakemake: what invalidates what

Full detail in `docs/snakemake-operations.md`. The four rules worth knowing before you edit anything:

- **Editing a QC script is INVISIBLE unless the script is an `input:`.** Snakemake hashes a rule's own code
  and params, never the contents of a file those params merely name. All six report scripts are now
  declared as `input:` as well as `params:`; do the same for any new one, and remember it cuts both ways —
  editing a script re-runs every rule that uses it.
- **Editing `config/genomes.yaml` is invisible the same way, and worse.** It is read at parse time and is a
  declared input of nothing, so a rule notices only if the changed value is interpolated into its own
  `params`. Treat a `genomes.yaml` change as needing an explicit `--forcerun`, and work out which rules
  consume the field you touched.
- **Use DEFAULT triggers for a config change and scope it with `--config experiments=`.**
  `--rerun-triggers mtime` compares timestamps of a job's **existing** inputs, so it **cannot see a new
  input** — adding an organelle or rDNA decoy is invisible to it. Keep `mtime` for an edited SCRIPT, where
  the trigger is an existing input with a fresh timestamp.
- **Editing the `trim` rule re-runs everything.** It has two persistent outputs (`fastp.json`, `fastp.html`)
  alongside its `temp()` FASTQs, so unlike the deeper rules it still has metadata to compare against —
  and that cascades through align, peaks and QC for all runs. Defensible, but it should be a decision, not
  a surprise. The reverse also holds: a change *below* trim usually will NOT be picked up, because the
  intermediate chain is `temp()` and already deleted. Force those explicitly.

**One stale timestamp near the top of the chain rebuilds from FASTQ**, because the `temp()` design that
makes the pipeline cheap to store is what makes a top-of-chain mtime expensive to satisfy. Asking for 42
plots once planned **701 jobs**. The zero-risk way to run just the step you want is **`--forcerun` PLUS
`--allowed-rules`** (forcing alone still drags the chain; restricting the rule set is what stops it, and it
changes no DAG state):

```bash
snakemake orientation -j 48 --rerun-triggers mtime \
    --forcerun orientation_qc --allowed-rules orientation_qc orientation
```

**`--touch` DOES NOT WORK on this pipeline** once the FASTQs are deleted — `fetch_fastq` declares
`protected()`, and `--touch` calls `os.lstat` on the output, so it takes the whole invocation down with
`FileNotFoundError`. The repo's own lifecycle deletes FASTQs after mapping, so most DAGs are in this state.
**The escape hatch is that every QC script runs standalone** against what is already on disk — that is why
they take `-e`/`--all`. Three traps in doing it that way: run `rrna_content` **before**
`experiment_stats --all`, loop `orientation_qc` per experiment rather than `--all --tsv` (one path reopened
per experiment leaves a file describing only the last one), and check `ls data/annotation/` before
parallelising it, since concurrent jobs race for the same fetched file.

## Data conventions to preserve

- **Strand sign.** Minus-strand bigwigs may store signal as negative (UCSC convention) or positive. The
  codebase normalizes with `torch.abs()` on every signal/control tensor after `extract_loci`, and
  `make_negatives.py` abs-values the minus bigwig before merging strands — without it the strands cancel
  instead of summing. `src/bpnet/fit/data_loader.py` exists *only* to add these `abs()` calls around
  `bpnetlite`'s `PeakGenerator`. Any new code that reads signal must do the same.
- **Non-ACGT.** Every `extract_loci` call passes `ignore=list("QWERYUIOPSDFHJKLZXVBNM")`.
- **Windows** are `in_window=2114` / `out_window=1000` throughout; `trimming` is always
  `(in_window - out_window) // 2`.
- **Outlier peaks ARE dropped, at `quantile(0.99) * 1.2`**, inside `src/bpnet/fit/data_loader.py` — which
  is byte-identical to procap-atlas's, so upstream drops them too. `max_counts=None` in both fit scripts is
  a *different* knob (tangermeme's own cutoff inside `extract_loci`); this quantile filter is applied
  afterwards, on top. **Do not assume it is off.** The objection is live: the threshold is data-dependent,
  so every species and library gets a different effective cutoff, which is corrosive in a repo whose point
  is cross-species comparison — and the top of a PRO-cap distribution is real biology (snRNA, histone,
  ribosomal-protein promoters), i.e. the most informative loci for an initiation model. Decide deliberately.
- **Artifact removal is the exclusion lists' job, and those are canonical published lists** — **do not
  hand-curate regions into them**, or folds and preprocessing stop being comparable with procap-atlas.
  The real gap this leaves: S. cerevisiae and S. pombe have no published list *and* no outlier filter
  beyond the above, so their in-assembly rDNA arrays will be among the highest-signal PINTS calls. Those
  are real Pol I loci rather than mismapping artifacts and `log1p` compresses their influence, so it is a
  known and probably tolerable exposure — but check it before publishing yeast numbers.
- **Negatives live in `main_chromosomes`**, the same allow-list the bigWig and PINTS steps use, so
  negatives are drawn from exactly the space the peaks occupy. This applies to the peak set, the merged
  bigWig, **and the negative sampling space itself** — the last was silently missing while the script
  shelled out to `bpnet negatives`, whose `chroms` is the whole assembly and is used to *build* the
  candidate space. It is now a direct `tangermeme.match.extract_matching_loci(..., chroms=keep)` call;
  **do not go back to the CLI** unless bpnet-lite grows a `--chroms` flag. The filter matches column 0
  exactly, `str()`s the YAML names, and raises if `main_chromosomes` and the FASTA disagree. This replaced
  a hand-written per-*experiment* `CHROM_EXCLUDE` regex map that both under-covered (3 of 38 experiments
  had an entry) and was actively wrong (substring regexes over the whole BED line, so dm6's `_` dropped
  any peak whose *name* contained an underscore). **`ALPHA` stays per-experiment**, because it is a tuning
  parameter rather than a property of the genome.
- **A purely numeric chromosome column silently drops EVERY peak** inside `extract_matching_loci`: its
  `read_csv` has no `dtype`, so an all-digit column infers as `int64`, `numpy.isin` against string
  `chroms` is False everywhere, and the run dies later on `zero-size array to reduction operation`. It took
  out A. thaliana, C. reinhardtii and P. patens and spared C. griseus only because it has an `X`.
  `sample_negatives` reads the BED itself with `dtype={0: str}` and passes the **DataFrame**, skipping
  tangermeme's read entirely. **A fixture with `chrA`/`chrI`-style names cannot catch this** — the
  regression test uses all three naming styles on purpose.
- **The same upstream bug has a second instance in `tangermeme.io._load_exclusion_zones`**, loud rather
  than silent (`KeyError: 1`), and it could not be fixed at the call site because that function does its
  own `read_csv`. Renaming the BED's contigs would be worse than the bug — the published list was
  deliberately stripped to bare `1`-`5` to match the Ensembl FASTA, and a list whose names do not match
  **excludes nothing, silently**. So `src/tangermeme_compat.py` patches the one function, and
  **`patch_numeric_chroms()` is SELF-RETIRING**: it functionally probes the installed tangermeme and does
  not patch if the probe passes, and refuses to install a patch that fails its own probe. Called from all
  five scripts that pass a blacklist into `extract_loci`. **The tangermeme PR is worth two
  `dtype={0: str}` edits**, in `match.extract_matching_loci` and `io._load_exclusion_zones`.
- **`-j` in `make_negatives.py` is a PROCESS pool, and has to be** — the strand merge and GC matching are
  in-process now, so threads serialise on the GIL. It buys parallelism ACROSS experiments, not within one
  (`extract_matching_loci`'s internal `joblib` scan is pinned to `n_jobs=1`). Each worker holds a genome,
  so a large `-j` is memory-hungry on the multi-gigabase species.
- **Negatives ratio follows each family's OWN library default: 1/7 for BPNet, 1/4 for Cherimoya** (so
  negatives are 1/8 and 1/5 of a batch). The two libraries disagree and this number has been wrong here in
  three different ways — both fit scripts pass `params["negatives_ratio"]` explicitly, so bpnet-lite's own
  0.1 never applies at all.
- **`negative_ratio` is CAPPED at the available pool** in both fit scripts —
  `min(configured, len(negatives) / len(peaks))` — so no negative is drawn more than once per epoch, and
  the cap is printed when it engages. Computed from whole-genome counts, since `pool/peaks` is
  near-constant across folds and the per-fold number is not knowable without duplicating `extract_loci`.
  **The cap addresses RECYCLING, not DIVERSITY**: it lowers how often each negative is seen and with it the
  negative share of a batch, but adds no new background sequence. `--no-ratio-cap` keeps the configured
  composition on **both** scripts.
- **`NO_SIGNAL_FILTER = {"S.cerevisiae", "S.pombe"}`** is a deliberate, documented departure: for those
  two, negatives are representative peak-free background rather than the genome's silent tail, so anything
  derived from their negatives is not strictly comparable with the other ten species. The switch is
  three-state — `auto` (default), `--no-signal-filter`, `--force-signal-filter` — and
  `resolve_signal_filter()` returns the reason alongside the decision so every run prints which state it is
  in. **`NEGATIVE_WINDOW` is deliberately empty**: set an entry only with a measured `pct_peak_overlap` in
  front of you, and record why. The density measurements behind all of this, and why the four densest yeast
  experiments cannot be rescued by any setting, are in `docs/investigations.md`.
- **Cherimoya's `dtype=torch.bfloat16` is the one place this repo diverges from BOTH cherimoya and
  procap-atlas**, which use `float32`. `fit()` applies it through `torch.autocast`, so it is training-loop
  precision, not a storage detail — an unflagged numerical divergence in a repo that otherwise tracks
  upstream. **Decide it deliberately rather than inheriting it.**
- Cherimoya optimization splits parameters: **Muon** for 2-D weight matrices except `linear.weight` and
  `conv_weight`, **AdamW** for everything else, `lw0`/`lw1` to neither, each with linear warmup -> cosine
  decay. `warmup_epochs` and `decay_epochs` are config + CLI, both no-ops at their defaults. Setting
  `decay_epochs` shorter adds a **`ConstantLR` hold at `eta_min`**, which is load-bearing rather than
  cosmetic: `CosineAnnealingLR` is *periodic*, so without it the LR would climb again past `T_max`. Its
  `total_iters` is deliberately far larger than the hold, because `ConstantLR` reverts to the base LR once
  reached and would snap the LR back up on the last step.
- `load_config()` uses `yaml.safe_load` for both YAML and JSON (YAML is a JSON superset), so config files
  can be either format.
- `load_bed()` reads only columns 0-2 with `dtype={"chrom": str}` — chromosome names must stay strings
  (S. cerevisiae uses roman numerals, dm has `4`/`X`).

## Upstream reference: procap-atlas

[kundajelab/procap-atlas](https://github.com/kundajelab/procap-atlas) is the mature sibling of this repo
(ENCODE PRO-cap atlas, human K562), the source of truth for **io, sampling and training standards**, and
lives at `~/github/procap-atlas` locally.

**Copy its standards, not its assumptions.** It is human-only and hardcodes `data/hg38.fa`, its blacklist,
its cCREs and a 7-fold `% 7` split. This repo is multi-species: per-species FASTA from
`experiment_config.yaml`, species-keyed `chrom_splits.yaml`, peak-level folds where chromosomes cannot
serve. **Never** replace that logic with a wholesale copy of theirs.

Already synced — do not fork these: `src/bpnet/fit/data_loader.py` (byte-identical to theirs),
`src/cherimoya/fit/data_loader.py` (a ~46-line wrapper over `cherimoya.io.PeakGenerator`, with
`torch.abs()` layered on because upstream cherimoya no longer un-negates minus-strand values), both fit
scripts, and `attribute.py`'s DeepLIFT reference. Hyperparameters were aligned 2026-09-03; the table of
what differed is in `docs/decisions-and-history.md`.

**`early_stopping: null` is a decision with evidence behind it, so do not "restore" a value.** Upstream
swept it and settled on 50 epochs / no early stopping / 5 warmup; re-enabling it underperformed on **every**
benchmark metric, profile and count alike. The proposed mechanism applies to bpnet-lite identically:
`fit()` checkpoints whenever `valid_count_corr > best_corr`, a bare validation count-correlation
comparison, so more epochs give that rule more chances to overfit the validation set — and stopping on the
*same* metric compounds it. **Upstream states the caveat itself and it should travel with the number**:
none of those comparisons control for random initialization, and no `torch.manual_seed` is set anywhere in
either script. Treat it as upstream's considered default, weakly evidenced.

Deliberately not copied: `--background NAME:RATIO` multi-source negatives (its `ccre` source is
GRCh38-only), `--min-reads` (needs their `config/n_reads.txt`), `--head orientation`, upstream's
`hitcall/` tree (depends on Linux-only `finemo`), `modisco/relaunch_timeout.py` (site policy, not a
pipeline stage), and their predict/motifcompendium/model_upload trees.

### Cherimoya API compatibility

`cherimoya >= 0.2` broke the API this repo was written against. Both breaks are fixed, ported from
procap-atlas:

- `Cherimoya.__init__` has **no `n_outputs`**; use `signal_groups=[len(signals)]` — one 2-element group is
  one stranded (pl, mn) pair. A checkpoint saved by a pre-0.2 cherimoya whose stored config contains
  `n_outputs` will fail to load under the pinned version, since `load()` reconstructs via
  `cls(**payload['config'])`.
- `fit()` requires **`lw_optimizer` and `lw_scheduler`** (no defaults) — a third optimizer for the
  `lw0`/`lw1` Kendall uncertainty loss weights, stepped separately: `SGD(lw_params, ...)` with linear
  warmup then a flat `ConstantLR`, defaults in `config/cherimoya_params.json`.
- `PeakGenerator(signals=[params["signals"]])` — **nested**. A flat 2-element list now means two
  independent unstranded groups, which breaks reverse-complement channel swapping. `params["signals"]`
  itself stays flat for `extract_loci` and for `signal_groups`.
- `Cherimoya.load()` **defaults to `compile=True`**, so the benchmark was compiling unconditionally until
  2026-09-04. Both scripts now take an explicit flag with **deliberately opposite defaults**, because the
  warmup economics differ: `benchmark_cherimoya.py --compile` is opt-in (one inference pass does not
  amortise compilation), `fit_cherimoya.py --no-compile` is opt-out (50 epochs do). Both stay gated on
  `sys.version_info < (3, 14) or torch.__version__ >= "2.10"`, and the benchmark warns when `--compile`
  cannot be honoured rather than silently ignoring it.

## Standing decisions

- **Nothing is excluded: every dataset is analysed and modelled.** Model every experiment, including the
  problematic ones, and QC at the end. The reason is structural rather than optimistic — one experiment ==
  one model, no multi-tasking and no shared heads, so a weak dataset cannot contaminate a strong one, and
  a trained model is *better* evidence about a library than a pre-hoc read count is. So `tier` gates
  preprocessing only, `launch.py` deliberately does not filter on it or on `qc_flags`, and proposals to add
  an `exclude` tier have been declined. **The flags exist to tell you which numbers to distrust when
  reading results, not to decide what runs.**
- **`G.hirsutum-ovule_GROcap` is excluded from FINAL REPORTING ONLY.** Every analysis still runs and its
  outputs are kept and still read as diagnostics — it is what exposed the one-signed-attribution crash, and
  its contrast with the diploid is what established that `profile_pearson` is largely a depth statistic.
  What is excluded is reporting its motifs, attributions or metrics as evidence about biology: its
  `profile_pearson` is the corpus's lowest at 0.0621, its attributions are 100% sign-inverted, and its
  modisco report puts nearly everything in `neg_patterns` (so do **not** read that as "no motifs found",
  and do not read it as biology either). **Nothing in the code implements this and nothing should** — no
  `tier` change, no launcher filter, no skip in the analysis scripts. The exclusion is applied by a person
  deciding what goes in a figure. The remedy, if a reportable model is ever wanted, is **depth** (~2x the
  reads) or a stricter peak set, not reprocessing.
- **Keep the S. cerevisiae perturbation experiments.** The Ino80/Spt5 depletions carry the strongest,
  correctly placed initiator motif of any yeast library here and are not close to the corpus's worst data.
  The real caveat is narrower: *within-study, cross-condition* comparison in the Spt5 series is confounded,
  because cap-selection quality varies 11-fold across its conditions in the same direction as the peak
  counts. That does not touch a per-experiment model, so train them and deprioritise analyses that read
  *across* the Spt5 conditions — which the `tier` column already encodes.
- **Merge nothing further.** Within an experiment, replicates are already pooled at the BAM level. Across
  projects several experiments are the same biology and stay separate: depth does not force it (each is
  25 M+ reads), merging would pool batch with biology (different cap chemistry, different labs), and these
  clusters are the only cross-assay, cross-lab held-out sets in the repo — train on Kim2018 BMDM PRO-cap,
  evaluate on Link2018 BMDM GRO-cap. Merging destroys the one honest generalisation test available.
- **Depth requirements scale with the nascent transcriptome, not with a constant.** Ranking libraries by
  raw `signal_reads` across species penalises the compact genomes for being compact — S. pombe on 23 M
  reads is better sampled than mouse on 150 M. Hence `THIN_COVERAGE_READS_PER_PEAK = 500` rather than an
  absolute read floor. The threshold is a heuristic recalibrated from this corpus with no clean gap in the
  distribution; do not treat 500 as principled.
- **Assay provenance rests on the manifest.** All 64 runs were swept 2026-09-06 and every one names a
  cap-selected initiation assay, but **no archive field can verify cap selection** — `library_strategy` is
  `OTHER` for 51/64 and is `OTHER` for csRNA-seq and 5'GRO-seq alike. Re-run such a sweep by title, never
  by strategy, and treat a new dataset's assay as curated-until-proven.

## Scope note

Human K562 configs from other work in this lab are deliberately excluded from this repo.
`planning/nonhuman_capped_runon_manifest.xlsx` tracks candidate non-human datasets not yet wired into
`config/`. The manifest TSVs in `planning/` are the pipeline's actual inputs and hold curation the
workbook never received — **apply a workbook update by appending rows, never by regenerating the TSVs from
the xlsx.**
