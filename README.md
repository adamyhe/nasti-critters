# nasti-critters

**NAScent Transcription Initiation models in critters.**

Sequence-to-function models of PRO-cap / GRO-cap / ChRO-cap initiation profiles
across non-human species. 42 experiments over 12 species are defined in
`config/experiment_config.yaml`, all mapped uniformly from FASTQ:

| Species | Experiments | Assemblies |
| --- | --- | --- |
| *Drosophila melanogaster* | 5 (S2 cells, LacZ KD, 5'GRO, embryo 3-4 h / 6-8 h) | dm6 |
| *Mus musculus* | 13 (ESC, BMDM, liver young/old × female/male, B cells, 5'GRO, GRO-cap, MEF CoPRO ± heat shock) | mm10 |
| *Caenorhabditis elegans* | 4 (embryo, L1 starved, L3, sdc-2) | ce11 |
| *Saccharomyces cerevisiae* | 6 (WT, Ino80 ctl/KD, Spt5 EtOH/IAA 1 h/IAA 4 h) | R64-1-1 |
| *Schizosaccharomyces pombe* | 1 (WT) | ASM294v2 |
| *Arabidopsis thaliana* | 1 (6-day seedlings) | TAIR10 |
| *Cricetulus griseus* | 7 (CHO-K1, BMDM ± KLA, brain, kidney, liver, lung) | CriGri-PICRH-1.0 |
| *Chlamydomonas reinhardtii* | 1 (liquid culture) | v5.5 |
| *Physcomitrium patens* | 1 (plate culture) | Phypa_V3 |
| *Selaginella moellendorffii* | 1 (stems and leaves) | v1.0 |
| *Gossypium arboreum* | 1 (ovule, diploid AA) | ASM2569848v2 |
| *Gossypium hirsutum* | 1 (ovule, tetraploid AD) | Gossypium_hirsutum_v2.1 |

The mouse liver experiments are split by **sex**, not pooled: the manifest
records sex as a distinct replicate group, mouse liver is one of the most
strongly sex-dimorphic transcriptional programs known, and each sex carries
~97-116 M reads on its own.

Each experiment is one species x one biological condition and trains **its own
model** — conditions and assay families are not multi-tasked into shared heads.

Dataset selection, accessions, and caveats come from the curated manifest in
`planning/` (also exported as TSVs there). Human K562 PRO-cap configs from the
lab's other workspaces are intentionally not included here.

## Install

**Prerequisites:** [mamba or micromamba](https://mamba.readthedocs.io/) (conda
works, but is slow to solve this environment), [uv](https://docs.astral.sh/uv/),
and git. Training and benchmarking additionally need a **CUDA GPU** —
`fit_bpnet.py` hard-codes `.to("cuda")`. Everything else, including all config
and metadata work, runs on a laptop.

```bash
git clone https://github.com/adamyhe/nasti-critters.git
cd nasti-critters
```

**The two environments have separate jobs.** Preprocessing — fetch, map, signal,
peaks, negatives — runs entirely from the mamba env and never touches uv: it
needs `snakemake` plus the external binaries plus `pyyaml`/`pandas`/`numpy`, and
nothing in that path imports `torch`, `bpnet-lite`, `cherimoya` or `tangermeme`
(verified by loading every preprocessing entry point and inspecting
`sys.modules`). Model work — training, benchmarking, attribution — is the uv
side. So on a CPU or transfer node you can generate labels with no GPU stack and
no venv at all:

```bash
mamba env create -f environment.yml     # or: conda-lock install -n nasti-critters conda-lock.yml
mamba activate nasti-critters
snakemake -c16 --config tier=include
```

QC runs as part of the DAG, so a normal run produces it. Two diagnostics per
experiment land in `qc/`:

- `qc/orientation/{exp}.orientation_qc.png` — initiator logo around in-peak
  signal maxima, and a stranded metaplot around annotated TSSs. These are what
  catch a wrong `five_prime_mate` or `reverse_strand`, both of which are
  otherwise silent.
- `qc/umi/{exp}.tsv` — entropy screen for an undeclared UMI, checked against
  the manifest's `umi_len`.

`{exp}.tsv` next to each PNG carries the automated verdict, so you can scan all
24 at once:

```bash
head -1 qc/orientation/*.tsv | head -2; awk 'FNR==2' qc/orientation/*.tsv
awk 'FNR>1 && $NF!="no UMI signature"' qc/umi/*.tsv
```

Run just the QC against existing signal with `snakemake qc -c8`.

The DAG stops at peak calls. **GC-matched negatives are not part of it** — they
are a model-training input rather than a label. What keeps them out of the DAG is
`tangermeme`, which is PyPI-only and so can never live in `environment.yml`.
Build them from the venv before training:

```bash
uv run python src/make_negatives.py
```

For model work, add the uv venv on top. Activate it **last** so its interpreter
wins over anything conda pulled in:

```bash
uv sync                      # creates .venv from pyproject.toml + uv.lock
source .venv/bin/activate    # or run jobs with `uv run ...`
```

Both sides are lockable. `uv.lock` pins the Python environment;
**`conda-lock.yml` pins the conda side** — 124 packages for `linux-64`, with the
ENCODE pins (`fastp` 0.23.4, `star` 2.7.11a, `samtools` 1.18, `umi_tools` 1.1.5,
`pypints` 1.1.10) resolved together with every transitive dependency. Use it when
you need byte-reproducible label generation:

```bash
conda-lock install --name nasti-critters conda-lock.yml
```

`environment.yml` remains the source of truth; regenerate after editing it:

```bash
uvx --from conda-lock conda-lock lock -f environment.yml -p linux-64 --lockfile conda-lock.yml
```

The lock covers **`linux-64` only**, deliberately: `star 2.7.11a` has no
`osx-arm64` build, so adding that platform fails to solve. That matches how the
repo is used — preprocessing and training need the cluster, and the config and
metadata steps need no conda env at all.


### Check it worked

```bash
# config parses and all 42 experiments resolve
python src/data_preprocessing/run_procap_pipeline.py --list

# fold assignments agree with the YAML *and* with the real contig names
# (off-cluster it checks the YAML/CSV agreement only, and says so)
python config/write_split_csvs.py --check

# mapped reads + peak counts per experiment (blank columns = not mapped yet)
python src/qc/experiment_stats.py --all \
    -o qc/stats/experiment_stats.tsv --markdown qc/stats/experiment_stats.md

# external tools are on PATH (warns per missing tool under --dry-run)
python src/data_preprocessing/run_procap_pipeline.py --tier include --dry-run

# GPU visible, for training only
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The first two work anywhere. The third expects `fastp`, `STAR`, `samtools`,
`bedtools`, `bgzip`, `bedGraphToBigWig`, `pints_caller` and `wget`; mamba
installs all of them, so a missing tool usually means the env is not active.

### Dependency notes

`environment.yml` pins `fastp=0.23.4`, `star=2.7.11a`, and `samtools=1.18` to the
ENCODE PRO-cap spec; `pyproject.toml` pins `pyPINTS==1.1.10` and
`umi-tools==1.1.5` for the same reason. `uv.lock` is committed — use `uv sync`
rather than `uv pip install` to get the locked versions.

Two package names are traps, both avoided deliberately:

- the peak caller is **`pyPINTS`**; PyPI `pints` is an unrelated
  time-series inference library.
- **do not** install PyPI `muon` — it is a multi-omics framework, not the
  optimizer. `torch>=2.10` provides `torch.optim.Muon`, which is what
  `fit_cherimoya.py` uses.

Resolution is verified for linux x86_64 on Python 3.11, 3.12, and 3.13 (131
pinned packages; `torch==2.13.0`, `triton==3.7.1`, both with manylinux wheels).

**No compiler is needed, and no Python dependency builds from sdist.** Two
changes get you there, and both are deliberate:

- **`macs3` is blocked.** It is a transitive dependency of `bpnet-lite` and
  `cherimoya`, ships no wheels at any version, and nothing here imports it.
  `[tool.uv] override-dependencies` gates it behind an unsatisfiable marker,
  which is how a transitive dependency is dropped in uv. That also strands
  `cykhash` and `hmmlearn`, which only `macs3` wanted. The upstream sherlock
  image does the same thing with `pip install --no-deps`.
- **`umi_tools` and `pypints` live in conda, not uv.** The repo never imports
  either — it only runs `umi_tools dedup` and `pints_caller` via subprocess and
  probes for them with `shutil.which` — so they belong with `fastp`/`STAR`/
  `samtools` as pipeline binaries. bioconda ships `umi_tools` 1.1.5 for
  linux-64/aarch64 and osx-64/arm64, and `pypints` 1.1.10 as `noarch`.
  `umi_tools` additionally has no wheel at any version, so under uv it was the
  last sdist build in the tree. Moving `pypints` out also drops its exclusive
  tail — `biopython`, `pybedtools`, `statsmodels` and the latter's
  `formulaic`/`patsy`/`interface-meta`.

Consequently `environment.yml` carries **no `c-compiler`/`cxx-compiler`**. If you
add a Python dependency that needs compiling, add them back there rather than to
`pyproject.toml` — a compiler is not a Python dependency.

Everything else resolves to a wheel. The only source installs reachable from the
declared dependencies are `cherimoya` (pinned by git commit) and
`connection-pool`, both pure Python. The uv environment is 131 packages, down
from 145 before these two moves.

A container is **optional**. This repo does not define one: the authoritative
Apptainer images live in [adamyhe/sherlock](https://github.com/adamyhe/sherlock) and are maintained there.
Point the launchers at one with `APPTAINER_IMAGE`; they run natively when it is
unset.

`cherimoya` is pinned by git commit in `pyproject.toml` to the same commit that
image installs (0.2.1, reachable from git only — PyPI stops at 0.2.0). The two
are functionally identical: every module except `__init__.py` is byte-identical,
and the addition is only a `__version__` attribute. The pin exists for exact
agreement with the image and because the upstream `v0.2.0` tag has been
force-moved once, not because 0.2.0 behaves differently. `uv.lock` resolves
`torch==2.13.0`, matching that image's
`pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` base, so native and container
runs agree.

Nothing in the repo hard-codes a cluster path, partition, or GPU constraint.
Pass those at submit time (`sbatch --partition=... -C ...`,
`launch.py --partition ... --constraint ...`) and supply site module loads with
`launch.py --setup-file`.


## Runbook

Only the metadata steps (1) run anywhere. Everything from (2) on needs the
cluster: `fastp`, `STAR`, `samtools`, GNU `sort` (coreutils), `bgzip` (htslib),
`bedGraphToBigWig` (UCSC), `pints_caller`, and `umi_tools` on `PATH`, plus a GPU
for training. On Sherlock: `mamba activate torch` and the `ml` module loads from
`src/bpnet/fit/slurm.sh`.

### 1. Metadata (seconds, no cluster)

```bash
python src/data_preprocessing/resolve_runs.py            # GEO/SRA -> run accessions via ENA
python src/data_preprocessing/build_experiment_config.py # -> experiment_config.yaml, datasets.tsv
python src/data_preprocessing/run_procap_pipeline.py --list
```

### 2. Storage and fetch (transfer/login node — do not burn GPU time on this)

```bash
ln -s /scratch/users/$USER/procap_fastq data/fastq      # or export PROCAP_FASTQ_DIR
python src/data_preprocessing/fetch_fastqs.py --tier include --dry-run   # 40.2 GiB, 20 runs
python src/data_preprocessing/fetch_fastqs.py --tier include conditional -j 4
# --tier takes several values after one flag; repeating the flag also works.
# -j is concurrent ENA connections. Keep it low: ENA throttles by refusing
# connections, so a high -j produces FAILED transfers that look like data
# problems. 4-8 is the useful range; the Snakemake path caps downloads at 4.
python src/data_preprocessing/fetch_fastqs.py --tier include --verify-only
```

Add `--tier conditional` later for the other 59.0 GiB. Transfers are resumable
and md5-verified; rerunning skips anything already verified.

### 3. Genomes and STAR indices

```bash
python src/data_preprocessing/run_procap_pipeline.py --fetch-genomes --tier include
python src/data_preprocessing/run_procap_pipeline.py --index-only --tier include -t 16
```

Submit the mm10 index as a job, not on a login node — STAR `genomeGenerate` for a
mammalian genome needs roughly 32+ GB RAM.

### 4. Map one experiment and QC it before fanning out

```bash
python src/data_preprocessing/run_procap_pipeline.py -e S.cerevisiae-Ino80ctl_PROcap --dry-run
python src/data_preprocessing/run_procap_pipeline.py -e S.cerevisiae-Ino80ctl_PROcap -t 16
```

Then run the QC suite. It exists because several things in this pipeline are
conventions rather than documented facts, and every one of them is **silent**
when wrong — the pipeline still runs, still calls peaks and still trains:

```bash
snakemake qc -c8 --config tier=include,conditional   # target BEFORE --config
```

| output | what it catches |
| --- | --- |
| `qc/reads/{exp}.tsv` | adapter content, insert length, poly-G, interleaved mates. Survey the raw FASTQs *before* alignment burns the compute |
| `qc/orientation/{exp}.png` | initiator logo and a stranded TSS metaplot — whether the 5' assignment and the strand convention are right |
| `qc/rrna/{exp}.tsv` | rRNA + organellar share, which sets the ceiling on `pct_unique` |
| `qc/stats/experiment_stats.md` | one table over every experiment, with `FAIL:`/`WARN:` flags |

**The orientation plot is the check — look at it.** A flat initiator logo means
the maxima are not initiation sites, i.e. the 5'/mate assignment is wrong. If
the plus/minus tracks are inverted, re-run that dataset with `--reverse-strand`;
R1/R2 conventions differ across these deposits, so it is a per-dataset check.
The metaplot is deliberately **not** scored — a numeric TSS-enrichment score was
added and removed, because it scored 27.37 for a displaced-signal case against
6.31 for the correct one.

Read `pct_short_untrimmed`, not `pct_adapter`: every library carries a few
percent of adapter dimers that cost nothing, and the former is the share STAR
will actually discard. A `FAIL:adapter_mismatch` means the manifest's curated
adapter name disagrees with what is in the reads — believe the reads.

### 5. Map the rest, then build negatives and splits

Preferred: run the Snakemake workflow, which parallelises and resumes properly.

```bash
snakemake -n --config tier=include     # dry run
snakemake -c16 --config tier=include   # 16 cores
snakemake -c16                         # all tiers
snakemake -c16 signal_only             # stop at bigWigs (QC gate)
```

It covers fetch -> trim -> align -> unique -> dedup -> signal -> peaks ->
negatives, terminating at `data/procap/{experiment}_{pl,mn}.bw`,
`_peaks.bed.gz`, and `_negatives.bed.gz`. Job counts: 396 for all tiers, 211 for
`tier=include`, 25 for a single experiment. Experiments with unresolved runs or
an unresolved TAP+/TAP- split are skipped with a warning; override the latter
with `--config allow_bundled_tap=1`.

**A big `-c` does nothing during the fetch phase.** Transfers are capped at 4
concurrent by the `downloads` resource (ENA throttles by refusing connections),
and each is a single thread, so `-c64` runs 4 one-thread jobs. Fetch first on a
transfer node, then compute where the cores can actually be used:

```bash
snakemake fetch_only -c4  --config tier=include,conditional   # I/O bound, ~4 cores
snakemake          -c64 --config tier=include,conditional     # now CPU bound
```

**Pass a memory budget when you raise `-c`.** `align` declares `mem_mb=40000`
for mouse, but declared resources are ignored unless you supply a limit — so
`-c64` will happily start 5 concurrent mouse alignments and ask for 200 GB:

```bash
snakemake -c64 --resources mem_mb=180000 --config tier=include,conditional
```

`-c16` is a global budget, not a per-job setting: Snakemake packs jobs so the
sum of running `threads:` stays within it, and feeds `{threads}` straight into
each tool's own flag. Declared counts are `star_index` 16, `align` 12, `trim`
(fastp) 8, `pints` up to 8, `filter_unique`/`merge_runs` 4, everything else 1.
Note `--cores` *caps* `threads:`, so `-c8` silently runs `star_index` at 8.

`pints` scales its threads to the species' chromosome count, since PINTS
parallelises across chromosomes and reserving more just idles scheduler slots
(S. pombe has 3 chromosomes, A. thaliana 5, fly and worm 6).

Concurrent ENA transfers are capped at 4 by default via a `downloads` resource;
raise or lower it with `--resources downloads=N`.

Per-rule resources are declared (`star_index` asks 40 GB for mm10 vs 16 GB for
compact genomes), so adding a SLURM profile later is a small step:
`snakemake --executor slurm --profile <dir>`. No profile ships yet — for now run
inside one allocation with `-c`.

The single-experiment path remains available and implements the same steps:

```bash
python src/data_preprocessing/run_procap_pipeline.py --tier include -t 16
uv run python src/make_negatives.py -j 4
```

Two species use **peak-level** folds and need them built once their peaks
exist — S. pombe because 3 chromosomes cannot give 5 balanced folds, and
S. moellendorffii because v1.0 has no chromosomes at all:

```bash
python src/data_preprocessing/make_random_splits.py                    # S. pombe (defaults)
python src/data_preprocessing/make_random_splits.py \
    -e S.moellendorffii-stemleaf_5GRO \
    --output config/splits/S.moellendorffii_random_fold_assignments.csv
```

### 6. Train

```bash
python src/bpnet/fit/fit_bpnet.py -e S.cerevisiae-Ino80ctl_PROcap -f 0 -v   # smoke test
python src/bpnet/fit/launch.py --dry-run
python src/bpnet/fit/launch.py
```

Without SLURM, `--print-commands` emits the same job selection as bare shell
commands, one per line, and submits nothing:

```bash
python src/bpnet/fit/launch.py --print-commands | bash
```

### Open questions and resolved ones

Every experiment now has a fold assignment — `launch.py` reports **214
(experiment × fold) jobs and 0 unusable**. What is left is not blocking, but is
worth knowing before trusting numbers:

| Open | Effect |
| --- | --- |
| Unexplained mapping residuals | *P. patens* 38 pts, *C. reinhardtii* 48 pts, *A. thaliana* ~50%, all **after** adjusting for rRNA. Not explained by adapter, assembly or rRNA content |
| 18% of the cotton libraries in STAR's `unmapped: other` | not the match fraction, not the mismatch filter (0.00%), not multimapping — so no ENCODE parameter accounts for it. `--winAnchorMultimapNmax` is the untested guess |
| `P.patens` / `C.griseus` rDNA unresolved | both `null`; no reference sequence exists to use as a sink. C. griseus is a rodent, so it is the one to watch |
| `C.elegans` / `C.griseus` / both cottons folds originate here | carry them across before using those species elsewhere, or a locus in test here becomes train there |
| In-assembly rDNA in both yeasts | no published exclusion list and no outlier filter, so their rDNA arrays will be among the highest-signal PINTS calls. Real Pol I loci, not artifacts, and `log1p` compresses them — but check before publishing yeast numbers |

Resolved since this list was written, kept here so they are not re-investigated:

| Was | Resolution |
| --- | --- |
| `five_prime_mate: R1` unverified | validated by `snakemake qc`: all experiments put the initiator maximum at offset −1 or +0, with zero orientation flags |
| `C.griseus` / `S.moellendorffii` / cotton had no folds | all assigned. C. griseus and both cottons chromosome-level from peak counts; S. moellendorffii peak-level. C. elegans was also retuned onto **six** folds, one chromosome each |
| `S.moellendorffii` may fail PINTS | it did not; 32,973 peaks called over 189 scaffolds |

Resolved since this list was written, kept here so they are not re-investigated:

| Was | Resolution |
| --- | --- |
| E-MTAB-6154 (2 fly embryo rows) had no SRA data | ENA study `ERP106976` / `PRJEB25091` via the BioStudies API; all 4 runs resolved |
| `rdna_accession: null` for every species | mouse uses `BK000964.3`; the other five original species already carry their rDNA in-assembly, so they need none |
| `Tome2018_mm_CoPRO` "paired-end" | curation error of the same kind as Liver: ENA reports both runs SINGLE (1 FASTQ, no `nominal_length`, 76.0 bp). The paired description is of CoPRO the assay, not of the deposit |
| `Shamie2021_cg_5GRO` organism looked wrong | it is not: `BMDM/Brain/Kidney/Liver/Lung` read like Glass-lab mouse names, but ENA reports 72/72 runs as *Cricetulus griseus* — a Chinese hamster TSS atlas |
| `C.reinhardtii` / `P.patens` folds | reused unchanged from earlier lab work, after checking the two copies we hold against each other |
| `make_negatives.py` `CHROM_EXCLUDE` under-covered | replaced by `main_chromosomes` from `genomes.yaml`, applied to peaks, signal and chrom.sizes for every species. The old regexes were also actively wrong, not just sparse: matching `_` over the whole BED line dropped legitimate `chr2L` peaks whose name contained an underscore |
| `biodatatools` subcommand unrecorded | immaterial — see below |
| Booth2016 TAP+/TAP- "bundle" | curation error; the 2 runs per sample are technical replicates under one SRX, and no TAP- data was deposited. Both experiments now run |
| `Liver_ChROcap_mm` layout disagreement | curation error; all 8 runs are single-end. The manifest's "R1/R2" was the biological replicate, not the read |
| 5 weak `sample_name ~ sample_alias` matches | not weak: the Spt5 aliases are the sample_name with `rep`->`r`, bijective across all six runs, and every condition agrees. `resolve_runs.py` now reports ambiguity rather than flagging every name match |

### Why the unknown `biodatatools` subcommand does not matter

ENCODE produces its per-replicate bigWigs with `biodatatools`, and the source
document does not record the subcommand. That looked like a gap, but it is not,
because the *quantity* is pinned from both ends:

- procap-atlas consumes ENCODE's per-replicate bigWigs directly by accession
  (`ENCFF*.bigWig`, listed as `pl_bigwigs`/`mn_bigwigs` in its config) and sums
  them with `bigWigMerge`. Summing replicates is only meaningful for raw,
  unnormalised per-base counts, so that is what those files are.
- We produce the same quantity from the same read-level spec, then merge at the
  BAM level — arithmetically the same sum.

So the two pipelines agree on values without needing the subcommand. What the
subcommand would buy is byte-identity, which was never the goal. The distinction
that *does* matter is raw versus normalised: summing normalised tracks is
meaningless, which is exactly why the GEO-provided `R1Normed` yeast bigWigs had
to be re-mapped rather than reused.

## Data Preprocessing

### ENCODE PRO-cap pipeline (current)

Labels are generated with the ENCODE PRO-cap pipeline
(`planning/20240501_PRO-cap_Computational_Pipeline.pdf`, transcribed into
`config/procap_pipeline.yaml`): fastp -> STAR -> samtools unique -> umi_tools
dedup -> 5' stranded bigWigs -> PINTS. The training/validation/test locus set is
PINTS **unidirectional + bidirectional** peaks, concatenated and sorted (not
interval-merged), following kundajelab/ProCapNet. Many non-human species have a
large fraction of unidirectional TSSs, so both classes are needed.

```bash
# Resolve archive run accessions (all 64 manifest rows resolve via ENA)
python src/data_preprocessing/resolve_runs.py

# Regenerate config/experiment_config.yaml + config/datasets.tsv from the manifest
python src/data_preprocessing/build_experiment_config.py

# Bulk-download raw FASTQs (~202 GiB over 82 files, md5-verified, resumable)
python src/data_preprocessing/fetch_fastqs.py --dry-run
python src/data_preprocessing/fetch_fastqs.py --tier include -j 4

# Inspect what is defined and which experiments have resolved runs
python src/data_preprocessing/run_procap_pipeline.py --list

# Build STAR indices, then run the pipeline
python src/data_preprocessing/run_procap_pipeline.py --index-only --species S.cerevisiae
python src/data_preprocessing/run_procap_pipeline.py -e S.cerevisiae-Ino80ctl_PROcap --dry-run
python src/data_preprocessing/run_procap_pipeline.py --tier include -t 16
```

**Adapters are assigned per experiment from the manifest, and this was the
single largest correctness fix here.** Nothing configured a sequencing adapter
before, and it cost most of the reads in 14 of the 38 experiments that
existed when the survey was run: inserts sit
mid-read at a different offset every time, so fastp's auto-detection reported
`No adapter detected` and STAR discarded 95-99.85% of mouse liver as
`unmapped: too short`. Two adapters cover the corpus — `smallRNA_RA3` for the
PRO-cap/ChRO-cap/CoPRO lineage, `truseq_universal` for 5'GRO/GRO-cap — mapped to
sequences in `config/procap_pipeline.yaml` and resolved identically by both
drivers.

Both pass `--adapter_fasta`, not just `--adapter_sequence`, because a dimer can
be deposited **truncated**: 43.4% of `SRR12774945` begins five bases into the
TruSeq adapter, and fastp matches an adapter by looking for its *beginning*, so
those reads survive trimming as pure adapter. Four guards fire on a bad value —
a conflict within a project fails the generator, an unknown name fails DAG
construction *and* the serial driver, and a name contradicted by the reads
raises `FAIL:adapter_mismatch`.

Note the fix mostly **removes** reads rather than recovering them. For the
small-RNA libraries it recovers real inserts, but a truncated TruSeq dimer has
no insert behind it, so `input_reads` roughly halves while `unique_reads` holds
— an honest mapping rate, not extra depth.

The `--allow-bundled-tap` guard still exists but no longer fires for anything.
It was added for `S.cerevisiae_PROcap` / `S.pombe_PROcap` (Booth2016) on the
belief that TAP+ and TAP- runs shared one GEO sample; the archive says
otherwise — each sample's two runs sit under a single SRA experiment, so they
are technical replicates and merging them is correct. Both now run. The guard is
kept because it is right for a genuinely bundled deposit.

Raw FASTQs are fetched once, up front, rather than inside each mapping job.
Only runs referenced by `config/experiment_config.yaml` are downloaded: whole
studies would be several times larger, since many deposits bundle unrelated
assays and other organisms. `PRJNA978596` (McDonald2024) is the clearest case —
68 runs over 8 species, of which 4 are wanted. On a cluster, put them on scratch:

```bash
ln -s /scratch/users/$USER/procap_fastq data/fastq   # or set PROCAP_FASTQ_DIR
```

Outputs land in `data/procap/`. Until the pipeline has run, `launch.py` and
`make_negatives.py` report `SKIP: missing data` for every experiment — that is
the expected state mid-re-map. The three original experiments retain their
previous paths under `legacy_processed:` in `config/experiment_config.yaml`.

**`make_negatives.py` runs entirely in-process, and must stay that way.** It is
the one script here that runs from the **uv venv** rather than the mamba env, so
any binary from `environment.yml` is off `PATH` for it. It used to shell out to
three of them — `bigWigToBedGraph`/`bigWigMerge` to sum the strands, `bgzip` to
write the filtered peaks, `samtools faidx` to build a missing index — and the
first of those is what produced
`ERROR ...: [Errno 2] No such file or directory: 'bigWigToBedGraph'`. All three
are now `pybigtools`, the `gzip` module and `pyfaidx`. **Before adding a
subprocess call here, check which environment provides it.**

Generate GC-matched negatives from `config/experiment_config.yaml`:

```bash
uv run python src/make_negatives.py
```

To process only one dataset:

```bash
uv run python src/make_negatives.py -e S.cerevisiae_PROcap
```

**The two yeasts are sampled without the signal restriction, and that is a
deliberate departure from bpnet-lite and procap-atlas.** S. cerevisiae and
S. pombe are transcribed densely enough that the default threshold — window
signal at or below `signal_beta x` the 1st percentile of peak signal — leaves
only a few hundred candidate windows in the whole genome. Dropping it recovers
3-4x more at the sparse end, with 0.0% peak overlap and median signal well
under a random window, so there is no contamination traded for the gain. GC
matching, the N-content filter and peak-tile masking all still apply.

The species are recorded in `NO_SIGNAL_FILTER` in `src/make_negatives.py`, so
the choice lives in the repo rather than in shell history, and every run prints
which state it is in. Two flags override it for one run:

```bash
uv run python src/make_negatives.py --no-signal-filter     # off for every species
uv run python src/make_negatives.py --force-signal-filter  # on for every species
```

It changes what a negative *means* for those two species — representative
peak-free background rather than the silent tail of the genome — so treat yeast
negatives-derived metrics as not strictly comparable with the other ten.

**Training caps `negatives_ratio` at the pool that actually exists.** The
configured ratio is 1/7 for BPNet and 1/4 for Cherimoya (negatives per peak), and
the dense yeast experiments have far fewer negatives than that implies, so both
fit scripts lower it to `len(negatives) / len(peaks)` and say so. No negative is
then drawn more than once per epoch. Pass `--no-ratio-cap` to keep the configured
batch composition and accept the repeats instead.

### Cross-validation splits

`config/chrom_splits.yaml` is the single source of truth for chromosome folds,
keyed by species. The per-species CSVs in `config/splits/` are derived from it:

```bash
python config/write_split_csvs.py           # regenerate all
python config/write_split_csvs.py --check   # verify, non-zero exit on drift
```

For fold `i`: test = fold `i`, validation = fold `(i+1) % n_folds`, train = the
rest. Assignments are made by matching **peak counts** across folds, not sequence
length — bp totals are not expected to match:

```bash
python config/write_split_csvs.py --peak-counts -e S.cerevisiae-Ino80ctl_PROcap
```

For a species with **no** entry yet, the same flag reports peaks per
*chromosome* instead — which is the input you need to build one. `--check` also
verifies every fold member against the real contig names, because `extract_loci`
matches literally and a readable-but-wrong name yields **zero loci in silence**.

There is **no public source for any of these fold assignments** — they are all
the lab's own. Six are reused unchanged from earlier projects (A. thaliana,
D. melanogaster, M. musculus, S. cerevisiae, C. reinhardtii, P. patens) so
models stay comparable with our other work. Four **originate here** and should
be carried across before those species are used elsewhere:

| species | folds | note |
| --- | --- | --- |
| *C. elegans* | **6**, one chromosome each | six near-equal chromosomes mean any 5-fold split pairs two and lands at 76% of target; six folds give 12% |
| *C. griseus* | 5 | the two unplaced scaffolds share a fold — they are the arms of the unassembled chromosome 1 |
| *G. arboreum* | 5 | 13 diploid chromosomes, exhaustively optimal at 14.1% spread |
| *G. hirsutum* | 5 | **homoeologs share a fold** — A0i and D0i are ~96%+ identical, so splitting a pair puts the same sequence on both sides of train/test. Costs 19.0% spread against 2.8% unpaired |

**S. pombe and S. moellendorffii are absent from `chrom_splits.yaml` on
purpose**, for different reasons: three chromosomes cannot give five balanced
folds, and *S. moellendorffii* v1.0 has no chromosomes at all (759 scaffolds,
empty karyotype). Both use random peak-level folds and nothing else — asking for
their chromosome folds is an error, not a fallback. Generate them once peaks
exist:

```bash
python src/data_preprocessing/make_random_splits.py
python src/data_preprocessing/make_random_splits.py --check   # verify
```

That writes `config/splits/S.pombe_random_fold_assignments.csv`, which both
training scripts pick up automatically. The file carries a `#`-comment header
recording the peaks it was built from (`peaks_sha256`, `n_peaks`) and the
parameters (`n_folds`, `seed`, `window`, `jitter`), so `--check` can tell you
whether it is still current. Peaks whose training windows could overlap
(centers within `in_window + 2*max_jitter` = 2514 bp) are grouped and never
straddle a split — that is the leakage guard.

The digest is order-sensitive: the CSV is written in its input's row order and
joined to the peaks positionally, so a peak set with the same content in a
different order is not interchangeable. A mismatch raises rather than silently
re-joining on coordinates, which on a concatenated peak set containing duplicate
intervals could otherwise assign the wrong folds.

## rDNA sink

ENCODE aligns to genome + rDNA. The decoy works as a **sink**: it is the true,
full-length rDNA sequence, so rDNA-derived reads align to it better than to the
degenerate rDNA-like fragments scattered through the nuclear genome, and STAR
takes the best alignment. Reads whose real source is a nuclear locus still map
there and are kept. This matters for PRO-cap because a nuclear run-on captures
Pol I as well as Pol II, so these libraries carry plenty of rDNA reads.

**Sink only where the array is missing from the assembly; never mask rRNA that
lives on a real chromosome.** Reads on a genuine in-assembly rDNA locus are
mapping to their true source — masking that discards real signal. The problem
worth solving is spurious alignment *elsewhere*.

Only mouse qualifies: mm10 lacks the array (it has just `Rn18s-rs5`, a dispersed
18S copy on chr17), so it uses `BK000964.3` — 45,306 bp, "complete repeating
unit", the analogue of the human `U13369.1`. It is fetched from ENA into
`data/decoy/`.

**Decoys go in the STAR index only, and are deliberately absent from
`chrom.sizes`.** `bedgraph` filters to `main_chromosomes` before writing, so
decoy rows never reach `bedGraphToBigWig` — the same treatment `chrM` already
gets. A decoy can therefore never reach a fold, a peak or a bigWig.

Every other species already has its rDNA. C. elegans, both yeasts and
Arabidopsis carry theirs on real chromosomes; Drosophila carries it on the
unplaced scaffold `chrUn_CP007120v1` (76,973 bp, "chromosome X; Y rDNA
sequence"), which is a built-in sink that can never enter a fold.

Note dm6's rDNA is invisible to a name grep — UCSC renames scaffolds to
accession-based `chrUn_*` names — and Ensembl types rRNA features under
`ncRNA_gene` rather than `rRNA_gene`. Both make rDNA look absent when it is not.
See `config/genomes.yaml` for coordinates and the full audit.

**Organelle decoys use the same mechanism for the opposite problem.** Where the
rDNA sink prevents *mis*mapping, these fix reads that cannot map at all: the two
plant references carry **no organelle contigs**, which strands 12.2% of the
*C. reinhardtii* library and 8.6% of *P. patens* as `unmapped: too short`. Both
get chloroplast and mitochondrial decoys; C. griseus gets a mitochondrial one.

Verify presence by **sequence, not by length** — 15 random 30-mers against the
assembly on both strands. `KZ454947` is within 5% of the Chlamydomonas
chloroplast's length and contains none of its sequence, and a 1/15 hit is what a
NUMT looks like, not a present genome. Adding a decoy raises `pct_unique`
without adding usable signal, since those reads are filtered out again at the
`bedgraph` step; the gain is correctness, not depth.

## Alignment analysis set

ENCODE aligns to no_alt references, so all twelve references here were checked
against their real contig lists. **All twelve are already alt-free**, and mm10's
contig set is identical to ENCODE's `mm10_no_alt_analysis_set_ENCODE`. No
filtering step is needed or wanted.

*S. pombe*'s `MTR` and `AB325691` look like alts but are not: `MTR` (FP565355) is
the silent mat2/mat3 mating-type cassettes, a distinct locus from mat1 on
chromosome II, and `AB325691` is gap-filling sequence *missing* from chromosome
II's left arm. Both are unique sequence the assembly lacks, so excluding either
would lose real sequence or misplace its reads. See `config/genomes.yaml` for
the full audit.

## Assemblies and exclusion lists

Data are remapped to the ENCODE/modENCODE standard assembly where one exists, and
chromosome naming follows it:

| species | assembly | naming | exclusion list |
| --- | --- | --- | --- |
| *M. musculus* | mm10 | `chr1`… | Boyle-Lab v2, 3,435 regions |
| *D. melanogaster* | dm6 | `chr2L`… | Boyle-Lab v2, 182 regions |
| *C. elegans* | ce11 | `chrI`… | Boyle-Lab v2, 97 regions |
| *S. cerevisiae* | R64-1-1 | `I`… | none published |
| *S. pombe* | ASM294v2 | `I`… | none published |
| *A. thaliana* | TAIR10 | `1`… | Boyle-Lab software on 20 inputs, 83 regions (in-repo) |
| *C. reinhardtii* | v5.5 | `1`… | none published |
| *P. patens* | Phypa_V3 | `1`… | none published |
| *S. moellendorffii* | v1.0 | `scaffold_1`… | none published |
| *C. griseus* | CriGri-PICRH-1.0 | `1`… | none published |
| *G. arboreum* | ASM2569848v2 | `NC_069070.1`… | none published |
| *G. hirsutum* | Gossypium_hirsutum_v2.1 | `NC_053424.1`… | none published |

Mouse is **mm10, not mm39**, following ENCODE — this overrides the manifest's
recommendation. Yeast and Arabidopsis are not ENCODE organisms, so they keep
Ensembl bare names.

**The two cottons are the only NCBI-sourced genomes**, since neither species is
in Ensembl Plants. FASTA and annotation come from the same RefSeq release, which
is what keeps their names consistent — but those names are **RefSeq accessions**
rather than the community `Chr01` / `A01`–`D13` labels, so `main_chromosomes`
and `chrom_splits.yaml` are accessions too. `extract_loci` matches literally, and
a readable-but-wrong name yields zero loci in silence; the accession-to-name
mapping is in each assembly's `*_assembly_report.txt`. Exclusion lists are fetched alongside the genomes:

```bash
python src/data_preprocessing/run_procap_pipeline.py --fetch-genomes --tier include
```

## Upstream reference

io, sampling, and training standards are tracked from
[kundajelab/procap-atlas](https://github.com/kundajelab/procap-atlas), the mature human-K562 sibling of this repo.
Its standards are copied; its assumptions are not — procap-atlas is human-only
(hardcoded `hg38.fa`, `GRCh38-cCREs.bed.gz`, 7-fold splits), whereas this repo is
multi-species with per-species FASTAs, species-keyed `chrom_splits.yaml`, and
peak-level random folds for S. pombe.

Synced so far: `src/bpnet/fit/data_loader.py` (byte-identical), the
`data_loader.PeakGenerator` delegation in `fit_bpnet.py`, the cherimoya >= 0.2 API
port in `src/cherimoya/fit/`, and the `.final.torch` completion check in
`launch.py`. See CLAUDE.md for the full list and for what was deliberately not
copied.

## Train BPNet Models

Run from the repository root in the `torch` conda environment. Experiment IDs
are the keys of `config/experiment_config.yaml`:

```bash
python src/bpnet/fit/fit_bpnet.py -e D.melanogaster-S2_PROcap -f 0
python src/bpnet/fit/fit_bpnet.py -e S.cerevisiae-Ino80ctl_PROcap -f 0
python src/bpnet/fit/fit_bpnet.py -e M.musculus-liver-young-female_ChROcap -f 0

# Submit all (experiment x fold) jobs. --requeue helps on preemptible partitions.
python src/bpnet/fit/launch.py --dry-run
python src/bpnet/fit/launch.py --partition gpu --requeue

# No SLURM: the same selection as bare commands, nothing submitted
python src/bpnet/fit/launch.py --print-commands | bash
```

`launch.py` has three emission modes over one selection rule — `--print-commands`
(bare commands on stdout), `--dry-run` (full sbatch scripts) and the default
(submit). The two flags are mutually exclusive. With `--print-commands`, skip
messages and the summary go to **stderr**, so stdout stays pipeable; the env
setup block is not included, so activate the mamba env and uv venv first. These
are GPU jobs, so `| bash` runs them serially — use
`| xargs -P N -I{} bash -c '{}'` only if N models are known to fit in VRAM.
Cherimoya has the same launcher, over the same selection rule:

```bash
python src/cherimoya/fit/launch.py --dry-run
python src/cherimoya/fit/launch.py --print-commands | bash
```

Both are thin wrappers over `src/launcher.py`. The selection logic — which
(experiment, fold) pairs exist, which finished, which lack inputs — is
family-agnostic because `src/experiments.py` is keyed by family, so there is one
implementation rather than two that can drift. The only family difference is
`--controls`, which exists on the BPNet launcher and not the Cherimoya one,
because `fit_cherimoya.py` has no such flag.

A fold counts as done only when `{experiment}.fold{f}.final.torch` exists.
bpnet-lite also writes `{experiment}.fold{f}.torch` whenever validation loss
improves, which can happen after a single epoch — `launch.py` deliberately does
not treat that as completion.

Submit all configured experiments and folds on SLURM:

```bash
python src/bpnet/fit/launch.py --dry-run
python src/bpnet/fit/launch.py
```
