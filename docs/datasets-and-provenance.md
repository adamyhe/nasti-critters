# Datasets, curation and provenance

Per-dataset record: what was added when, what the archives say versus what the reads say, which
curation errors were found and resolved, and the merge/exclude decisions. `CLAUDE.md` carries the live
rules that came out of this work; this file carries the provenance.

The manifest TSVs in `planning/` are the pipeline's actual inputs and hold curation the xlsx never
received. **Apply a workbook update by appending rows, never by regenerating the TSVs from the xlsx** —
regenerating silently reverts every correction recorded here.

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

- ~~**`C.griseus` and `S.moellendorffii` have no folds yet.**~~ **Closed.** C. griseus was assigned
  chromosome-level from CHO peak counts on 2026-09-01 and S. moellendorffii got peak-level folds on
  2026-09-02. Both originate here and still need pushing upstream — see
  [cross-validation-folds.md](cross-validation-folds.md).
- **`S.moellendorffii` is the species most likely to hit the PINTS `bw_pl and bw_mn should have the same
  chromosomes` failure**, because `main_chromosomes` is 189 scaffolds and sparse ones can have reads on
  one strand only. Remedy is in `config/genomes.yaml`: raise the length cutoff (>= 500 kb keeps 118
  scaffolds and 87.7% of the assembly) rather than dropping the species.
- **`P.patens` and `C.griseus` rDNA are unresolved** (see the rDNA table in
  [assemblies-and-references.md](assemblies-and-references.md)). Neither has a reference
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

**C. reinhardtii's rRNA is measured from its GFF3 `rRNA` features again, following the 2026-09-07
revert to v5.5; `rdna_regions` is back to `[]` and `pct_rrna` back to the 39.1% basis.** The paragraph
below describes why the coordinates had to be measured while ASM4749649v1 was in use, and is kept because
the failure mode it documents is general.
**Under ASM4749649v1 the annotation-derived measurement was unavailable.** Its `pct_rrna` came from v5.5's **GFF3 `rRNA` features**, and
ASM4749649v1 has no annotation at all — so with `rdna_regions: []` the only thing left to score against
was `data/decoy/`, i.e. the two organelles, giving an **organellar-only ~12%** where the previous figure
was **39.1% rRNA + organellar**. That is exactly the false negative `rdna_regions` exists to prevent for
dm6 and ce11: a number that looks measured while nuclear rRNA is silently absent, which `rrna_indexed`
cannot catch because the entry *is* indexed. And it is load-bearing here, because `pct_unique_adj`
**20.7%** — the corpus's one `FAIL:very_low_mapping` — was computed from that 39.1%.

**The array is LOCALIZED TO CHROMOSOMES in BOTH assemblies, so no sink is needed either way** (the rule
is sink only where the array is missing). v5.5, which is back in use, shows it via its GFF3 `rRNA`
features; ASM4749649v1 was measured 2026-09-06 with the repo's usual k-mer method and agreed: 29 30-mers tiled
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

**P. patens is back on its Ensembl GFF3 `rRNA` features following the 2026-09-07 revert, so
`rdna_regions` is `[]` and `pct_rrna` returns to the 47.4% basis — EXPECT THE `FAIL:very_low_mapping`
FLAG BACK.** That flag had cleared (adj 27.8% -> 60.7%) on the V7 measurement's 76.3%, which this file
flagged as unsafe for over-broad spans; retiring that figure is part of the point of reverting. The
paragraph below describes the V7 state and is kept because the failure mode is general.
**Under V7 P. patens lost its rRNA source, and its FAIL flag silently changed basis.** Same
shape as C. reinhardtii: `pct_rrna` came from Phypa_V3's Ensembl GFF3 `rRNA` features, and **V7's GWH
annotation has ZERO rRNA features** — verified by downloading it, the only types present are `gene`,
`mRNA`, `exon`, `CDS`, `five_prime_UTR`, `three_prime_UTR` over 33,075 genes. So `rrna_content.py` scored
organelles only, wrote `rrna_indexed: no`, and `pct_rrna` came back blank. **The blank is correct
behaviour** — that is exactly what `rrna_indexed` is for — but it moved this experiment's flag from
`FAIL:very_low_mapping(28%,adj)`-class onto `FAIL:very_low_mapping(14%,raw)`, i.e. judged on a number
whose ceiling is set by 8.6% organellar plus ~47% rRNA it can no longer see.
`rdna_regions` is now measured for V7 as well (three arrays, see `config/genomes.yaml`), so both plants
are back on an adjusted basis.

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


## Archive metadata, and the curation errors it produced

Four findings that changed what the pipeline does. The general lesson is in `CLAUDE.md`: the archives are
authoritative for layout and run structure, the reads are authoritative over both the archive and the
paper, and cap selection is recorded nowhere but the manifest.

**ArrayExpress submissions need their ENA counterpart filled in by hand.** `resolve_runs.py` queries ENA
with `bioproject`/`sra_study` from the manifest and skips a project that has neither — which is what
`Embryo_PROcap_dm: no bioproject/sra_study in manifest; skipping query` means. `E-MTAB-6154` is an
ArrayExpress accession, and ENA's portal does not resolve it. The sequencing data *is* in ENA, brokered
under a different accession; find it via the BioStudies API
(`https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-6154`), whose record links out to the ENA study.
For this one that is **ERP106976 / PRJEB25091**, now recorded per sample in the manifest, and all 44
manifest rows resolve. If another ArrayExpress row appears, do the same lookup rather than teaching
`resolve_runs.py` a second archive.


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

