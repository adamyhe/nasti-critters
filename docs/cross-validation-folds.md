# Cross-validation fold assignments

Per-species record of how each fold assignment was derived, which are reused from earlier lab work, which
originate here and still need pushing upstream, and the balance costs that are structural rather than
fixable. `CLAUDE.md` carries the convention and the source-of-truth rule; this file carries the
derivations.

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


