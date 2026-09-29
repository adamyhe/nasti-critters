# Investigations: measured findings and refuted hypotheses

Analyses run against the real corpus, kept so they are neither repeated nor re-proposed. Several entries
record a hypothesis that was **refuted** — the reasoning is preserved because it is what the result had to
overturn. `CLAUDE.md` carries only the conclusions that change how you work.

## `modisco motifs` goes through `run_modisco.py`, because it crashes on a ONE-SIGNED track

Hit 2026-09-07 on the first full modisco run: 81 of 84 jobs finished, G. hirsutum was still running, and
**exactly two failed — `S.cerevisiae_PROcap` and `S.pombe_PROcap`, both on the `counts` head** — with a
message that names nothing about attributions:

    ValueError: Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required.

**The cause is upstream and mechanical.** `extract_seqlets` splits the smoothed contribution track into
`values[values >= 0]` and `values[values < 0]` and computes a threshold for **each side
unconditionally**. When no windowed sum is negative, `_isotonic_thresholds` gets an empty `values`:

    w = len(values) / len(null_values)                  # 0 / 10000 = 0.0
    sample_weight = concat([ones(0), ones(n2) * w])     # ALL ZEROS
    model.fit(X, y, sample_weight=sample_weight)

sklearn's `IsotonicRegression._build_y` drops zero-weight rows (`mask = sample_weight > 0`), which
empties `y`, so the error surfaces four frames deeper in sklearn with no mention of the real problem.
**modiscolite already supports this outcome and simply cannot reach it**: `TFMoDISco` sets
`neg_patterns = None` when there are too few negative seqlets, so a one-signed track is an anticipated
result — it dies while thresholding a side that has no data.

**Read the EXCEPTION TYPE to tell the three degenerate cases apart** — measured on fixtures, and worth
keeping because they present as the same "modisco crashed":

| track | negative windows | modisco raises |
| --- | --- | --- |
| signed (normal) | ~50% | — works |
| **non-negative, mode ≈ 0** | **0** | **`ValueError: 0 sample(s)`** in sklearn |
| all zero | 0 | `IndexError` in `_laplacian_null` |
| signed + large positive offset | 0 | `ZeroDivisionError` at `w` |

So the `ValueError` **rules out a blank or corrupt npz** on its own: an all-zero array fails differently.

**It is a CONTINUUM, not two broken experiments — and the fragility is corpus-wide.** Measured over all
83 finished attribution files, as the fraction of per-position contributions (`(ohe * hyp).sum(axis=2)`)
that are negative:

| counts head | negative per-position | windowed min |
| --- | --- | --- |
| `S.pombe_PROcap` | **0.0%** | **+0.037** (crashed) |
| `S.cerevisiae_PROcap` | **0.7%** | **+0.025** (crashed) |
| the other 5 S. cerevisiae | 4.4 – 5.4% | −0.13 to −0.34 |
| C. griseus brain / lung | 7.6 / 8.7% | — |
| mouse, hamster, A. thaliana, plants | 11 – 28% | — |
| C. elegans L3, P. patens | 41 / 45% | — |
| D. melanogaster | 52 – 66% | — |
| G. hirsutum | **95.6%** | — |

The five surviving yeast counts runs cleared zero by 0.13–0.34 against maxima of +1.4 to +2.1, i.e.
**narrowly**. Every `profile` run is far from the boundary. So treat this as fragile for any
low-variance counts head — a retrain, a reseed or a depth change could tip the others over — rather than
as a property of those two libraries.

**Hypothesis refuted, recorded so it is not re-proposed: it is NOT the starved yeast negative pools.**
The prediction was exactly backwards. The two failures have the *largest* yeast pools and the *lowest*
peak density, while `Spt5IAA4h` — 327 negatives for 23,640 peaks, 4.14 peaks/window, the worst in the
corpus — passed:

| | peaks/window | negative pool | counts modisco |
| --- | --- | --- | --- |
| `S.cerevisiae_PROcap` | 1.18 | **1,905** (largest) | **FAILED** |
| `S.pombe_PROcap` | 1.56 | 947 | **FAILED** |
| `Spt5EtOH` → `Spt5IAA4h` | 1.89 → 4.14 | 738 → **327** | all ok |

`NO_SIGNAL_FILTER` is keyed by species and covers all seven, so it cannot discriminate either. What the
two failures do share is being the **Booth2016 pair** (`PRJNA306424`), the least dense and most
unidirectional yeast libraries — but why their counts heads are one-signed is **not established**, only
that the track is positive-shifted with low dynamic range (both compressed into a narrow positive band,
maxima 0.385 and 0.477, the smallest in the corpus).

**The fix is `src/bpnet/modisco/run_modisco.py` + `src/modiscolite_compat.py`, and it had to be a CLI
front end.** Nothing here imports modiscolite — `src/bpnet/modisco/` only shells out, like `umi_tools`
and `pints_caller` — so unlike `tangermeme_compat.py` there is **no in-process call site to patch**, and
a shim can only reach the library by owning the process. `run_modisco.py` patches, then hands `argv` to
the real `modisco` script via `runpy`, so every subcommand, flag and default stays whatever the installed
version provides and there is no argument parsing to drift.

- **`launcher.py` emits `python .../run_modisco.py motifs …` for motifs only**; `modisco report` is
  unaffected and still calls `modisco` directly. The `NUMBA_NUM_THREADS=N` prefix rides on it unchanged.
- **The sentinel is ±inf, verified inert by reading every consumer** rather than assumed:
  `idxs = (tracks >= pos) | (tracks <= neg)` selects nothing on the empty side,
  `transformed_neg_threshold` becomes exactly −1.0 via `sign * searchsorted(distribution, inf) / len`,
  and `weak_thresh` takes `min(transformed_pos, abs(−1.0)) − 0.0001` so the positive side still decides
  it.
- **Self-retiring**, same discipline as `tangermeme_compat.py`: it functionally probes the installed
  version with an empty `values` and does nothing if the probe passes, and refuses to install a patch
  that fails its own probe. **modisco 2.5.2 is the latest release** (checked 2026-09-07), so there is no
  version to upgrade to; the real fix is an early return in `_isotonic_thresholds` and is worth a
  modisco-lite issue.
- Verified end to end on a 400-locus one-signed fixture: the **stock CLI reproduces the exact
  `ValueError`** and writes no `.h5`, the wrapper completes and writes one containing **`pos_patterns`
  only, with no `neg_patterns` group** — upstream's own supported path. The patch is inert on a
  two-signed track (identical output patched and unpatched, and `_isotonic_thresholds` returns the same
  value for non-empty input).

**Diagnose a future instance by probing the arrays, not the log.** Reproduce modisco's own slicing
(central `-w` window, transpose to `(N, L, 4)`, 20 bp rolling sums) and count negative windows; zero
means it will crash. `neg hyp` near 50% with `neg pos` near 0 is the fingerprint — the *hypothetical*
array is signed, and it is the value **at the observed base** that is one-signed.

**This survey also turned up `G.hirsutum-ovule_GROcap` as the corpus outlier in the OPPOSITE
direction**, which proved to be a broken model rather than a modisco problem — see "G. hirsutum is the
corpus's WORST model" below. It is the only file where `neg hyp` is *below* 50% (39.8%
counts, 38.7% profile) while `neg pos` is **95.6% for BOTH heads**, i.e. the observed base is almost
always the negative one. Consequence for reading its report: nearly every seqlet lands in the
**negative** metacluster, so its motifs are in `neg_patterns` while `pos_patterns` is close to empty —
the reverse of every other experiment. Do not read that as "no motifs found", and do not read it as
biology either, given the model behind it.


## G. hirsutum is the corpus's WORST model, and five folds agree — so the ceiling is in the DATA

Found 2026-09-07 while diagnosing a `modisco motifs` crash in the yeasts, which is the only reason
anyone looked: **nothing in the benchmark pipeline flags this experiment.** Ranked on
`genome_wide.profile_pearson` across all 42 benchmarked experiments it is last by a discontinuity:

| | `profile_pearson` |
| --- | --- |
| **`G.hirsutum-ovule_GROcap`** | **0.0621** |
| `C.reinhardtii-liquidculture_5GRO` | 0.1644 (**2.65x higher**) |
| `D.melanogaster-S2_5GROcap` | 0.1825 |
| `P.patens-plateculture_5GRO` | 0.1973 |
| `C.griseus-BMDM-KLA1h_GROcap` (8th) | 0.2228 |

**The worst-to-second gap (+0.102) is LARGER than the gap spanning second through eighth (+0.058)**, so
it sits off the bottom of the distribution rather than merely last.

**All five folds agree to within 0.017** (0.0491-0.0663, sd 0.0066), against the diploid's 0.2581-0.3053.
That is the load-bearing observation and it rules out three explanations at once: **not** one diverged
fold (which `attribute.py`'s fold-mean would then have inherited), **not** training instability, and
**not** the `valid_count_corr` checkpoint-selection hazard — all three produce scatter across folds, not
five-fold consensus. Retraining, reseeding or changing the schedule will not move it. **The ceiling is
in the data or the labels.**

**Its attributions are 100% sign-inverted, which is how this was found.** Every one of 171,449 loci has
`sum(ohe * hypothetical) < 0` for BOTH heads, against **0.0%** of the diploid's 95,685:

| | f(x) − f(ref) | implied counts ratio |
| --- | --- | --- |
| G. hirsutum | −16.14 counts, −16.18 profile | **e^−16.1 ≈ 1e-7** |
| G. arboreum | +5.90, +4.33 | e^+5.9 ≈ 366x |

**Do NOT read the −16 as biology; it is most likely a soft-reference artifact.** The default
`--reference-mode frequency` reference is a **soft PFM, not a one-hot input** — which is exactly why it
has to reach tangermeme as a callable, to get past its one-hot validator — so `f(ref)` is an
extrapolation, and a model with no sequence skill has no reason to extrapolate sensibly. The diploid's
`+5.9` is equally large, merely signed the other way. Cheap decisive test:
`attribute.py -e G.hirsutum-ovule_GROcap --reference-mode dinucleotide`, whose shuffles are hard one-hot
and in-distribution; the output path carries the mode, so it cannot overwrite the frequency run.

**RESOLVED 2026-09-08: IT IS PER-PEAK COVERAGE, AND HOMOEOLOGY IS REFUTED.** Depth-matching (test (b),
`src/analysis/matched_comparison.py`) accounts for **84% of the 4.45x gap** — raw 4.45x falls to
**1.56x** once loci are matched on observed counts — and depth predicts per-locus correlation strongly
*within* each experiment (Spearman **0.649** for AA, **0.581** for AD). The candidates below are kept
because the reasoning is what the result had to overturn, and because the nesting trap they describe is
what made the answer interpretable.

**The mechanism is arithmetic, and the read budgets are what settle it.** The two libraries have
*essentially identical* budgets — **141.0 M vs 140.1 M signal reads, 142.9 M vs 143.5 M unique** — so
mappability loss cannot be the cause: had MAPQ 255 been discarding homoeologous reads, AD would have
ended with FEWER. It did not. The whole depth difference is that one budget was divided across 1.79x
more peaks:

    peak-count ratio = genome size 1.449x  x  peaks/Mb 1.237x  =  1.793x
    reads/peak ratio =                                            1.805x   <- the same number

They agree *because* the budgets are equal. So the chain is **ploidy -> ~1.8x more initiation sites ->
same sequencing depth -> half the reads per peak -> low per-locus profile correlation.** A tetraploid
needs roughly twice its diploid progenitor's depth to reach equal per-peak coverage, and it was
sequenced to the same depth. That is an experimental-design gap, not a pipeline defect and not a
subgenome artifact.
**THE EQUAL-BUDGET ARGUMENT REFUTES ONE HOMOEOLOGY MECHANISM, NOT BOTH — do not overstate it, as an
earlier version of this note did by concluding "test (c) is dropped".** Homoeology has two distinct
routes to a lower score and they need separating:

- **Read depletion** — MAPQ 255 discarding homoeologous reads. **Refuted**, by the equal budgets above:
  nothing was lost, so nothing was depleted.
- **Sequence ambiguity** — two near-identical inputs (~85 differences per 2114 bp window) carrying
  genuinely different profiles, which imposes irreducible error **regardless of depth**. **Untouched by
  any matching done here**, because matching on depth or confidence cannot remove it.

A residual that survives both covariates uniformly is exactly what the second predicts.

**THE CORPUS-WIDE CONSEQUENCE IS THE BIGGER FINDING: `profile_pearson` IS LARGELY A DEPTH STATISTIC.**
Within G. arboreum alone, per-locus correlation runs **0.0040 in the shallowest count decile to 0.6617
in the deepest — a 165x range**. The largest *between-experiment* difference in the entire 42-experiment
corpus is 4.5x. So every cross-experiment model comparison in this repo is confounded by per-peak
coverage, and **both cotton models are far better than their headline numbers**: AA reaches 0.66 and AD
0.41 on their top decile, against published 0.28 and 0.06. Quote a locus set, or match, or say neither
and expect the number to mean little. This is measured for the cottons only; the other 40 experiments
are unchecked.

**The three candidates as they stood, CAUSALLY NESTED rather than rival**, with the trap that an
"explanation" at one level may be a symptom of the next:

1. **Peak-set quality.** 171,640 peaks, the corpus's largest, **96% unidirectional**, and the weakest
   initiator logo here at 0.11 bits — this file already says to treat that set as low-confidence-heavy.
   A model cannot fit a profile at loci that are not really initiation sites.
2. **Thin per-peak coverage: 816 reads/peak against the diploid's 1,473, 1.8x thinner** (140.1 M signal
   over 171,640 peaks vs 141.0 M over 95,714). Both clear `THIN_COVERAGE_READS_PER_PEAK = 500`, so
   neither is flagged. Noisier observed profiles cap the achievable correlation directly, and this is
   downstream of (1): calling 1.8x more peaks from the same depth is what makes them thin.
3. **Homoeology / mappability.** A05 and D05 are 96%+ identical (~85 differences per 2114 bp window) and
   the paired-homoeolog fold constraint deliberately puts both in the SAME fold, so the model sees
   near-duplicate sequences carrying different measured signal, because MAPQ 255 discards reads that
   cannot be assigned to one subgenome.
   **But note mappability is a DETERMINISTIC function of the reference**, not sampling noise — so it is
   reproducible, learnable in principle, and its main effect is to *deplete* unique reads. That is to
   say it most likely acts THROUGH (2) rather than as irreducible label noise, which is why a
   depth-matched test could "explain away" homoeology while homoeology is still the root cause.

**How to test, in order:**

- **Split per-locus metrics by PINTS class and confidence.** Free: `combine_peaks` keeps upstream's
  strand/confidence/class/summit columns precisely so this is possible, and
  `benchmark_predictions.py --output-fname` saves the per-locus predictions. If the bad loci are the
  low-confidence unidirectional calls, it is (1).
- **Depth-match across the two cottons.** Bin loci by observed window counts and compare
  G. hirsutum against G. arboreum within bins. If the gap closes, depth is sufficient; if it survives
  depth matching, something else is live. Same logic as GC-matching negatives.
- **2D stratify depth x mappability**, which is the only way to separate (2) from (3). A mappability
  score needs no new tooling: subsample ~20 30-mers per peak window, align them with **STAR against the
  index that already exists**, and take the uniquely-mapped fraction per locus. Then ask whether the
  gradient follows uniqueness with depth held fixed.

**TEST (a) IS DONE, 2026-09-08, AND ITS ANSWER IS A DISSOCIATION: peak confidence explains a lot of
WITHIN-experiment variance and NONE of the between-species gap.** Run by splitting the peak file on
PINTS confidence (`src/analysis/stratify_peaks.py`) and scoring each stratum with
`benchmark_predictions.py --loci`, so the numbers come from the same code that wrote the canonical JSON.
`profile_pearson`, q1 = most confident quartile of unidirectional calls:

| stratum | AD hirsutum | AA arboreum | AA/AD |
| --- | --- | --- | --- |
| uniq1 | 0.1601 | **0.5883** | 3.67x |
| uniq2 | 0.0661 | 0.4314 | 6.53x |
| uniq3 | 0.0475 | 0.1697 | 3.57x |
| uniq4 | 0.0437 | 0.1198 | 2.74x |
| bi | 0.1064 | 0.2492 | 2.34x |
| all | 0.0621 | 0.2764 | 4.45x |

**The gradient is real and monotone in BOTH species** — 3.66x from q1 to q4 in the tetraploid, 4.91x in
the diploid — so peak-set quality is a genuine effect and **generic, not tetraploid-specific**. `jsd`
agrees independently throughout (0.5814 vs 0.3487 at q1).
**But the gap does not close at matched confidence: 3.67x at q1 against 4.45x unstratified.** The
sharpest statement is that **the tetraploid's most confident quartile (0.1601) scores BELOW the
diploid's third quartile (0.1697)**, barely clearing its worst (0.1198). So (1) is ruled out as the
explanation for the AD-vs-AA gap while being confirmed as a large within-experiment effect. Next is (2),
depth-matching, which confidence stratification cannot control for: the q-value ranks peaks *within* an
experiment, and AD still runs 816 reads/peak against AA's 1,473.

**Two corpus-wide consequences, neither cotton-specific:**

- **`profile_pearson` is a peak-set-composition number as much as a model number.** The diploid's own
  headline 0.2764 understates its model, which reaches **0.5883** on its confident quarter. If a 3.7-4.9x
  within-experiment gradient holds for both cottons it very likely holds everywhere, so **any quoted
  figure should say which locus set it is on.** This is not yet measured for the other 40 experiments.
- **Only the PROFILE columns are safe to compare across strata.** `profile_pearson` and `profile_jsd`
  are per-locus medians (`pearson_corr` runs over each locus's flattened profile), so subsetting merely
  chooses which loci to median over. `log_counts_pearson` and `counts_spearman` are correlations ACROSS
  loci, so subsetting restricts the count range and depresses them mechanically — which is why `bi`
  reads 0.3627 against 0.4885 while being the *better* stratum on profile. That is range restriction,
  not evidence.

**`benchmark_predictions.py --per-locus-tsv` exists for the next tests** — coordinates, fold, per-locus
profile Pearson and JSD, observed counts and predicted log counts, one row per evaluated locus. It makes
depth-matching and the depth x mappability join analyses rather than further reruns. `--output-fname`
cannot serve: it dumps `{preds, signals}` with **no coordinates**, and `load_bed` reads columns 0-2, so
its rows cannot be joined to anything.
**Rows are aligned with `extract_loci(return_mask=True)`, which is REQUIRED rather than tidy** — that
call drops loci falling off a contig end or inside an exclusion zone, so a positional join would
silently misalign.

**THE MASK INDEXES THE INTERLEAVED LOCI, NOT THE LOCI AS PASSED.** `extract_loci` runs
`_interleave_loci(loci, chroms)` *before* its loop, so the chromosome filter is already applied by the
time the first `kept_mask` entry is appended — the docstring's "complete set of interleaved peaks" means
**interleaved**, which is easy to read as "provided". Indexing the passed frame is therefore wrong by
exactly the size of the fold's chromosome subset, and on the real tree that was **a mask of 36,732
against 171,640 loci**. The frame is rebuilt with tangermeme's own `_interleave_loci` rather than by
reproducing its filter, so the two cannot drift; for a single DataFrame it takes columns 0-2, applies
`numpy.isin(chrom, chroms)` and reindexes by `arange`, i.e. **filters while preserving order**. It is
private, so a rename upstream gives an immediate `ImportError` rather than a silent misalignment.
**A fixture without `chroms` CANNOT catch this** — that is how it got through: with `chroms=None` the
mask length equals the provided length and naive indexing coincidentally works. Any test of this must
pass a real chromosome subset; verified on a two-contig fixture for `None`, `["chr1"]` and `["chr2"]`.

**The mask is appended LAST**, so it must be popped before the existing `len(data) == 3` control-track
test: without popping, a signals-only call plus a mask looks exactly like a call with controls, and the
mask would be used AS a control track. Verified across all four combinations of controls x mask. Both
the mask length and the per-fold row counts are asserted, and a mismatch raises rather than writing a
misaligned table — which is what caught the interleave bug on the first real run.


**JOINT depth x confidence matching adds almost nothing over depth alone — run 2026-09-09.** The
per-locus tables were annotated with the peak file's own q-value (`annotate_per_locus.py`, joining on
coordinate; 99.86-99.89% joined) and standardised over the 6x6 cell cross product:

| matched on | raw | matched | explained |
| --- | --- | --- | --- |
| depth | 4.45x | 1.56x | 84% |
| depth x confidence | 4.64x | **1.52x** | 86% |

Confidence removes only **7% of the depth-matched residual**. The raw ratios differ because the joint
run is restricted to unidirectional loci — `uni_qval` is undefined for the bidirectional class.
**So a ~1.5x residual survives everything measurable, and it is uniform rather than localised**:
per-cell ratio median 1.51x, IQR 1.31-1.71, 91% of 35 usable cells favouring the diploid, covering 100%
of loci. It is not an artefact of thin cells.

**Secondary observation, consistent with something tetraploid-specific being real:** `rho(uni_qval)` is
**-0.345** for the diploid against only **-0.180** for the tetraploid, so PINTS confidence is about half
as informative in AD. Plausibly its background estimation is affected by the duplicated genome; not
established.

**DECISION 2026-09-11: `G.hirsutum-ovule_GROcap` IS EXCLUDED FROM FINAL REPORTING ONLY. EVERY ANALYSIS
STILL RUNS.** The boundary is narrow and deliberate — it sits at the write-up, not anywhere in the
pipeline:

- **Nothing is excluded from the PIPELINE.** It preprocesses, trains, benchmarks, attributes and goes
  through modisco exactly as before, and those outputs are kept. That follows "Nothing is excluded:
  every dataset is analysed and modelled" — `tier` gates preprocessing only and `launch.py` deliberately
  does not filter on `qc_flags`. Nothing about the DATA is known to be wrong: the library maps fine and
  its peak calls are sound.
- **Nothing in the CODE implements this.** No `tier` change, no launcher filter, no skip in
  `compare_bpnet_cherimoya.py`. The exclusion is applied by a person deciding what goes in a figure or
  a table, which is the only place it belongs. Do not add a switch for it.
- **The outputs stay worth GENERATING and worth LOOKING AT — they are diagnostic.** This experiment is
  what exposed the one-signed-attribution crash, and its contrast with the diploid is what established
  that `profile_pearson` is largely a depth statistic. Both of those are results, produced by an
  uninterpretable model. Keep running it for exactly that reason.
- **What is excluded is REPORTING it as evidence about biology**: its motifs, its attributions, and its
  metrics. `profile_pearson` 0.0621 is the corpus's lowest, its attributions are 100% sign-inverted, and
  the modisco report puts nearly everything in `neg_patterns`.
- **The AA-vs-AD MODEL comparison: compute it, do not report it.** The tetraploid's model cannot carry a
  biological claim. The diploid's is fine (0.2764 overall, 0.5883 on its confident quartile) and can be
  reported on its own.
- **Peak-level AA-vs-AD results are UNAFFECTED and remain reportable** — the peaks/Mb comparison, the
  mapping rates and the adapter work depend on no model and stand as written.
- **The remedy, if it is ever wanted as a reportable model, is DEPTH, not reprocessing**: ~2x the reads,
  or a stricter peak set. Nothing in the pipeline needs changing.

Test (c) — a per-locus sequence-uniqueness score to test the surviving ambiguity mechanism — is
therefore **parked, not refuted**. It is worth doing only if tetraploid modelling becomes a question in
its own right; it would be establishing whether tetraploids are intrinsically harder, not diagnosing
this experiment.

**The matching tooling, and what it is validated against.** `matched_comparison.py` reports three things
in reading order: `Spearman(covariate, profile_pearson)` *within* each experiment (a rho near 0 means
that covariate cannot explain a between-experiment gap, and the rest is moot); a cell-wise table on bin
edges from the POOLED distribution so both are on one scale; and a **directly standardised** score per
experiment, the weighted mean of its per-cell medians under common weights, whose ratio against the raw
ratio is the answer. Cells with fewer than 50 loci in *either* experiment are dropped and counted, and
if none qualifies it **exits nonzero** rather than reporting a number built from nothing — non-overlapping
populations cannot be rescued by matching, and that case is reported as itself.
Validated on four fixtures with known ground truth: an identical `prof_r = f(counts)` sampled at two
depths standardises 0.52x -> 0.98x (reported as depth accounting for 96%); identical depths with
different levels leave 4.00x -> 4.00x (0%); and a two-covariate fixture where `prof_r =
g(depth)*h(qval)` is identical in both gives **71% for depth alone, 40% for confidence alone, and 96%
jointly**, which is the property that matters — joint matching must close more than either margin.
**Only `profile_pearson` is a valid response here.** It is a per-locus median, so reweighting merely
chooses which loci to summarise. The JSON's `log_counts_pearson` and `counts_spearman` are correlations
ACROSS loci and are range-restricted by any subsetting or matching, so they serve only as the covariate.

**What this does NOT overturn:** the peaks/Mb result under "Do NOT adopt the paper's looser STAR
settings" is a peak-COUNT comparison and depends on no model. It does add a caveat to reading the
AA-vs-AD pair as a MODELLING comparison — **the tetraploid's model is the worst in the corpus and the
diploid's is mid-pack** — so any cross-species claim resting on those two models is confounded by model
quality before biology enters.



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
- **Summit-anchored metaplot: the statistic is a RATIO, and antisense owns the y-axis.** Both of those
  are corrections made after the first full run, which showed the panel looking **basically identical
  across all 42 experiments** — the observation that prompted them.
  **Why it looked identical: sense and antisense shared one y-scale, and the sense spike at 0 is
  guaranteed by the anchor.** It set the scale, so the antisense channel was compressed onto the axis
  line. `M.musculus-GCB_PROcap` (antisense peak at −137 bp) and `S.cerevisiae-Ino80ctl_PROcap` (no
  localized antisense peak at all) were indistinguishable by eye. Antisense is now the primary trace on
  its own axis, sense is faint context on a twin axis, and the band the statistic uses is shaded so the
  printed number and the picture cannot disagree.
  **The numbers were discriminative all along, and they split along known biology:**

  | antisense argmax | experiments |
  | --- | --- |
  | **−97 to −159 bp** — an upstream peak | all M. musculus, all C. griseus, all C. elegans, C. reinhardtii |
  | **\|offset\| 360–500** — at the ±500 window edge, i.e. NO localized peak | all D. melanogaster, A. thaliana, P. patens, S. pombe, 4 of 6 S. cerevisiae |
  | **−1 to +39** — on the summit | S. moellendorffii, both cottons, fly LacZ-KD, Spt5EtOH |

  **AN UPSTREAM PEAK AT ~−100 bp IS A TETRAPOD EXPECTATION, NOT A CORPUS-WIDE ONE — and an earlier
  version of this table called that row "canonical divergent distance", which was wrong.** Divergent
  initiation from the SAME NFR as the main TSS is a vertebrate promoter architecture; most taxa here do
  not have it, so for them the absence of an upstream peak is the **null**, not a deficiency, and the
  middle row above is the *expected* result rather than a weaker version of the first. Only the
  DOWNSTREAM reading is ever a fault. Measured over the corpus (2026-09-06):

  | | n | localized peak reported | median log2 |
  | --- | --- | --- | --- |
  | mouse + hamster | 20 | **18** | **+1.39** |
  | every other taxon | 22 | 10 | +0.09 |

  This also removes a contradiction with the annotated-TSS section below, which already said divergent
  upstream antisense "is NOT an expectation at all" for C. elegans while this table listed worm in the
  canonical group.

  **THE ANTISENSE *FRACTION* IS NEARLY CIRCULAR — it is close to a restatement of PINTS' bidirectional
  class.** `peaks_bidirectional / peaks_total` against `antisense % of windowed signal` is **Pearson
  +0.939** across all 42 experiments. A bidirectional call means PINTS already found an opposite-strand
  peak nearby, so anchoring on those summits puts antisense in the window by construction. Print it for
  context; never read it as independent evidence.
  **The log2 ratio is NOT confounded that way** (r = **−0.287** with the same share), and S. cerevisiae
  is the proof the two are different quantities: **92.8% bidirectional, 40.4% antisense, log2 −0.03.**
  Maximal bidirectional calling with zero upstream asymmetry — so bidirectional calls fill the window
  SYMMETRICALLY rather than manufacturing an upstream peak. The taxonomic signal lives in the ratio and
  the position, not in the fraction.

  **C. ELEGANS IS THE ONE NON-TETRAPOD SHOWING THE TETRAPOD PATTERN. Tested 2026-09-06; DIVERGENT GENE
  PAIRS ARE REFUTED, and the question is PARKED as answered well enough for a QC read-out.**
  `src/qc/divergent_annotation_test.py` partitions summits by whether an annotated opposite-strand
  protein-coding start sits in the band and recomputes the statistic per class. Shared-NFR divergent
  transcription is largely *unannotated*; a divergent gene pair is an annotated minus-strand gene start.
  Over four worm libraries plus three tetrapod positive and two fly negative controls:

  | | log2 | peak | prominence | vs sense anchor |
  | --- | --- | --- | --- | --- |
  | tetrapod (hamster, 2× mouse) | **+1.38 to +2.25** | −101 to −129 | 2.08–2.28 | 0.43–0.69% |
  | C. elegans (all 4) | **+0.73 to +1.14** | −97 to −124 | 1.80–2.11 | 0.38–0.81% |
  | D. melanogaster (both) | **−0.18, −0.38** | *band max ON the summit*, +37/+58 | 1.33–1.45 | 0.20–0.21% |

  All four worm libraries keep the upstream peak among the ~92% of summits with **no** annotated
  opposite-strand partner, so annotation does not explain it — and annotation explains **least** in worm
  (3.5–4.7% of upstream-band antisense) against 17.3% for mouse BMDM. Fly is the clean negative: no
  localized upstream peak at all, in two libraries from different labs.
  **The amplitude does NOT separate worm from tetrapod; the RATIO does.** Worm's prominence overlaps
  theirs and its peak is as large relative to the sense anchor (embryo is the highest of all nine), while
  its absolute height is ~1.6× lower. So the honest statement is that worm's divergent peak is comparable
  in size but **less directional** — it sits on more downstream antisense — not that it is small. Do not
  restate this as "worm looks tetrapod-like"; the log2 ranges do not overlap.
  **The one caveat left untested** is that ce11 refGene 5′ ends are SL trans-splice acceptor positions,
  not TSSs (`annotation_tss_anchored: false`), so a genuine partner whose annotated start is displaced
  beyond the 50 bp pairing halfwidth is scored unpaired, biasing toward the result obtained.
  `--halfwidth 300` probes it; the paired fraction climbing steeply with halfwidth would mean the ~8% is
  a displacement artifact. Not run — the question was closed first.
  **Four artefacts were found and fixed while getting here, all of which had produced confident wrong
  readings**, and they are the reason to distrust a first number from this script: an asymmetric
  partition (defining "paired" upstream-only makes the complement downstream-biased, which drove one
  unpaired class to log2 −2.54 and tripped the strand-swap flag); raw-summed rather than per-site
  normalised profiles (one locus in 995 inverted a profile's sign — the failure this file documents
  elsewhere at 78.8% from 1 site in 1001); a single shift null aliasing against gene spacing; and a
  regex that could not match a positive peak offset because `summit_notes` formats with `{at:+d}`.
  **So `argmax` was the wrong readout**: on a channel with no peak it lands wherever noise is highest,
  and the third group above then tripped a naive `argmax > 0` swap test. Most of the flags on the first
  run were that artefact. It is now `log2(upstream / downstream)` antisense summed over
  `|offset| ∈ [20, 300]`, which degrades to 0 on a flat channel and has no edge behaviour, with a
  position reported only when the peak is prominent against the band median and away from the edge.
  **The band's outer bound is what keeps dense genomes out of it.** S. cerevisiae runs 1.2–4.1 peaks per
  2114 bp, so a ±500 window usually contains another promoter — which is why yeast shows the corpus's
  highest antisense fraction (35–43%) *with* a monotone rise to the edge. Unbounded, that pattern scored
  log2 −1.51 and was flagged as a strand swap; bounded at 300 it scores −0.88 and is not. Both 300 and
  the −1.0 threshold are **provisional**, calibrated from this corpus like `FLAT_BITS`.
  **Severity is split by whether a PEAK backs the asymmetry**: downstream weight *with* a localized peak
  is a flag, downstream weight *without* one is a `REVIEW`, because in a dense genome that is
  neighbouring promoters rather than a swap. Verified on eight synthetic regimes — divergent broad and
  narrow, swapped, flat, on-summit, weak, edge-rising, and zero antisense — each landing on its intended
  verdict.
  **And the on-summit check has to be measured against the band MAXIMUM, not its median.** Against the
  median it fired on a genuine broad upstream peak, which still carries real signal at offset 0 while the
  median is pulled down by the quiet downstream half. Against the maximum it asks the right question: is
  the summit itself the largest antisense feature in the window?
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

**A null `annotation_url` is a SUPPORTED state rather than a DAG failure — and as of the 2026-09-07
C. reinhardtii revert, NO species exercises it.** Keep the handling: it was earned by a real failure, and
it is what makes an annotation-free assembly adoptable at all. C. reinhardtii under ASM4749649v1 was the
case that forced it, and the notes below describe that state.
**Under ASM4749649v1 C. reinhardtii had NO annotation, which is one of the reasons the swap was
reverted.** `annotation_path()` interpolated
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


## Depth requirements scale with the nascent transcriptome, NOT with a constant

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


## Negative sampling in densely transcribed genomes

Why the two yeasts get a fraction of the negatives every other species gets, what the levers do, and the
measurements behind `NO_SIGNAL_FILTER` and the ratio cap.

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


### Negatives ratio, and how hard the small yeast pools get recycled

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

## Where the Spt5 UMI actually sits

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
