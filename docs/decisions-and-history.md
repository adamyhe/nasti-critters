# Decisions and history

Why the repo has the shape it does: the pre-unification regimes it grew out of, the upstream standards it
tracks, and the standing decisions that should not be quietly reopened. `CLAUDE.md` carries the
conventions; this file carries the reasons and the superseded state.

## Repo history in brief

- Renamed from `dm-procap-models`. The `dm-` prefix dated from when this was one *D. melanogaster*
  dataset; the repo now covers 42 experiments across 12 species and three assay families.
- `eefd528` "Reviving this project" (2026-05-23) added a flat `config/data_paths.json` for a single dm3
  Drosophila dataset, and `c9b1311` added cherimoya onto that same scaffolding one commit later.
- `a806e0d` "Added yeast genomes" needed multiple species, introduced `experiment_config.yaml` +
  `chrom_splits.yaml`, and converted **only** `fit_bpnet.py` and `launch.py`. The five downstream scripts
  kept reading `data_paths.json`, so the repo held two genome builds of the same Drosophila data, two
  model filename conventions, and duplicated hyperparameters that had silently diverged
  (`count_loss_weight` 100 vs 50, `max_epochs` 100 vs 200). All of that is now unified and
  `data_paths.json` is deleted. To reproduce the old dm3 dataset, add it as an experiment rather than
  resurrecting a parallel path.
- **Cherimoya was never D. melanogaster-only**, though `CLAUDE.md` said so until 2026-09-03. What made it
  look single-species was a set of hand-written `slurm.sh` / `cmd.sh` scripts left at the pre-unification
  interface: cherimoya's fit script ran `-f $SLURM_ARRAY_TASK_ID` with **no `-e`**, so every array task
  exited 2; its benchmark `cmd.sh` hard-coded `D.melanogaster-S2_PROcap.json` as its already-done check
  while passing `"$@"` through, so once fly was benchmarked every other experiment printed "Skipping" and
  exited 0; and bpnet's fit script had the same missing `-e` plus a relative path and a
  `--job-name=s2_fit` from the dm3 era. **All four are deleted** and replaced by the eight `launch.py`
  launchers over `src/launcher.py`, which get per-species fold counts right where a static `--array=0-4`
  cannot (C. elegans has six folds). Do not add hand-written SLURM scripts back.
- **The six legacy `src/data_preprocessing/*.sh` download scripts are deleted** —
  `download_genomes.sh`, `download_{drosophila_s2_procap,scer,spombe}.sh`, `download_S2_PROcap_dm3.sh`
  and `utils.sh`. Everything they did is now in `workflow/Snakefile` (which covers all twelve species
  where the old script covered three) or was part of the deleted dm3 regime. `utils.sh` was sourced by
  nothing and hard-coded `/programs/...` paths. Read them at `a806e0d` for legacy provenance; do not
  restore them.
- A project-local `src/cherimoya/apptainer/` existed and was **deleted deliberately**. Container images
  are maintained externally at [adamyhe/sherlock](https://github.com/adamyhe/sherlock).

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



## Cherimoya has three sources of defaults and they disagree

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


