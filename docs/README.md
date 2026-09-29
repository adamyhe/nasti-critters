# docs/

Handoff and record docs for **nasti-critters**. These were split out of `CLAUDE.md` on 2026-09-15, which
had grown to ~4,600 lines by accumulating the evidence behind every rule alongside the rules themselves.

**The division of labour:**

- **`CLAUDE.md`** — the live rules. What to do, what not to do, and the traps that are still live. Loaded
  into context every session, so it stays lean.
- **`docs/`** — why. Measurements with their dates and methods, per-dataset provenance, rejected options,
  and hypotheses that were **refuted**. Several entries exist specifically so a plausible idea is not
  re-proposed, and some preserve superseded reasoning because it is what a later result had to overturn.
- **`README.md`** (repo root) — install and the operator runbook.

**Check the relevant doc before reopening a question.** If you find yourself about to re-measure
something, re-derive a pin, or re-argue an assembly choice, it is probably already recorded here with the
command that produced the number.

| doc | what is in it |
| --- | --- |
| [environment.md](environment.md) | the two-environment split, Sherlock's wheel ceiling and every pin's justification, package-name traps, why conda-forge torch was rejected |
| [snakemake-operations.md](snakemake-operations.md) | what invalidates a rule, the 701-job cascade, incomplete metadata, why `--touch` cannot work here, scoping a run |
| [assemblies-and-references.md](assemblies-and-references.md) | assembly choices per species, exclusion lists, the alt-contig audit, the 2026-09-07 NCBI survey, the two plant swaps and their revert, rDNA and organelle decoys |
| [cross-validation-folds.md](cross-validation-folds.md) | how each species' folds were derived, which are reused from earlier lab work, which originate here and still need pushing upstream |
| [datasets-and-provenance.md](datasets-and-provenance.md) | dataset additions, the curation errors the archives and papers produced, the merge/exclude analysis, `experiment_stats` detail |
| [investigations.md](investigations.md) | G. hirsutum's depth analysis, the adapter survey, orientation QC calibration, yeast negative sampling, the one-signed modisco track, the Spt5 UMI |
| [decisions-and-history.md](decisions-and-history.md) | the pre-unification regimes and deleted scripts, the procap-atlas sync, cherimoya's three sources of defaults |

Two conventions to keep if you add to these:

1. **Record the date and the method** beside any measurement, as the existing entries do — the point is
   that the next person can tell whether the situation has moved rather than re-deriving it.
2. **Keep a refuted hypothesis rather than deleting it**, with what refuted it. Most of the traps in
   `CLAUDE.md` were found by someone believing a reasonable thing that turned out to be wrong.
