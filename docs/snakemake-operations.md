# Running the Snakemake pipeline: rerun triggers, stale metadata, and scoping

Operational record for what invalidates a rule, why a cheap-looking target can plan hundreds of jobs, and
which flags actually scope a run. `CLAUDE.md` carries the working recipes; this file carries the
measurements and the failure modes behind them.

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

**Editing `config/genomes.yaml` is invisible the same way, and worse.** It is read at parse time
(`GENOMES = load_yaml(...)`) and is a declared input of nothing, so a rule only notices a change if the
changed value happens to be interpolated into its own `params`. For rules whose script reads the config
itself that is never true: `rrna_content`'s params are the script path and the FASTQ dir, so adding
`rdna_regions` for C. reinhardtii changes **nothing** Snakemake can see — not under `mtime`, not under
default triggers. Force it: `snakemake --forcerun rrna_content --config experiments=<exp>`.

Declaring `genomes.yaml` as an input of every rule that reads it would fix the class, at the cost of
re-running the whole corpus for any one species' edit. Until that trade is taken, treat a `genomes.yaml`
change as needing an explicit `--forcerun` and work out which rules consume the field you touched.

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

## One timestamp on a `.fai` re-fetches 74 FASTQs — use the standalone scripts

Observed 2026-09-06 on the real tree: `snakemake orientation -n --rerun-triggers mtime`, asking for
nothing but 42 plots, planned **701 jobs** — 74 `fetch_fastq`, 56 `trim`/`align`/`filter_unique`/
`final_bam`, 10 `star_index`, 80 `bedgraph`/`bigwig`, 40 `pints`/`combine_peaks`. `snakemake stats` does
the same. Nothing was wrong with the tree; the script edit was not the cause.

**The chain is short and worth memorising, because any invalidation near the top of it rebuilds from
FASTQ.** `chrom_sizes`' output is a **declared input of `bigwig`** (`sizes=`). So one `.fai` newer than
one `.chrom.sizes` re-runs `chrom_sizes`, whose output is then newer than every bigWig ->
`bigwig` -> needs `{strand}.bg`, `temp()` and deleted -> `bedgraph` -> needs `merged.bam`, `temp()` and
deleted -> `merge_runs` -> `final_bam` -> `align` -> `trim` -> the FASTQs, deleted after mapping ->
`fetch_fastq`. **The `temp()` design that makes the pipeline cheap to store is exactly what makes a
top-of-chain mtime expensive to satisfy**, and on a tree whose STAR indices have been cleaned up it
rebuilds those too. Read the `Reasons:` block to find the roots: "updated input files" lists the rules
mtime actually triggered, and anything else is downstream of them.

**The escape hatch is that every QC script runs standalone against what is already on disk**, which is
why they take `-e`/`--all` at all. None of these touch the DAG:

    python src/qc/experiment_stats.py --all -o qc/stats/experiment_stats.tsv \
        --markdown qc/stats/experiment_stats.md
    python src/qc/rrna_content.py -e <exp> --fastq-dir data/fastq -o qc/rrna/<exp>.tsv
    xargs -P 8 -I{} python src/qc/orientation_qc.py -e {} --outdir qc/orientation \
        --tsv qc/orientation/{}.tsv < experiments.txt

Three traps in doing it that way. Run `rrna_content` **before** `experiment_stats --all`, which reads
`qc/rrna/` opportunistically, or the adjusted mapping rate is computed from a stale rRNA figure. Loop
`orientation_qc` per experiment rather than using `--all`: `--tsv` is a single path reopened per
experiment, so `--all --tsv` leaves one file describing only the last one. And check
`ls data/annotation/` before parallelising it — without `--annotation` the script fetches its own, and
concurrent jobs on one species race for the same file, which is the whole reason the rule passes the
path.

**The zero-risk way to run just the step you want is `--forcerun` PLUS `--allowed-rules`.** Forcing
alone is not enough — the upstream jobs are genuinely out of date, so `--forcerun orientation_qc` still
drags the whole chain. Restricting the rule set is what stops it, and it changes no DAG state at all:

    snakemake orientation -j 48 --rerun-triggers mtime \
        --forcerun orientation_qc --allowed-rules orientation_qc orientation

Verified on a fixture reproducing the cascade (stale `.fai`, deleted `temp()` chain, absent FASTQ):
7 jobs unrestricted, **1** with the two flags. The same shape works for the table —
`--forcerun experiment_stats stats_table --allowed-rules experiment_stats stats_table stats`. It does
**not** work for a rule whose inputs are genuinely missing: `rrna_content` reads the raw FASTQs, so if
those are gone, forbidding `fetch_fastq` just moves the failure. Run that one from the script instead.

**`--touch` DOES NOT WORK ON THIS PIPELINE once the FASTQs are deleted, and that is structural.**
`fetch_fastq` declares `protected()`, and `--touch` runs each job's postprocess, which calls
`handle_protected()` -> `IOFile.protect()` -> `os.lstat` on the output. On a deleted FASTQ that is:

    FileNotFoundError: [Errno 2] No such file or directory: '.../data/fastq/DRR991137_1.fastq.gz'

and it takes the whole invocation down. The repo's own lifecycle is fetch -> map -> **delete the FASTQs
to save space** (~202 GiB), so any DAG that reaches `fetch_fastq` is in this state, which is most of
them. Reproduced on a fixture by adding `protected()` to its fetch rule — without the decorator `--touch`
completes and skips missing files, which is why an earlier version of this note recommended it. Three of
this pipeline's five `protected()` uses are files that legitimately get cleaned up, so do not expect
`--touch` to be available.

**So the working answer is the `--forcerun` + `--allowed-rules` pair above.** It needs no DAG state, so
it is repeatable, and the cost is having to pass both flags every time.

If you genuinely want the DAG quiet, the lever that DID work is `--cleanup-metadata` on the offending
outputs: deleting a record leaves nothing to compare, and Snakemake then reports the file as up to date
(this is why `chrom_sizes` disappeared from the job list after its records were cleared, even though its
`.fai` is still newer). Weigh it carefully — a file with no provenance also stops re-triggering on code
and params changes, which is the silent-staleness failure this file warns about elsewhere.

**And `--touch -n` tells you nothing either way**: it is a dry run OF THE TOUCH, so it prints the whole
DAG it would touch — the same several-hundred-job list you are trying to eliminate — which reads exactly
like the touch having failed.

Two things measured on the fixture rather than assumed: `--touch` does **not** fabricate a missing
input — the absent FASTQ stayed absent, so it cannot manufacture an empty file at a path the fetch
convention treats as complete-and-verified — and the subsequent `--forcerun` genuinely re-executes the
rule rather than touching it (the fixture's output picked up the edited script's content).

`--touch` still asserts the existing outputs are correct. If a `.fai` differs in *content* from what the
bigWigs were built against, that assertion is false and a real rebuild is owed — the one case where
paying for the 701 jobs is the right answer, and the reason to check before touching.

**Check it by CONTENT, not by reasoning about the timestamps** — `chrom_sizes` is
`cut -f1,2 {fai} | grep -E '^<chrom>\t|…'`, so its output is a pure function of the `.fai` and
`main_chromosomes` and can be reproduced in a few lines:

    python - <<'EOF'
    import yaml, pathlib
    G = yaml.safe_load(open('config/genomes.yaml'))['species']
    R = pathlib.Path('.')
    for sp, d in sorted(G.items()):
        fai, cs = R/(d['fasta']+'.fai'), R/'data/procap_work/genome'/f'{sp}.chrom.sizes'
        if not (fai.exists() and cs.exists()):
            print(f'{sp:18} SKIP'); continue
        keep = {str(c) for c in d['main_chromosomes']}
        want = ''.join(f'{p[0]}\t{p[1]}\n' for p in
                       (l.split('\t') for l in fai.read_text().splitlines()) if p[0] in keep)
        print(f'{sp:18} {"IDENTICAL" if want == cs.read_text() else "DIFFERS"}')
    EOF

Run on the real tree 2026-09-06: **all 12 IDENTICAL**, so the entire 701-job plan would have rewritten
byte-identical files.

**There are TWO independent causes here, and the second one is the blocker. Do not stop at the
timestamps.** An earlier version of this note attributed the whole thing to genome prep done outside the
DAG — all 8 `.chrom.sizes` share the second `2026-09-03 02:45:43` while every `.fai` is an hour or more
later (`03:56`-`04:35`), and Snakemake does report "missing provenance/metadata" for `faidx`,
`fetch_genome` and others. That inference was wrong about the `chrom_sizes` jobs specifically:

    IncompleteFilesException:
    Incomplete files:
    .../genome/M.musculus.chrom.sizes   (+ 7 more)

**Snakemake HAS metadata for those 8 and it says INCOMPLETE** — a `chrom_sizes` batch was interrupted
mid-write, which is exactly what 8 files sharing one second looks like. An incomplete marker forces a
rerun whatever the mtimes say, and once `--touch` is attempted it **aborts DAG construction entirely**,
so nothing proceeds until it is cleared. The 8 incomplete files are precisely the `chrom_sizes 8` in the
original 701-job plan.

**Clear it with `--cleanup-metadata`, NOT with `--rerun-incomplete`** — and the content check above is
what licenses that. **Pass ABSOLUTE paths**, exactly as the exception printed them:

    snakemake --cleanup-metadata \
        /path/to/repo/data/procap_work/genome/{A.thaliana,C.elegans,C.griseus,\
        D.melanogaster,G.arboreum,G.hirsutum,M.musculus,S.moellendorffii}.chrom.sizes

**That error is misleading in TWO ways, and the second one matters more.**

First, metadata is keyed by the LITERAL path string — `Persistence._get_key()` is `str(f)` with no
normalization — and every path in this Snakefile is absolute because
`REPO_ROOT = Path(workflow.basedir).parent`. A relative argument therefore looks up a key that was never
written, and reports `Failed to clean up metadata ... the reason might be file system latency or still
running jobs`, which points at neither real cause.

Second, and the reason to run the dry-run before believing the error: **with absolute paths it can fail
while having already done the thing you wanted.** There are two separate stores,
`.snakemake/incomplete/` for markers and `.snakemake/metadata/` for records, and
`Persistence.cleanup_metadata()` is:

    key = self._get_key(target)
    self._unmark_incomplete(key)      # deletes the MARKER
    return self._delete_record(key)   # deletes the RECORD -- only this is reported

An interrupted job wrote a marker (`started()`) and never wrote a record (`finished()` never ran), so
the marker is deleted and the record was never there — failure is reported for the half that was already
absent. **Re-run the dry-run rather than trusting the exit status**; the `IncompleteFilesException` is
usually gone. Confirm on disk if you want, markers being urlsafe-base64 of the absolute path:

    python - <<'EOF'
    from base64 import urlsafe_b64encode
    from pathlib import Path
    root = Path('/path/to/repo')
    for store in (root/'.snakemake/incomplete', root/'workflow/.snakemake/incomplete'):
        for sp in ('A.thaliana', 'M.musculus'):        # etc.
            key = str(root/'data/procap_work/genome'/f'{sp}.chrom.sizes')
            b = urlsafe_b64encode(key.encode()).decode()
            print(store, sp, 'PRESENT' if (store/b).exists() else 'gone')
    EOF

Check **both** stores: `.snakemake` is resolved against the WORKING directory, so running from `workflow/`
and from the repo root build two independent ones, and a marker cleared in one still blocks the other.

Note `--cleanup-metadata` deletes the whole record, not just the marker, so those files end up with NO
provenance and fall back to mtime comparisons — which is why the `--touch` step below is usually needed
after it rather than instead of it.

`--rerun-incomplete` is the trap, and it is the remedy the error message lists second. Regenerating
`chrom_sizes` is trivially cheap in itself — a `cut` and a `grep` — but it gives those files a NEW mtime,
which makes them newer than every bigWig and triggers the full `bigwig -> bedgraph -> align -> trim ->
fetch_fastq` cascade this section is about. The cheap fix causes the expensive one.
`--cleanup-metadata` only clears the flag and leaves the timestamps alone.

Then re-run the dry-run. If `chrom_sizes` still appears, the mtime reason is live as well and `--touch`
handles that; if it does not, the incomplete marker was the whole story. Both are worth knowing because
they present identically in the job counts and only the `Reasons:` block and the exception message tell
them apart.

## `--rerun-triggers mtime` CANNOT see a new input, so it cannot see a new decoy

Measured on a fixture 2026-09-06, after
`snakemake --rerun-triggers mtime --config experiments=...` reported **"Nothing to be done"** for a
re-map that had just gained two organelle decoys.

`mtime` compares the timestamps of a job's **existing** inputs against its outputs. A decoy that has
never been fetched has no timestamp, and Snakemake does not schedule a missing input's rule when the
downstream output already exists -- the same behaviour that makes a deleted `temp()` chain fail to
retrigger. So adding an `organelle_accessions` or `rdna_accession` entry is **invisible** under `mtime`:
no `fetch_decoy`, no `star_index`, no `align`. The `input` trigger, which is ON by default, is the one
that notices a rule's input *set* changed.

| invocation | result on the fixture |
| --- | --- |
| `--rerun-triggers mtime` | **Nothing to be done** |
| default triggers | `fetch_decoy` -> `index` -> `align` |
| `--rerun-triggers mtime --forcerun index` | `fetch_decoy` -> `index` -> `align` |

**So use DEFAULT triggers for a config change and scope it with `--config experiments=`**, which is what
bounds the cascade -- the reason to reach for `mtime` in the first place was avoiding a corpus-wide
re-run, and a subset achieves that without disabling the trigger you need. It is also the more correct
choice for a `genomes.yaml` edit generally, since those values reach `bedgraph` and `pints` as `params`,
which `mtime` equally cannot see. Keep `mtime` for an edited SCRIPT, where the trigger is an existing
input with a fresh timestamp.


