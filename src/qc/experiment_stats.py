#!/usr/bin/env python3
"""Per-experiment mapping and peak statistics, for exclude/merge decisions.

Answers, for every experiment the pipeline has processed: how many reads went
in, how many survived unique-mapping and dedup to actually become signal, and
how many peaks PINTS called from them. Archive read counts alone do not tell
you this -- a library can arrive deep and map badly -- so this reads the real
pipeline outputs.

Two modes:

    # one experiment, called by the `experiment_stats` rule in workflow/Snakefile
    python src/qc/experiment_stats.py -e EXP --work DIR --peaks F --pints-dir D -o F

    # every experiment in one pass, no Snakemake -- for a quick refresh
    python src/qc/experiment_stats.py --all -o qc/stats/experiment_stats.tsv \
        --markdown qc/stats/experiment_stats.md

    # aggregate every per-experiment TSV into one sorted table + a markdown copy
    python src/qc/experiment_stats.py --combine qc/stats/per_experiment/*.tsv \
        -o qc/stats/experiment_stats.tsv --markdown qc/stats/experiment_stats.md

Column notes, because several are easy to misread:

* `archive_reads`   -- ENA read_count, summed over runs. What was deposited.
* `input_reads`     -- STAR "Number of input reads", summed. POST-trim, so it
                       is below archive_reads by whatever fastp dropped.
* `unique_reads`    -- STAR "Uniquely mapped reads number", summed. The ENCODE
                       MAPQ 255 filter keeps exactly these.
* `signal_reads`    -- BLANK means NOT MEASURED, never 0: a BAM holding none of
                       the configured main_chromosomes is a BAM from another
                       assembly, not an empty library, and reporting its 0 hid
                       exactly that after two assembly swaps. Counted from the
                       SAMPLE-keyed BAMs, falling back to run-keyed ones only
                       for a tree predating the per-sample alignment refactor.
                       Reads in the merged BAM ON main_chromosomes, i.e. what
                       actually reaches a bigWig. Below unique_reads by dedup
                       (UMI libraries only), by the one-mate filter (paired
                       libraries only), and by the main_chromosomes restriction.
                       This is the number that matters.
                       It used to count EVERY contig, which overstated usable
                       depth wherever chrM, Pt or a decoy is present, because
                       `bedgraph` filters those out downstream.
* `pct_unique`      -- unique_reads / input_reads. Low values mean a mapping
                       problem: wrong assembly, contamination, or spike-in
                       reads that were never split off.
* `reads_per_peak`  -- signal_reads / peaks_total. A crude signal-density
                       measure; very low values mean the peak set is being
                       called from thin coverage.
* `qc_flags`        -- comma-joined failures, each prefixed FAIL: or WARN: so
                       a consumer can filter on severity without knowing the
                       vocabulary. See the threshold block below for what each
                       one means and the corpus value it was set from.

FLAGS ARE ADVISORY. Nothing in this repo excludes an experiment automatically:
"bad" depends on what the model is for, and a silent auto-exclusion would
change the training set without leaving a record. Exclusion is a manual
decision, recorded as `tier` in planning/manifest_samples.tsv. What this column
buys is that a bad library is discoverable in the table everyone already reads,
instead of only in a plot someone happens to look at.

A blank `qc_flags` on an unmapped experiment does NOT mean it is clean -- its
pipeline columns are blank, and blank is not the same as good.

`peaks_below_species` needs the whole table, so it is computed in the --all and
--combine paths and is always empty in a single-experiment TSV. --combine
recomputes every flag rather than trusting what the per-experiment jobs wrote.

--combine's ROW SET IS config/experiment_config.yaml, not the directory it
globs. A TSV naming no configured experiment is skipped with a warning, because
it can never contribute a row: after an experiment is renamed or split nothing
has the wildcards to rewrite its file, so it is frozen at whatever schema it had
and will either fail the header check or silently add a stale row. A configured
experiment with a wrong header is the opposite case and still exits nonzero --
that is stale output to rebuild, not a file to ignore.

`--flagged-only` writes just the flagged rows, for a short worklist.

For paired-end libraries `signal_reads` counts one mate per fragment, since
that is what final_bam keeps. Missing or unreadable inputs give empty cells
rather than an error -- this is a reporting tool and must not fail a DAG.
"""

import argparse
import csv
import glob
import gzip
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = REPO_ROOT / "config" / "experiment_config.yaml"
RESOLVED = REPO_ROOT / "planning" / "manifest_runs_resolved.tsv"

COLUMNS = [
    "experiment", "species", "tier", "assay_family", "layout", "has_umi",
    "n_samples", "n_runs",
    "archive_reads", "input_reads", "unique_reads", "pct_unique",
    "signal_reads", "pct_signal_of_unique",
    "peaks_total", "peaks_unidirectional", "peaks_bidirectional",
    "reads_per_peak", "pct_rrna", "pct_unique_adj", "qc_flags",
]

# --- Failure thresholds -------------------------------------------------------
#
# These make a bad library DISCOVERABLE. Nothing here excludes anything: the
# flags are advisory and exclusion stays a manual decision recorded as `tier`
# in planning/manifest_samples.tsv, because "bad" depends on what the model is
# for and because an auto-exclusion would silently change the training set.
#
# Every threshold is set from the observed corpus, quoted here so the next
# person can see whether it still separates anything.

#: Mapping rate below this is a WARN, below VERY_LOW a FAIL. Both are tested
#: against `pct_unique_adj` -- unique reads over NON-rRNA input -- whenever
#: src/qc/rrna_content.py has measured the library, and against raw pct_unique
#: only as a fallback.
#:
#: THE ADJUSTMENT IS THE POINT. Raw pct_unique is not a quality metric for a
#: species whose rDNA array sits in the assembly in 2+ near-identical copies:
#: every rRNA read is then a multimapper, gets MAPQ ~3, and is correctly dropped
#: by the -q 255 filter. Measured rRNA + organellar content against the ceiling
#: it implies:
#:
#:   S.cerevisiae Booth  70.4%  ceiling 29.6%  observed 16.3%
#:   S.pombe Booth       46.0%  ceiling 53.9%  observed 42.0%
#:   C.reinhardtii       67.4%  ceiling 32.6%  observed 10.8%
#:   P.patens            47.4%  ceiling 52.6%  observed 12.3%
#:
#: On the raw number the first three were FAIL:very_low_mapping, which said
#: little more than "this organism has rDNA in its assembly".
LOW_MAPPING_PCT = 50.0
VERY_LOW_MAPPING_PCT = 25.0
#: Coverage per called peak, below which the labels are thin for a
#: base-resolution model.
#:
#: **This replaced an ABSOLUTE read count (10M), which was wrong in a way that
#: correlated with species.** The depth a library needs is proportional to the
#: size of the nascent transcriptome being sampled, not a constant: a small
#: genome with few distal elements reaches the same coverage per initiation site
#: on far fewer reads than mouse or human. Judging all 12 species against one
#: read count therefore penalised the compact genomes for being compact, and it
#: was measurably backwards in both directions on the real corpus:
#:
#:                                    signal   reads/peak   old flag
#:   S.pombe_PROcap                    23.1M       2,513    (none)  <- best sampled
#:   S.cerevisiae_PROcap                4.9M         721    SHALLOW <- false positive
#:   C.reinhardtii-liquidculture        6.3M         770    SHALLOW <- false positive
#:   M.musculus-BMDM_5GRO-ctl          18.2M         450    (none)  <- false negative
#:   D.melanogaster-S2_5GROcap         22.2M         518    (none)  <- false negative
#:
#: S. cerevisiae at 4.9M matches C.griseus-CHO's 723 reads/peak on a tenth of
#: the reads, and S. pombe on 23.1M is better sampled than M.musculus-GCB on
#: 150.4M. Meanwhile two libraries above 18M were sampling their (much larger)
#: transcriptomes more thinly than several flagged ones, and escaped silently.
#:
#: `reads_per_peak` is the normalisation the corpus already carries: peaks are
#: this pipeline's own measure of how much transcribed space exists, so dividing
#: by them asks "how deeply is each initiation site covered" rather than "how
#: many reads are there". It is not perfectly independent of depth — peak
#: calling saturates, so a shallow library calls fewer peaks and its denominator
#: shrinks too — but it errs conservatively, since peaks fall more slowly than
#: reads.
#:
#: **The threshold is a heuristic recalibrated from this corpus, not a
#: principled constant** — same status as the FLAT initiator threshold. There is
#: no clean gap in the distribution; 500 sits below a loose cluster at 595-770
#: and above the genuinely thin tail at 209-518. Revisit it against a fuller
#: corpus rather than treating it as settled.
THIN_COVERAGE_READS_PER_PEAK = 500
#: The "excessive adapter content" flag: a run is majority adapter dimer when
#: its MEDIAN insert is below fastp's --length_required (so most reads are
#: discarded before alignment) AND the adapter is actually prevalent.
#:
#: BOTH conditions are needed, and each alone gives false positives that were
#: observed here:
#:
#: * median_insert alone flagged all four C. elegans runs, whose 30 bp read is
#:   SHORTER than the insert -- so the adapter is present in 0.01% of reads and
#:   the median is taken over that unrepresentative 0.01%. It also flagged
#:   Lam2013, whose ~5% dimer background CLAUDE.md already records as harmless,
#:   and R2 of the paired yeast runs, where the survey looks for the wrong
#:   adapter (R2 reads into the 5' adapter's reverse complement, not RA3).
#: * pct_adapter alone is what CLAUDE.md warns against: every library carries a
#:   few percent of dimers, which cost nothing to leave in.
#:
#: Observed separation, R1 only: genuine dimer libraries are 42-88% adapter
#: (SRR29037352 88.3% at insert 4, SRR29037355 87.5%, SRR29037350 85.9%,
#: SRR6660402 42.4%); every harmless case is at or below 7.8%. 25% sits with a
#: 3.2x margin over the highest harmless value.
MIN_MEDIAN_INSERT = 18
#: A detected adapter this prevalent in R1 contradicts a different configured
#: name outright. Well above the few percent of cross-matching every library
#: shows, so a disagreement means the manifest is wrong, not that two adapters
#: are both weakly present.
ADAPTER_CONFLICT_PCT = 50.0
#: Survey columns the read-level checks need. A survey TSV predating any of them
#: silently disables the check that reads it, which is why the schema is
#: verified rather than assumed -- see report_flags.
SURVEY_REQUIRED = ("best_adapter", "interleave_suspect", "declared_interleaved")
DIMER_ADAPTER_PCT = 25.0
#: Peak count below this fraction of the species median is a WARN. Absolute
#: peak counts are not comparable across species -- 6,000 peaks is normal for
#: yeast and thin for mouse -- so the comparison has to be within species.
#: This is what made S.cerevisiae-Spt5EtOH_PROcap visible: 5,897 against
#: siblings at 15,051 and 17,315.
PEAKS_SPECIES_FRACTION = 0.5
#: Species needed before a within-species median means anything.
MIN_SPECIES_N = 3


def load_config() -> dict:
    import yaml
    with CONFIG.open() as f:
        return yaml.safe_load(f)["experiments"]


def star_logs(work: Path, exp_id: str, sample: str, runs: list[str]) -> list[Path]:
    """Every STAR log belonging to one sample, across all pipeline layouts.

    Alignment is now per SAMPLE, so the current layout has ONE log per sample. Trees
    mapped before that change have one per RUN, and a sample's runs are the SRRs it was
    built from -- so a sample-keyed lookup alone finds nothing on an existing tree, and
    the row silently reports no STAR metrics. Return whatever exists, and let the caller
    sum: one log in the new layout, N in the old, and the totals agree either way.
    """
    found = [work / "samples" / sample / "Log.final.out"]
    found = [c for c in found if c.exists()]
    if found:
        return found
    legacy = []
    for run in runs:
        for candidate in (work / "runs" / run / "Log.final.out",
                          work / exp_id / f"{run}.Log.final.out"):
            if candidate.exists():
                legacy.append(candidate)
                break
    return legacy


def star_log(work: Path, exp_id: str, unit: str) -> Path | None:
    """Locate a STAR Log.final.out for one alignment unit, whichever driver made it.

    `unit` is a SAMPLE for the current Snakefile and a RUN for the older layouts, which
    is why all three candidates are tried rather than one being chosen by driver:

      {work}/samples/{sample}/Log.final.out   workflow/Snakefile (alignment is per sample
                                              since resequencing runs merge as FASTQ)
      {work}/runs/{run}/Log.final.out         workflow/Snakefile before that change
      {work}/{exp}/{run}.Log.final.out        run_procap_pipeline.py

    Silence is the failure mode here -- a missing log yields None and the row simply
    reports no STAR metrics, so a path that has moved looks like "not mapped yet".
    """
    for candidate in (
        work / "samples" / unit / "Log.final.out",
        work / "runs" / unit / "Log.final.out",
        work / exp_id / f"{unit}.Log.final.out",
    ):
        if candidate.exists():
            return candidate
    return None


def star_metrics(log: Path | None) -> tuple[int | None, int | None]:
    """(input reads, uniquely mapped reads) from a STAR Log.final.out."""
    if log is None or not log.exists():
        return None, None
    inp = uniq = None
    for line in log.read_text().splitlines():
        if "Number of input reads" in line:
            inp = _int(line)
        elif "Uniquely mapped reads number" in line:
            uniq = _int(line)
    return inp, uniq


def _int(line: str) -> int | None:
    m = re.search(r"\|\s*([0-9]+)\s*$", line.strip())
    return int(m.group(1)) if m else None


def load_genomes() -> dict:
    import yaml
    with open(REPO_ROOT / "config" / "genomes.yaml") as f:
        return yaml.safe_load(f)["species"]


def mapped_reads(bam: Path, keep: set | None = None) -> int | None:
    """Mapped reads via `samtools idxstats`, which reads the index only.

    `keep` restricts the count to the contigs that actually reach the labels.
    This matters and used to be wrong: idxstats sums EVERY contig, including
    chrM, Pt and any decoy, but `bedgraph` greps to main_chromosomes afterwards,
    so those reads never enter a bigWig or a peak. Counting them made
    `signal_reads` -- documented as "the number that matters" -- overstate usable
    depth for the 9 of 12 species with in-assembly organelles, and the error
    grows with every decoy added (12.2% of the C. reinhardtii library lands on
    the plastid decoy alone).
    """
    if not bam.exists():
        return None
    try:
        out = subprocess.run(
            ["samtools", "idxstats", str(bam)],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    total, matched = 0, False
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] != "*":
            if keep is not None and parts[0] not in keep:
                continue
            matched = True
            total += int(parts[2])
    # NOT-MEASURED, NOT ZERO. A BAM holding none of the current
    # main_chromosomes is a BAM from a different naming regime -- almost always
    # an older assembly left on disk -- and returning its 0 makes that
    # indistinguishable from a library with no usable reads. That is exactly the
    # distinction pct_rrna already draws (blank means NOT measured, which is not
    # 0), and it bit here: after the C. reinhardtii and P. patens assembly
    # swaps, both reported `signal_reads 0` beside 9,216 and 9,833 called peaks,
    # which cannot both be true. A real library legitimately having zero reads on
    # its main chromosomes would also have no peaks, so blank is the honest cell
    # either way.
    if keep is not None and not matched:
        print(f"WARNING: {bam} holds none of the configured main_chromosomes "
              "-- wrong assembly for this config? Reporting signal_reads as "
              "NOT MEASURED rather than 0.")
        return None
    return total


def count_bed(path: Path) -> int | None:
    if not path.exists():
        return None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return sum(1 for line in fh if line.strip() and not line.startswith("#"))


def peak_classes(pints_dir: Path) -> dict:
    """Row counts for each PINTS peak class present."""
    counts = {}
    for cls in ("unidirectional", "bidirectional"):
        n = 0
        found = False
        for f in sorted(pints_dir.glob(f"*{cls}_peaks.bed")):
            c = count_bed(f)
            if c is not None:
                n += c
                found = True
        counts[cls] = n if found else None
    return counts


def archive_reads(exp_id: str, samples: list[str]) -> int | None:
    """Deposited read count, summed over the experiment's runs."""
    if not RESOLVED.exists():
        return None
    wanted, total, seen = set(samples), 0, False
    with RESOLVED.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["sample_accession"] not in wanted:
                continue
            for v in (row.get("read_count") or "").split(";"):
                if v.strip().isdigit():
                    total += int(v)
                    seen = True
    return total if seen else None


def pct(num, den) -> str:
    if not num or not den:
        return ""
    return f"{100.0 * num / den:.1f}"


def collect(exp_id: str, entry: dict, work: Path, peaks: Path,
            pints_dir: Path, genomes: dict | None = None) -> dict:
    runs = entry["raw"]["runs"]
    samples = entry["raw"]["sample_accessions"]
    # main_chromosomes as strings: bare-numeric names parse as ints from YAML
    # and would match nothing against idxstats output.
    keep = None
    if genomes and entry["species"] in genomes:
        keep = {str(c) for c in genomes[entry["species"]]["main_chromosomes"]}

    inputs = uniques = 0
    saw_input = saw_unique = False
    # Alignment units, not runs: STAR now runs once per SAMPLE, so a sample whose two
    # resequencing runs were merged has ONE log. Iterating runs would look for two, find
    # neither, and silently report no metrics. raw.samples is authoritative; fall back to
    # runs for configs predating it.
    # Alignment units, not runs: STAR runs once per SAMPLE now, so a sample whose
    # resequencing runs were merged has one log where an older tree has several.
    # star_logs() resolves both, so this works before and after a re-map.
    grouping = entry.get("raw", {}).get("samples") or {r: [r] for r in runs}
    for sample, sample_runs in grouping.items():
        for log in star_logs(work, exp_id, sample, list(sample_runs)):
            i, u = star_metrics(log)
            if i is not None:
                inputs += i
                saw_input = True
            if u is not None:
                uniques += u
                saw_unique = True

    signal = mapped_reads(work / "exp" / exp_id / "merged.bam", keep)
    if signal is None:                      # merged.bam is temp(); fall back
        # SAMPLE-keyed first, RUN-keyed only for a tree that predates the
        # per-sample alignment refactor. This fallback was still run-keyed after
        # that move, which is worse than merely finding nothing: the old
        # `runs/{run}/final.bam` files are NOT temp() and survive on disk, so on
        # a re-mapped tree it silently read BAMs aligned to the PREVIOUS
        # assembly. Combined with the zero above, that is how C. reinhardtii and
        # P. patens came out at `signal_reads 0`. star_logs() was taught both
        # layouts for the same reason; this is the counterpart it missed.
        bams = [work / "samples" / s / "final.bam" for s in grouping]
        if not any(b.exists() for b in bams):
            bams = [work / "runs" / r / "final.bam" for r in runs]
        per_bam = [mapped_reads(b, keep) for b in bams]
        if any(v is not None for v in per_bam):
            signal = sum(v for v in per_bam if v is not None)

    cls = peak_classes(pints_dir)
    total_peaks = count_bed(peaks)
    umi = entry["raw"].get("umi") or {}

    return {
        "experiment": exp_id,
        "species": entry["species"],
        "tier": entry.get("tier", ""),
        "assay_family": entry.get("assay_family", ""),
        "layout": entry["raw"].get("library_layout", ""),
        "has_umi": "yes" if umi.get("enabled") else "no",
        "n_samples": len(samples),
        "n_runs": len(runs),
        "archive_reads": archive_reads(exp_id, samples) or "",
        "input_reads": inputs if saw_input else "",
        "unique_reads": uniques if saw_unique else "",
        "pct_unique": pct(uniques if saw_unique else 0, inputs if saw_input else 0),
        "signal_reads": signal if signal is not None else "",
        "pct_signal_of_unique": pct(signal, uniques if saw_unique else 0),
        "peaks_total": total_peaks if total_peaks is not None else "",
        "peaks_unidirectional": cls["unidirectional"] if cls["unidirectional"] is not None else "",
        "peaks_bidirectional": cls["bidirectional"] if cls["bidirectional"] is not None else "",
        "reads_per_peak": (
            f"{signal / total_peaks:.0f}"
            if signal and total_peaks else ""
        ),
    }


def write_tsv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t",
                           lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows: list[dict], out: Path) -> None:
    """A readable subset. The TSV keeps every column."""
    show = ["experiment", "species", "tier", "n_runs", "archive_reads",
            "unique_reads", "pct_unique", "signal_reads", "peaks_total",
            "reads_per_peak", "pct_rrna", "pct_unique_adj", "qc_flags"]
    head = {"archive_reads": "archive", "unique_reads": "unique",
            "signal_reads": "signal", "peaks_total": "peaks",
            "pct_unique": "%uniq", "reads_per_peak": "rd/pk",
            "pct_rrna": "%rRNA", "pct_unique_adj": "%uniq_adj"}

    def cell(r, c):
        v = r.get(c, "")
        if c.endswith("_reads") and str(v).isdigit():
            return f"{int(v) / 1e6:.1f}M"
        return str(v)

    lines = [
        "| " + " | ".join(head.get(c, c) for c in show) + " |",
        "| " + " | ".join("---" for _ in show) + " |",
    ]
    lines += ["| " + " | ".join(cell(r, c) for c in show) + " |" for r in rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    # Note the explicit +: writing the banner and "\n".join(lines) as adjacent
    # string literals concatenates them BEFORE .join runs, which silently makes
    # the banner the row separator.
    banner = ("<!-- GENERATED by src/qc/experiment_stats.py. Do not edit; rerun\n     `snakemake stats -c8` or `python src/qc/experiment_stats.py --all\n     -o qc/stats/experiment_stats.tsv --markdown qc/stats/experiment_stats.md`.\n     Blank pipeline columns mean that experiment has not been mapped yet. -->\n\n")
    out.write_text(banner + "\n".join(lines) + "\n")


def _num(v):
    """A stats cell as a float, or None. Cells arrive as str via --combine."""
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def rrna_pct(qc_rrna: Path, exp_id: str) -> tuple[bool, float | None]:
    """(entry_found, measured rRNA + organellar share) for one experiment.

    Returns the pair rather than a bare value so that "the rrna output says this
    experiment has no rRNA source" is distinguishable from "there is no rrna
    output". Without that distinction the `pct_rrna` fallback below silently
    resurrected a stale organellar-only number for D. melanogaster and adjusted a
    mapping rate with it.

    Read opportunistically, like the read survey: it needs the raw FASTQs, which
    are routinely deleted after mapping. None means NOT MEASURED and is
    deliberately distinct from 0.0 -- D. melanogaster and C. elegans cannot be
    measured at all, because refGene GTF carries no `rRNA` feature and neither
    has an rDNA accession to fall back on.
    """
    for name in (f"{exp_id}.tsv", "rrna_content.tsv"):
        path = qc_rrna / name
        if not path.exists():
            continue
        with path.open() as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r.get("experiment") != exp_id:
                    continue
                if not _num(r.get("n_scored")):
                    return True, None
                # `rrna_indexed: no` means the run scored organellar content but
                # had no rRNA source at all, so pct_rrna_total is not an rRNA
                # measurement and must not adjust a mapping rate.
                if str(r.get("rrna_indexed", "yes")).strip().lower() == "no":
                    return True, None
                return True, _num(r.get("pct_rrna_total"))
    return False, None


def configured_adapter(exp_id: str) -> str | None:
    """`raw.adapter` for one experiment, or None if the config cannot be read.

    Needed because the manifest's adapter name is CURATED (from the paper or GEO
    record) while the survey MEASURES what is in the reads, and until now
    nothing compared the two. `G.{arboreum,hirsutum}-ovule_GROcap` were set to
    smallRNA_RA3 from Wen et al.'s TrimGalore `--small_rna`, while all four runs
    are 97.2-97.7% truseq_universal -- a silent disagreement, and exactly the
    failure the adapter column exists to prevent.
    """
    global _ADAPTER_CACHE
    if _ADAPTER_CACHE is None:
        try:
            _ADAPTER_CACHE = {
                k: (v.get("raw", {}) or {}).get("adapter") or ""
                for k, v in load_config().items()
            }
        except Exception:                      # no config here; skip the check
            _ADAPTER_CACHE = {}
    return _ADAPTER_CACHE.get(exp_id)


_ADAPTER_CACHE: dict[str, str] | None = None


def read_survey_flags(qc_reads: Path, exp_id: str) -> list[str]:
    """Per-run read-structure failures for one experiment, from the survey TSV.

    Read OPPORTUNISTICALLY and not declared as a Snakemake input, on purpose.
    read_structure_qc.py surveys the raw FASTQs, which are ~202 GiB and are
    routinely deleted once mapping is done; making the stats table depend on
    them would mean the table could no longer be rebuilt from surviving
    outputs. Absent survey -> no read-level flags, and the mapping and peak
    flags still apply.
    """
    path = qc_reads / f"{exp_id}.tsv"
    if not path.exists():
        path = qc_reads / "read_structure.tsv"
        if not path.exists():
            return []
    dimer, interleaved, wrong_adapter = [], [], {}
    want = configured_adapter(exp_id)
    with path.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("experiment") not in (exp_id, None):
                continue
            if not _num(r.get("n_sampled")) or r.get("mate") != "R1":
                continue
            ins, adapt = _num(r.get("median_insert")), _num(r.get("pct_adapter"))
            if (ins is not None and ins < MIN_MEDIAN_INSERT
                    and adapt is not None and adapt >= DIMER_ADAPTER_PCT):
                dimer.append(r["run"])
            # An already-declared interleaved deposit is HANDLED, not a defect:
            # the deinterleave rule splits it and the one-mate filter fires. Only
            # an undeclared suspicion is worth flagging.
            if (r.get("interleave_suspect")
                    and str(r.get("declared_interleaved", "no")).lower() != "yes"):
                interleaved.append(r["run"])
            # The manifest's curated adapter name against what the reads show.
            # Only a prevalent detection disagrees; a blank configured value
            # means "none detected, let fastp auto-detect", which is a
            # deliberate default and not a conflict.
            got = (r.get("best_adapter") or "").strip()
            if (want and got and got not in ("none", "missing") and got != want
                    and adapt is not None and adapt >= ADAPTER_CONFLICT_PCT):
                wrong_adapter[got] = wrong_adapter.get(got, 0) + 1
    out = []
    for label, runs in (("dimer_run", dimer), ("interleave_suspect", interleaved)):
        if runs:
            out.append(f"{label}:" + "/".join(sorted(set(runs))))
    if wrong_adapter:
        got = max(wrong_adapter, key=wrong_adapter.get)
        out.append(f"adapter_mismatch(config={want},reads={got})")
    return out


def add_flags(rows: list[dict], qc_reads: Path | None = None,
              qc_rrna: Path | None = None) -> None:
    """Set `qc_flags` on every row, in place.

    Flags are prefixed FAIL: or WARN: so a downstream consumer can filter on
    severity with a substring match and does not have to know the vocabulary.
    An unmapped experiment gets no flags rather than a spurious clean bill --
    its pipeline columns are blank, and blank is not the same as good.
    """
    import statistics
    by_species = {}
    for r in rows:
        n = _num(r.get("peaks_total"))
        if n:
            by_species.setdefault(r.get("species", ""), []).append(n)
    medians = {sp: statistics.median(v) for sp, v in by_species.items()
               if len(v) >= MIN_SPECIES_N}

    for r in rows:
        flags = []
        uniq = _num(r.get("pct_unique"))
        seen, rrna = (rrna_pct(qc_rrna, r.get("experiment", "")) if qc_rrna
                      else (False, None))
        if not seen:
            rrna = _num(r.get("pct_rrna"))     # already present via --combine
        r["pct_rrna"] = "" if rrna is None else f"{rrna:.1f}"
        adj = None
        if uniq is not None and rrna is not None and rrna < 100:
            adj = min(100.0, uniq / (1.0 - rrna / 100.0))
        r["pct_unique_adj"] = "" if adj is None else f"{adj:.1f}"
        # Judge on the rRNA-adjusted rate where it exists. The suffix records
        # which number was used, so a flag is never ambiguous about its basis.
        rate, basis = (adj, "adj") if adj is not None else (uniq, "raw")
        if rate is not None:
            if rate < VERY_LOW_MAPPING_PCT:
                flags.append(f"FAIL:very_low_mapping({rate:.0f}%,{basis})")
            elif rate < LOW_MAPPING_PCT:
                flags.append(f"WARN:low_mapping({rate:.0f}%,{basis})")
        signal = _num(r.get("signal_reads"))
        rpp = _num(r.get("reads_per_peak"))
        if rpp is not None and rpp < THIN_COVERAGE_READS_PER_PEAK:
            # No absolute depth in the message: qc_flags is COMMA-JOINED, so a
            # comma inside one flag splits it in two for anything parsing the
            # field. signal_reads is an adjacent column anyway.
            flags.append(f"WARN:thin_coverage({rpp:.0f}/peak)")
        peaks, med = _num(r.get("peaks_total")), medians.get(r.get("species", ""))
        if peaks and med and peaks < PEAKS_SPECIES_FRACTION * med:
            flags.append(f"WARN:peaks_below_species({peaks:.0f}v{med:.0f})")
        if qc_reads is not None:
            # Both survey flags are FAIL-class: a majority-dimer run and an
            # interleaved deposit are defects in the data, not thin data.
            for f in read_survey_flags(qc_reads, r.get("experiment", "")):
                flags.append(f"FAIL:{f}")
        r["qc_flags"] = ",".join(flags)


def report_flags(rows: list[dict], qc_reads: Path | None = None) -> None:
    """Print the flagged experiments, worst first. Discoverability is the point.

    Says so out loud when the read survey is absent. The survey is read
    opportunistically rather than declared as a Snakemake input (see
    read_survey_flags), which means the read-level flags could otherwise be
    silently missing -- and a table that looks clean because half the checks
    did not run is worse than no table.
    """
    if qc_reads is not None and not any(qc_reads.glob("*.tsv")):
        print(f"\nWARNING: no read survey under {qc_reads} -- dimer_run and "
              "interleave_suspect were NOT checked. Run:\n"
              "    python src/qc/read_structure_qc.py --tsv qc/reads/read_structure.tsv")
    elif qc_reads is not None:
        # A survey written before a column existed disables the check that
        # reads it, and says nothing while doing so. That is the same class of
        # failure as a missing survey, so report it the same way. It bites
        # hardest on the corpus-wide read_structure.tsv, which read_survey_flags
        # falls back to when a per-experiment file is absent: a stale one there
        # satisfies the fallback and silently answers for every experiment.
        # Only files that ANSWER for something. read_survey_flags reads
        # qc/reads/{exp}.tsv and falls back to the corpus-wide
        # read_structure.tsv, so those are the only files whose schema can
        # disable a check. An orphaned survey for a removed experiment is
        # consulted by nothing, and warning about it was pure noise in the one
        # place whose value is that a warning means something.
        consulted = {r.get("experiment", "") for r in rows} | {"read_structure"}
        for path in sorted(qc_reads.glob("*.tsv")):
            if path.stem not in consulted:
                continue
            with path.open() as f:
                cols = (csv.reader(f, delimiter="\t").__next__()
                        if path.stat().st_size else [])
            missing = [c for c in SURVEY_REQUIRED if c not in cols]
            if missing:
                print(f"\nWARNING: {path} predates {', '.join(missing)} -- the "
                      "check(s) reading those columns did NOT run for anything "
                      "it answers for. Re-run read_structure_qc.py.")
    flagged = [r for r in rows if r.get("qc_flags")]
    if not flagged:
        print("\nNo QC flags raised.")
        return
    fails = [r for r in flagged if "FAIL:" in r["qc_flags"]]
    print(f"\n{len(flagged)} experiment(s) flagged ({len(fails)} with a FAIL). "
          "Advisory only -- exclude by setting `tier` in the manifest.")
    for r in sorted(flagged, key=lambda x: ("FAIL:" not in x["qc_flags"],
                                            x["experiment"])):
        print(f"    {r['experiment']:<34} {r['qc_flags']}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-e", "--experiment")
    p.add_argument("--work", type=Path, help="pipeline work dir (data/procap_work)")
    p.add_argument("--peaks", type=Path)
    p.add_argument("--pints-dir", type=Path)
    p.add_argument("--combine", nargs="*", help="per-experiment TSVs to aggregate")
    p.add_argument("--all", action="store_true",
                   help="collect every experiment directly, without Snakemake")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--markdown", type=Path)
    p.add_argument("--qc-reads", type=Path,
                   default=REPO_ROOT / "qc" / "reads",
                   help="read_structure_qc output dir, for adapter/dimer flags")
    p.add_argument("--qc-rrna", type=Path, default=REPO_ROOT / "qc" / "rrna",
                   help="rrna_content output dir, for the rRNA-adjusted rate")
    p.add_argument("--flagged-only", action="store_true",
                   help="write only experiments carrying a QC flag")
    p.add_argument("--sort-by", default="species",
                   help="column to sort the combined table by (default: species)")
    args = p.parse_args()

    if args.all:
        # Same table the DAG builds, derived in one pass. Useful mid-run or
        # after the fact, since counting is index-only and costs nothing.
        experiments = load_config()
        genomes = load_genomes()
        work = args.work or (REPO_ROOT / "data" / "procap_work")
        rows = [
            collect(
                exp_id, entry, work,
                args.peaks or (REPO_ROOT / "data" / "procap" / f"{exp_id}_peaks.bed.gz"),
                work / "exp" / exp_id / "PINTS", genomes,
            )
            for exp_id, entry in experiments.items()
        ]
        rows.sort(key=lambda r: (str(r.get(args.sort_by, "")), r["experiment"]))
        add_flags(rows, args.qc_reads, args.qc_rrna)
        if args.flagged_only:
            rows = [r for r in rows if r.get("qc_flags")]
        write_tsv(rows, args.output)
        if args.markdown:
            write_markdown(rows, args.markdown)
        done = sum(1 for r in rows if r["signal_reads"] != "")
        print(f"Wrote {args.output}: {len(rows)} experiments "
              f"({done} with pipeline output, {len(rows) - done} not yet run)")
        if args.markdown:
            print(f"Wrote {args.markdown}")
        report_flags(rows, args.qc_reads)
        return

    if args.combine is not None:
        paths = [Path(x) for pat in args.combine for x in sorted(glob.glob(pat))] \
            if any("*" in x for x in args.combine) else [Path(x) for x in args.combine]
        # THE ROW SET IS THE CURRENT CONFIG, not whatever is on disk. The
        # Snakefile globs the whole per_experiment/ directory so that a
        # subsetting --config cannot narrow this global table, which means the
        # glob also picks up files for experiments that NO LONGER EXIST --
        # M.musculus-liver-{old,young}_ChROcap after the 2026-09-01 sex split.
        # Nothing can ever rewrite those: no rule has their wildcards to fill,
        # so each is frozen at whatever schema it was written under. Both
        # failure modes were observed from that one pair. One froze at 18
        # columns and hard-failed the header check below, killing a
        # two-experiment run that had nothing to do with it; its twin froze at
        # the current 21 and was accepted, giving a 44-row table over 42
        # experiments -- the worse outcome, because it reads as complete.
        # Skipping by name fixes both, and it is the honest rule: a file naming
        # no configured experiment cannot contribute a row whatever its header.
        configured = set(load_config())
        rows = []
        for path in paths:
            if not path.exists():
                continue
            if path.stem not in configured:
                # Deliberately not phrased as "delete it": this also catches
                # a too-broad glob that swept in something which is not a
                # stats TSV at all, and telling someone to delete their source
                # tree is worse advice than the bug.
                print(f"WARNING: ignoring {path} -- names no configured "
                      "experiment, so it cannot contribute a row. If it is a "
                      "stats TSV for an experiment that no longer exists, "
                      "delete it; nothing can regenerate it.")
                continue
            with path.open() as f:
                reader = csv.DictReader(f, delimiter="\t")
                # A CONFIGURED experiment with the wrong header is stale output
                # that must be regenerated, not ignored -- so this stays a hard
                # failure. It is also what still catches a too-broad --combine:
                # DictReader takes whatever the first line offers as
                # fieldnames, so `--combine {input}` in the Snakefile once swept
                # in qc/rrna/*.tsv and qc/reads/*.tsv, which ARE named by
                # experiment, for a 685-row table with every pipeline column
                # blank.
                if reader.fieldnames != COLUMNS:
                    sys.exit(
                        f"{path}: header has "
                        f"{len(reader.fieldnames or [])} column(s), expected "
                        f"{len(COLUMNS)}. {path.stem} IS a configured "
                        f"experiment, so this is stale output and must be "
                        f"rebuilt rather than skipped:\n"
                        f"    python {Path(__file__).name} -e {path.stem} "
                        f"-o {path}\n"
                        f"or `snakemake --forcerun experiment_stats --config "
                        f"experiments={path.stem}`. If instead this is not a "
                        f"stats TSV at all, --combine takes only "
                        f"qc/stats/per_experiment/*.tsv; qc/rrna and qc/reads "
                        f"are found automatically via --qc-rrna/--qc-reads."
                    )
                rows.extend(reader)
        # Zero rows is always a mistake and must not be written. The rule's
        # inputs guarantee the per-experiment TSVs for TARGETS exist before
        # this runs, so an empty result means the glob matched nothing that
        # counts -- and an empty global table reads as complete, which is the
        # same trap the whole-directory glob and the header check exist to
        # avoid. This is also what still fails a --combine pointed somewhere
        # entirely wrong, now that a name mismatch alone only warns.
        if not rows:
            sys.exit(
                f"--combine matched no per-experiment stats TSV "
                f"({len(paths)} file(s) examined). Expected "
                f"qc/stats/per_experiment/*.tsv."
            )
        rows.sort(key=lambda r: (str(r.get(args.sort_by, "")), r.get("experiment", "")))
        # Recomputed here rather than trusted from the per-experiment TSVs:
        # peaks_below_species needs the whole table, which a single-experiment
        # job cannot see.
        add_flags(rows, args.qc_reads, args.qc_rrna)
        if args.flagged_only:
            rows = [r for r in rows if r.get("qc_flags")]
        write_tsv(rows, args.output)
        if args.markdown:
            write_markdown(rows, args.markdown)
        print(f"Wrote {args.output}: {len(rows)} experiments")
        if args.markdown:
            print(f"Wrote {args.markdown}")
        report_flags(rows, args.qc_reads)
        return

    if not args.experiment:
        p.error("one of -e/--experiment, --all or --combine is required")
    experiments = load_config()
    if args.experiment not in experiments:
        sys.exit(f"unknown experiment: {args.experiment}")
    work = args.work or (REPO_ROOT / "data" / "procap_work")
    peaks = args.peaks or (REPO_ROOT / "data" / "procap" /
                           f"{args.experiment}_peaks.bed.gz")
    pints = args.pints_dir or (work / "exp" / args.experiment / "PINTS")
    row = collect(args.experiment, experiments[args.experiment], work, peaks,
                  pints, load_genomes())
    write_tsv([row], args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
