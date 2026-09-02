#!/usr/bin/env python3
"""Survey raw read structure: adapter content, insert length, dead cycles.

Written after the first full run showed 14 of 38 experiments losing most of
their reads to STAR's `unmapped: too short`. The cause was not read length or
the mismatch filter but an UNTRIMMED ADAPTER: PRO-seq/ChRO-cap libraries carry
the Illumina small-RNA 3' adapter (TGGAATTCTCGGGTGCCAAGG, also proseq2.0's
ADAPT1 default), inserts are 20-45 bp inside a 74 bp read, and fastp's
auto-detection missed it -- `reads with adapter trimmed: 0` -- because the
adapter starts at a different offset in every read. STAR then aligns the short
insert, soft-clips the adapter, and fails
`--outFilterMatchNminOverLread 0.66`, which on a 74 bp read demands >=49 bp
aligned. Nothing in the pipeline configured an adapter at all.

This tool measures, per run and mate, what is actually in the reads, so the
trimming fix can be configured from evidence rather than assumed:

  * `best_adapter` / `pct_adapter` -- which candidate adapter is present, and in
    what share of sampled reads. Determines whether one global adapter suffices
    or the config needs per-experiment overrides.
  * `median_insert` -- distance from read start to the adapter, i.e. the true
    insert length.
  * `pct_short_untrimmed` -- share of adapter-bearing reads whose insert is
    below MATCH_FRACTION * read_len, i.e. the share STAR will discard as
    "too short" if the adapter is NOT removed. This is the predicted recovery
    from trimming, and is the number to act on.
  * `dead_cycles` -- positions where >DEAD_CYCLE_FRAC of reads are N. A dead
    cycle matters out of proportion to its size because STAR counts an N as a
    mismatch, so with `--outFilterMismatchNmax 1` a single systematic N spends
    every read's entire mismatch budget before real variation is considered.
  * `pct_any_n`, `pct_polyg` -- N-padding and the NovaSeq/NextSeq poly-G
    artifact.
  * `interleave_suspect` -- whether this run looks like two mates deposited
    inside one "single-end" file, which is a silent-wrong-answer bug rather
    than a mapping-rate one: R2's 5' end is the RNA 3' END on the opposite
    strand, so the track becomes half strand-flipped 3'-end signal. See
    `interleave_screen()` for the two tells and the confirming test.

VALIDATED against STAR's own accounting, which is the point of the
`pct_short_untrimmed` column -- it predicts the `unmapped: too short` rate
without running an alignment:

    run          predicted %short   STAR "too short"
    SRR826225          5.4%              4.72%
    SRR826226          4.9%              5.30%
    synthetic BAD     92.9%             (95.29% for the real SRR12513902 it
                                         was built to mimic)

Note SRR826225/6 carry ~5% adapter at insert length 0 -- adapter DIMERS, which
every library has and which cost nothing to leave in. That is why the summary
flags on pct_short_untrimmed rather than pct_adapter.

stdlib only (gzip, collections), like umi_report.py, so this stays inside the
Snakemake DAG and needs no venv.

VALIDATED for the interleaving screen too: over all 69 surveyed files it flags
exactly one, SRR19034544, on both tells at once -- and no false positives, with
the nearest miss being McDonald2024_plant_5GRO's legitimate 51/75 bp spread.

Usage:
    python src/qc/read_structure_qc.py                     # every experiment
    python src/qc/read_structure_qc.py -e M.musculus-liver-old_ChROcap
    python src/qc/read_structure_qc.py --tsv qc/reads/read_structure.tsv
"""

import argparse
import gzip
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

#: Candidate adapters, most specific first. The small-RNA adapter is listed
#: first because it is the one PRO-seq/ChRO-cap/CoPRO protocols use; a prefix is
#: matched rather than the full sequence so reads that end mid-adapter still
#: count.
ADAPTERS = {
    "smallRNA_RA3": "TGGAATTCTCGGGTGCCAAGG",
    "truseq_universal": "AGATCGGAAGAGC",
    "nextera": "CTGTCTCTTATACACATCT",
}
#: Shortest adapter prefix that still counts as a hit. 12 bp is specific enough
#: to be effectively absent by chance in a genome-sized sample.
MIN_ADAPTER_PREFIX = 12
#: STAR's --outFilterMatchNminOverLread default. Reads whose ALIGNED length
#: falls below this fraction of the read are reported "unmapped: too short".
MATCH_FRACTION = 0.66
#: A position is a dead cycle if this fraction of sampled reads is N there.
DEAD_CYCLE_FRAC = 0.20
POLYG = "G" * 12
#: Report a file as needing trimming when this share of reads would be lost to
#: STAR's match-fraction filter with the adapter left on. Set well above the few
#: percent of adapter dimers every library carries.
TRIM_BENEFIT_PCT = 20.0
#: Poly-G share above which a file is suspected of holding interleaved mates.
#: Every genuine library surveyed here sits at or below 1.1%; SRR19034544 --
#: 202,303,680 mate pairs deposited as one single-end run -- is 65%, because
#: both mates of a 36 bp fragment run ~100 dark cycles past its end. 5% leaves
#: a wide margin over the real libraries without needing a corpus-wide
#: comparison, so a single file can be screened on its own.
POLYG_SUSPECT_PCT = 5.0
#: A file's read length must exceed this multiple of its project-mates' median
#: to count as an outlier. Only LONGER matters: a read shorter than its
#: siblings cannot hide two mates.
#:
#: 2.0, not 1.5, and the margin was measured rather than guessed.
#: SRR19034544 is 3.00x its project-mates (150 bp against 50), but
#: McDonald2024_plant_5GRO legitimately spans 51 and 75 bp -- three species
#: from one GEO series, sequenced differently -- which is 1.47x and leaves 1.5
#: with almost no room. Nothing else in the corpus exceeds 1.02x.
LENGTH_OUTLIER_RATIO = 2.0

COLUMNS = [
    "experiment", "run", "mate", "n_sampled", "read_len",
    "best_adapter", "pct_adapter", "median_insert", "pct_short_untrimmed",
    "dead_cycles", "pct_any_n", "pct_polyg", "project", "interleave_suspect",
    # Carried into the TSV so a consumer can tell a SUSPECTED interleaved deposit
    # from one the pipeline already handles. Without it,
    # M.musculus-GCB_PROcap -- whose interleaving is declared and deinterleaved
    # -- was reported as FAIL:interleave_suspect forever, since the per-file
    # printout noted "(already declared)" but the column did not.
    "declared_interleaved",
]


def survey(path: Path, n_reads: int) -> dict:
    """Read structure of the first `n_reads` reads of a FASTQ."""
    probes = {name: seq[:MIN_ADAPTER_PREFIX] for name, seq in ADAPTERS.items()}
    hits = Counter()
    inserts = {name: [] for name in probes}
    n_at = Counter()
    lengths = Counter()
    n_any = n_polyg = n = 0

    with gzip.open(path, "rt") as fh:
        for i, line in enumerate(fh):
            if i >= n_reads * 4:
                break
            if i % 4 != 1:
                continue
            seq = line.rstrip("\n")
            n += 1
            lengths[len(seq)] += 1
            if "N" in seq:
                n_any += 1
                for j, base in enumerate(seq):
                    if base == "N":
                        n_at[j] += 1
            if POLYG in seq:
                n_polyg += 1
            for name, probe in probes.items():
                pos = seq.find(probe)
                if pos >= 0:
                    hits[name] += 1
                    inserts[name].append(pos)

    if not n:
        return {}
    read_len = lengths.most_common(1)[0][0]
    best = max(probes, key=lambda k: hits[k]) if hits else None
    ins = sorted(inserts[best]) if best else []
    median = ins[len(ins) // 2] if ins else None
    # Of the reads that carry the adapter, how many have an insert too short to
    # survive STAR's match-fraction filter when the adapter is left on?
    threshold = MATCH_FRACTION * read_len
    short = sum(1 for x in ins if x < threshold)
    dead = [j + 1 for j, c in sorted(n_at.items()) if c >= DEAD_CYCLE_FRAC * n]
    return {
        "n_sampled": n,
        "read_len": read_len,
        "best_adapter": best or "none",
        "pct_adapter": 100.0 * hits[best] / n if best else 0.0,
        "median_insert": median,
        "pct_short_untrimmed": 100.0 * short / n if best else 0.0,
        "dead_cycles": ",".join(str(d) for d in dead) or "-",
        "pct_any_n": 100.0 * n_any / n,
        "pct_polyg": 100.0 * n_polyg / n,
    }


def interleave_screen(rows: list[dict]) -> None:
    """Flag files that may hold two interleaved mates inside a "single-end" run.

    Sets `interleave_suspect` in place: a comma-joined list of the tells that
    fired, or "" for none.

    SRR19034544 is why this exists. It is 202,303,680 mate pairs deposited as
    one single-end run, and ENA and SRA both report SINGLE, so nothing in the
    metadata could catch it. The pipeline counted the 5' end of BOTH mates, and
    R2's 5' end is the RNA 3' END on the opposite strand -- so ~half the track
    became strand-flipped 3'-end signal, and the experiment modelled 3' ends
    until the orientation metaplot showed antisense peaking just downstream of
    the TSS instead of divergent upstream.

    Two tells, both measured on that run rather than reasoned about:

    * `polyg` -- both mates of a fragment shorter than the read run out of
      template and sequence ~100 dark cycles as G. That run is 65% poly-G; no
      other file surveyed here exceeds 1.1%.
    * `long_for_project` -- interleaving needs a read long enough to hold two
      mates, so the offending run tends to be sequenced longer than its
      project-mates (150 bp against 50). Compared within `project_key` because
      read length is a property of the sequencing submission, not the species.

    NEITHER IS PROOF, and this is a screen, not a verdict. A genuinely long
    single-end run in a project of short ones trips the second on its own, and
    a real G-rich library trips the first. The confirming test is read names --
    interleaved mates share the instrument name, so:

        zcat file.fastq.gz | awk 'NR%4==1' | cut -d" " -f1 \
          | sed 's:/[12]$::' | sort | uniq -c | awk '{print $1}' \
          | sort | uniq -c

    should report every name once. Two occurrences of every name means
    interleaved, and the two records will share flowcell:lane:tile:x:y.

    Paired deposits are skipped: their mates are in separate files, which is
    the case the pipeline already handles.

    LIMITATION: `long_for_project` can only fire when a project-mate is in the
    SAME invocation. The Snakemake rule runs this per experiment (`-e {exp}`), so
    a project spanning two experiments never has both in `rows` -- which is
    exactly M.musculus-GCB_PROcap, whose only project-mate is priB. Under the DAG
    that run is caught by `polyg` alone (65% against <=1.1%), which is why the
    tell was worth having two of. For the full screen, run the standalone sweep:
    `python src/qc/read_structure_qc.py --tsv qc/reads/read_structure.tsv`.
    """
    import statistics
    by_project = {}
    for r in rows:
        if r.get("n_sampled") and isinstance(r.get("read_len"), int):
            by_project.setdefault(r.get("project") or "", []).append(r)
    for r in rows:
        r["interleave_suspect"] = ""
    for r in rows:
        if not r.get("n_sampled") or r.get("mate") != "R1":
            continue
        if r.get("paired"):
            continue
        tells = []
        if isinstance(r.get("pct_polyg"), float) and r["pct_polyg"] >= POLYG_SUSPECT_PCT:
            tells.append("polyg")
        peers = [q["read_len"] for q in by_project.get(r.get("project") or "", [])
                 if q["run"] != r["run"]]
        if peers and isinstance(r.get("read_len"), int):
            if r["read_len"] >= LENGTH_OUTLIER_RATIO * statistics.median(peers):
                tells.append("long_for_project")
        r["interleave_suspect"] = ",".join(tells)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-e", "--experiments", nargs="*", default=None)
    ap.add_argument("--fastq-dir", type=Path, default=None)
    ap.add_argument("-n", "--n-reads", type=int, default=200000,
                    help="reads to sample per file (default: %(default)s)")
    ap.add_argument("--tsv", type=Path, default=None, help="also write a TSV")
    args = ap.parse_args()

    import yaml
    with open(REPO_ROOT / "config" / "experiment_config.yaml") as f:
        exps = yaml.safe_load(f)["experiments"]
    fastq_dir = args.fastq_dir or Path(
        os.environ.get("PROCAP_FASTQ_DIR", REPO_ROOT / "data" / "fastq"))

    selected = args.experiments or list(exps)
    unknown = [e for e in selected if e not in exps]
    if unknown:
        sys.exit(f"Error: unknown experiment(s): {unknown}")

    rows = []
    for exp_id in selected:
        raw = exps[exp_id]["raw"]
        paired = str(raw.get("library_layout", "")).upper().startswith("PAIRED")
        for run in (raw.get("runs") or []):
            mates = ([f"{run}_1.fastq.gz", f"{run}_2.fastq.gz"] if paired
                     else [f"{run}.fastq.gz"])
            for mate_i, name in enumerate(mates, start=1):
                path = fastq_dir / name
                # `project` groups the read-length comparison; `paired` and
                # `declared_interleaved` keep the screen from re-flagging cases
                # the pipeline already handles.
                base = {"experiment": exp_id, "run": run, "mate": f"R{mate_i}",
                        "project": exps[exp_id].get("project_key", ""),
                        "paired": paired,
                        "declared_interleaved": "yes" if raw.get("interleaved") else "no"}
                if not path.exists():
                    rows.append({**base, "n_sampled": 0, "read_len": "",
                                 "best_adapter": "missing", "pct_adapter": "",
                                 "median_insert": "", "pct_short_untrimmed": "",
                                 "dead_cycles": "", "pct_any_n": "",
                                 "pct_polyg": ""})
                    continue
                rows.append({**base, **survey(path, args.n_reads)})

    interleave_screen(rows)

    hdr = f"{'run':<13}{'mate':<5}{'len':>5}  {'adapter':<18}{'%adapt':>7}" \
          f"{'insert':>7}{'%short':>8}{'deadCyc':>9}{'%N':>6}{'%polyG':>8}  experiment"
    print(hdr)
    def fmt(v, spec):
        return format(v, spec) if isinstance(v, (int, float)) else str(v)
    for r in rows:
        flag = "  <-- TRIM" if (isinstance(r.get("pct_short_untrimmed"), float)
                                and r["pct_short_untrimmed"] >= TRIM_BENEFIT_PCT) else ""
        if r.get("interleave_suspect"):
            flag += ("  <-- INTERLEAVED? " + r["interleave_suspect"]
                     + (" (already declared)" if (str(r.get("declared_interleaved", "")).lower() == "yes") else ""))
        print(f"{r['run']:<13}{r['mate']:<5}{fmt(r.get('read_len',''),'>5')}  "
              f"{str(r.get('best_adapter','')):<18}"
              f"{fmt(r.get('pct_adapter',''),'>7.1f')}"
              f"{fmt(r.get('median_insert') if r.get('median_insert') is not None else '-','>7')}"
              f"{fmt(r.get('pct_short_untrimmed',''),'>8.1f')}"
              f"{str(r.get('dead_cycles','')):>9}"
              f"{fmt(r.get('pct_any_n',''),'>6.1f')}"
              f"{fmt(r.get('pct_polyg',''),'>8.2f')}  {r['experiment']}{flag}")

    present = [r for r in rows if r.get("n_sampled")]
    if present:
        # Flag on pct_short_untrimmed, not pct_adapter: a few percent of adapter
        # DIMERS (insert 0) is normal in any library and costs nothing to leave.
        # What matters is the share of reads whose insert is too short to survive
        # STAR untrimmed, because that is exactly what trimming would recover.
        adapted = [r for r in present
                   if r["pct_short_untrimmed"] >= TRIM_BENEFIT_PCT]
        print(f"\n{len(present)} file(s) surveyed; {len(adapted)} where trimming "
              f"would recover >={TRIM_BENEFIT_PCT:.0f}% of reads")
        kinds = Counter(r["best_adapter"] for r in adapted)
        for k, c in kinds.most_common():
            print(f"    {c:>3}  {k}")
        suspect = [r for r in present if r.get("interleave_suspect")]
        if suspect:
            print(f"    {len(suspect)} file(s) suspected of interleaved mates "
                  "-- confirm with duplicate read names, see interleave_screen():")
            for r in suspect:
                seen = " (already declared interleaved)" if (str(r.get("declared_interleaved", "")).lower() == "yes") else ""
                print(f"         {r['run']}  {r['interleave_suspect']}  "
                      f"len={r['read_len']} polyG={r['pct_polyg']:.2f}%  "
                      f"{r['experiment']}{seen}")
        dead = [r for r in present if r["dead_cycles"] not in ("-", "")]
        if dead:
            print(f"    {len(dead)} file(s) with a dead cycle: "
                  + ", ".join(f"{r['run']}@{r['dead_cycles']}" for r in dead[:8])
                  + (" ..." if len(dead) > 8 else ""))

    if args.tsv:
        import csv
        args.tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.tsv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t",
                               lineterminator="\n", extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.tsv}")


if __name__ == "__main__":
    main()
