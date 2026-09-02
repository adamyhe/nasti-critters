#!/usr/bin/env python3
"""How much of a library is rRNA or organellar, measured from the raw reads.

Exists because `pct_unique` is NOT a quality metric for every species, and was
being read as one. Where the rDNA array sits in the assembly in two or more
near-identical copies, every rRNA read becomes a multimapper, gets MAPQ ~3, and
is dropped by the ENCODE `-q 255` filter. That is correct behaviour -- a Pol II
initiation model has no use for Pol I reads -- but it caps `pct_unique` at a
level that has nothing to do with library quality:

    library                 rRNA + organellar    ceiling   observed pct_unique
    S.cerevisiae Booth            70.4%           29.6%          16.3%
    S.pombe Booth                 46.0%           53.9%          42.0%
    C.reinhardtii                 67.4%           32.6%          10.8%
    P.patens                      47.4%           52.6%          12.3%

So `pct_unique` alone flagged Booth and the two plants as failures when most of
what it was measuring was rRNA being correctly discarded. `pct_unique_adj`
(unique / non-rRNA input) is the number that says something about the library.

METHOD, and two things it gets right that the obvious version does not:

* rRNA features are read from the Ensembl GFF3 as type `rRNA`, NOT `rRNA_gene`.
  Ensembl types these under `ncRNA_gene`, so filtering on `rRNA_gene` silently
  returns nothing -- it returned nothing for S. pombe chromosome III, which made
  a real rDNA array look absent.
* The 35S components (SSU, 5.8S, LSU) are MERGED per contig with a gap
  tolerance, so the span covers the whole rDNA unit INCLUDING ITS1 and ITS2. A
  nascent assay reads the precursor, not the mature rRNA, so scoring against
  mature gene bodies alone undercounts. The commonest read in the
  C. reinhardtii library spans the 5.8S/ITS2 junction and would be missed.

Organellar reads are scored against the decoy sequences in `data/decoy/`, since
for C. reinhardtii, P. patens and C. griseus the reference omits the organelle
entirely and those reads have nowhere to map at all.

Reads are classified by exact k-mer membership, not alignment: for each
adapter-trimmed insert, the first k-mer found in any class index wins. That is a
proxy for STAR and slightly over-calls, because an rDNA k-mer can occur
elsewhere -- but it agrees with an independent coordinate-based count on
S. cerevisiae (70.4% here at k=20 including ITS, 63.6% at k=24 excluding it) and
the resulting ceilings bracket the observed `pct_unique` for all four libraries.

Only contigs carrying an rRNA feature are held in memory, so this runs on a
473 Mb genome without loading it whole.

Usage:
    python src/qc/rrna_content.py -e S.cerevisiae_PROcap -o qc/rrna/exp.tsv
    python src/qc/rrna_content.py --all --tsv qc/rrna/rrna_content.tsv
"""

import argparse
import csv
import gzip
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

#: k-mer length. 20 rather than 24 so an insert just above fastp's
#: `--length_required 18` is still scorable; chance matches are ~1e-3 even on a
#: 473 Mb genome.
K = 20
#: Gap tolerated when merging 35S components into one rDNA unit. The unit is
#: 8-10 kb across these species, so 5 kb joins SSU/5.8S/LSU of one copy without
#: bridging two.
MERGE_GAP = 5000
#: Reads sampled per FASTQ, and scored per experiment.
N_SAMPLE = 60000
N_SCORE = 5000
#: Adapter prefixes to trim before scoring. Matches the set in
#: config/procap_pipeline.yaml, including the 5-truncated TruSeq form.
ADAPTER_PREFIXES = [
    "TGGAATTCTCGGG",        # smallRNA RA3
    "AGATCGGAAGAGC",        # TruSeq universal
    "GGAAGAGCACACGTCTGAAC"[:13],
]

COLUMNS = [
    "experiment", "species", "n_scored", "rrna_indexed",
    "pct_rrna_35s", "pct_rrna_5s", "pct_organellar", "pct_rrna_total",
]


def rc(seq: str) -> str:
    return seq[::-1].translate(str.maketrans("ACGTN", "TGCAN"))


def rrna_intervals(gff3: Path) -> tuple[dict, dict]:
    """(35S intervals, 5S intervals) per contig, from GFF3 `rRNA` features."""
    n35, n5 = defaultdict(list), defaultdict(list)
    with gzip.open(gff3, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "rRNA":
                continue
            name = ""
            for kv in p[8].split(";"):
                if kv.startswith("Name="):
                    name = kv[5:]
                elif kv.startswith("ID=") and not name:
                    name = kv[3:]
            span = (int(p[3]) - 1, int(p[4]))
            up = name.upper()
            target = n5 if ("5S_RRNA" in up or "RDN5-" in up) else n35
            target[p[0]].append(span)
    return dict(n35), dict(n5)


def parse_region(spec: str) -> tuple[str, int | None, int | None]:
    """`contig` or `contig:start-end` (1-based inclusive) -> (contig, s0, e)."""
    spec = str(spec)
    if ":" not in spec:
        return spec, None, None
    contig, coords = spec.rsplit(":", 1)
    start, end = coords.split("-")
    return contig, int(start) - 1, int(end)


def merge(spans: list, gap: int) -> list:
    out = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def contig_subset(fasta: Path, wanted: set) -> dict:
    """Only the requested contigs, so a 473 Mb genome need not be held whole."""
    seqs, name, buf, keep = {}, None, [], False
    opener = gzip.open if str(fasta).endswith(".gz") else open
    with opener(fasta, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if keep:
                    seqs[name] = "".join(buf)
                name = line[1:].split()[0]
                keep, buf = name in wanted, []
            elif keep:
                buf.append(line.strip().upper())
    if keep:
        seqs[name] = "".join(buf)
    return seqs


def kmers(seq: str) -> set:
    out = set()
    for strand in (seq, rc(seq)):
        for i in range(len(strand) - K + 1):
            out.add(hash(strand[i:i + K]))
    return out


def build_index(species: str, genome: dict) -> dict:
    """Class name -> set of k-mer hashes."""
    gff3 = REPO_ROOT / "data" / "annotation" / f"{species}.gff3.gz"
    if not gff3.exists():
        gff3 = REPO_ROOT / "data" / "annotation" / f"{species}.gtf.gz"
    fasta = REPO_ROOT / genome["fasta"]
    idx = {}
    # Organelle contigs already in the assembly, so pct_organellar is measured
    # for the 7 of 10 species that have them. Scoring only against data/decoy/
    # would report 0% for all of those, which is a false negative rather than a
    # missing measurement.
    in_asm = [str(c) for c in (genome.get("organelle_contigs") or [])]
    regions = [parse_region(r) for r in (genome.get("rdna_regions") or [])]
    has_gff3 = gff3.exists() and ".gff3" in gff3.name
    n35, n5 = rrna_intervals(gff3) if has_gff3 else ({}, {})
    # One FASTA pass for everything needed: rRNA-bearing contigs, in-assembly
    # organelles, and any explicitly configured rDNA region.
    wanted = set(n35) | set(n5) | set(in_asm) | {c for c, _, _ in regions}
    seqs = contig_subset(fasta, wanted)
    # FAIL LOUDLY on a configured region whose contig is not in the FASTA. This
    # is a config typo, not a data condition, and the failure mode it prevents is
    # the one this repo keeps hitting: CLAUDE.md's rDNA table writes C. elegans'
    # array as `I:15062083-15071033`, but ce11 is chr-prefixed, so the bare name
    # would have indexed nothing and reported 0% rRNA as if measured.
    missing = [c for c, _, _ in regions if c not in seqs]
    if missing:
        have = sorted(seqs)[:8]
        raise SystemExit(
            f"{species}: rdna_regions contig(s) {missing} not in {fasta.name}. "
            f"Check chrom_style -- contigs read from the FASTA include {have}. "
            "Fix rdna_regions in config/genomes.yaml."
        )
    for label, table, gap in (("rrna_35s", n35, MERGE_GAP), ("rrna_5s", n5, 0)):
        acc = set()
        for contig, spans in table.items():
            if contig not in seqs:
                continue
            for a, b in merge(spans, gap):
                acc |= kmers(seqs[contig][a:b])
        if acc:
            idx[label] = acc
    # Explicitly configured in-assembly rDNA, indexed as 35S.
    for contig, a, b in regions:
        seq = seqs[contig] if a is None else seqs[contig][a:b]
        idx.setdefault("rrna_35s", set())
        idx["rrna_35s"] |= kmers(seq)

    # An rDNA SINK is itself the rDNA sequence, so index it as 35S. This is the
    # only rRNA source for the three UCSC species: their annotation is refGene
    # GTF, which types everything exon/CDS/transcript and carries no `rRNA`
    # feature at all. M. musculus has BK000964.3 and is therefore measurable;
    # D. melanogaster and C. elegans have their arrays in-assembly and no
    # accession, so they get NO measurement -- reported as n_scored 0 and a
    # blank column, which must not be read as 0% rRNA.
    if genome.get("rdna_accession"):
        fa = REPO_ROOT / "data" / "decoy" / f"{genome['rdna_accession']}.fa"
        if fa.exists():
            idx.setdefault("rrna_35s", set())
            idx["rrna_35s"] |= kmers("".join(
                l.strip().upper() for l in fa.open() if not l.startswith(">")))

    org = set()
    for contig in in_asm:
        if contig in seqs:
            org |= kmers(seqs[contig])
    for acc in (genome.get("organelle_accessions") or []):
        fa = REPO_ROOT / "data" / "decoy" / f"{acc}.fa"
        if fa.exists():
            org |= kmers("".join(
                l.strip().upper() for l in fa.open() if not l.startswith(">")))
    if org:
        idx["organellar"] = org
    return idx


def insert_of(read: str) -> str:
    hits = [p for p in (read.find(a) for a in ADAPTER_PREFIXES) if p >= 0]
    return read[:min(hits)] if hits else read


def sample_inserts(paths: list, n: int) -> list:
    out = []
    per = max(1, n // max(len(paths), 1))
    for path in paths:
        got = 0
        try:
            with gzip.open(path, "rt") as f:
                for i, line in enumerate(f):
                    if i % 4 == 1:
                        out.append(insert_of(line.strip()))
                        got += 1
                        if got >= per:
                            break
        except Exception:
            # A QC reporter must never fail the DAG: an unreadable or truncated
            # FASTQ contributes whatever reads it yielded and is otherwise
            # skipped. n_scored in the output says how much was actually read.
            continue
    return out


def classify(inserts: list, idx: dict, limit: int) -> tuple[Counter, int]:
    hits = Counter()
    scored = [x for x in inserts if len(x) >= K][:limit]
    order = ["rrna_35s", "rrna_5s", "organellar"]
    for x in scored:
        got = None
        for o in range(0, len(x) - K + 1):
            h = hash(x[o:o + K])
            for label in order:
                if label in idx and h in idx[label]:
                    got = label
                    break
            if got:
                break
        hits[got or "other"] += 1
    return hits, len(scored)


def collect(exp_id: str, entry: dict, genomes: dict, fastq_dir: Path) -> dict:
    species = entry["species"]
    raw = entry["raw"]
    paired = str(raw.get("library_layout", "")).upper().startswith("PAIRED")
    paths = []
    for run in (raw.get("runs") or []):
        # Read 1 only: read 2 samples the same fragments from the other end.
        paths.append(fastq_dir / (f"{run}_1.fastq.gz" if paired
                                  else f"{run}.fastq.gz"))
    paths = [p for p in paths if p.exists()]
    base = {"experiment": exp_id, "species": species}
    if not paths:
        return {**base, "n_scored": 0}
    idx = build_index(species, genomes[species])
    if not idx:
        return {**base, "n_scored": 0}
    hits, n = classify(sample_inserts(paths, N_SAMPLE), idx, N_SCORE)
    if not n:
        return {**base, "n_scored": 0}
    total = sum(hits[k] for k in ("rrna_35s", "rrna_5s", "organellar"))
    # `rrna_indexed` distinguishes "measured 0%" from "rRNA was never in the
    # index". Adding organelle_contigs made n_scored > 0 for the species with no
    # rRNA source, which populated pct_rrna with an organellar-only number and
    # destroyed the blank-means-unmeasured signal this column relies on.
    return {
        **base,
        "n_scored": n,
        "rrna_indexed": "yes" if ("rrna_35s" in idx or "rrna_5s" in idx) else "no",
        "pct_rrna_35s": round(hits["rrna_35s"] / n * 100, 2),
        "pct_rrna_5s": round(hits["rrna_5s"] / n * 100, 2),
        "pct_organellar": round(hits["organellar"] / n * 100, 2),
        "pct_rrna_total": round(total / n * 100, 2),
    }


def write_tsv(rows: list, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t",
                           lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-e", "--experiment")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fastq-dir", type=Path, default=None)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--tsv", type=Path)
    args = ap.parse_args()
    if not (args.experiment or args.all):
        ap.error("one of -e/--experiment or --all is required")

    import yaml
    with open(REPO_ROOT / "config" / "experiment_config.yaml") as f:
        exps = yaml.safe_load(f)["experiments"]
    with open(REPO_ROOT / "config" / "genomes.yaml") as f:
        genomes = yaml.safe_load(f)["species"]
    fastq_dir = args.fastq_dir or Path(
        os.environ.get("PROCAP_FASTQ_DIR", REPO_ROOT / "data" / "fastq"))

    selected = [args.experiment] if args.experiment else list(exps)
    unknown = [e for e in selected if e not in exps]
    if unknown:
        sys.exit(f"Error: unknown experiment(s): {unknown}")

    rows = []
    print(f"{'experiment':<34}{'35S':>8}{'5S':>7}{'organelle':>11}"
          f"{'total':>8}{'n':>8}")
    for exp_id in selected:
        row = collect(exp_id, exps[exp_id], genomes, fastq_dir)
        rows.append(row)
        if row["n_scored"]:
            print(f"{exp_id:<34}{row['pct_rrna_35s']:>8.1f}"
                  f"{row['pct_rrna_5s']:>7.1f}{row['pct_organellar']:>11.1f}"
                  f"{row['pct_rrna_total']:>8.1f}{row['n_scored']:>8}")
        else:
            print(f"{exp_id:<34}{'(no FASTQ or no rRNA annotation)':>42}")

    out = args.output or args.tsv
    if out:
        write_tsv(rows, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
