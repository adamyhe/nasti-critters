#!/usr/bin/env python3
"""Screen raw FASTQs for an undeclared UMI, and check the declared ones.

Why this exists: UMI presence is curated by hand in the manifest
(umi_len/umi_loc) and CANNOT be cross-checked against the archives -- ENA's
library_construction_protocol was queried for all 45 runs then resolved and
mentions a UMI for none of them, including the three Spt5 experiments that
demonstrably have one. The scheme is default-deny, so a library whose UMI was
never noted keeps its PCR duplicates and nothing in the pipeline complains.
This is the empirical check.

How it works: a UMI is a random N-mer at the 5' end of a read, so its base
composition is near-uniform at every position. Genuine PRO-cap 5' ends are not
-- initiation is strongly biased (the Inr, and a pyrimidine/purine preference
at the TSS), and library structure biases the first bases further. So a run of
near-maximum-entropy positions at the read start, followed by a clear drop, is
the signature of a UMI.

Read this as a screen, not a verdict. It flags candidates for a human to check
against the paper; it does not prove absence. Confirmation for a declared UMI
is stronger: the length reported here should match umi_len in the manifest.

Usage:
    python src/data_preprocessing/umi_report.py                 # every experiment
    python src/data_preprocessing/umi_report.py -e S.cerevisiae-Spt5EtOH_PROcap
    python src/data_preprocessing/umi_report.py --tsv report.tsv
"""

import argparse
import gzip
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# A uniform random base is 2 bits. Real sequence rarely sustains this.
MAX_BITS = 2.0
UNIFORM_BITS = 1.95      # per-position entropy that counts as "random"
MIN_RUN = 4              # shortest prefix worth calling a UMI


def entropy(counts: Counter) -> float:
    n = sum(counts[b] for b in "ACGT")
    if n == 0:
        return 0.0
    h = 0.0
    for b in "ACGT":
        p = counts[b] / n
        if p > 0:
            h -= p * math.log2(p)
    return h


#: fastp --umi_loc value -> which mate carries the UMI.
_FASTP_LOC_TO_MATE = {"read1": 1, "read2": 2}


def umi_mate(loc: str | None) -> int:
    """Which mate carries the UMI, from the manifest's prose `umi_loc`.

    Resolved through `steps.dedup.umi_locations` in config/procap_pipeline.yaml
    -- the same table both drivers use for fastp's --umi_loc -- so this report
    and the pipeline cannot disagree about which mate to look at. They did:
    this was hardcoded to mate 2.

    An unmapped prose value raises rather than defaulting, matching the guards
    in workflow/Snakefile and run_procap_pipeline.py. A silent default here is
    what produced two spurious MISMATCH rows per Spt5 run.
    """
    import yaml
    with open(REPO_ROOT / "config" / "procap_pipeline.yaml") as fh:
        table = (yaml.safe_load(fh)["steps"]["dedup"].get("umi_locations") or {})
    if loc not in table:
        raise SystemExit(
            f"umi_loc {loc!r} has no mapping in steps.dedup.umi_locations "
            f"(known: {sorted(table)}). Add it to config/procap_pipeline.yaml."
        )
    fastp_loc = table[loc]
    if fastp_loc not in _FASTP_LOC_TO_MATE:
        raise SystemExit(
            f"umi_loc {loc!r} maps to fastp value {fastp_loc!r}, which is not "
            f"one of {sorted(_FASTP_LOC_TO_MATE)}."
        )
    return _FASTP_LOC_TO_MATE[fastp_loc]


def profile(path: Path, n_reads: int, width: int) -> list[float]:
    """Per-position base entropy over the first `width` bases."""
    cols = [Counter() for _ in range(width)]
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= n_reads * 4:
                break
            if i % 4 != 1:
                continue
            for j, base in enumerate(line.rstrip()[:width]):
                cols[j][base] += 1
    return [entropy(c) for c in cols]


def candidate_len(bits: list[float]) -> int:
    """Length of the leading run of near-uniform positions."""
    n = 0
    for b in bits:
        if b >= UNIFORM_BITS:
            n += 1
        else:
            break
    return n if n >= MIN_RUN else 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-e", "--experiments", nargs="+", action="extend", default=[])
    ap.add_argument("--fastq-dir", type=Path, default=None)
    ap.add_argument("-n", "--n-reads", type=int, default=200_000,
                    help="reads to sample per file (default: %(default)s)")
    ap.add_argument("-w", "--width", type=int, default=20,
                    help="leading bases to profile (default: %(default)s)")
    ap.add_argument("--tsv", type=Path, default=None, help="also write a TSV")
    args = ap.parse_args()

    import yaml
    with open(REPO_ROOT / "config" / "experiment_config.yaml") as f:
        exps = yaml.safe_load(f)["experiments"]
    fastq_dir = (args.fastq_dir or Path(
        __import__("os").environ.get("PROCAP_FASTQ_DIR", REPO_ROOT / "data" / "fastq")))

    selected = args.experiments or list(exps)
    unknown = [e for e in selected if e not in exps]
    if unknown:
        print(f"Error: unknown experiment(s): {unknown}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for exp_id in selected:
        raw = exps[exp_id]["raw"]
        umi = raw.get("umi") or {}
        declared = umi.get("len") or 0
        paired = str(raw.get("library_layout", "")).upper().startswith("PAIRED")
        # Single-end has one mate, so everything lands on R1 regardless.
        expect_mate = 1 if not paired else umi_mate(umi.get("loc"))
        for run in (raw.get("runs") or []):
            mates = ([f"{run}_1.fastq.gz", f"{run}_2.fastq.gz"] if paired
                     else [f"{run}.fastq.gz"])
            for mate_i, name in enumerate(mates, start=1):
                path = fastq_dir / name
                # The UMI sits on one mate only, and WHICH mate comes from the
                # manifest via steps.dedup.umi_locations -- the same mapping
                # both drivers hand to fastp as --umi_loc. This used to be
                # hardcoded as "R2, because a 3' adaptor UMI is on read 2",
                # which is the exact claim the Spt5 investigation overturned:
                # the UMI is at the START OF READ 1 and the manifest records
                # `5' adaptor`. The pipeline was corrected; this report was not,
                # so it checked the wrong mate and printed
                # `MISMATCH (manifest says 10)` on R2 while R1 read
                # `no UMI signature`. Both were artifacts of this line.
                expect = declared if mate_i == expect_mate else 0
                if not path.exists():
                    rows.append((exp_id, run, f"R{mate_i}", expect, None, "", "missing"))
                    continue
                bits = profile(path, args.n_reads, args.width)
                found = candidate_len(bits)
                if found and not expect:
                    verdict = "UNDECLARED UMI?"
                elif expect and found == expect:
                    verdict = "ok (matches manifest)"
                elif expect and found != expect:
                    verdict = f"MISMATCH (manifest says {expect})"
                else:
                    verdict = "no UMI signature"
                rows.append((exp_id, run, f"R{mate_i}", expect, found,
                             " ".join(f"{b:.2f}" for b in bits[:12]), verdict))

    hdr = ("experiment", "run", "mate", "declared", "detected", "entropy[0:12]", "verdict")
    print(f"{hdr[0]:34s} {hdr[1]:12s} {hdr[2]:4s} {hdr[3]:>8s} {hdr[4]:>8s}  {hdr[6]}")
    for r in rows:
        det = "-" if r[4] is None else str(r[4])
        print(f"{r[0]:34s} {r[1]:12s} {r[2]:4s} {r[3]:>8} {det:>8}  {r[6]}")

    flagged = [r for r in rows if r[6].startswith(("UNDECLARED", "MISMATCH"))]
    missing = [r for r in rows if r[6] == "missing"]
    print(f"\n{len(rows)} files: {len(flagged)} flagged, {len(missing)} not on disk")
    if flagged:
        print("Flagged files need a human check against the paper or GEO record; "
              "this screen cannot prove a UMI's absence.", file=sys.stderr)

    if args.tsv:
        args.tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.tsv, "w") as f:
            f.write("\t".join(hdr) + "\n")
            for r in rows:
                f.write("\t".join("" if x is None else str(x) for x in r) + "\n")
        print(f"wrote {args.tsv}")


if __name__ == "__main__":
    main()
