#!/usr/bin/env python3
"""Resolve archive run accessions for the non-human capped run-on manifest.

The planning manifest deliberately leaves `run_accession` blank wherever the
sample -> experiment -> run crosswalk was not independently verified (42 of 44
rows), and its curation rule forbids inferring run IDs from accession
adjacency. This script resolves them from archive metadata instead: for every
study in the manifest it pulls the full ENA read_run table, then matches each
manifest row by SRA experiment accession when one is known, otherwise by GEO
sample accession found in the archive's own sample_alias/sample_title fields.

Rows that cannot be matched that way are reported as unresolved rather than
guessed.

Output columns are the manifest's identifying fields plus resolved
run_accession, library_layout, read_count, base_count, fastq_bytes, fastq_ftp,
fastq_md5, and the
match_basis that produced them.

Usage:
    python src/data_preprocessing/resolve_runs.py
    python src/data_preprocessing/resolve_runs.py --project Kim2018_mm_ESC_BMDM
    python src/data_preprocessing/resolve_runs.py --include-only
    python src/data_preprocessing/resolve_runs.py -o planning/runs.tsv
"""

import argparse
import csv
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "planning" / "manifest_samples.tsv"
DEFAULT_OUTPUT = REPO_ROOT / "planning" / "manifest_runs_resolved.tsv"

ENA_PORTAL = "https://www.ebi.ac.uk/ena/portal/api/filereport"
ENA_FIELDS = [
    "run_accession",
    "experiment_accession",
    "sample_accession",
    "sample_alias",
    "sample_title",
    "experiment_title",
    "library_layout",
    "library_strategy",
    "library_selection",
    "read_count",
    "base_count",
    "fastq_bytes",
    "fastq_ftp",
    "fastq_md5",
    "scientific_name",
]

OUT_FIELDS = [
    "project_key",
    "study_accession",
    "sample_accession",
    "sample_name",
    "sra_experiment",
    "organism",
    "assay_label",
    "target_use",
    "default_include",
    "run_accession",
    "library_layout",
    "read_count",
    # base_count and fastq_bytes come along so library depth, mean read length
    # and download size are all answerable from this file alone -- they are what
    # the depth column of config/datasets.tsv is built from, and what the
    # exclude/merge decisions in CLAUDE.md are argued from. Without them every
    # such question meant re-querying ENA ad hoc.
    "base_count",
    "fastq_bytes",
    "fastq_ftp",
    "fastq_md5",
    "match_basis",
]


def ena_read_run(accession: str, retries: int = 3, pause: float = 0.34) -> list[dict]:
    """Fetch the ENA read_run table for a study/experiment accession."""
    query = urllib.parse.urlencode(
        {
            "accession": accession,
            "result": "read_run",
            "fields": ",".join(ENA_FIELDS),
            "format": "tsv",
            "limit": "0",
        }
    )
    url = f"{ENA_PORTAL}?{query}"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                text = resp.read().decode("utf-8")
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            if attempt == retries:
                print(f"  ! ENA query failed for {accession}: {exc}", file=sys.stderr)
                return []
            time.sleep(2 * attempt)
    time.sleep(pause)  # be polite to the ENA portal

    lines = text.strip().splitlines()
    if len(lines) < 2:
        return []
    reader = csv.DictReader(lines, delimiter="\t")
    return [row for row in reader if row.get("run_accession")]


def study_ids(row: dict) -> list[str]:
    """Archive identifiers worth querying for a manifest row, best first."""
    return [v for v in (row["bioproject"], row["sra_study"]) if v]


def normalize_name(value: str) -> str:
    """Collapse library-name spelling differences for a last-resort match.

    Submitters spell the same library differently in GEO and ENA (e.g. GEO
    "CAP_SPT5_EtOH_1h_rep1" vs ENA sample_alias "CAP_SPT5_EtOH_1h_r1"), so
    compare on lowercased alphanumerics with "rep" folded to "r".
    """
    value = re.sub(r"[^a-z0-9]", "", (value or "").lower())
    return re.sub(r"rep(\d)", r"r\1", value)


def index_runs(runs: list[dict]) -> tuple[dict, dict, dict]:
    """Index a study's runs by experiment accession, GEO sample ID, and name."""
    by_experiment = defaultdict(list)
    by_gsm = defaultdict(list)
    by_name = defaultdict(list)
    for run in runs:
        if run.get("experiment_accession"):
            by_experiment[run["experiment_accession"]].append(run)
        for field in ("sample_alias", "sample_title", "experiment_title"):
            value = (run.get(field) or "").strip()
            if not value:
                continue
            # GEO IDs are often embedded mid-string, e.g.
            # "Illumina HiSeq 2500 sequencing: GSM6090082: priBd3_B18hi_cap2 ..."
            found = re.findall(r"\b(GSM\d+)\b", value)
            for gsm in found:
                if run["run_accession"] not in {
                        h["run_accession"] for h in by_gsm[gsm]}:
                    by_gsm[gsm].append(run)
            if not found:
                # Index the name as given AND with a trailing parenthetical
                # stripped. Archives that mirror another archive often append the
                # source accession to the title -- DDBJ's copy of CNCB
                # PRJCA024429 reports sample_title "AA_GROcap_Rep1 (SAMC4531003)"
                # while the manifest curates the library name alone, and the
                # whole-string match then fails. Only ever adds keys, so an
                # ambiguity it creates is caught by the existing bijectivity
                # check rather than silently mis-assigning.
                for candidate in (value, re.sub(r"\s*\([^)]*\)\s*$", "", value)):
                    key = normalize_name(candidate)
                    if not key:
                        continue
                    if run["run_accession"] not in {
                            h["run_accession"] for h in by_name[key]}:
                        by_name[key].append(run)
    return by_experiment, by_gsm, by_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--project", nargs="+", default=None, help="restrict to these project_key values"
    )
    parser.add_argument(
        "--include-only",
        action="store_true",
        help="only resolve rows with default_include == include",
    )
    args = parser.parse_args()

    with args.manifest.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    if args.project:
        rows = [r for r in rows if r["project_key"] in set(args.project)]
    if args.include_only:
        rows = [r for r in rows if r["default_include"] == "include"]
    if not rows:
        print("No manifest rows selected.", file=sys.stderr)
        sys.exit(1)

    # One ENA query per study, not per sample.
    studies: dict[str, list[str]] = {}
    for row in rows:
        studies.setdefault(row["project_key"], study_ids(row))

    cache: dict[str, tuple[dict, dict, dict]] = {}
    for project, ids in studies.items():
        if not ids:
            print(f"{project}: no bioproject/sra_study in manifest; skipping query")
            cache[project] = ({}, {}, {})
            continue
        for accession in ids:
            runs = ena_read_run(accession)
            if runs:
                print(f"{project}: {accession} -> {len(runs)} runs")
                cache[project] = index_runs(runs)
                break
        else:
            print(f"{project}: no runs returned for {ids}")
            cache[project] = ({}, {}, {})

    resolved = []
    n_resolved = n_manifest = n_unresolved = 0
    for row in rows:
        by_experiment, by_gsm, by_name = cache[row["project_key"]]
        out = {k: row.get(k, "") for k in OUT_FIELDS if k in row}

        if row["run_accession"]:
            basis = "manifest (already resolved)"
            n_manifest += 1
            # Still look the run up so fastq_ftp/md5/layout get filled in --
            # otherwise the fetcher has a run ID but no file to download.
            by_run = {h["run_accession"]: h
                      for hits_ in (by_experiment, by_gsm, by_name)
                      for group in hits_.values() for h in group}
            hits = [by_run[row["run_accession"]]] if row["run_accession"] in by_run else []
            out["run_accession"] = row["run_accession"]
        elif row["sra_experiment"] and by_experiment.get(row["sra_experiment"]):
            hits = by_experiment[row["sra_experiment"]]
            basis = "sra_experiment"
        elif by_gsm.get(row["sample_accession"]):
            hits = by_gsm[row["sample_accession"]]
            basis = "geo sample_alias"
        elif by_name.get(normalize_name(row["sample_name"])):
            hits = by_name[normalize_name(row["sample_name"])]
            # A name match is only weak if it is AMBIGUOUS. Report that, rather
            # than labelling every name match "verify" and leaving a permanent
            # to-do: the Spt5 aliases are just this sample_name with "rep"
            # abbreviated to "r", mapping one-to-one across all six runs, which
            # is not a weak match in substance.
            key = normalize_name(row["sample_name"])
            n_samples_here = sum(
                1 for r2 in rows
                if r2["project_key"] == row["project_key"]
                and normalize_name(r2["sample_name"]) == key
            )
            ambiguous = n_samples_here > 1 or len(hits) > 1
            basis = ("sample_name ~ sample_alias (AMBIGUOUS -- verify)" if ambiguous
                     else "sample_name ~ sample_alias (unique)")
        else:
            hits, basis = [], "UNRESOLVED"
            n_unresolved += 1

        if hits:
            seen, unique_hits = set(), []
            for h in hits:
                if h["run_accession"] not in seen:
                    seen.add(h["run_accession"])
                    unique_hits.append(h)
            # ENA returns runs in no particular order, so without this the
            # accession lists reshuffle on every re-run and experiment_config.yaml
            # and datasets.tsv churn. Sort the hits themselves, not just the
            # joined string, so read_count below stays aligned with run_accession.
            hits = sorted(unique_hits, key=lambda h: h["run_accession"])
            if not row["run_accession"]:
                out["run_accession"] = ";".join(h["run_accession"] for h in hits)
            out["library_layout"] = ";".join(
                sorted({h.get("library_layout", "") for h in hits})
            )
            out["read_count"] = ";".join(h.get("read_count", "") for h in hits)
            out["base_count"] = ";".join(h.get("base_count", "") for h in hits)
            out["fastq_bytes"] = ";".join(h.get("fastq_bytes", "") for h in hits)
            out["fastq_ftp"] = ";".join(h.get("fastq_ftp", "") for h in hits)
            out["fastq_md5"] = ";".join(h.get("fastq_md5", "") for h in hits)
            n_resolved += 1
        out["match_basis"] = basis
        resolved.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        # lineterminator="\n": csv defaults to CRLF, which .gitattributes then
        # normalizes on commit -- leaving the working copy permanently
        # "modified" after every regeneration.
        writer = csv.DictWriter(
            f, fieldnames=OUT_FIELDS, delimiter="\t", extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for out in resolved:
            writer.writerow({k: out.get(k, "") for k in OUT_FIELDS})

    print(
        f"\n{len(resolved)} rows: {n_resolved} newly resolved from archive metadata, "
        f"{n_manifest} already in manifest, {n_unresolved} unresolved"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
