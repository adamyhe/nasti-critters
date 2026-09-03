#!/usr/bin/env python3
"""Bulk-download raw FASTQs for every experiment we intend to re-map.

Fetching is separated from mapping on purpose. A uniform re-map needs every
FASTQ eventually, so pulling them in one resumable, md5-verified pass beats
downloading inside each mapping job: transfers can run on a login/transfer node
instead of burning GPU allocation, a failed mapping run never re-downloads, and
integrity is checked once up front.

Only runs referenced by config/experiment_config.yaml are fetched. Downloading
whole studies instead would be several times larger than the ~202 GiB here, because several
deposits bundle unrelated assays and other species (PRJNA834081 is 8/11 human
Ramos libraries; SRP131922 is 294 runs of which one is wanted).

FASTQ URLs and md5 checksums come from planning/manifest_runs_resolved.tsv,
which is produced by resolve_runs.py from ENA metadata.

Storage: ~202 GiB total. On a cluster, point --fastq-dir at scratch (or set
PROCAP_FASTQ_DIR) rather than filling the repo checkout:

    ln -s /scratch/users/$USER/procap_fastq data/fastq

Usage:
    python src/data_preprocessing/fetch_fastqs.py --dry-run
    python src/data_preprocessing/fetch_fastqs.py --tier include -j 4
    python src/data_preprocessing/fetch_fastqs.py -e S.cerevisiae-Ino80ctl_PROcap
    python src/data_preprocessing/fetch_fastqs.py --verify-only
"""

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_PATH = REPO_ROOT / "config" / "experiment_config.yaml"
RESOLVED = REPO_ROOT / "planning" / "manifest_runs_resolved.tsv"


def default_fastq_dir() -> Path:
    env = os.environ.get("PROCAP_FASTQ_DIR")
    return Path(env) if env else REPO_ROOT / "data" / "fastq"


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_run_files() -> dict[str, list[tuple[str, str]]]:
    """run_accession -> [(url, md5), ...] from the resolved manifest."""
    files: dict[str, list[tuple[str, str]]] = {}
    with RESOLVED.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            runs = [r for r in (row["run_accession"] or "").split(";") if r]
            ftps = [u for u in (row["fastq_ftp"] or "").split(";") if u]
            md5s = [m for m in (row["fastq_md5"] or "").split(";") if m]
            if not runs:
                continue
            # ENA returns one ftp/md5 entry per file; a paired run has two.
            # Group them back onto their run by the run ID in the path.
            for run in runs:
                pairs = [
                    (u, md5s[i] if i < len(md5s) else "")
                    for i, u in enumerate(ftps)
                    if f"/{run}/" in u or Path(u).name.startswith(run)
                ]
                if pairs:
                    files.setdefault(run, [])
                    for p in pairs:
                        if p not in files[run]:
                            files[run].append(p)
    return files


def fetch_one(url: str, md5: str, dest: Path, dry_run: bool, force: bool,
              verify_only: bool) -> tuple[str, str]:
    """Download one FASTQ and verify md5. Returns (status, detail).

    Transfers stage into `<fastq_dir>/.incoming/` and are only moved to the
    final path after the md5 matches. A file at the final path therefore always
    means "complete and verified" -- a truncated transfer can never be mistaken
    for a finished one, which matters because these are multi-hundred-MB files
    and an interrupted run is the normal case, not the exception.
    """
    name = dest.name
    if dest.exists() and not force:
        if not md5:
            return ("kept", f"{name} (no md5 published; not verified)")
        actual = md5 if dry_run else md5sum(dest)
        if actual == md5:
            return ("ok", name)
        if verify_only:
            return ("CORRUPT", f"{name} md5 {actual} != {md5}")
        print(f"  {name}: md5 mismatch on disk, re-fetching")
    if verify_only:
        return ("missing", name)
    if dry_run:
        return ("would fetch", f"{name} <- {url}")

    staging = dest.parent / ".incoming"
    staging.mkdir(parents=True, exist_ok=True)
    part = staging / name
    # `wget -c -P <dir>` has well-defined resume semantics (unlike -c with -O),
    # and the ENA basename is already the name we want.
    result = subprocess.run(
        # -nv, not -q: quiet suppresses the error text too, which turned every
        # failure into a bare "FAILED: <name>:" with no reason.
        # ENA throttles by refusing connections, so retry rather than give up.
        ["wget", "-nv", "-c", "--tries=3", "--waitretry=10",
         "--retry-connrefused", "--timeout=60", "-P", str(staging),
         url if url.startswith("http") else f"https://{url}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not part.exists():
        why = result.stderr.strip() or result.stdout.strip()
        if not why:
            why = f"wget exit {result.returncode}, no output"
        return ("FAILED", f"{name}: {why[-300:]}")
    if md5:
        actual = md5sum(part)
        if actual != md5:
            return ("CORRUPT",
                    f"{name} md5 {actual} != {md5} (left in {staging}/ to resume)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(part, dest)   # atomic within the same filesystem
    return ("fetched", name)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-e", "--experiments", nargs="+", action="extend",
                        default=[])
    parser.add_argument("--species", nargs="+", action="extend", default=[])
    # action="extend" so `--tier include --tier conditional` accumulates.
    # With plain nargs="+" the second flag REPLACED the first, silently
    # selecting only the last tier named.
    parser.add_argument("--tier", nargs="+", action="extend", default=[],
                        choices=["include", "conditional", "exclude"])
    parser.add_argument("--fastq-dir", type=Path, default=None,
                        help="destination (default: $PROCAP_FASTQ_DIR or data/fastq)")
    parser.add_argument("-j", "--jobs", type=int, default=4,
                        help="parallel downloads (default: %(default)s)")
    parser.add_argument("--verify-only", action="store_true",
                        help="check md5 of what is already on disk; download nothing")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with EXPERIMENTS_PATH.open() as f:
        config = yaml.safe_load(f)["experiments"]

    selected = list(config)
    if args.experiments:
        unknown = [e for e in args.experiments if e not in config]
        if unknown:
            print(f"Error: unknown experiment(s): {', '.join(unknown)}",
                  file=sys.stderr)
            sys.exit(1)
        selected = args.experiments
    if args.species:
        selected = [e for e in selected if config[e]["species"] in set(args.species)]
    if args.tier:
        selected = [e for e in selected if config[e]["tier"] in set(args.tier)]

    run_files = load_run_files()
    fastq_dir = args.fastq_dir or default_fastq_dir()

    # run -> the experiments that need it (runs can be shared across experiments)
    wanted: dict[str, list[str]] = {}
    no_runs = []
    for exp_id in selected:
        runs = config[exp_id]["raw"]["runs"]
        if not runs:
            no_runs.append(exp_id)
            continue
        for run in runs:
            wanted.setdefault(run, []).append(exp_id)

    missing_meta = sorted(r for r in wanted if r not in run_files)
    jobs = [
        (url, md5, fastq_dir / Path(url).name)
        for run in sorted(wanted)
        for url, md5 in run_files.get(run, [])
    ]

    print(f"experiments selected : {len(selected)}")
    print(f"runs to fetch        : {len(wanted)}  ({len(jobs)} FASTQ files)")
    print(f"destination          : {fastq_dir}")
    if args.jobs > 8 and not (args.dry_run or args.verify_only):
        print(f"WARNING: -j {args.jobs} opens that many concurrent connections to "
              "ENA, which throttles by refusing them; transfers then fail for "
              "reasons unrelated to the data. 4-8 is the useful range (the "
              "Snakemake path caps downloads at 4). Retries are on, but if you "
              "see FAILED lines, lower -j before investigating anything else.")
    if no_runs:
        print(f"no resolved runs     : {', '.join(no_runs)}")
    if missing_meta:
        print(f"no FASTQ URL in ENA  : {', '.join(missing_meta)}")
    print()

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(fetch_one, url, md5, dest, args.dry_run, args.force,
                        args.verify_only): dest
            for url, md5, dest in jobs
        }
        for fut in as_completed(futures):
            status, detail = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status in ("FAILED", "CORRUPT", "missing", "would fetch", "kept"):
                print(f"  {status}: {detail}")

    print("\n" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if counts.get("FAILED") or counts.get("CORRUPT"):
        print("\nSome transfers failed integrity checks; rerun to resume.",
              file=sys.stderr)
        sys.exit(1)
    if not args.dry_run and not args.verify_only:
        print(f"\nnext: python src/data_preprocessing/run_procap_pipeline.py "
              f"--fastq-dir {fastq_dir} --tier include")


if __name__ == "__main__":
    main()
