#!/usr/bin/env python3
"""Download the JASPAR CORE MEME databases `modisco report` matches against.

Standalone, and deliberately NOT a Snakemake rule. The database has no effect on
which motifs modisco discovers -- it only names the matches in the report -- so
putting it in the DAG would make a reporting convenience a network dependency of
the pipeline. Same placement argument as src/make_negatives.py, for a different
reason.

WHICH files are needed is derived from `experiments.motif_db_path()`, one per
distinct `jaspar_collection` in config/genomes.yaml, so the filename convention
has exactly one definition. The RELEASE is read out of that filename rather than
being a flag here: the downloaded name has to be the name launch_report.py looks
for, and a --release that disagreed with motif_db_path() would produce files
nothing reads.

Two guards, both earned by failures recorded in CLAUDE.md:

  - Content is checked for MEME's magic line, not just an HTTP 200. bedbase's
    SPA catch-all serves HTML with status 200, so a status code alone is not
    evidence of a real file.
  - Downloads stage through a temporary file and are renamed only after that
    check passes, matching fetch_fastqs.py's `.incoming/` convention: a file at
    the final path always means complete and valid.

Usage:
    python src/bpnet/modisco/fetch_motif_dbs.py --dry-run
    python src/bpnet/modisco/fetch_motif_dbs.py
    python src/bpnet/modisco/fetch_motif_dbs.py --verify-only
    python src/bpnet/modisco/fetch_motif_dbs.py --species C.elegans --force
"""

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from experiments import motif_db_path  # noqa: E402

GENOMES_PATH = REPO_ROOT / "config" / "genomes.yaml"
JASPAR_BASE = "https://jaspar.elixir.no/download/data"

#: First line of a MEME motif file. What separates a real download from an
#: HTML error page served with status 200.
MEME_MAGIC = "MEME version"

#: What motif_db_path() is expected to produce. Parsed rather than rebuilt so
#: the release and collection in the URL cannot drift from the filename.
NAME_RE = re.compile(
    r"^JASPAR(?P<release>\d{4})_CORE_(?P<collection>[a-z]+)"
    r"_non-redundant_pfms_meme\.txt$"
)


def wanted(species: list[str]) -> dict[Path, str]:
    """Destination path -> URL, one entry per distinct database needed.

    Keyed on the resolved path, so the six plant species collapse to one file.
    """
    genomes = yaml.safe_load(GENOMES_PATH.read_text())["species"]
    if species:
        missing = [s for s in species if s not in genomes]
        if missing:
            raise SystemExit(
                f"not in {GENOMES_PATH.name}: {', '.join(missing)}\n"
                f"  known: {', '.join(sorted(genomes))}"
            )
    else:
        species = sorted(genomes)

    out: dict[Path, str] = {}
    for sp in species:
        path = motif_db_path(sp)
        m = NAME_RE.match(path.name)
        if not m:
            raise SystemExit(
                f"cannot build a URL for {path.name}: motif_db_path()'s filename "
                f"format has changed and this script needs updating to match "
                f"(expected JASPAR<release>_CORE_<collection>_non-redundant_pfms_meme.txt)"
            )
        out[path] = (f"{JASPAR_BASE}/{m['release']}/CORE/{path.name}")
    return out


def check(path: Path) -> tuple[bool, str]:
    """Is this a real MEME motif file? Returns (ok, one-line description)."""
    try:
        text = path.read_text()
    except OSError as exc:
        return False, f"unreadable: {exc}"
    if not text.startswith(MEME_MAGIC):
        first = text.lstrip()[:60].replace("\n", " ")
        return False, f"not a MEME file (starts {first!r})"
    n = sum(1 for line in text.splitlines() if line.startswith("MOTIF "))
    if n == 0:
        return False, "MEME header but zero MOTIF records"
    return True, f"{n:,} motifs, {len(text) / 1024:.0f} KiB"


def download(url: str, dest: Path) -> tuple[bool, str]:
    """Fetch `url` to `dest` via a staged temporary file. Returns (ok, note)."""
    tmp = dest.with_name(dest.name + ".incoming")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            tmp.write_bytes(resp.read())
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        return False, f"download failed: {exc}"

    ok, note = check(tmp)
    if not ok:
        # Leave the staged file for inspection rather than the final path, so a
        # bad fetch can never look like a complete one.
        return False, f"{note}; kept at {tmp.name}"
    tmp.replace(dest)
    return True, note


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--species", nargs="+", action="extend", default=[],
                        help="limit to these species' collections "
                             "(default: every species in config/genomes.yaml)")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="override the destination directory; the FILENAMES "
                             "still come from motif_db_path(), since those are "
                             "what launch_report.py looks for")
    parser.add_argument("--verify-only", action="store_true",
                        help="check what is on disk, download nothing")
    parser.add_argument("--force", action="store_true",
                        help="re-download files that are already present and valid")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = wanted(args.species)
    if args.outdir:
        targets = {args.outdir / p.name: url for p, url in targets.items()}

    checked = present = broken = fetched = 0
    for dest in sorted(targets):
        url = targets[dest]
        rel = dest.relative_to(REPO_ROOT) if dest.is_relative_to(REPO_ROOT) else dest
        checked += 1
        exists = dest.exists()
        ok, note = check(dest) if exists else (False, "missing")

        if args.verify_only:
            if ok:
                print(f"  have    {rel}  ({note})")
                present += 1
            else:
                print(f"  {'BAD' if exists else 'MISSING':7} {rel}  ({note})")
                broken += 1
            continue

        if ok and not args.force:
            print(f"  have    {rel}  ({note})")
            present += 1
            continue
        if exists and not ok:
            print(f"  BAD     {rel}  ({note}) -- re-downloading")
        elif exists:
            print(f"  force   {rel}  ({note})")

        if args.dry_run:
            print(f"  would fetch {rel}\n           from {url}")
            fetched += 1
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        ok, note = download(url, dest)
        print(f"  {'got' if ok else 'FAIL':7} {rel}  ({note})")
        if ok:
            fetched += 1
        else:
            broken += 1

    if args.verify_only:
        print(f"\n{checked} checked: {present} present and valid, "
              f"{broken} missing or invalid")
    else:
        verb = "would fetch" if args.dry_run else "fetched"
        print(f"\n{checked} database(s): {fetched} {verb}, "
              f"{present} already present, {broken} failed")
        if broken:
            print("A staged .incoming file means the server answered but the "
                  "content was not a MEME file -- read it before retrying.")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
