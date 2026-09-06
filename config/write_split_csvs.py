#!/usr/bin/env python3
"""Derive config/splits/<species>_data_fold_assignments.csv from chrom_splits.yaml.

This repo has two consumers of chromosome folds and they read different files:

  * config/chrom_splits.yaml   -- fit_bpnet.py, launch.py (species-keyed)
  * config/splits/<species>_data_fold_assignments.csv
                                -- fit_cherimoya.py and every benchmark /
                                   attribute / compare_bigwigs script

Keeping both by hand invites drift, and the CSVs were missing entirely for
species added later (M.musculus, C.elegans, A.thaliana), which silently left the
second group of scripts falling back to the D.melanogaster default. This script
makes chrom_splits.yaml the single source of truth and regenerates the CSVs from
it.

Peak-level split files (config/splits/<species>_random_fold_assignments.csv,
written by make_random_splits.py) are a different mechanism and are never
touched here.

--peak-counts reports peaks per fold for an experiment. Folds in this project are
assigned by manually matching *peak counts* across folds, not sequence length --
a fold can be well balanced in bp and badly balanced in loci, and the latter is
what affects training. Use this to check or retune an assignment.

For a species with NO entry in chrom_splits.yaml it reports peaks per CHROMOSOME
instead, over the species' main_chromosomes, which is the input you need to
build the assignment in the first place. It used to exit 1 with
"<species> not in chrom_splits.yaml" -- refusing precisely when it is most
useful, and forcing the counts to be gathered by hand (which is how the
C. griseus assignment was done). --by-chrom forces that view for an assigned
species too, e.g. to retune from a deeper library.

Usage:
    python config/write_split_csvs.py --check      # verify, exit 1 on drift
    python config/write_split_csvs.py              # (re)write all species
    python config/write_split_csvs.py -s M.musculus
    python config/write_split_csvs.py --peak-counts -e S.cerevisiae-Ino80ctl_PROcap
    python config/write_split_csvs.py --peak-counts -e G.hirsutum-ovule_GROcap
"""

import argparse
import csv
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CHROM_SPLITS = REPO_ROOT / "config" / "chrom_splits.yaml"
SPLITS_DIR = REPO_ROOT / "config" / "splits"
EXPERIMENTS = REPO_ROOT / "config" / "experiment_config.yaml"
GENOMES = REPO_ROOT / "config" / "genomes.yaml"


def rows_for(folds: dict) -> list[tuple[str, int]]:
    """(chrom, fold) rows in fold order, matching the YAML's within-fold order."""
    return [
        (str(chrom), int(fold))
        for fold in sorted(folds, key=int)
        for chrom in folds[fold]
    ]


def peak_counts(experiment: str, splits: dict, by_chrom: bool = False) -> int:
    """Report peaks per fold for one experiment, or per chromosome. Returns 0 on success."""
    import csv as _csv
    import gzip

    with EXPERIMENTS.open() as f:
        experiments = yaml.safe_load(f)["experiments"]
    if experiment not in experiments:
        print(f"Error: unknown experiment {experiment!r}", file=sys.stderr)
        return 1

    entry = experiments[experiment]
    species = entry["species"]
    by_chrom = by_chrom or species not in splits

    peaks_path = REPO_ROOT / entry["processed"]["peaks"]
    if not peaks_path.exists():
        print(f"Peaks not built yet: {peaks_path}\n"
              "Run the ENCODE pipeline first "
              "(src/data_preprocessing/run_procap_pipeline.py).", file=sys.stderr)
        return 1

    if by_chrom:
        # Group by chromosome rather than by fold. main_chromosomes is the right
        # universe: it is the allow-list bedgraph and PINTS already applied, so a
        # name outside it cannot be a fold member anyway.
        with GENOMES.open() as f:
            main = [str(c) for c in
                    yaml.safe_load(f)["species"][species]["main_chromosomes"]]
        chrom_to_fold = {c: c for c in main}
        counts = {c: 0 for c in main}
    else:
        chrom_to_fold = {
            str(chrom): int(fold)
            for fold, chroms in splits[species].items()
            for chrom in chroms
        }
        counts = {int(f): 0 for f in splits[species]}
    unassigned = {}
    opener = gzip.open if peaks_path.suffix == ".gz" else open
    with opener(peaks_path, "rt") as f:
        for row in _csv.reader(f, delimiter="\t"):
            if not row or row[0].startswith(("#", "track")):
                continue
            fold = chrom_to_fold.get(row[0])
            if fold is None:
                unassigned[row[0]] = unassigned.get(row[0], 0) + 1
            else:
                counts[fold] += 1

    total = sum(counts.values())
    unit = "chromosomes" if by_chrom else "folds"
    print(f"{experiment}  ({species}, {len(counts)} {unit})")
    if by_chrom and species not in splits:
        print(f"  NOTE: {species} has no entry in chrom_splits.yaml. These are "
              "per-chromosome\n        counts, for building one -- match peak "
              "counts across folds, not bp.")
    print(f"  peaks assigned to a {unit[:-1]}: {total:,}")
    order = sorted(counts) if not by_chrom else list(counts)
    width = max((len(str(k)) for k in order), default=1)
    for key in order:
        n = counts[key]
        share = n / total if total else 0
        label = f"fold {key}" if not by_chrom else str(key).rjust(width)
        print(f"    {label}: {n:>8,}  {share:6.1%}  "
              f"{'|' * round(share * 40)}")
    if total and not by_chrom:
        lo, hi = min(counts.values()), max(counts.values())
        mean = total / len(counts)
        print(f"  spread: {hi - lo:,} peaks ({(hi - lo) / mean:.1%} of mean) "
              f"-- lower is better balanced")
    if unassigned:
        shown = sorted(unassigned.items(), key=lambda kv: -kv[1])[:8]
        print(f"  NOT in any fold ({sum(unassigned.values()):,} peaks): "
              + ", ".join(f"{c}={n:,}" for c, n in shown))
        print("    (expected for scaffolds/organelles; unexpected names may mean "
              "a chrom-naming mismatch with config/genomes.yaml)")
    return 0


def verify_genome(splits: dict, work: Path) -> list[str]:
    """Check every fold member against the real contig names. Returns problems.

    --check compared chrom_splits.yaml to the derived CSVs and nothing else, so
    it could not see the failure that actually matters: `extract_loci` matches
    chromosome names LITERALLY, so a name that is readable but not in the FASTA
    yields zero loci in silence. Same class as an exclusion list that excludes
    nothing. Three things are compared, per species:

      * fold members vs the FASTA's own contigs (`<fasta>.fai`) -- the ground
        truth for what extract_loci will see;
      * fold members vs genomes.yaml `main_chromosomes` -- these must be the
        same set, since a fold member outside the allow-list can never carry a
        peak or a bigWig value, and a main chromosome in no fold is silently
        dropped from training;
      * `main_chromosomes` vs the derived {work}/genome/<species>.chrom.sizes,
        which is what bedGraphToBigWig and PINTS actually saw.

    A missing .fai or chrom.sizes is SKIPPED, not failed, so this stays runnable
    on a dev machine where data/ is empty. What it must never do is report a
    clean bill for a species it could not check, so skips are named.
    """
    with GENOMES.open() as f:
        genomes = yaml.safe_load(f)["species"]
    problems, checked, skipped = [], [], []
    for sp in sorted(splits):
        if sp not in genomes:
            problems.append(f"{sp}: no entry in genomes.yaml")
            continue
        members = [str(c) for fold in splits[sp].values() for c in fold]
        dupes = sorted({c for c in members if members.count(c) > 1})
        if dupes:
            problems.append(f"{sp}: chromosome in more than one fold: {dupes}")
        members = set(members)

        main = {str(c) for c in genomes[sp]["main_chromosomes"]}
        if members - main:
            problems.append(f"{sp}: in a fold but not in main_chromosomes: "
                            f"{sorted(members - main)}")
        if main - members:
            problems.append(f"{sp}: in main_chromosomes but in no fold "
                            f"(silently untrained): {sorted(main - members)}")

        fai = REPO_ROOT / (genomes[sp]["fasta"] + ".fai")
        sizes = work / "genome" / f"{sp}.chrom.sizes"
        if not fai.exists() and not sizes.exists():
            skipped.append(sp)
            continue
        if fai.exists():
            with fai.open() as f:
                contigs = {ln.split("\t", 1)[0] for ln in f if ln.strip()}
            missing = sorted(members - contigs)
            if missing:
                problems.append(
                    f"{sp}: NOT IN {fai.name} -- extract_loci would return zero "
                    f"loci for these: {missing}")
        if sizes.exists():
            with sizes.open() as f:
                listed = {ln.split("\t", 1)[0] for ln in f if ln.strip()}
            if main - listed:
                problems.append(f"{sp}: in main_chromosomes but absent from "
                                f"{sizes.name}: {sorted(main - listed)}")
            if listed - main:
                problems.append(f"{sp}: in {sizes.name} but not in "
                                f"main_chromosomes: {sorted(listed - main)}")
        checked.append(sp)

    for sp in checked:
        print(f"  {sp:16s} genome ok ({len(splits[sp])} folds)")
    if skipped:
        print("  NOT CHECKED against a genome (no .fai or chrom.sizes here): "
              + ", ".join(skipped))

    # Species with no chrom_splits.yaml entry are invisible to everything above,
    # so an all-green report would quietly cover 10 of 12. They are absent by
    # DESIGN (peak-level folds), but "absent by design" and "absent and not
    # built yet" look identical from here, and the second means the species
    # cannot train at all. Report which, without failing: building the CSV needs
    # peaks, so its absence is a to-do, not drift.
    for sp in sorted(set(genomes) - set(splits)):
        csv_path = SPLITS_DIR / f"{sp}_random_fold_assignments.csv"
        if csv_path.exists():
            print(f"  {sp:16s} peak-level folds ({csv_path.name})")
        else:
            print(f"  {sp:16s} NO FOLDS AT ALL -- not in chrom_splits.yaml and "
                  f"{csv_path.name}\n  {'':16s} does not exist, so this species "
                  f"cannot train. Build it with\n  {'':16s} "
                  f"src/data_preprocessing/make_random_splits.py")

    verify_genome.checked = checked        # so main() cannot overclaim
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--species", nargs="+", action="extend",
                        default=[])
    parser.add_argument("--check", action="store_true",
                        help="report drift without writing; exit 1 if any found")
    parser.add_argument("--peak-counts", action="store_true",
                        help="report peaks per fold for -e EXPERIMENT and exit")
    parser.add_argument("-e", "--experiment", default=None,
                        help="experiment ID, for --peak-counts")
    parser.add_argument("--work", type=Path,
                        default=REPO_ROOT / "data" / "procap_work",
                        help="pipeline work dir, for {work}/genome/*.chrom.sizes")
    parser.add_argument("--by-chrom", action="store_true",
                        help="with --peak-counts, report per chromosome rather "
                             "than per fold (automatic for a species with no "
                             "chrom_splits.yaml entry)")
    args = parser.parse_args()

    with CHROM_SPLITS.open() as f:
        splits = yaml.safe_load(f)

    if args.peak_counts:
        if not args.experiment:
            print("Error: --peak-counts requires -e EXPERIMENT", file=sys.stderr)
            sys.exit(1)
        sys.exit(peak_counts(args.experiment, splits, args.by_chrom))

    species = args.species or sorted(splits)
    unknown = [s for s in species if s not in splits]
    if unknown:
        print(f"Error: not in chrom_splits.yaml: {unknown}", file=sys.stderr)
        sys.exit(1)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    drift = []
    for sp in species:
        rows = rows_for(splits[sp])
        path = SPLITS_DIR / f"{sp}_data_fold_assignments.csv"

        existing = None
        if path.exists():
            with path.open() as f:
                existing = [(r["chrom"], int(r["fold"])) for r in csv.DictReader(f)]

        if args.check:
            if existing is None:
                drift.append(f"{sp}: CSV missing")
            elif existing != rows:
                drift.append(f"{sp}: CSV disagrees with chrom_splits.yaml")
            else:
                print(f"  {sp:16s} ok ({len(rows)} chroms, "
                      f"{len(splits[sp])} folds)")
            continue

        with path.open("w", newline="") as f:
            # lineterminator="\n": csv defaults to CRLF, which .gitattributes
            # then normalizes on commit -- leaving the working copy permanently
            # "modified" after every regeneration.
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["chrom", "fold"])
            w.writerows(rows)
        status = "unchanged" if existing == rows else (
            "created" if existing is None else "rewritten")
        print(f"  {sp:16s} {status} ({len(rows)} chroms, {len(splits[sp])} folds)")

    if args.check:
        drift += verify_genome({sp: splits[sp] for sp in species}, args.work)
        for d in drift:
            print(f"  DRIFT {d}", file=sys.stderr)
        if drift:
            print(f"\n{len(drift)} problem(s); run without --check to fix.",
                  file=sys.stderr)
            sys.exit(1)
        n = len(getattr(verify_genome, "checked", []))
        tail = (f" and all {n} fold member set(s) exist in the genome they name"
                if n else "; NO species was checked against a genome here, so "
                "naming is UNVERIFIED -- run this where data/ lives")
        print(f"\nAll CSVs agree with chrom_splits.yaml{tail}.")


if __name__ == "__main__":
    main()
