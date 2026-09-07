#!/usr/bin/env python3
"""Run the ENCODE PRO-cap pipeline for one or more experiments.

Implements the steps in config/procap_pipeline.yaml, transcribed from
planning/20240501_PRO-cap_Computational_Pipeline.pdf:

    fastp -> STAR -> samtools (unique) -> umi_tools dedup -> 5' stranded
    bigWigs (merged across replicates) -> PINTS peaks

ENCODE drives these steps with rmsp; this script drives them directly and uses
output existence for caching (--force to redo). Experiment definitions, run
accessions, layouts, UMI settings, and spike-in genomes come from
config/experiment_config.yaml; genome resources from config/genomes.yaml.

The peak set written for each experiment is the CONCATENATION of PINTS
unidirectional and bidirectional calls, sorted -- never interval-merged, and not
cut to BED3. `divergent` calls are excluded (a subset of bidirectional). Many
non-human species have a large fraction of unidirectional TSSs, so this union --
not bidirectional calls alone -- is the locus set used for training, validation,
and testing. workflow/Snakefile's combine_peaks rule must stay identical.

Usage:
    python src/data_preprocessing/run_procap_pipeline.py --list
    python src/data_preprocessing/run_procap_pipeline.py -e S.cerevisiae_PROcap --dry-run
    python src/data_preprocessing/run_procap_pipeline.py -e S.cerevisiae_PROcap S.pombe_PROcap
    python src/data_preprocessing/run_procap_pipeline.py --species S.cerevisiae -t 16
    python src/data_preprocessing/run_procap_pipeline.py --index-only --species M.musculus
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_PATH = REPO_ROOT / "config" / "experiment_config.yaml"
GENOMES_PATH = REPO_ROOT / "config" / "genomes.yaml"
PIPELINE_PATH = REPO_ROOT / "config" / "procap_pipeline.yaml"
RESOLVED_PATH = REPO_ROOT / "planning" / "manifest_runs_resolved.tsv"

REQUIRED_TOOLS = [
    "wget",
    "fastp",
    "STAR",
    "samtools",
    "bedtools",
    "bgzip",             # htslib
    "bedGraphToBigWig",  # UCSC
    "pints_caller",
]
# Only needed for libraries that carry a UMI (currently Spt5_PROcap_sc).
UMI_TOOLS = ["umi_tools"]

ENA_FASTQ_BASE = "https://ftp.sra.ebi.ac.uk/vol1/fastq"

#: Deinterleave an interleaved-mates FASTQ. Odd records are mate 1, even are
#: mate 2; the read name is rewritten to the instrument name with /1 stripped
#: so both mates share a name. Keep identical to the deinterleave rule in
#: workflow/Snakefile.
AWK_DEINTERLEAVE = (
    '{ name = $1; sub(/^@[^ \\t]+[ \\t]+/, "", name); sub(/\\/[12]$/, "", name); rec = "@" name "\\n" $2 "\\n+\\n" $4; if (NR % 2 == 1) print rec | ("gzip > " o1); else print rec | ("gzip > " o2) }'
)


def load(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run(cmd: list, dry_run: bool, check: bool = True) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"  $ {printable}")
    if not dry_run:
        subprocess.run([str(c) for c in cmd], check=check)


def load_ena_urls() -> dict[str, list[str]]:
    """run_accession -> FASTQ URLs, as reported by ENA in the resolved manifest.

    Preferred over constructing paths: ENA's layout varies (some runs sit at
    vol1/fastq/<prefix>/<run>/, others at
    vol1/fastq/<prefix>/<zero-padded suffix>/<run>/), so guessing produces 404s.
    """
    urls: dict[str, list[str]] = {}
    if not RESOLVED_PATH.exists():
        return urls
    with RESOLVED_PATH.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            runs = [r for r in (row["run_accession"] or "").split(";") if r]
            ftps = [u for u in (row["fastq_ftp"] or "").split(";") if u]
            for run_id in runs:
                for u in ftps:
                    if f"/{run_id}/" in u or Path(u).name.startswith(run_id):
                        urls.setdefault(run_id, [])
                        full = u if u.startswith("http") else f"https://{u}"
                        if full not in urls[run_id]:
                            urls[run_id].append(full)
    return urls


def ena_fastq_urls(run_id: str, layout: str,
                   known: dict[str, list[str]] | None = None) -> list[str]:
    """FASTQ URLs for a run: ENA-reported if available, else constructed."""
    if known and known.get(run_id):
        return known[run_id]
    prefix = run_id[:6]
    if len(run_id) > 9:
        sub = f"{int(run_id[9:]):03d}" if run_id[9:].isdigit() else run_id[9:]
        base = f"{ENA_FASTQ_BASE}/{prefix}/{sub}/{run_id}"
    else:
        base = f"{ENA_FASTQ_BASE}/{prefix}/{run_id}"
    print(f"  NOTE: {run_id} not in {RESOLVED_PATH.name}; constructing URL")
    if layout.upper().startswith("PAIRED"):
        return [f"{base}/{run_id}_1.fastq.gz", f"{base}/{run_id}_2.fastq.gz"]
    return [f"{base}/{run_id}.fastq.gz"]


def check_tools(dry_run: bool) -> None:
    missing = [t for t in REQUIRED_TOOLS if shutil.which(t) is None]
    if missing:
        msg = f"missing required tools: {', '.join(missing)}"
        if dry_run:
            print(f"WARNING: {msg} (continuing because --dry-run)\n")
        else:
            print(f"Error: {msg}", file=sys.stderr)
            sys.exit(1)


def fetch_genome(species: str, genome: dict, dry_run: bool, force: bool) -> None:
    """Download and index a species FASTA from the URL in config/genomes.yaml."""
    fasta = REPO_ROOT / genome["fasta"]
    if fasta.exists() and not force:
        print(f"  {species}: {fasta.name} present")
    else:
        gz = Path(f"{fasta}.gz")
        fasta.parent.mkdir(parents=True, exist_ok=True)
        run(["wget", "-q", "-c", "-O", gz, genome["fasta_url"]], dry_run)
        run(["gunzip", "-f", gz], dry_run)
    if not Path(f"{fasta}.fai").exists() or force:
        run(["samtools", "faidx", fasta], dry_run)
    bl_url, bl = genome.get("blacklist_url"), genome.get("blacklist")
    if bl_url and bl:
        dest = REPO_ROOT / bl
        if dest.exists() and not force:
            print(f"  {species}: {dest.name} present")
        else:
            run(["wget", "-q", "-c", "-O", dest, bl_url], dry_run)
    elif not bl:
        print(f"  {species}: no exclusion list published for this assembly")

    if not genome.get("rdna_accession"):
        print(f"  {species}: no rDNA decoy configured "
              "(ENCODE aligns to genome + rDNA; see rdna_note)")
    orgs = genome.get("organelle_accessions") or []
    if orgs:
        print(f"  {species}: organelle decoys configured: {', '.join(orgs)}")


def decoy_accessions(genome: dict) -> list[str]:
    """Non-assembly sequences to concatenate into the STAR index.

    Two rationales, one mechanism -- keep in step with decoy_accessions() in
    workflow/Snakefile:

    * `rdna_accession` is a SINK, only where the array is missing from the
      assembly.
    * `organelle_accessions` covers a reference that omits its chloroplast or
      mitochondrion, which for C. reinhardtii and P. patens leaves 12.2% and
      8.6% of the library with nowhere to map at all.

    Decoys go in the INDEX ONLY. They are absent from main_chromosomes, so
    `bedgraph` filters them out before bigwigs, peaks or folds -- the same
    treatment mm10's chrM already gets, since no organelle is in
    main_chromosomes for any species here.
    """
    accs = [genome["rdna_accession"]] if genome.get("rdna_accession") else []
    return accs + list(genome.get("organelle_accessions") or [])


def star_index(species: str, genome: dict, threads: int, dry_run: bool,
               force: bool) -> Path:
    """Build (or reuse) the STAR index: genome + every decoy."""
    fasta = REPO_ROOT / genome["fasta"]
    index_dir = REPO_ROOT / "data" / "star" / species
    if (index_dir / "SA").exists() and not force:
        print(f"  STAR index exists: {index_dir}")
        return index_dir
    if not fasta.exists() and not dry_run:
        print(f"Error: genome FASTA not found: {fasta}\n"
              f"  fetch it from {genome['fasta_url']}", file=sys.stderr)
        sys.exit(1)

    # Genome + decoys, matching workflow/Snakefile's star_index rule.
    fastas = [fasta]
    for acc in decoy_accessions(genome):
        decoy_fa = REPO_ROOT / "data" / "decoy" / f"{acc}.fa"
        if not decoy_fa.exists():
            url = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{acc}"
            decoy_fa.parent.mkdir(parents=True, exist_ok=True)
            run(["wget", "-q", "-O", decoy_fa, url], dry_run)
        fastas.append(decoy_fa)
    if not genome.get("rdna_accession"):
        print(f"  NOTE: no rDNA decoy configured for {species}; ENCODE aligns to "
              "genome + rDNA. See rdna_note in config/genomes.yaml.")

    index_dir.mkdir(parents=True, exist_ok=True)
    run(
        ["STAR", "--runThreadN", threads, "--runMode", "genomeGenerate",
         "--genomeDir", index_dir, "--genomeFastaFiles", *fastas,
         "--genomeSAindexNbases", genome["star_sa_index_nbases"]],
        dry_run,
    )
    return index_dir


def adapter_arg(pipeline: dict, adapter: str, work: Path,
                dry_run: bool = False) -> list:
    """fastp --adapter_fasta for an adapter NAME, or [] to auto-detect.

    Keep in step with adapter_fasta() and the adapter_fasta rule in
    workflow/Snakefile. Passing the adapter explicitly is load-bearing: fastp's
    auto-detection looks for an overrepresented read TAIL and cannot find a
    PRO-cap adapter, which starts at a different offset in every read because
    inserts range from 4 to 55 bp. It reported "No adapter detected" and trimmed
    0 reads on libraries measured at 96% adapter content.

    A FASTA rather than --adapter_sequence because one name can carry several
    sequences, and TruSeq has to: 43.4% of SRR12774945 and 18.7% of SRR6660402
    begin FIVE BASES INTO the adapter, and fastp matches an adapter by looking
    for its BEGINNING in the read, so those reads are never trimmed and reach
    STAR as full-length runs of pure adapter.
    """
    if not adapter:
        return []
    seqs = pipeline["steps"]["trim"].get("adapters", {}) or {}
    if adapter not in seqs:
        raise SystemExit(
            f"unknown adapter name {adapter!r}; known: {sorted(seqs)}. "
            "Add it to steps.trim.adapters in config/procap_pipeline.yaml."
        )
    entries = seqs[adapter]
    if isinstance(entries, str):        # a scalar config still works
        entries = [entries]
    fasta = work / f"{adapter}.adapters.fa"
    if not dry_run:
        fasta.parent.mkdir(parents=True, exist_ok=True)
        fasta.write_text("".join(
            f">{adapter}_{i}\n{seq}\n" for i, seq in enumerate(entries)
        ))
    # Both flags: the fasta carries every variant (and is what catches a
    # truncated dimer), while --adapter_sequence pins the canonical one and
    # thereby suppresses fastp's auto-detection, making trimming deterministic.
    # Keep in step with params.adapter in workflow/Snakefile's trim rule.
    return ["--adapter_sequence", entries[0], "--adapter_fasta", fasta]


def umi_arg(pipeline: dict, umi: dict, paired: bool, run_id: str) -> list:
    """fastp --umi flags for a run, or [] for a library without a UMI.

    --umi_loc comes from the MANIFEST via steps.dedup.umi_locations, not from
    the layout. It used to be `read2 if paired else read1`, which was backwards
    for the only three experiments it applied to: the Spt5 UMI sits at the start
    of READ 1. Keep in step with umi_arg() in workflow/Snakefile.
    """
    if not umi.get("enabled"):
        return []
    locations = pipeline["steps"]["dedup"].get("umi_locations", {}) or {}
    loc = umi.get("loc", "")
    if loc not in locations:
        raise SystemExit(
            f"run {run_id}: umi_loc {loc!r} has no fastp mapping; known: "
            f"{sorted(locations)}. Add it to steps.dedup.umi_locations in "
            "config/procap_pipeline.yaml."
        )
    fastp_loc = locations[loc]
    if fastp_loc == "read2" and not paired:
        raise SystemExit(
            f"run {run_id}: umi_loc {loc!r} maps to fastp read2, but the run is "
            "single-end. A 3'-adaptor UMI on a single-end read sits at the "
            "read's 3' END, which fastp --umi_loc cannot extract."
        )
    return ["--umi", "--umi_loc", fastp_loc, "--umi_len", umi["len"]]


def process_run(run_id: str, layout: str, index_dir: Path, work: Path,
                pipeline: dict, umi: dict, threads: int, dry_run: bool,
                force: bool, fastq_dir: Path | None = None,
                known_urls: dict[str, list[str]] | None = None,
                sort_ram_gb: float = 10.0, adapter: str = "",
                interleaved: bool = False) -> Path:
    """fastp -> STAR -> unique filter -> dedup for a single run. Returns BAM."""
    steps = pipeline["steps"]
    # `interleaved` overrides the archive layout: a deposit reported SINGLE
    # that actually holds interleaved mates must be PROCESSED as paired, or
    # every R2 contributes its 5' end (the RNA 3' end, opposite strand,
    # displaced downstream by the fragment length) to the signal. Keep in step
    # with is_paired()/is_interleaved() in workflow/Snakefile.
    paired = layout.upper().startswith("PAIRED") or interleaved
    final_bam = work / f"{run_id}.final.bam"
    if final_bam.exists() and not force:
        print(f"  {run_id}: final BAM exists, skipping")
        return final_bam

    # -- inputs: prefer FASTQs already fetched by fetch_fastqs.py
    urls = ena_fastq_urls(run_id, layout, known_urls)
    fastqs = []
    for url in urls:
        name = Path(url).name
        prefetched = (fastq_dir / name) if fastq_dir else None
        if prefetched is not None and prefetched.exists():
            print(f"  {run_id}: using pre-fetched {prefetched}")
            fastqs.append(prefetched)
            continue
        dest = work / name
        if not dest.exists() or force:
            run(["wget", "-q", "-c", "-O", dest, url], dry_run)
        fastqs.append(dest)

    # -- 0. split an interleaved deposit into real mate files.
    # Mates alternate record-by-record. Read names are rewritten to the
    # instrument name with /1 stripped so both mates share a name -- SRA gives
    # the mates different accession names while labelling both /1, which is what
    # hid the pairing. Single pass: awk holds both gzip pipes open.
    if interleaved:
        m1 = work / f"{run_id}.mate_1.fastq.gz"
        m2 = work / f"{run_id}.mate_2.fastq.gz"
        if not (m1.exists() and m2.exists()) or force:
            awk_prog = AWK_DEINTERLEAVE
            cmd = (f"zcat {fastqs[0]} | paste - - - - | "
                   f"awk -F'\\t' -v o1={m1} -v o2={m2} '{awk_prog}'")
            print(f"  $ {cmd}")
            if not dry_run:
                subprocess.run(["bash", "-o", "pipefail", "-c", cmd], check=True)
        fastqs = [m1, m2]

    # -- 1. fastp: adapter trimming (+ UMI if this library has one)
    trimmed = [work / f"{f.name.replace('.fastq.gz', '')}.trimmed.fastq.gz"
               for f in fastqs]  # always in work/, fastq_dir may be read-only
    cmd = ["fastp", *steps["trim"]["encode_params"].split()]
    cmd += adapter_arg(pipeline, adapter, work, dry_run)
    cmd += steps["trim"].get("poly_params", "").split()
    if paired:
        cmd += ["-i", fastqs[0], "-I", fastqs[1],
                "-o", trimmed[0], "-O", trimmed[1]]
    else:
        cmd += ["-i", fastqs[0], "-o", trimmed[0]]
    cmd += umi_arg(pipeline, umi, paired, run_id)
    cmd += ["--json", work / f"{run_id}.fastp.json",
            "--html", work / f"{run_id}.fastp.html",
            "--thread", min(threads, 16)]
    run(cmd, dry_run)

    # -- 2. STAR alignment
    #
    # --limitBAMsortRAM defaults to the size of the GENOME INDEX, not to
    # anything about the library, so on a small genome the sort budget is tiny
    # and a deep library dies with "not enough memory for BAM sorting".
    # Counter-intuitively it is the small genomes that break. Keep this in step
    # with the align rule in workflow/Snakefile.
    prefix = work / f"{run_id}."
    run(
        ["STAR", "--genomeDir", index_dir, "--runThreadN", threads,
         "--readFilesIn", *(trimmed if paired else [trimmed[0]]),
         "--readFilesCommand", "zcat", "--outFileNamePrefix", prefix,
         "--outSAMtype", "BAM", "SortedByCoordinate",
         "--limitBAMsortRAM", int(sort_ram_gb * 1e9),
         *steps["align"]["encode_params"].split()],
        dry_run,
    )
    aligned = work / f"{run_id}.Aligned.sortedByCoord.out.bam"

    # -- 3. uniquely mapped reads only (STAR gives unique reads MAPQ 255)
    unique = work / f"{run_id}.unique.bam"
    run(["samtools", "view", "-b", *steps["filter"]["encode_params"].split(),
         "-@", max(1, int(threads) - 1), "-o", unique, aligned], dry_run)
    run(["samtools", "index", unique], dry_run)

    # -- 4. dedup (UMI libraries only)
    if umi.get("enabled"):
        # --paired (layout) and --method (clustering) are orthogonal; the
        # separator must match what fastp wrote into the read name. Keep in
        # step with the dedup rule in workflow/Snakefile.
        dd = steps["dedup"]
        run(["umi_tools", "dedup", "-I", unique, "-S", final_bam,
             f"--method={dd['method']}",
             f"--umi-separator={dd['umi_separator']}",
             *(["--paired"] if paired else [])], dry_run)
    else:
        print(f"  {run_id}: no UMI, skipping umi_tools dedup "
              "(PRO-cap stacks real reads on one base; do not collapse)")
        if not dry_run:
            shutil.copy(unique, final_bam)
        else:
            print(f"  $ cp {unique} {final_bam}")

    # Paired-end: keep ONE mate only. bedtools genomecov -5 reports the 5' end
    # of every record, so both mates would contribute and half of each track
    # would be strand-flipped 3'-end signal. After dedup, because umi_tools
    # --paired needs both mates. Keep in step with final_bam in the Snakefile.
    if paired:
        mate_flag = 64 if steps["signal"].get("five_prime_mate", "R1") == "R1" else 128
        mated = work / f"{run_id}.mate.bam"
        run(["samtools", "view", "-b", "-f", mate_flag, "-o", mated, final_bam],
            dry_run)
        if not dry_run:
            os.replace(mated, final_bam)
        else:
            print(f"  $ mv {mated} {final_bam}")
    run(["samtools", "index", final_bam], dry_run)
    return final_bam


def sort_cmd(threads: int = 1, buffer: str = "2G") -> list[str]:
    """GNU sort for BED/bedGraph, with parallelism and buffer size pinned.

    GNU sort already multithreads by default (up to 8) and auto-sizes its
    buffer from free memory, so several concurrent jobs each size themselves
    against the same total. Both are pinned here so a job's real cost matches
    what the caller budgeted. Requires GNU coreutils; BSD sort has neither flag.
    """
    return ["sort", f"--parallel={threads}", "-S", buffer, "-k1,1", "-k2,2n"]


def make_signal(bams: list[Path], exp_id: str, genome: dict, out_dir: Path,
                work: Path, reverse_strand: bool, threads: int,
                dry_run: bool) -> tuple[Path, Path]:
    """Merged, strand-separated single-base 5'-end bigWigs across replicates."""
    fasta = REPO_ROOT / genome["fasta"]
    chrom_sizes = work / "chrom.sizes"
    fai = Path(f"{fasta}.fai")
    if not fai.exists():
        run(["samtools", "faidx", fasta], dry_run)
    # Main chromosomes only -- keep in step with the chrom_sizes rule in
    # workflow/Snakefile. Scaffolds cannot reach a fold, and their sparse
    # coverage makes the pl and mn bigWigs carry different contig sets, which
    # PINTS rejects outright ("bw_pl and bw_mn should have the same
    # chromosomes"). Decoys -- the rDNA sink and any organelle -- are excluded
    # for the same reason: they belong in the STAR index, not in the
    # peak-calling space, which is also how mm10's chrM is already treated.
    keep = set(genome["main_chromosomes"])
    if not dry_run:
        with chrom_sizes.open("w") as f_out, fai.open() as f_in:
            for line in f_in:
                parts = line.split("\t")
                if parts[0] in keep:
                    f_out.write(f"{parts[0]}\t{parts[1]}\n")
    else:
        print(f"  $ cut -f1,2 {fai} | grep -w -E '{'|'.join(sorted(keep))}' "
              f"> {chrom_sizes}")

    merged = work / f"{exp_id}.merged.bam"
    if len(bams) > 1:
        run(["samtools", "merge", "-f", "-o", merged, *bams], dry_run)
    else:
        merged = bams[0]
    run(["samtools", "index", merged], dry_run)

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    # `-5` reports only the read 5' end, which for PRO-cap is the initiation
    # base. Strand assignment depends on library orientation, hence
    # reverse_strand: validate at known unidirectional promoters before trusting.
    for label, strand in (("pl", "+"), ("mn", "-")):
        read_strand = strand
        if reverse_strand:
            read_strand = "-" if strand == "+" else "+"
        bg = work / f"{exp_id}_{label}.bg"
        bw = out_dir / f"{exp_id}_{label}.bw"
        if dry_run:
            print(f"  $ bedtools genomecov -5 -bg -strand {read_strand} "
                  f"-ibam {merged} | grep -w -E '{'|'.join(sorted(keep))}' "
                  f"| {' '.join(sort_cmd(threads))} > {bg}")
        else:
            with bg.open("w") as f_out:
                gc = subprocess.Popen(
                    ["bedtools", "genomecov", "-5", "-bg", "-strand",
                     read_strand, "-ibam", str(merged)],
                    stdout=subprocess.PIPE,
                )
                # Drop scaffolds before sorting, so the bedGraph matches the
                # restricted chrom.sizes above and both strands agree.
                grep = subprocess.Popen(
                    ["grep", "-w", "-E", "|".join(sorted(keep))],
                    stdin=gc.stdout, stdout=subprocess.PIPE,
                )
                subprocess.run(sort_cmd(threads), stdin=grep.stdout,
                               stdout=f_out, check=True)
                gc.wait(); grep.wait()
        run(["bedGraphToBigWig", bg, chrom_sizes, bw], dry_run)
        outputs[label] = bw
    return outputs["pl"], outputs["mn"]


def call_peaks(pl_bw: Path, mn_bw: Path, exp_id: str, pipeline: dict,
               genome: dict, out_dir: Path, work: Path, threads: int,
               dry_run: bool) -> Path:
    """PINTS, then CONCATENATE unidirectional + bidirectional calls and sort.

    Deliberately not `bedtools merge`, and columns are not cut to BED3 --
    see the combine_peaks rule in workflow/Snakefile and the notes under
    steps.peaks in config/procap_pipeline.yaml. Both drivers must agree here:
    a different locus set would make their models incomparable.
    """
    peaks_cfg = pipeline["steps"]["peaks"]
    pints_dir = work / f"{exp_id}_PINTS"
    # PINTS defaults --chromosome-start-with to "chr", which matches NOTHING for
    # the Ensembl-named species and would silently return zero peaks.
    # Keep in step with PINTS_CHROM_PREFIX in workflow/Snakefile.
    # "ncbi" is GenBank-accession naming (CM...), which C. reinhardtii's
    # ASM4749649v1 uses; "gwh" is CNCB's Genome Warehouse, which P. patens V7
    # uses. Both were added to workflow/Snakefile with those assemblies and NOT
    # here, so this driver raised SystemExit for both species -- the exact drift
    # the comment above warns about. Both take the empty prefix, as ensembl does.
    prefixes = {"ucsc": "chr", "ensembl": "", "ncbi_refseq": "",
                "ncbi": "", "gwh": ""}
    style = genome["chrom_style"]
    if style not in prefixes:
        raise SystemExit(
            f"chrom_style {style!r} has no PINTS prefix; known: {sorted(prefixes)}. "
            "Add it here and to PINTS_CHROM_PREFIX in workflow/Snakefile."
        )
    prefix = prefixes[style]
    run(["pints_caller", "--save-to", pints_dir, "--file-prefix", exp_id,
         "--bw-pl", pl_bw, "--bw-mn", mn_bw, "--thread", threads,
         "--chromosome-start-with", prefix,
         "--exp-type", peaks_cfg["exp_type"],
         *peaks_cfg["encode_params"].split()], dry_run)

    out = out_dir / f"{exp_id}_peaks.bed.gz"
    classes = peaks_cfg["peak_classes"]
    globs = " ".join(f"{pints_dir}/*{c}_peaks.bed" for c in classes)
    print(f"  combining PINTS classes {classes} -> {out.name}")
    if dry_run:
        print(f"  $ cat {globs} | {' '.join(sort_cmd(threads, '1G'))} | bgzip > {out}")
        return out

    beds = [p for c in classes for p in sorted(pints_dir.glob(f"*{c}_peaks.bed"))]
    if not beds:
        print(f"Error: PINTS produced no peak files in {pints_dir}",
              file=sys.stderr)
        sys.exit(1)
    cat = subprocess.Popen(["cat", *[str(b) for b in beds]],
                           stdout=subprocess.PIPE)
    sort = subprocess.Popen(sort_cmd(threads, "1G"), stdin=cat.stdout,
                            stdout=subprocess.PIPE)
    with out.open("wb") as f_out:
        subprocess.run(["bgzip", "-c"], stdin=sort.stdout, stdout=f_out,
                       check=True)
    for proc in (cat, sort):
        proc.wait()
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-e", "--experiments", nargs="+", action="extend",
                        default=[],
                        help="experiment IDs (default: all with resolved runs)")
    parser.add_argument("--species", nargs="+", action="extend", default=[],
                        help="restrict to these species")
    # action="extend" so repeated flags accumulate; plain nargs="+" makes
    # `--tier include --tier conditional` silently mean just "conditional".
    parser.add_argument("--star-sort-ram", type=float, default=10.0,
                        metavar="GB",
                        help="STAR --limitBAMsortRAM in GB (default: %(default)s). "
                             "STAR's own default is the genome index size, which "
                             "is too small for a deep library on a compact genome.")
    parser.add_argument("--tier", nargs="+", action="extend", default=[],
                        choices=["include", "conditional", "exclude"])
    parser.add_argument("--list", action="store_true",
                        help="list experiments and their readiness, then exit")
    parser.add_argument("--fetch-genomes", action="store_true",
                        help="download + faidx the FASTAs for the selected "
                             "species, then exit")
    parser.add_argument("--index-only", action="store_true",
                        help="build STAR indices for the selected species only")
    parser.add_argument("--allow-bundled-tap", action="store_true",
                        help="process experiments whose TAP+/TAP- run split is "
                             "unresolved (merges cap-selected and background "
                             "libraries into one track -- almost never correct)")
    parser.add_argument("--reverse-strand", action="store_true",
                        help="library sequences the antisense strand; swap pl/mn")
    parser.add_argument("--fastq-dir", type=Path, default=None,
                        help="directory of pre-fetched FASTQs from "
                             "fetch_fastqs.py (default: $PROCAP_FASTQ_DIR or "
                             "data/fastq). Missing files are fetched on demand.")
    parser.add_argument("-t", "--threads", type=int, default=8)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load(EXPERIMENTS_PATH)["experiments"]
    genomes = load(GENOMES_PATH)["species"]
    pipeline = load(PIPELINE_PATH)

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

    if args.list:
        print(f"{'experiment':40s} {'species':16s} {'tier':12s} {'reps':5s} runs")
        for e in selected:
            c = config[e]
            runs = c["raw"]["runs"]
            print(f"{e:40s} {c['species']:16s} {c['tier']:12s} "
                  f"{c['n_biological_replicates']:<5d} "
                  f"{len(runs) if runs else 'UNRESOLVED'}")
        return

    check_tools(args.dry_run)
    if any(config[e]["raw"].get("umi", {}).get("enabled") for e in selected):
        if shutil.which("umi_tools") is None:
            print("WARNING: a selected experiment has a UMI but umi_tools is "
                  "not on PATH; dedup would be skipped.\n")
    work_root = args.work_dir or (REPO_ROOT / "data" / "procap_work")
    known_urls = load_ena_urls()
    fastq_dir = args.fastq_dir or Path(
        os.environ.get("PROCAP_FASTQ_DIR", REPO_ROOT / "data" / "fastq"))
    out_dir = REPO_ROOT / "data" / "procap"

    if args.fetch_genomes:
        for species in sorted({config[e]["species"] for e in selected}):
            print(f"\n=== genome: {species} ===")
            fetch_genome(species, genomes[species], args.dry_run, args.force)
        return

    if args.index_only:
        for species in sorted({config[e]["species"] for e in selected}):
            print(f"\n=== STAR index: {species} ===")
            star_index(species, genomes[species], args.threads, args.dry_run,
                       args.force)
        return

    n_done = n_skipped = 0
    for exp_id in selected:
        entry = config[exp_id]
        raw = entry["raw"]
        species = entry["species"]
        print(f"\n=== {exp_id} ({species}, {entry['tier']}) ===")

        if not raw["runs"]:
            print("  SKIP: no resolved run accessions "
                  "(see planning/manifest_runs_resolved.tsv)")
            n_skipped += 1
            continue
        if raw.get("tap_bundled") and not args.allow_bundled_tap:
            print("  SKIP: TAP+ and TAP- runs are bundled under one GEO sample "
                  "and the run-level split is unresolved.\n"
                  f"        Merging {raw['runs']} would mix cap-selected signal "
                  "with protocol background.\n"
                  "        Resolve which run is TAP+ (then set raw.runs to just "
                  "that run), or pass --allow-bundled-tap to override.")
            n_skipped += 1
            continue
        if raw.get("spike_in"):
            print(f"  NOTE: spike-in present ({raw['spike_in']}). Spike-in reads "
                  "must not contribute to target labels; record the mapped "
                  "fraction and confirm they are absorbed by the decoy/other "
                  "genome before trusting counts.")

        final = out_dir / f"{exp_id}_peaks.bed.gz"
        if final.exists() and not args.force:
            print(f"  SKIP: {final.name} exists (use --force to redo)")
            n_skipped += 1
            continue

        work = work_root / exp_id
        work.mkdir(parents=True, exist_ok=True)
        index_dir = star_index(species, genomes[species], args.threads,
                               args.dry_run, force=False)

        layout = raw["library_layout"]
        bams = [
            process_run(run_id, layout, index_dir, work, pipeline,
                        raw.get("umi", {}), args.threads, args.dry_run,
                        args.force, fastq_dir, known_urls,
                        args.star_sort_ram, raw.get("adapter", ""),
                        bool(raw.get("interleaved")))
            for run_id in raw["runs"]
        ]
        pl_bw, mn_bw = make_signal(bams, exp_id, genomes[species], out_dir, work,
                                   args.reverse_strand, args.threads, args.dry_run)
        call_peaks(pl_bw, mn_bw, exp_id, pipeline, genomes[species], out_dir,
                   work, args.threads, args.dry_run)
        print(f"  done -> {out_dir}/{exp_id}_{{pl,mn}}.bw, {final.name}")
        print("  next: uv run python src/make_negatives.py "
              f"-e {exp_id}")
        n_done += 1

    action = "Would process" if args.dry_run else "Processed"
    print(f"\n{action} {n_done} experiments, skipped {n_skipped}")


if __name__ == "__main__":
    main()
