#!/usr/bin/env python3
"""Regenerate config/experiment_config.yaml from the planning manifest.

Each trainable experiment is one biological condition of one species, pooling
biological replicates. Conditions are kept as separate experiments rather than
separate heads of one model: this project trains one model per experiment and
does not multi-task, so a perturbation (Ino80 depletion, Spt5 IAA) is its own
experiment, not a covariate.

Target rows become the experiment's signal; TAP-/noTAP rows from the same
project become `controls`, which are background/specificity references and must
never be used as positive TSS labels.

`processed` paths point at the ENCODE PRO-cap pipeline outputs
(config/procap_pipeline.yaml), which is where uniform re-mapping writes. Where a
legacy hand-built track already exists it is preserved under
`legacy_processed` for provenance and comparison, not used for training.

Usage:
    python src/data_preprocessing/build_experiment_config.py
    python src/data_preprocessing/build_experiment_config.py --dry-run
"""

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES = REPO_ROOT / "planning" / "manifest_samples.tsv"
RESOLVED = REPO_ROOT / "planning" / "manifest_runs_resolved.tsv"
OUTPUT = REPO_ROOT / "config" / "experiment_config.yaml"
DATASETS_TSV = REPO_ROOT / "config" / "datasets.tsv"
GENOMES = REPO_ROOT / "config" / "genomes.yaml"

PROCAP_DIR = "data/procap"

#: Projects whose experiments are split by SEX as well as by condition, i.e. the
#: key below gains a fifth field. Only where sex is a biological variable worth
#: its own model and depth allows -- see SEX_SPLIT_PROJECTS notes in CLAUDE.md.
#:
#: Liver_ChROcap_mm qualifies on both counts. The manifest already records sex as
#: a distinct `replicate_group` (liver old female / liver old male / ...), so
#: pooling was overriding its own grouping; mouse liver is one of the most
#: strongly sex-dimorphic transcriptional programs known, driven by growth
#: hormone pulsatility; and each sex carries ~97-116 M archive reads (~84 M
#: signal), deeper than every experiment in the corpus bar GCB. Pooling averaged
#: over a bimodal program for no gain in depth that mattered.
SEX_SPLIT_PROJECTS = {"Liver_ChROcap_mm"}


def experiment_key(row: dict) -> tuple:
    """(project_key, condition, stage, material) -- plus sex where it matters.

    Kept as a 4-tuple by default rather than always carrying a fifth empty field,
    so the 36 experiments where sex is meaningless (cell lines, pooled embryos)
    do not each grow a `""` that invites someone to fill it in.
    """
    key = (row["project_key"], row["condition"],
           row["developmental_stage_age"], row["biological_material"])
    if row["project_key"] in SEX_SPLIT_PROJECTS:
        key += (row["sex"],)
    return key


# Curated experiment IDs, keyed by experiment_key() above:
# (project_key, condition, developmental_stage_age, biological_material), with a
# fifth `sex` field for projects in SEX_SPLIT_PROJECTS.
# The first three IDs are the repo's pre-existing experiments and must not be
# renamed -- config/splits/*.csv, models/, and trained checkpoints key off them.
EXPERIMENT_IDS = {
    ("Kwak2013_dm_S2", "untreated", "cell culture", "S2 cells"):
        "D.melanogaster-S2_PROcap",
    ("Booth2016_yeast_PROcap", "wild type", "log-phase culture", "W303a cells"):
        "S.cerevisiae_PROcap",
    ("Booth2016_yeast_PROcap", "wild type", "log-phase culture", "S. pombe cells"):
        "S.pombe_PROcap",

    ("Kim2018_mm_ESC_BMDM", "untreated", "cell culture", "embryonic stem cells"):
        "M.musculus-ESC_PROcap",
    ("Kim2018_mm_ESC_BMDM", "untreated", "cell culture", "macrophages"):
        "M.musculus-BMDM_PROcap",
    # Cotton, Wen2026. The two species share project/stage/material, so PLOIDY
    # AND GENOTYPE carry the distinction in `condition` -- which is honest
    # curation rather than a key hack, since diploid AA vs tetraploid AD is the
    # comparison the paper makes. (McDonald2024's three species are separated by
    # biological_material instead, because there the tissue genuinely differs.)
    ("Wen2026_cotton_GROcap", "G. arboreum cv. Shixiya1 (diploid AA)",
     "ovule outer integument protoplasts", "ovule outer integument"):
        "G.arboreum-ovule_GROcap",
    ("Wen2026_cotton_GROcap", "G. hirsutum cv. Xuzhou 142 (tetraploid AD)",
     "ovule outer integument protoplasts", "ovule outer integument"):
        "G.hirsutum-ovule_GROcap",

    # Split by sex; see SEX_SPLIT_PROJECTS. Each is 2 biological replicates.
    ("Liver_ChROcap_mm", "untreated", "old (22–25 months)", "frozen liver", "female"):
        "M.musculus-liver-old-female_ChROcap",
    ("Liver_ChROcap_mm", "untreated", "old (22–25 months)", "frozen liver", "male"):
        "M.musculus-liver-old-male_ChROcap",
    ("Liver_ChROcap_mm", "untreated", "young (2–4 months)", "frozen liver", "female"):
        "M.musculus-liver-young-female_ChROcap",
    ("Liver_ChROcap_mm", "untreated", "young (2–4 months)", "frozen liver", "male"):
        "M.musculus-liver-young-male_ChROcap",
    ("Bcell_PROcap_mm", "B18hi", "primary cells", "primary activated B cells"):
        "M.musculus-priB_PROcap",
    ("Bcell_PROcap_mm", "B18hi", "primary cells", "germinal center B cells"):
        "M.musculus-GCB_PROcap",
    ("Lam2013_mm_BMDM_5GRO", "biotin-tag control", "primary cells",
     "bone-marrow-derived macrophages"): "M.musculus-BMDM_5GRO-ctl",
    ("Lam2013_mm_BMDM_5GRO", "BLRP-Rev-Erbα", "primary cells",
     "bone-marrow-derived macrophages"): "M.musculus-BMDM_5GRO-RevErb",
    ("Link2018_mm_BMDM_GROcap", "untreated wild-type/control subset",
     "primary cells", "bone-marrow-derived macrophages"):
        "M.musculus-BMDM_GROcap",

    ("Embryo_PROcap_dm", "wild type", "3–4 h after egg laying", "whole embryos"):
        "D.melanogaster-embryo-3to4h_PROcap",
    ("Embryo_PROcap_dm", "wild type", "6–8 h after egg laying", "whole embryos"):
        "D.melanogaster-embryo-6to8h_PROcap",
    ("LacZ_PROcap_dm", "LacZ knockdown", "cell culture", "S2 cells"):
        "D.melanogaster-S2-LacZKD_PROcap",
    ("Duttke2017_dm_S2", "wild type / control", "cell culture", "S2 cells"):
        "D.melanogaster-S2_5GROcap",

    ("Kruesi2013_ce_GROcap", "N2", "embryo", "whole-animal/embryo material"):
        "C.elegans-embryo_GROcap",
    ("Kruesi2013_ce_GROcap", "N2", "L3", "whole-animal/embryo material"):
        "C.elegans-L3_GROcap",
    ("Kruesi2013_ce_GROcap", "N2 starved", "L1", "whole-animal/embryo material"):
        "C.elegans-L1starved_GROcap",
    ("Kruesi2013_ce_GROcap", "sdc-2(y93); sdc-2 RNAi", "embryo",
     "whole-animal/embryo material"): "C.elegans-embryo-sdc2_GROcap",

    ("Ino80_PROcap_sc", "control", "log-phase culture", "yeast culture"):
        "S.cerevisiae-Ino80ctl_PROcap",
    ("Ino80_PROcap_sc", "Ino80 depletion", "log-phase culture", "yeast culture"):
        "S.cerevisiae-Ino80KD_PROcap",
    ("Spt5_PROcap_sc", "EtOH 1 h control", "log-phase culture",
     "Spt5-AID yeast culture"): "S.cerevisiae-Spt5EtOH_PROcap",
    ("Spt5_PROcap_sc", "IAA 1 h; Spt5 depletion", "log-phase culture",
     "Spt5-AID yeast culture"): "S.cerevisiae-Spt5IAA1h_PROcap",
    ("Spt5_PROcap_sc", "IAA 4 h; Spt5 depletion", "log-phase culture",
     "Spt5-AID yeast culture"): "S.cerevisiae-Spt5IAA4h_PROcap",

    ("Hetzel2016_at_5GRO", "Col-0 wild type", "6-day seedlings", "whole seedlings"):
        "A.thaliana-seedling_5GRO",

    # --- added with the McDonald2024 / Shamie2021 / Tome2018 manifest update ---

    # Tome2018: CoPRO capped fraction, MEFs, one library per heat-shock state.
    # Kept as two experiments because heat shock is a biological condition, the
    # same way the Ino80/Spt5 perturbations are.
    ("Tome2018_mm_CoPRO", "no heat shock", "cell culture",
     "mouse embryonic fibroblasts"): "M.musculus-MEF_CoPRO",
    ("Tome2018_mm_CoPRO", "60 min heat shock at 42 \u00b0C", "cell culture",
     "mouse embryonic fibroblasts"): "M.musculus-MEF-HS_CoPRO",

    # McDonald2024: 5'GRO-seq in three new species. C. reinhardtii pools its two
    # biological replicates into one experiment; the other two have n=1.
    ("McDonald2024_plant_5GRO", "untreated", "late-log culture",
     "liquid culture"): "C.reinhardtii-liquidculture_5GRO",
    ("McDonald2024_plant_5GRO", "untreated", "plate culture",
     "plate culture"): "P.patens-plateculture_5GRO",
    ("McDonald2024_plant_5GRO", "untreated", "stems/leaves after 1 week growth",
     "stems and leaves"): "S.moellendorffii-stemleaf_5GRO",

    # Shamie2021: Chinese hamster GRO-cap atlas. Seven experiments -- CHO-K1
    # pools its two biological replicates, every other tissue is n=1. The
    # matched *_GRO1 samples in the same study are ordinary GRO-seq inputs and
    # are not targets, so they never appear here.
    ("Shamie2021_cg_5GRO", "untreated", "cell culture",
     "CHO-K1 cells"): "C.griseus-CHO_GROcap",
    ("Shamie2021_cg_5GRO", "untreated", "primary cells",
     "bone-marrow-derived macrophages"): "C.griseus-BMDM_GROcap",
    ("Shamie2021_cg_5GRO", "KLA 10 ng/mL, 1 h", "primary cells",
     "bone-marrow-derived macrophages"): "C.griseus-BMDM-KLA1h_GROcap",
    ("Shamie2021_cg_5GRO", "untreated", "not reported",
     "brain"): "C.griseus-brain_GROcap",
    ("Shamie2021_cg_5GRO", "untreated", "not reported",
     "kidney"): "C.griseus-kidney_GROcap",
    ("Shamie2021_cg_5GRO", "untreated", "not reported",
     "liver"): "C.griseus-liver_GROcap",
    ("Shamie2021_cg_5GRO", "untreated", "not reported",
     "lung"): "C.griseus-lung_GROcap",
}

SPECIES = {
    "Mus musculus": "M.musculus",
    "Drosophila melanogaster": "D.melanogaster",
    "Caenorhabditis elegans": "C.elegans",
    "Saccharomyces cerevisiae": "S.cerevisiae",
    "Schizosaccharomyces pombe": "S.pombe",
    "Arabidopsis thaliana": "A.thaliana",
    "Chlamydomonas reinhardtii": "C.reinhardtii",
    "Physcomitrium patens": "P.patens",
    "Selaginella moellendorffii": "S.moellendorffii",
    "Cricetulus griseus": "C.griseus",
    "Gossypium arboreum": "G.arboreum",
    "Gossypium hirsutum": "G.hirsutum",
}

# Legacy hand-built tracks, kept for provenance only. The S. cerevisiae and
# S. pombe entries are GEO-provided *normalized* bigWigs (R1Normed), which is
# precisely why uniform re-mapping was needed: BPNet's count head was being fed
# normalized signal for yeast and raw-read counts for fly.
LEGACY = {
    "D.melanogaster-S2_PROcap": {
        "peaks": "data/GSE233927_D.melanogaster-S2cells-10.tss.bed.gz",
        "pl_bigwig": "data/D.melanogaster-S2_PROcap_pl.bw",
        "mn_bigwig": "data/D.melanogaster-S2_PROcap_mn.bw",
        "gc_negatives": "data/GSE233927_D.melanogaster-S2cells-10.tss.negatives.bed.gz",
        "sequences": "data/dm6.fa",
    },
    "S.cerevisiae_PROcap": {
        "peaks": "data/GSE179468_scer-tsr.bed.gz",
        "pl_bigwig": "data/GSM1974987_3870_7157_12387_C53ARACXX_cerevisiaeW303-aProCap_TTAGGC_R1Normed_plus.bw",
        "mn_bigwig": "data/GSM1974987_3870_7157_12387_C53ARACXX_cerevisiaeW303-aProCap_TTAGGC_R1Normed_minus.bw",
        "gc_negatives": "data/GSE179468_scer-tsr.negatives.bed.gz",
        "sequences": "data/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa",
        "caveat": "GEO-normalized signal (R1Normed), not raw-read counts",
    },
    "S.pombe_PROcap": {
        "peaks": "data/GSE179468_spombe-tsr.bed.gz",
        "pl_bigwig": "data/GSM1974988_3870_7157_12385_C53ARACXX_pombe972h-ProCap_ATCACG_R1Normed_plus.bw",
        "mn_bigwig": "data/GSM1974988_3870_7157_12385_C53ARACXX_pombe972h-ProCap_ATCACG_R1Normed_minus.bw",
        "gc_negatives": "data/GSE179468_spombe-tsr.negatives.bed.gz",
        "sequences": "data/Schizosaccharomyces_pombe.ASM294v2.dna.toplevel.fa",
        "caveat": "GEO-normalized signal (R1Normed), not raw-read counts",
    },
}

# Genome facts -- FASTA path, assembly name, assembly accession -- are read
# from config/genomes.yaml at runtime rather than duplicated here. They used to
# be local FASTA/ASSEMBLY dicts, i.e. a second copy of the same three fields
# that had to be edited in lockstep every time a species was added.

# UMI presence comes from the manifest's umi_len/umi_loc columns, NOT from a
# list here. It used to be a hardcoded dict, transcribed by hand from free text
# buried in the library_layout column -- nothing validated the transcription and
# nothing made it visible next to the sample it describes.
#
# It cannot be cross-checked against the archives: ENA's
# library_construction_protocol was queried for all 45 runs then resolved and mentions a UMI
# for NONE of them, including the three Spt5 experiments that demonstrably have
# one (their read names carry a 10-base tag, e.g.
# SRR29037352.25948720:ACTAGATAGC). So the manifest is authoritative and must be
# curated from the paper or GEO record.
#
# This is default-deny: a library whose UMI was never noted is treated as having
# none, and its PCR duplicates stay in the signal. Getting it wrong the other
# way is worse -- deduplicating a non-UMI PRO-cap library destroys real stacked
# 5' ends -- which is why the default errs this way.


def interleaved_from_rows(rows: list[dict]) -> bool:
    """True if the deposited FASTQ holds interleaved mate pairs.

    A curated override for an ARCHIVE error, which is why it cannot be derived:
    `library_layout` is resolved from ENA precisely because the manifest was
    wrong twice (Liver_ChROcap_mm, Tome2018_mm_CoPRO both claimed paired and are
    single). SRR19034544 is the reverse -- ENA and SRA both say SINGLE, because
    SRA labels every read /1 when it dumps a run as single-end, but the file is
    interleaved pairs. Nothing in the metadata can tell you; only the reads can.

    Left as SINGLE in library_layout on purpose: that is what the archive says
    and it stays recorded. This flag is the measured correction on top, and it is
    what makes the pipeline process the run as paired.
    """
    return any(r.get("deposited_interleaved", "").strip().lower() in ("yes", "true")
               for r in rows)


def adapter_from_rows(rows: list[dict]) -> str:
    """Adapter NAME for a project, from the manifest's `adapter` column.

    Same principle as umi_from_rows: the value is curated evidence in the
    manifest, not a lookup table in code. It was set from measurement --
    `src/qc/read_structure_qc.py` over every FASTQ -- and the survey found the
    adapter perfectly consistent within each project, so one value per project
    is the right granularity. The name resolves to a sequence via
    steps.trim.adapters in config/procap_pipeline.yaml.

    Empty means no adapter was detected and fastp is left to auto-detect. That
    is the honest default for a library with none, and it is what every library
    silently got before this column existed -- which is how 37 of 69 FASTQs came
    to be aligned with the adapter still on.
    """
    names = {r.get("adapter", "").strip() for r in rows} - {""}
    if len(names) > 1:
        raise ValueError(f"conflicting adapter within one project: {sorted(names)}")
    return names.pop() if names else ""


def umi_from_rows(rows: list[dict]) -> dict | None:
    """UMI spec for a project, from the manifest's umi_len/umi_loc columns."""
    lens = {r.get("umi_len", "").strip() for r in rows} - {""}
    if not lens:
        return None
    if len(lens) > 1:
        raise ValueError(f"conflicting umi_len within one project: {sorted(lens)}")
    locs = {r.get("umi_loc", "").strip() for r in rows} - {""}
    return {"enabled": True, "len": int(lens.pop()),
            "loc": (locs.pop() if len(locs) == 1 else "unspecified")}


def q(value: str) -> str:
    """Quote a YAML scalar."""
    return '"{}"'.format(str(value).replace('"', '\\"'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import yaml as _yaml
    with GENOMES.open() as f:
        genomes = _yaml.safe_load(f)["species"]

    with SAMPLES.open() as f:
        samples = list(csv.DictReader(f, delimiter="\t"))
    with RESOLVED.open() as f:
        runs = {(r["project_key"], r["sample_accession"]): r
                for r in csv.DictReader(f, delimiter="\t")}

    experiments: OrderedDict = OrderedDict()
    unmapped = []
    for row in samples:
        key = experiment_key(row)
        if row["target_use"] == "control":
            continue
        exp_id = EXPERIMENT_IDS.get(key)
        if exp_id is None:
            unmapped.append(key)
            continue
        experiments.setdefault(exp_id, {"rows": [], "key": key})
        experiments[exp_id]["rows"].append(row)

    if unmapped:
        for key in sorted(set(unmapped)):
            print(f"  ! no experiment ID mapped for {key}")

    # Controls, grouped by project so they can be attached to its experiments.
    controls = {}
    for row in samples:
        if row["target_use"] == "control":
            controls.setdefault(row["project_key"], []).append(row)

    lines = [
        "# Experiment configuration for PRO-cap model training.",
        "# All paths are relative to REPO_ROOT.",
        "#",
        "# GENERATED by src/data_preprocessing/build_experiment_config.py from",
        "# planning/manifest_samples.tsv + planning/manifest_runs_resolved.tsv.",
        "# Edit the generator, not this file.",
        "#",
        "# One experiment == one species x one biological condition, pooling",
        "# biological replicates. Each experiment trains its own model; conditions",
        "# are NOT multi-tasked into a single model.",
        "#",
        "# `processed` paths are the ENCODE PRO-cap pipeline outputs (see",
        "# config/procap_pipeline.yaml). They do not exist until",
        "# src/data_preprocessing/run_procap_pipeline.py has run; launch.py and",
        "# make_negatives.py skip experiments with missing inputs, so an unmapped",
        "# experiment is simply reported as SKIP rather than failing.",
        "",
        "experiments:",
    ]

    for exp_id, entry in experiments.items():
        rows = entry["rows"]
        project = rows[0]["project_key"]
        species = SPECIES[rows[0]["organism"]]
        resolved = [runs.get((project, r["sample_accession"]), {}) for r in rows]
        run_ids = [x for r in resolved for x in (r.get("run_accession") or "").split(";") if x]
        layouts = sorted({(r.get("library_layout") or "") for r in resolved if r.get("run_accession")})
        bases = sorted({(r.get("match_basis") or "") for r in resolved})
        spike = sorted({r["spike_in"] for r in rows if r["spike_in"] not in ("", "none reported")})

        lines += [
            f"  {exp_id}:",
            f"    biosample: {q(rows[0]['biological_material'] + ' ' + rows[0]['assay_label'])}",
            f"    species: {species}",
            f"    project_key: {project}",
            f"    study_accession: {q(rows[0]['study_accession'])}",
            f"    assay_family: {q(rows[0]['assay_family'])}",
            f"    condition: {q(rows[0]['condition'])}",
            f"    developmental_stage: {q(rows[0]['developmental_stage_age'])}",
            f"    tier: {q(rows[0]['default_include'])}",
            f"    n_biological_replicates: {len(rows)}",
            "    raw:",
            f"      sample_accessions: [{', '.join(q(r['sample_accession']) for r in rows)}]",
        ]
        if run_ids:
            lines.append(f"      runs: [{', '.join(q(r) for r in run_ids)}]")
        else:
            lines.append("      runs: []  # UNRESOLVED: no run accession found in ENA")
        lines.append(f"      library_layout: {q(';'.join(layouts) if layouts else 'unresolved')}")
        lines.append(f"      run_match_basis: {q('; '.join(b for b in bases if b))}")
        if spike:
            lines.append(f"      spike_in: {q('; '.join(spike))}")
        if any("bundled" in r["cap_status"] for r in rows):
            lines.append(
                "      tap_bundled: true  # TAP+ and TAP- runs share one GEO "
                "sample; the run-level split is UNRESOLVED"
            )
        if interleaved_from_rows(rows):
            lines.append(
                "      interleaved: true  # deposited SINGLE but holds "
                "interleaved mate pairs; see the manifest note"
            )
        adapter = adapter_from_rows(rows)
        if adapter:
            lines.append(f"      adapter: {q(adapter)}")
        else:
            lines.append("      # adapter: none detected; fastp auto-detects")
        umi = umi_from_rows(rows)
        if umi:
            lines.append(f"      umi: {{enabled: true, len: {umi['len']}, loc: {q(umi['loc'])}}}")
        else:
            lines.append("      umi: {enabled: false}")

        ctl = controls.get(project, [])
        if ctl:
            lines.append(
                f"      protocol_controls: [{', '.join(q(c['sample_accession']) for c in ctl)}]"
                "  # TAP-/noTAP: background only, never positive labels"
            )

        lines += [
            "    processed:",
            f"      peaks: {PROCAP_DIR}/{exp_id}_peaks.bed.gz",
            f"      pl_bigwig: {PROCAP_DIR}/{exp_id}_pl.bw",
            f"      mn_bigwig: {PROCAP_DIR}/{exp_id}_mn.bw",
            f"      gc_negatives: {PROCAP_DIR}/{exp_id}_negatives.bed.gz",
            f"      sequences: {genomes[species]['fasta']}",
        ]
        # Exclusion list, where one is published for this assembly. Threaded
        # through to extract_loci/PeakGenerator as exclusion_lists.
        bl = genomes[species].get("blacklist")
        if bl:
            lines.append(f"      blacklist: {bl}")
        else:
            lines.append(
                "      # blacklist: none published for this assembly "
                "(see config/genomes.yaml)"
            )
        legacy = LEGACY.get(exp_id)
        if legacy:
            lines.append("    legacy_processed:  # provenance only; not used for training")
            for k, v in legacy.items():
                lines.append(f"      {k}: {v if k != 'caveat' else q(v)}")
        lines.append("")

    # Provenance table, one row per experiment.
    tsv = ["\t".join([
        "dataset_name", "organism", "tissue_condition", "study_accession",
        "sample_accessions", "sra_accessions", "run_match_basis",
        "genome_build_name", "assembly_accession", "data_type", "assay_family",
        "tier", "n_replicates", "remapping_status",
    ])]
    for exp_id, entry in experiments.items():
        rows = entry["rows"]
        project = rows[0]["project_key"]
        resolved = [runs.get((project, r["sample_accession"]), {}) for r in rows]
        run_ids = [x for r in resolved for x in (r.get("run_accession") or "").split(";") if x]
        bases = sorted({(r.get("match_basis") or "") for r in resolved})
        species = SPECIES[rows[0]["organism"]]
        tsv.append("\t".join([
            exp_id,
            rows[0]["organism"],
            f"{rows[0]['biological_material']}; {rows[0]['condition']}; "
            f"{rows[0]['developmental_stage_age']}",
            rows[0]["study_accession"],
            ",".join(r["sample_accession"] for r in rows),
            ",".join(run_ids) if run_ids else "UNRESOLVED",
            "; ".join(b for b in bases if b),
            genomes[species]["assembly"],
            genomes[species]["assembly_accession"],
            rows[0]["assay_label"],
            rows[0]["assay_family"],
            rows[0]["default_include"],
            str(len(rows)),
            "pending ENCODE PRO-cap pipeline" if run_ids else "blocked: run unresolved",
        ]))

    text = "\n".join(lines)
    if args.dry_run:
        print(text)
        print("\n".join(tsv))
    else:
        OUTPUT.write_text(text)
        DATASETS_TSV.write_text("\n".join(tsv) + "\n")
        print(f"Wrote {DATASETS_TSV}: {len(tsv) - 1} rows")
        print(f"Wrote {OUTPUT}: {len(experiments)} experiments")
        print(f"  species: {sorted({SPECIES[e['rows'][0]['organism']] for e in experiments.values()})}")
        n_unresolved = sum(
            1 for e in experiments.values()
            if not any((runs.get((e['rows'][0]['project_key'], r['sample_accession']), {}) or {}).get('run_accession')
                       for r in e['rows'])
        )
        print(f"  experiments with no resolved runs: {n_unresolved}")


if __name__ == "__main__":
    main()
