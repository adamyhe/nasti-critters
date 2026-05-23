#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# Dataset:          S. cerevisiae PRO-cap
# Organism:         Saccharomyces cerevisiae
# Loci source:      GSE179468 TSR annotations
# PRO-cap source:   GSM1974987
# Genome Build:     R64-1-1
# Assembly:         GCA_000146045.2
# Publication:      Booth et al. 2016
# =============================================================================

wkdir=../../data
mkdir -p "$wkdir"
cd "$wkdir"

# TSR loci used for model training.
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE179nnn/GSE179468/suppl/GSE179468%5Fscer%2Dtsr.txt.gz
zcat GSE179468_scer-tsr.txt.gz | cut -f2-4 | awk 'NR > 1' | sort-bed - | bgzip \
    > GSE179468_scer-tsr.bed.gz

# PRO-cap data.
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1974nnn/GSM1974987/suppl/GSM1974987%5F3870%5F7157%5F12387%5FC53ARACXX%5FcerevisiaeW303%2DaProCap%5FTTAGGC%5FR1Normed%5Fminus.bw
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1974nnn/GSM1974987/suppl/GSM1974987%5F3870%5F7157%5F12387%5FC53ARACXX%5FcerevisiaeW303%2DaProCap%5FTTAGGC%5FR1Normed%5Fplus.bw

# Convert to bedGraph, strip the chr prefix to match the Ensembl FASTA, and
# remove mitochondrial reads.
bigWigToBedGraph GSM1974987_3870_7157_12387_C53ARACXX_cerevisiaeW303-aProCap_TTAGGC_R1Normed_plus.bw \
    GSM1974987_3870_7157_12387_C53ARACXX_cerevisiaeW303-aProCap_TTAGGC_R1Normed_plus.bg
bigWigToBedGraph GSM1974987_3870_7157_12387_C53ARACXX_cerevisiaeW303-aProCap_TTAGGC_R1Normed_minus.bw \
    GSM1974987_3870_7157_12387_C53ARACXX_cerevisiaeW303-aProCap_TTAGGC_R1Normed_minus.bg
sed 's/^chr//' GSM1974987_3870_7157_12387_C53ARACXX_cerevisiaeW303-aProCap_TTAGGC_R1Normed_plus.bg | \
    grep -v "mt" | sort-bed - \
    > GSM1974987_3870_7157_12387_C53ARACXX_cerevisiaeW303-aProCap_TTAGGC_R1Normed_plus_nochr.bg
sed 's/^chr//' GSM1974987_3870_7157_12387_C53ARACXX_cerevisiaeW303-aProCap_TTAGGC_R1Normed_minus.bg | \
    grep -v "mt" | sort-bed - \
    > GSM1974987_3870_7157_12387_C53ARACXX_cerevisiaeW303-aProCap_TTAGGC_R1Normed_minus_nochr.bg
bedGraphToBigWig GSM1974987_3870_7157_12387_C53ARACXX_cerevisiaeW303-aProCap_TTAGGC_R1Normed_plus_nochr.bg \
    Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.chromsizes \
    GSM1974987_3870_7157_12387_C53ARACXX_cerevisiaeW303-aProCap_TTAGGC_R1Normed_plus.bw
bedGraphToBigWig GSM1974987_3870_7157_12387_C53ARACXX_cerevisiaeW303-aProCap_TTAGGC_R1Normed_minus_nochr.bg \
    Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.chromsizes \
    GSM1974987_3870_7157_12387_C53ARACXX_cerevisiaeW303-aProCap_TTAGGC_R1Normed_minus.bw

rm -f *.bg
