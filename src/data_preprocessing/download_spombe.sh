#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# Dataset:          S. pombe PRO-cap
# Organism:         Schizosaccharomyces pombe
# Loci source:      GSE179468 TSR annotations
# PRO-cap source:   GSM1974988
# Genome Build:     ASM294v2
# Assembly:         GCA_000002945.2
# Publication:      Booth et al. 2016
# =============================================================================

wkdir=../../data
mkdir -p "$wkdir"
cd "$wkdir"

# TSR loci used for model training.
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE179nnn/GSE179468/suppl/GSE179468%5Fspombe%2Dtsr.txt.gz
zcat GSE179468_spombe-tsr.txt.gz | cut -f2-4 | awk 'NR > 1' | sort-bed - | bgzip \
    > GSE179468_spombe-tsr.bed.gz

# PRO-cap data.
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1974nnn/GSM1974988/suppl/GSM1974988%5F3870%5F7157%5F12385%5FC53ARACXX%5Fpombe972h%2DProCap%5FATCACG%5FR1Normed%5Fminus.bw
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1974nnn/GSM1974988/suppl/GSM1974988%5F3870%5F7157%5F12385%5FC53ARACXX%5Fpombe972h%2DProCap%5FATCACG%5FR1Normed%5Fplus.bw

# Convert to bedGraph, strip the chr prefix to match the Ensembl FASTA, and
# remove mitochondrial reads.
bigWigToBedGraph GSM1974988_3870_7157_12385_C53ARACXX_pombe972h-ProCap_ATCACG_R1Normed_plus.bw \
    GSM1974988_3870_7157_12385_C53ARACXX_pombe972h-ProCap_ATCACG_R1Normed_plus.bg
bigWigToBedGraph GSM1974988_3870_7157_12385_C53ARACXX_pombe972h-ProCap_ATCACG_R1Normed_minus.bw \
    GSM1974988_3870_7157_12385_C53ARACXX_pombe972h-ProCap_ATCACG_R1Normed_minus.bg
sed 's/^chr//' GSM1974988_3870_7157_12385_C53ARACXX_pombe972h-ProCap_ATCACG_R1Normed_plus.bg | \
    grep -v "mt" | sort-bed - \
    > GSM1974988_3870_7157_12385_C53ARACXX_pombe972h-ProCap_ATCACG_R1Normed_plus_nochr.bg
sed 's/^chr//' GSM1974988_3870_7157_12385_C53ARACXX_pombe972h-ProCap_ATCACG_R1Normed_minus.bg | \
    grep -v "mt" | sort-bed - \
    > GSM1974988_3870_7157_12385_C53ARACXX_pombe972h-ProCap_ATCACG_R1Normed_minus_nochr.bg
bedGraphToBigWig GSM1974988_3870_7157_12385_C53ARACXX_pombe972h-ProCap_ATCACG_R1Normed_plus_nochr.bg \
    Schizosaccharomyces_pombe.ASM294v2.dna.toplevel.chromsizes \
    GSM1974988_3870_7157_12385_C53ARACXX_pombe972h-ProCap_ATCACG_R1Normed_plus.bw
bedGraphToBigWig GSM1974988_3870_7157_12385_C53ARACXX_pombe972h-ProCap_ATCACG_R1Normed_minus_nochr.bg \
    Schizosaccharomyces_pombe.ASM294v2.dna.toplevel.chromsizes \
    GSM1974988_3870_7157_12385_C53ARACXX_pombe972h-ProCap_ATCACG_R1Normed_minus.bw

rm -f *.bg
