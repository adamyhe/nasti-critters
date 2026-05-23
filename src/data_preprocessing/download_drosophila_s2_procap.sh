#!/usr/bin/env bash
# =============================================================================
# Dataset:          D. melanogaster S2 cell PROcap
# Organism:         Drosophila melanogaster
# GEO Accession:    GSE42117
# SRA Accessions:   SRR611829, SRR611830
# Genome Build:     dm6
# Assembly:         GCF_000001215.4
# Data Type:        PROcap
# Publication:      Kwak et al. 2013
# Remapping Status: current
# =============================================================================

# S2 PRO-cap data

wget https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos5/sra-pub-run-32/SRR000/611/SRR611829/SRR611829.2
wget https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos5/sra-pub-run-32/SRR000/611/SRR611830/SRR611830.2
fasterq-dump SRR611829.2
fasterq-dump SRR611830.2
bash ~/proseq2.0/proseq2.0.bsh -SE -i dm6.fa -c dm6.chromsizes -I SRR611829.2.fastq.gz -O SRR611829/ -T SRR611829_tmp/ -G --thread=16 --ADAPT_SE=
bash ~/proseq2.0/proseq2.0.bsh -SE -i dm6.fa -c dm6.chromsizes -I SRR611830.2.fastq.gz -O SRR611830/ -T SRR611830_tmp/ -G --thread=16 --ADAPT_SE=
bigWigMerge SRR611829/SRR611829.2_QC_plus.bw SRR611830/SRR611830.2_QC_plus.bw S2_dm6_PROcap_pl.bg
bigWigMerge -threshold=-1000000 SRR611829/SRR611829.2_QC_minus.bw SRR611830/SRR611830.2_QC_minus.bw S2_dm6_PROcap_mn.bg
sort-bed S2_dm6_PROcap_pl.bg | grep -v -E "rDNA|211" > S2_dm6_PROcap_pl_sort.bg
sort-bed S2_dm6_PROcap_mn.bg | grep -v -E "rDNA|211" > S2_dm6_PROcap_mn_sort.bg
bedGraphToBigWig S2_dm6_PROcap_pl_sort.bg dm6.chromsizes S2_dm6_PROcap_pl.bw
bedGraphToBigWig S2_dm6_PROcap_mn_sort.bg dm6.chromsizes S2_dm6_PROcap_mn.bw
pints_caller --save-to S2_dm6_PROcap_PINTS --file-prefix S2_dm6_PROcap_PINTS --bw-pl S2_dm6_PROcap_pl.bw --bw-mn S2_dm6_PROcap_mn.bw --thread 32
cat pints/*.bed | cut -f1-3 | sort-bed - | bedtools merge > S2_dm6_PINTS_1_merged_peaks.bed

# Run pints_caller
pints_caller --save-to D.melanogaster-S2_PROcap --file-prefix D.melanogaster-S2_PROcap --bw-pl D.melanogaster-S2_PROcap_pl.bw --bw-mn D.melanogaster-S2_PROcap_mn.bw --thread 16
