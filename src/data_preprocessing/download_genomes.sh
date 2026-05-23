#!/usr/bin/env bash
set -euo pipefail

wkdir=../../data
mkdir -p "$wkdir"

# D. melanogaster dm6
wget https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz -P "$wkdir"
gunzip "$wkdir/dm6.fa.gz"
samtools faidx "$wkdir/dm6.fa"
cut -f1-2 "$wkdir/dm6.fa.fai" > "$wkdir/dm6.chromsizes"

# S. cerevisiae R64-1-1
wget http://ftp.ensemblgenomes.org//pub/release-35/fungi/fasta/saccharomyces_cerevisiae/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz -P "$wkdir"
gunzip "$wkdir/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz"
samtools faidx "$wkdir/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"
cut -f1-2 "$wkdir/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.fai" > "$wkdir/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.chromsizes"

# S. pombe ASM294v2
wget http://ftp.ensemblgenomes.org/pub/release-60/fungi/fasta/schizosaccharomyces_pombe/dna/Schizosaccharomyces_pombe.ASM294v2.dna.toplevel.fa.gz -P "$wkdir"
gunzip "$wkdir/Schizosaccharomyces_pombe.ASM294v2.dna.toplevel.fa.gz"
samtools faidx "$wkdir/Schizosaccharomyces_pombe.ASM294v2.dna.toplevel.fa"
cut -f1-2 "$wkdir/Schizosaccharomyces_pombe.ASM294v2.dna.toplevel.fa.fai" > "$wkdir/Schizosaccharomyces_pombe.ASM294v2.dna.toplevel.chromsizes"
