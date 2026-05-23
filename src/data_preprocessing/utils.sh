#!/usr/bin/env bash
# =============================================================================
# utils.sh — Shared utility functions for data preprocessing scripts
#
# Source this file at the top of each download script:
#   source "$(dirname "$0")/utils.sh"
#
# Functions:
#   trim_adapters  — homerTools adapter trimming (Illumina TruSeq adapter)
#   star_index     — STAR genome index generation
#   star_align     — STAR read alignment
#   sam_to_bam     — SAM → sorted BAM via samtools
#   bam_to_bw      — sorted BAM → bigWig via bam2bw
#   merge_bams     — merge multiple BAMs via samtools
# =============================================================================

# Set HOMER and STAR on PATH if not already present
export PATH="$PATH:/programs/homer-5.1/bin"
export PATH="/programs/STAR-2.7.11b:$PATH"

# -----------------------------------------------------------------------------
# trim_adapters
# Trim Illumina TruSeq Read 1 adapter from a single FASTQ file.
# Usage:  trim_adapters <input.fastq[.gz]>
# Output: <input.fastq[.gz]>.trimmed (created by homerTools in same directory)
# -----------------------------------------------------------------------------
trim_adapters() {
    local input="$1"
    homerTools trim -3 AGATCGGAAGAGCACACGTCT \
        -mis 2 -minMatchLength 4 -min 20 \
        "$input"
}
export -f trim_adapters

# -----------------------------------------------------------------------------
# star_index
# Build a STAR genome index.
# Usage:  star_index <genome_fa> <star_dir> <sa_index_nbases> <threads>
#   <sa_index_nbases>  value for --genomeSAindexNbases; pass "" to omit
# -----------------------------------------------------------------------------
star_index() {
    local genome_fa="$1"
    local star_dir="$2"
    local sa_index_nbases="$3"
    local threads="$4"
    local sa_arg=""
    if [ -n "$sa_index_nbases" ]; then
        sa_arg="--genomeSAindexNbases $sa_index_nbases"
    fi
    STAR --runThreadN "$threads" \
        --runMode genomeGenerate \
        --genomeDir "$star_dir" \
        --genomeFastaFiles "$genome_fa" \
        $sa_arg
}

# -----------------------------------------------------------------------------
# star_align
# Align a FASTQ to a pre-built STAR index.
# Usage:  star_align <star_dir> <input_fastq> <prefix> <threads>
# Output: <prefix>Aligned.out.sam  (plus STAR log files)
# -----------------------------------------------------------------------------
star_align() {
    local star_dir="$1"
    local input_fastq="$2"
    local prefix="$3"
    local threads="$4"
    STAR --genomeDir "$star_dir" \
        --runThreadN "$threads" \
        --readFilesIn "$input_fastq" \
        --outFileNamePrefix "$prefix" \
        --outSAMstrandField intronMotif \
        --outMultimapperOrder Random \
        --outSAMmultNmax 1 \
        --outFilterMultimapNmax 10000 \
        --limitOutSAMoneReadBytes 10000000
}

# -----------------------------------------------------------------------------
# sam_to_bam
# Convert a STAR SAM output to a sorted BAM.
# Usage:  sam_to_bam <prefix> <threads>
# Input:  <prefix>Aligned.out.sam
# Output: <prefix>.sorted.bam
# -----------------------------------------------------------------------------
sam_to_bam() {
    local prefix="$1"
    local threads="$2"
    samtools view -@ "$threads" -bS "${prefix}Aligned.out.sam" | \
        samtools sort - -o "${prefix}.sorted.bam"
}

# -----------------------------------------------------------------------------
# bam_to_bw
# Convert a sorted BAM to a bigWig using bam2bw.
# Usage:  bam_to_bw <bam> <chromsizes> <name>
# -----------------------------------------------------------------------------
bam_to_bw() {
    bam2bw "$1" -s "$2" -n "$3" -v
}

# -----------------------------------------------------------------------------
# merge_bams
# Merge multiple sorted BAM files into one.
# Usage:  merge_bams <output.bam> <threads> <input1.bam> [input2.bam ...]
# -----------------------------------------------------------------------------
merge_bams() {
    local output="$1"
    local threads="$2"
    shift 2
    samtools merge -@ "$threads" -o "$output" "$@"
}
