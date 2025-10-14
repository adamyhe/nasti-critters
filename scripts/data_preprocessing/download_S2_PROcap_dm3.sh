# Data from Kwak et al. PRO-seq paper (https://www.science.org/doi/10.1126/science.1229386)

wget https://hgdownload.soe.ucsc.edu/goldenPath/dm3/bigZips/dm3.fa.gz \
    -O ../../data/dm3.fa.gz
gunzip ../../data/dm3.fa.gz
samtools faidx ../../data/dm3.fa
cut -d $'\t' -f 1,2 ../../data/dm3.fa.fai > ../../data/dm3.chromsizes

wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1032nnn/GSM1032759/suppl/GSM1032759%5FPROcap.pl.bedgraph.gz \
    -P ../../data/
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1032nnn/GSM1032759/suppl/GSM1032759%5FPROcap.mn.bedgraph.gz \
    -P ../../data/

zcat ../../data/GSM1032759_PROcap.pl.bedgraph.gz | \
    tail -n +2 | \
    sort-bed - | \
    bgzip > ../../data/GSM1032759_PROcap.pl_sort.bedgraph
zcat ../../data/GSM1032759_PROcap.mn.bedgraph.gz | \
    tail -n +2 | \
    sort-bed - | \
    > ../../data/GSM1032759_PROcap.mn_sort.bedgraph

bedGraphToBigWig \
    ../../data/GSM1032759_PROcap.pl_sort.bedgraph \
    ../../data/dm3.chromsizes \
    ../../data/S2_dm3_PROcap_pl.bw
bedGraphToBigWig \
    ../../data/GSM1032759_PROcap.mn_sort.bedgraph \
    ../../data/dm3.chromsizes \
    ../../data/S2_dm3_PROcap_mn.bw
rm ../../data/*.bg*
rm ../../data/*.bedgraph*

pints_caller \
    --save-to ../../data/ \
    --file-prefix S2_dm3_PROcap \
    --bw-pl ../../data/S2_dm3_PROcap_pl.bw \
    --bw-mn ../../data/S2_dm3_PROcap_mn.bw \
    --thread 16 \
    --exp-type PROcap

cat ../../data/S2_dm3_PROcap_1_*_peaks.bed | \
    cut -f1-3 | \
    sort-bed - | \
    bedtools merge -i - | \
    bgzip > ../../data/S2_dm3_PROcap_merged_peaks.bed.gz
    
zcat ../../data/S2_dm3_PROcap_merged_peaks.bed.gz | \
    grep -Ev "Het|chrU" | \
    bgzip > ../../data/S2_dm3_PROcap_merged_peaks_main.bed.gz

bpnet negatives \
  -i ../../data/S2_dm3_PROcap_merged_peaks_main.bed.gz \
  -f ../../data/dm3.fa \
  -o ../../data/S2_dm3_PROcap_negatives.bed.gz \
  -v
