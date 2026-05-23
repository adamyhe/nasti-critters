# dm-procap-models

Sequence-to-function models of PRO-cap initiation profiles for:

- *Drosophila melanogaster* S2 cells
- *Saccharomyces cerevisiae* log-phase cells
- *Schizosaccharomyces pombe* log-phase cells

Human K562 PRO-cap configs from the csRNANet workspace are intentionally not
included here.

## Install Python Dependencies

```bash
pip install -r requirements.txt
```

The training scripts expect the same external command-line tools used by
csRNANet preprocessing: STAR, homerTools, samtools, bedtools/bedops,
SRA Toolkit, UCSC BigWig utilities, `pints_caller`, and `bpnet`.

## Data Preprocessing

Download genomes first:

```bash
bash src/data_preprocessing/download_genomes.sh
```

Then run the dataset-specific scripts as needed:

```bash
bash src/data_preprocessing/download_drosophila_s2_procap.sh
bash src/data_preprocessing/download_scer.sh
bash src/data_preprocessing/download_spombe.sh
```

Generate GC-matched negatives from `configs/experiment_config.yaml`:

```bash
python src/data_preprocessing/make_negatives.py
```

To process only one dataset:

```bash
python src/data_preprocessing/make_negatives.py -e S.cerevisiae_PROcap
```

S. pombe uses random peak-level folds rather than chromosome-held-out folds,
because it only has three chromosomes. Generate those fold assignments after
the S. pombe TSR BED exists:

```bash
python src/data_preprocessing/make_pombe_random_splits.py
```

That writes `configs/splits/S.pombe_random_fold_assignments.csv`; the BPNet
training script detects this file automatically and uses random peak splits for
`S.pombe_PROcap`.

## Train BPNet Models

Run from the repository root in the `torch` conda environment:

```bash
python src/bpnet/fit/fit_bpnet.py -e D.melanogaster-S2_PROcap -f 0
python src/bpnet/fit/fit_bpnet.py -e S.cerevisiae_PROcap -f 0
python src/bpnet/fit/fit_bpnet.py -e S.pombe_PROcap -f 0
```

Submit all configured experiments and folds on SLURM:

```bash
python src/bpnet/fit/launch.py --dry-run
python src/bpnet/fit/launch.py
```
