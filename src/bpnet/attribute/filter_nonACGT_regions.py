import argparse

import numpy as np
import pandas as pd
import pyfaidx
import tqdm

# from tangermeme.io import extract_loci


def filter_nonACGT_regions(bed_fp, fa_fp, in_window=2114, verbose=False):
    # pyfaidx, and this exact coordinate arithmetic, to match
    # procap-atlas/src/preprocess/_filter_nonACGT_regions.py. Two differences
    # from the version this replaces: it used pyfastx (1-based, inclusive-end)
    # where upstream and tangermeme both use pyfaidx, and it derived `end` from
    # the centre rather than from the clamped `start`, so a region near a contig
    # start produced a short window and was dropped instead of being shifted.
    snp_bed = pd.read_csv(bed_fp, sep="\t", header=None)
    fa = pyfaidx.Fasta(fa_fp)
    chroms = set(fa.keys())
    wholesome = []
    for row in tqdm.tqdm(
        snp_bed.itertuples(), total=snp_bed.shape[0], disable=not verbose
    ):
        chrom = str(row[1])
        center = (row[2] + row[3]) // 2
        start = max(0, center - in_window // 2)
        end = start + in_window
        if chrom in chroms:
            seq = str(fa[chrom][start:end]).upper()
            is_wholesome = all([c in "ACGT" for c in seq]) and len(seq) == in_window
        else:
            is_wholesome = False
        wholesome.append(is_wholesome)

    print(
        f"Filtered out {sum(~np.array(wholesome))} due to non-ACGT characters, "
        f"length != {in_window}, or invalid chromosome."
    )
    return snp_bed[wholesome]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bed_fp", type=str, required=True)
    parser.add_argument("-f", "--fa_fp", type=str, required=True)
    parser.add_argument("-o", "--out_fp", type=str, required=True)
    parser.add_argument("-w", "--in_window", type=int, default=2114)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    filter_bed = filter_nonACGT_regions(
        args.bed_fp, args.fa_fp, args.in_window, args.verbose
    )
    filter_bed.to_csv(args.out_fp, sep="\t", index=False, header=False)
