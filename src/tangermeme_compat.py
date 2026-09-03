#!/usr/bin/env python3
"""Local fixes for tangermeme bugs that only surface on numeric chromosome names.

Three of the twelve species here name their chromosomes with bare digits --
A. thaliana (`1`-`5`), C. reinhardtii (`1`-`17`) and P. patens (`1`-`27`) --
because that is what their Ensembl Plants FASTAs use, and `extract_loci` matches
chromosome names literally, so the names cannot be prettified. Every other
species is saved by an accident of naming: roman numerals, a `chr` prefix, an
`NC_` prefix, or (C. griseus) a single `X` among the digits is enough to make
pandas infer `object` rather than `int64`.

tangermeme reads BED files with `pandas.read_csv(..., usecols=(0,1,2))` and no
`dtype`, so for those three species the chromosome column comes back as `int64`
and then fails to match the string keys taken from the FASTA. It has bitten this
repo twice, in two different functions, and the two fail in OPPOSITE ways:

  * `match.extract_matching_loci` -- `numpy.isin(loci['chrom'], chroms)` is
    False for every row, `loci` becomes empty, and the run dies much later on
    `zero-size array to reduction operation maximum`, naming nothing. SILENT.
    Worked around in src/make_negatives.py by reading the BED there with
    `dtype={0: str}` and passing the DataFrame, which skips tangermeme's read.
  * `io._load_exclusion_zones` -- `exclusion_zones[chrom][start:end] = True`
    raises `KeyError: 1` outright. LOUD, and what this module patches.

The second cannot be worked around at the call site the way the first was:
`_load_exclusion_zones` calls `pandas.read_csv` on each element of
`exclusion_lists` itself, so there is no way to hand it a pre-typed DataFrame,
and no file-level trick makes pandas infer `object` for a purely numeric column.
Rewriting the BED with prefixed names is not an option either -- the published
list was deliberately stripped to bare `1`-`5` to match the FASTA, and an
exclusion list whose names do not match excludes nothing, silently.

The remaining options were to fork `data_loader.py` (which must not happen: it
is byte-identical to procap-atlas's) or to patch the one function. This patches.

**The real fix belongs upstream**, as `dtype={0: str}` in two `read_csv` calls,
and is worth a tangermeme PR. Until then `patch_numeric_chroms()` is
SELF-RETIRING: it functionally probes the installed version with a numeric BED
and does nothing if the probe passes, so the shim disappears the moment
tangermeme is fixed rather than silently shadowing a corrected implementation.
"""

import tempfile
from pathlib import Path

_PATCHED = False


def _fixed_load_exclusion_zones(chrom_lengths, exclusion_lists):
    """tangermeme.io._load_exclusion_zones with the chromosome column typed.

    Byte-for-byte the upstream body apart from `dtype={0: str}`. Keep it that
    way so the diff stays reviewable against whatever version is installed.
    """
    import numpy
    import pandas

    if exclusion_lists is not None:
        exclusion_zones = {}
        for chrom, size in chrom_lengths.items():
            exclusion_zones[chrom] = numpy.zeros(size // 100 + 1, dtype="bool")

        names = "chrom", "start", "end"

        exclusion_list = pandas.concat([
            pandas.read_csv(elist, sep="\t", names=names, header=None,
                            usecols=(0, 1, 2), dtype={0: str})
            for elist in exclusion_lists
        ])

        for _, (chrom, start, end) in exclusion_list.iterrows():
            start = start // 100
            end = end // 100 + 1

            exclusion_zones[chrom][start:end] = True

        return exclusion_zones


def _probe_ok(fn) -> bool:
    """True if `fn` already handles a numerically-named exclusion list."""
    with tempfile.TemporaryDirectory(prefix="tm_probe_") as tmp:
        bed = Path(tmp) / "numeric.bed"
        bed.write_text("1\t100\t200\n5\t300\t400\n")
        try:
            zones = fn({"1": 10_000, "5": 10_000}, [str(bed)])
        except (KeyError, TypeError, ValueError):
            return False
        # Not just "did not raise": the zone must actually be marked, or the
        # names silently failed to match and the list excludes nothing.
        return bool(zones["1"][1:2].all())


def patch_numeric_chroms(verbose: bool = False) -> bool:
    """Patch tangermeme for numeric chromosome names. True if a patch was applied.

    Idempotent, and a no-op where the installed tangermeme is already correct.
    Call it AFTER importing tangermeme and BEFORE the first extract_loci --
    inside main(), like every other heavy import in this repo.
    """
    global _PATCHED
    if _PATCHED:
        return True

    from tangermeme import io as tm_io

    if _probe_ok(tm_io._load_exclusion_zones):
        if verbose:
            print("tangermeme handles numeric chromosome names; no patch needed")
        return False

    if not _probe_ok(_fixed_load_exclusion_zones):
        # The replacement itself does not work against this version, so the
        # signature or the internals moved. Fail rather than install a patch
        # that breaks exclusion lists for every species.
        raise RuntimeError(
            "tangermeme_compat: the numeric-chromosome patch no longer works "
            "against the installed tangermeme. Check whether "
            "tangermeme.io._load_exclusion_zones still takes "
            "(chrom_lengths, exclusion_lists) and returns a dict of per-chrom "
            "boolean arrays, and update src/tangermeme_compat.py."
        )

    tm_io._load_exclusion_zones = _fixed_load_exclusion_zones
    _PATCHED = True
    if verbose:
        print("patched tangermeme.io._load_exclusion_zones for numeric "
              "chromosome names (A.thaliana, C.reinhardtii, P.patens)")
    return True
