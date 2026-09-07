#!/usr/bin/env python3
"""Local fix for modisco-lite crashing on a ONE-SIGNED attribution track.

`modisco motifs` dies in sklearn, with a message that names nothing:

    ValueError: Found array with 0 sample(s) (shape=(0,)) while a minimum of 1
                is required.

The cause is upstream and mechanical. `extract_seqlets` splits the smoothed
contribution track into `values[values >= 0]` and `values[values < 0]` and
computes a threshold for each side unconditionally. When NO windowed sum is
negative, `_isotonic_thresholds` is called with an empty `values`:

    w = len(values) / len(null_values)                  # 0 / 10000 = 0.0
    sample_weight = concat([ones(0), ones(n2) * w])     # ALL ZEROS
    model.fit(X, y, sample_weight=sample_weight)

sklearn's IsotonicRegression._build_y drops zero-weight rows (`mask =
sample_weight > 0`), which empties `y`, and the error surfaces four frames
deeper with no reference to attributions or signs.

**modiscolite already supports the outcome; it just cannot reach it.**
`tfmodisco.TFMoDISco` sets `neg_patterns = None` when there are too few
negative seqlets (tfmodisco.py, `if len(neg_seqlets) > min_metacluster_size`),
so a one-signed track is an anticipated result -- the crash happens earlier,
while computing a threshold for a side that has no data.

Measured in this repo 2026-09-07, over all 83 finished attribution files: the
counts head's fraction of NEGATIVE per-position contributions runs from 0.0% to
95.6% across the corpus, and this is the tail of that continuum rather than a
distinct failure. `S.cerevisiae_PROcap` (0.7%) and `S.pombe_PROcap` (0.0%)
crossed zero; the other five S. cerevisiae counts runs sit at 4.4-5.4% with
windowed minima of only -0.13 to -0.34, so they cleared it narrowly. Every
profile run is far from the boundary. Treat this as fragile for any low-variance
counts head, not as two broken experiments -- a retrain or a depth change could
tip the others over.

Three notes on the fix:

  * The sentinel is +/-inf, chosen because it is inert everywhere downstream --
    verified by reading the consumers rather than assumed. `idxs = (tracks >=
    pos) | (tracks <= neg)` selects nothing on the empty side;
    `transformed_neg_threshold` becomes exactly -1.0 via `sign * searchsorted(
    distribution, inf) / len(distribution)`; and `weak_thresh` takes
    `min(transformed_pos, abs(-1.0)) - 0.0001`, so the positive side still
    decides it. `_refine_thresholds` may then reset both to +/-percentile, which
    is also harmless because no window is on the empty side by construction.
  * **The real fix belongs upstream** -- an early return in
    `_isotonic_thresholds`, worth a modisco-lite issue. modisco 2.5.2 is the
    latest release as of 2026-09-07, so there is no version to upgrade to.
  * `patch_one_signed_attributions()` is SELF-RETIRING: it functionally probes
    the installed version with an empty `values` array and does nothing if the
    probe passes, so the shim disappears when modisco-lite is fixed rather than
    shadowing a corrected implementation. It also refuses to install a patch
    that fails its own probe.

Reached only through `src/bpnet/modisco/run_modisco.py`, because `modisco` is
invoked as a CLI: nothing in this repo imports modiscolite, so there is no
in-process call site to patch the way tangermeme_compat.py has one.
"""

import numpy as np

_PATCHED = False
_ORIGINAL = None


def _fixed_isotonic_thresholds(values, null_values, increasing, target_fdr,
                               min_frac_neg=0.95):
    """modiscolite's `_isotonic_thresholds`, returning early on no data.

    An empty `values` means no windowed sum fell on this side of zero, so the
    correct threshold is one nothing can pass: +inf for the positive side,
    -inf for the negative one.
    """
    if len(values) == 0:
        return np.inf if increasing else -np.inf
    return _ORIGINAL(values, null_values, increasing, target_fdr, min_frac_neg)


def _probe_ok(fn) -> bool:
    """Does `fn` survive the empty-`values` call that breaks upstream?"""
    try:
        out = fn(np.array([]), np.array([-3.0, -2.0, -1.0]),
                 increasing=False, target_fdr=0.05)
    except Exception:
        return False
    return np.isscalar(out) or np.ndim(out) == 0


def patch_one_signed_attributions(verbose: bool = False) -> bool:
    """Patch modiscolite so a one-signed attribution track does not crash.

    Returns True if the patch is installed (or already was), False if the
    installed modiscolite needs no patching.
    """
    global _PATCHED, _ORIGINAL
    if _PATCHED:
        return True

    from modiscolite import extract_seqlets

    if _probe_ok(extract_seqlets._isotonic_thresholds):
        if verbose:
            print("modiscolite handles one-signed attributions; not patching",
                  flush=True)
        return False

    _ORIGINAL = extract_seqlets._isotonic_thresholds
    if not _probe_ok(_fixed_isotonic_thresholds):
        _ORIGINAL = None
        raise RuntimeError(
            "the one-signed-attribution patch failed its own probe, so it is "
            "not being installed -- modiscolite's _isotonic_thresholds no "
            "longer has the signature (values, null_values, increasing, "
            "target_fdr, min_frac_neg) this shim wraps. Update "
            "src/modiscolite_compat.py against the installed version."
        )

    extract_seqlets._isotonic_thresholds = _fixed_isotonic_thresholds
    _PATCHED = True
    if verbose:
        print("patched modiscolite._isotonic_thresholds for one-signed "
              "attribution tracks (src/modiscolite_compat.py)", flush=True)
    return True
