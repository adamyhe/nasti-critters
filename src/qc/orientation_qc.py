#!/usr/bin/env python3
"""Validate 5'-end orientation: initiator PWM/logo, and stranded metaplots.

Two assumptions in this pipeline are conventions rather than documented facts,
and both are silent when wrong:

  * which MATE of a paired library carries the RNA 5' end
    (steps.signal.five_prime_mate, default R1 -- matching proseq2.0's
    --RNA5=R1_5prime default, but its Example 3 uses the opposite);
  * which READ STRAND becomes the plus track (reverse_strand in the config).

Get either backwards and the pipeline still runs, still calls peaks, and still
trains -- it just models 3' ends, or models the wrong strand. These are the two
read-outs that catch it.

1. INITIATOR PWM + LOGO around in-peak signal maxima.
   Real initiation sites carry an Inr-like signature: a pyrimidine at -1 and a
   purine at +1 (the TSS base), i.e. the classic Py-Pu dinucleotide, plus
   AT-rich flanks in the compact genomes. If the 5' assignment is inverted you
   are centred on 3' ends instead and the logo is flat or shows an unrelated
   bias. A flat logo is the alarm.

2. SUMMIT-ANCHORED METAPLOT, centred on each peak's own signal maximum.
   Annotation-free, like the PWM, and therefore the metaplot that works where the
   gene models do not -- which is most non-model species here. It costs nothing:
   peak_maxima() is already computed for the PWM.

   READ THE ANTISENSE CHANNEL, NOT THE SENSE ONE. Centring on the maximum makes
   a sense peak at 0 tautological; it is guaranteed by construction and is
   evidence of nothing. The informative signal is where the ANTISENSE weight
   sits: divergent initiation puts it UPSTREAM (negative offset, order -50 to
   -250 bp depending on species). If the two strand tracks are swapped it moves
   DOWNSTREAM, and the sign flip is unambiguous in a way the annotation-anchored
   version never was.

   THE STATISTIC IS log2(upstream / downstream) ANTISENSE, not an argmax, over a
   band that excludes the anchor's own footprint and stops before neighbouring
   promoters dominate. Argmax was tried first and is fragile in precisely the
   case this panel exists for: on a channel with no localized peak it lands
   wherever noise is highest, which across the corpus meant |offset| 360-500 for
   fly, A. thaliana, P. patens, S. pombe and four S. cerevisiae experiments, and
   a few bp off the anchor for cotton and others -- and that second group then
   tripped a naive `argmax > 0` swap test. A position is reported only when the
   peak is prominent and away from the edge; otherwise the verdict says there is
   no localized antisense peak, which for a unidirectional species is the
   correct answer rather than a fault.

   The panel gives antisense ITS OWN Y-AXIS for the same reason. Sharing one
   scale with the anchor spike compressed it onto the axis line and made every
   species look identical -- M.musculus-GCB (peak at -137) and
   S.cerevisiae-Ino80ctl (no localized peak) were indistinguishable by eye.

3. STRANDED METAPLOT around annotated gene TSSs.
   Plus-strand genes should show plus-track signal peaking just downstream of
   the annotated TSS, and minus-strand genes the mirror image on the minus
   track. If the two tracks are swapped, reverse_strand is wrong for that
   experiment. If signal centres on gene 3' ends, the mate choice is wrong.

   SKIPPED ENTIRELY where the species has no annotation (`annotation_url: null`
   in config/genomes.yaml -- currently only C. reinhardtii, whose ASM4749649v1
   assembly has none and never will). load_annotation() returns no TSSs, this
   panel draws "no metaplot", and no flag is raised. Panels 1 and 2 are
   annotation-free and still carry the orientation verdict, which is the reason
   the summit-anchored panel exists.

   DIVERGENT upstream antisense signal is expected in some species and NOT in
   others -- C. elegans promoters are predominantly unidirectional, so absent
   upstream antisense there is biology, not a pipeline fault. Do not read it as
   a pass/fail criterion. That is why verdict() below tests only
   sense-downstream-vs-upstream and antisense-vs-sense TOTALS, never the
   presence of divergence: those two hold regardless of how divergent the
   species' promoters are.

Annotation is used for QC ONLY -- never for training, peak calling, or fold
assignment, so no circularity is introduced into the model.

Readers are pybigtools and pyfaidx deliberately: those are what tangermeme's
extract_loci uses, so this QC sees the data through exactly the same path the
model does. Reading it with a different pair (pyBigWig/pyfastx, which arrive
only as transitive deps of bam2bw/biodatatools) would risk validating something
subtly different from what gets trained on.

Usage:
    uv run python src/qc/orientation_qc.py -e D.melanogaster-S2_PROcap
    uv run python src/qc/orientation_qc.py --all --outdir qc/
"""

import argparse
import gzip
import sys
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


#: Third-party modules this QC needs, with the conda package that supplies each.
#: All four are conda deps in environment.yml, NOT uv ones -- that is what keeps
#: the QC rules inside the Snakemake DAG (see the DAG-boundary note in
#: CLAUDE.md). They are imported lazily, deep in the call stack, which keeps
#: --help instant but means a stale environment surfaces as a ModuleNotFoundError
#: only after the peaks have been read. require_deps() front-loads that check so
#: the failure is immediate and says what to do about it.
REQUIRED = {
    "pybigtools": "pybigtools",
    "pyfaidx": "pyfaidx",
    "logomaker": "logomaker",
    "matplotlib": "matplotlib-base",
    "numpy": "numpy",
    "pandas": "pandas",
}


def require_deps() -> None:
    """Fail early, with the fix, if the environment predates the QC deps."""
    import importlib.util
    absent = sorted({pkg for mod, pkg in REQUIRED.items()
                     if importlib.util.find_spec(mod) is None})
    if not absent:
        return
    sys.exit(
        "Missing QC dependencies: " + ", ".join(absent) + "\n"
        "These are conda packages declared in environment.yml; an environment\n"
        "created before they were added will not have them. Fix with one of:\n"
        "    mamba env update -f environment.yml -n nasti-critters\n"
        "    conda-lock install --name nasti-critters conda-lock.yml\n"
        f"    mamba install -n nasti-critters -c conda-forge -c bioconda {' '.join(absent)}"
    )


def load_annotation(species: str, genome: dict, cache: Path,
                    local: Path | None = None) -> list[tuple]:
    """(chrom, tss, strand) per gene, in the assembly's own chromosome naming.

    `local` is the path to an already-fetched annotation. The Snakemake rule
    passes it, so the DAG owns the download and parallel jobs for the same
    species cannot race for the same file.
    """
    url, fmt = genome["annotation_url"], genome["annotation_format"]
    # No annotation for this assembly at all -- C. reinhardtii under
    # ASM4749649v1. Return no TSSs rather than raising: metaplot() already
    # returns (None, None, 0) for an empty site list, render() draws
    # "no metaplot", and verdict() raises no metaplot line. The initiator logo
    # and the summit-anchored metaplot are annotation-free and still carry the
    # verdict, which is the reason the summit panel exists.
    if url is None and local is None:
        print("  no annotation for this assembly -- TSS metaplot skipped "
              "(logo and summit metaplot are annotation-free)")
        return []
    if local is None:
        cache.mkdir(parents=True, exist_ok=True)
        local = cache / Path(url).name
        if not local.exists():
            print(f"  fetching annotation: {url}")
            urllib.request.urlretrieve(url, local)
    seen, out = set(), []
    with gzip.open(local, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9:
                continue
            feat = p[2]
            keep = (feat == "transcript") if fmt == "gtf" else (feat == "gene")
            if not keep:
                continue
            chrom, start, end, strand, attrs = p[0], int(p[3]), int(p[4]), p[6], p[8]
            tss = start if strand == "+" else end
            key = (chrom, tss, strand)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def peak_maxima(peaks, pl_bw, mn_bw, limit=None):
    """Signal maximum and its strand for each peak -- the presumed TSS base.

    `limit=None` means every peak, which is the default and what you want.
    combine_peaks writes the peak file `sort -k1,1 -k2,2n`, so it is
    COORDINATE-SORTED -- a `limit` therefore takes a genomic PREFIX, not a
    sample. At the old default of 20,000 that capped 16 of the 38 experiments
    then defined, and
    C.griseus-CHO used 30% of its peaks, i.e. roughly one third of the genome in
    lexicographic chromosome order. The PWM's standard error at n=20,000 was
    already negligible, so this is about not estimating a sequence preference
    from one slice of the genome, not about precision.
    """
    import numpy as np
    import pybigtools
    pl, mn = pybigtools.open(str(pl_bw)), pybigtools.open(str(mn_bw))
    chroms = set(pl.chroms()) & set(mn.chroms())
    out = []
    for chrom, start, end in peaks:
        if chrom not in chroms or end <= start:
            continue
        n = pl.chroms()[chrom]
        s, e = max(0, start), min(n, end)
        if e <= s:
            continue
        # fillna=0: pybigtools returns NaN for uncovered bases by default.
        a = np.asarray(pl.values(chrom, s, e, fillna=0), dtype=float)
        b = np.abs(np.asarray(mn.values(chrom, s, e, fillna=0), dtype=float))
        if a.max() >= b.max():
            if a.max() <= 0:
                continue
            out.append((chrom, s + int(a.argmax()), "+"))
        else:
            out.append((chrom, s + int(b.argmax()), "-"))
        if limit and len(out) >= limit:
            break
    pl.close(); mn.close()
    return out


def pwm_around(sites, fasta, flank):
    """Position frequency matrix around sites, strand-corrected."""
    import numpy as np
    import pyfaidx
    fa = pyfaidx.Fasta(str(fasta))
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    width = 2 * flank + 1
    counts = np.zeros((width, 4))
    idx = {b: i for i, b in enumerate("ACGT")}
    n = 0
    for chrom, pos, strand in sites:
        try:
            # pyfaidx slices 0-based half-open, like Python. pos is the maximum
            # itself, so this is [pos-flank, pos+flank] inclusive.
            seq = str(fa[chrom][pos - flank:pos + flank + 1]).upper()
        except Exception:
            continue
        if len(seq) != width:
            continue
        if strand == "-":
            seq = seq.translate(comp)[::-1]
        if any(c not in idx for c in seq):
            continue
        for j, c in enumerate(seq):
            counts[j, idx[c]] += 1
        n += 1
    return (counts / counts.sum(axis=1, keepdims=True)) if n else None, n


def metaplot(tss_list, pl_bw, mn_bw, flank, limit=None):
    """Mean PER-SITE-NORMALISED sense/antisense signal around annotated TSSs.

    Each site's window is divided by its own total (sense + antisense) before
    averaging, so every TSS contributes equal weight. Summing raw profiles let a
    single locus dominate: PRO-cap spans orders of magnitude, and one snRNA or
    ribosomal-protein promoter could outweigh a thousand ordinary ones, which is
    what made these panels spiky and hard to read. Increasing the number of
    sites does not fix that -- the statistic has to change.

    The two strands share ONE denominator per site, deliberately. Normalising
    each separately would equalise them and destroy the sense-versus-antisense
    comparison that the strand-swap check depends on.

    Sites with no signal in the window contribute nothing and are not counted.
    `limit=None` means every TSS.
    """
    import numpy as np
    import pybigtools
    pl, mn = pybigtools.open(str(pl_bw)), pybigtools.open(str(mn_bw))
    sense = np.zeros(2 * flank); anti = np.zeros(2 * flank); n = 0
    for chrom, tss, strand in tss_list:
        if chrom not in pl.chroms():
            continue
        s, e = tss - flank, tss + flank
        if s < 0 or e > pl.chroms()[chrom]:
            continue
        # fillna=0: pybigtools returns NaN for uncovered bases by default.
        a = np.asarray(pl.values(chrom, s, e, fillna=0), dtype=float)
        b = np.abs(np.asarray(mn.values(chrom, s, e, fillna=0), dtype=float))
        if strand == "+":
            site_s, site_a = a, b
        else:
            site_s, site_a = b[::-1], a[::-1]
        denom = site_s.sum() + site_a.sum()
        if denom <= 0:
            continue
        sense += site_s / denom
        anti += site_a / denom
        n += 1
        if limit and n >= limit:
            break
    pl.close(); mn.close()
    return (sense / n, anti / n, n) if n else (None, None, 0)


#: Positions at least this far from the signal maximum are treated as local
#: background when measuring information content. The initiator occupies roughly
#: -4..+4, so 5 keeps the motif out of its own null while staying inside the same
#: promoter context.
BACKGROUND_MIN_OFFSET = 5
#: Offsets where the initiator's information should peak: the Pu at 0, or the Py
#: at -1 if the pyrimidine preference is the stronger of the two.
INITIATOR_OFFSETS = (-1, 0)
#: Information below this counts as low amplitude. PROVISIONAL -- calibrated
#: against the older uniform-background measure.
FLAT_BITS = 0.15


def information_content(pwm, flank):
    """Per-position information in bits, RELATIVE TO LOCAL BASE COMPOSITION.

    Returns (bits_per_position, background_frequencies).

    This used to be `2 + sum(p*log2 p)` -- information against a UNIFORM
    background -- computed separately here and in render(), which is why it now
    lives in one function.

    Uniform background is wrong for a multi-species repo, and wrong in the
    direction opposite to what one might guess: it INFLATES compositionally
    skewed genomes rather than penalising them. A position whose frequencies
    exactly match a 64% GC background still scores `2 - H(background)` =
    0.057 bits of pure artifact, which is visible as the 0.02-0.03 bit flanking
    letters in the C. reinhardtii logo where mouse's flanks sit near 0.005. So
    the old numbers gave GC-rich genomes free apparent signal, and C.
    reinhardtii's 0.13 bits will DROP under this measure, not rise.

    The null is the composition of the flanks of the same windows, not the whole
    genome, and deliberately so: the relevant question is "does this base differ
    from a random base NEAR A PEAK", and promoters are compositionally unlike
    genome average even in an AT-rich genome. Using genome-wide frequencies
    would credit a GC-rich promoter context as signal at every position,
    including the flanks, and inflate the maximum along with them.

    A flat PWM still reads ~0 everywhere, so the FLAT flag keeps working; a real
    initiator in a skewed genome now reads high relative to its own context.
    """
    import numpy as np
    offsets = np.abs(np.arange(-flank, flank + 1))
    outer = offsets >= BACKGROUND_MIN_OFFSET
    if not outer.any():                       # tiny flank: fall back to uniform
        bg = np.full(4, 0.25)
    else:
        bg = pwm[outer].mean(axis=0)
    bg = np.clip(bg, 1e-9, None)
    bg = bg / bg.sum()
    ratio = np.log2(np.clip(pwm, 1e-9, None) / bg)
    bits = np.where(pwm > 0, pwm * ratio, 0.0).sum(axis=1)
    return bits, bg


def summit_metaplot(sites, pl_bw, mn_bw, flank):
    """Metaplot anchored on peak maxima rather than annotated TSSs.

    `sites` is peak_maxima() output, already strand-resolved, so this is just
    metaplot() over a different anchor set -- no extra bigWig pass beyond the
    windows themselves.
    """
    return metaplot(sites, pl_bw, mn_bw, flank)


#: Offsets inside this are the anchor's own footprint and are excluded from
#: every summit statistic. The summit IS the sense maximum, so the first few
#: tens of bp carry the peak's own signal on both strands.
SUMMIT_ANTI_MIN_OFFSET = 20
#: An argmax this close to the BAND's outer edge is not a peak -- it is where
#: the noise happened to be highest on a profile with no peak in it. Measured
#: across the corpus: fly, A. thaliana, P. patens, S. pombe and four
#: S. cerevisiae experiments all reported an "antisense peak" at |offset|
#: 360-500 against a 500 bp flank, which is this artefact, not a reading.
#:
#: MEASURED AGAINST SUMMIT_ANTI_MAX_OFFSET, NOT THE WINDOW FLANK. The first
#: version compared against the flank (500), which was wrong once argmax was
#: restricted to the band: the edge where noise piles up is then 300, not 500,
#: so a "peak" at -257 or +243 passed a check meant to exclude exactly that.
#: The corpus showed three -- MEF_CoPRO +243 and priB_PROcap +241, both with
#: log2 ratios of +0.13 and +0.00 (i.e. no asymmetry at all, so no peak to
#: report), and P.patens -257, which is separately too weak to interpret.
#: Nothing real is near the bound: every genuine peak measured sits at
#: -97 to -159.
SUMMIT_EDGE_MARGIN = 60
#: Outer edge of the band. Divergent initiation sits at -50 to -250 bp; the
#: measured corpus puts every real antisense peak at -97 to -159. Past this,
#: what the window contains is NEIGHBOURING PROMOTERS, not divergence -- which
#: is the dominant effect in the dense genomes (S. cerevisiae runs 1.2-4.1
#: peaks per 2114 bp, so a +-500 window usually holds another peak, and yeast
#: accordingly shows the corpus's highest antisense fraction, 35-43%, with a
#: monotone rise to the edge). PROVISIONAL, like FLAT_BITS: calibrated from
#: this corpus, not from anything principled.
SUMMIT_ANTI_MAX_OFFSET = 300
#: A peak must exceed this multiple of the band's own median to count as
#: localized rather than as a ripple on a flat channel.
SUMMIT_PROMINENCE = 1.5
#: Antisense mean inside the excluded anchor footprint, over the band MAXIMUM,
#: above which the two strands look collapsed onto one point. Against the band
#: median this fired on a broad genuine upstream peak, which still carries real
#: signal at offset 0 while the median is dragged down by the quiet downstream
#: half. Against the maximum it asks the right question: is the summit itself
#: the largest antisense feature anywhere in the window? Measured this way
#: rather than by argmax because the footprint is outside the band, so an
#: on-summit peak would otherwise surface only as an argmax pinned to the band
#: edge -- which is what the first version of this check did.
SUMMIT_ONSUMMIT_FACTOR = 2.0
#: log2(upstream / downstream antisense) at or below this, with enough
#: antisense to be real, is the strand-swap signature.
SUMMIT_SWAP_LOG2 = -1.0
#: Below this share of windowed signal the antisense channel is too weak to
#: interpret either way.
SUMMIT_MIN_ANTI_FRAC = 0.05


def summit_notes(sense, anti, flank):
    """Read of the summit-anchored panel. NEVER gated on annotation quality --
    it uses none.

    Only the antisense channel is interpreted. See the module docstring: the
    sense peak at offset 0 is an artefact of the anchor and says nothing.
    """
    import numpy as np
    if sense is None or anti is None or anti.sum() <= 0:
        return ["summit metaplot: no antisense signal (unidirectional promoters "
                "are normal in some species; not a flag)"]
    x = np.arange(-flank, flank)
    frac = anti.sum() / max(sense.sum() + anti.sum(), 1e-9)

    # THE STATISTIC IS A RATIO, NOT AN ARGMAX. Reporting argmax was fragile in
    # exactly the case this panel is for: on a channel with no localized peak,
    # argmax lands wherever noise is highest -- at the window edge (|offset|
    # 360-500 for fly, A. thaliana, P. patens, S. pombe and four
    # S. cerevisiae experiments) or a few bp off the anchor -- and the second
    # of those then tripped the `at > 0` strand-swap test. Most of the flags
    # raised on the first full run were that artefact.
    #
    # Total antisense weight upstream vs downstream degrades to 0 on a flat
    # channel instead, has no edge behaviour, and is the quantity the
    # strand-swap question actually asks.
    band = ((np.abs(x) >= SUMMIT_ANTI_MIN_OFFSET)
            & (np.abs(x) <= SUMMIT_ANTI_MAX_OFFSET))
    up = float(anti[band & (x < 0)].sum())
    dn = float(anti[band & (x > 0)].sum())
    eps = 1e-12
    lr = float(np.log2((up + eps) / (dn + eps)))

    # Is there a peak at all? Judged inside the same band, and required to be
    # both prominent against the band's own median and away from the edge.
    idx = np.where(band)[0]
    j = int(idx[int(np.argmax(anti[idx]))])
    at = int(x[j])
    baseline = float(np.median(anti[band]))
    prominent = baseline > 0 and float(anti[j]) >= SUMMIT_PROMINENCE * baseline
    localized = (prominent
                 and (SUMMIT_ANTI_MAX_OFFSET - abs(at)) > SUMMIT_EDGE_MARGIN)

    note = (f"summit metaplot: antisense {frac:.1%} of windowed signal, "
            f"upstream/downstream log2 ratio {lr:+.2f}")
    if localized:
        note += f", peak at {at:+d} bp"
    else:
        note += (f", NO localized antisense peak (argmax {at:+d} bp is at the "
                 "window edge or not prominent) -- unidirectional promoters, "
                 "or neighbouring peaks inside the window; not a fault")

    if frac <= SUMMIT_MIN_ANTI_FRAC:
        note += "  (antisense too weak to interpret)"
        return [note]

    # Severity depends on whether a PEAK backs the asymmetry. A real strand
    # swap puts a localized antisense peak downstream; a dense genome's
    # neighbouring promoters produce the same sign with no peak at all, which
    # is a reading to look at rather than a fault to assert. The REVIEW marker
    # is counted separately from flags for exactly this distinction.
    inner = float(anti[np.abs(x) < SUMMIT_ANTI_MIN_OFFSET].mean())
    if lr <= SUMMIT_SWAP_LOG2 and localized:
        note += ("  <-- antisense weight is DOWNSTREAM of the summit and peaks "
                 f"at {at:+d} bp; expected upstream. Suspect the two strand "
                 "tracks are swapped (reverse_strand)")
    elif lr <= SUMMIT_SWAP_LOG2:
        note += ("  <-- REVIEW: antisense weight is downstream but with no "
                 "localized peak -- in a dense genome this is neighbouring "
                 "promoters inside the window, not necessarily a strand swap")
    elif float(anti[j]) > 0 and inner >= SUMMIT_ONSUMMIT_FACTOR * float(anti[j]):
        note += ("  <-- antisense is concentrated ON the summit rather than "
                 "upstream; suspect an unextracted UMI or a 5' offset "
                 "collapsing the two strands")
    return [note]


def render(exp_id, pwm, n_pwm, sense, anti, n_tss, flank_pwm, flank_meta, outdir,
           s_sense=None, s_anti=None, n_summit=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    fig, axes = plt.subplots(1, 3, figsize=(19, 4))
    if pwm is not None:
        import logomaker
        bits, _ = information_content(pwm, flank_pwm)
        info = pwm * bits[:, None]
        df = pd.DataFrame(info, columns=list("ACGT"),
                          index=range(-flank_pwm, flank_pwm + 1))
        logomaker.Logo(df, ax=axes[0])
        axes[0].axvline(0, color="k", lw=0.8, ls="--")
        axes[0].set_title(f"initiator, {n_pwm:,} peak maxima\n"
                          "expect Py at -1, Pu at 0; flat = 5' assignment wrong")
        axes[0].set_xlabel("position relative to signal maximum")
        axes[0].set_ylabel("bits vs local background")
    else:
        axes[0].text(.5, .5, "no PWM", ha="center"); axes[0].axis("off")

    if sense is not None:
        x = np.arange(-flank_meta, flank_meta)
        axes[1].plot(x, sense, label="sense", lw=1.2)
        axes[1].plot(x, -anti, label="antisense", lw=1.2)
        axes[1].axvline(0, color="k", lw=0.8, ls="--")
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_title(f"stranded signal at {n_tss:,} annotated TSSs\n"
                          "expect sense peak just downstream of 0")
        axes[1].set_xlabel("position relative to annotated TSS")
        axes[1].set_ylabel("mean fraction of site signal")
        axes[1].legend(frameon=False)
    else:
        axes[1].text(.5, .5, "no metaplot", ha="center"); axes[1].axis("off")

    if s_sense is not None:
        x = np.arange(-flank_meta, flank_meta)
        # ANTISENSE OWNS THE Y-AXIS, and this is the whole reason the panel is
        # worth looking at. Sharing one scale with sense made it useless: the
        # sense spike at 0 is guaranteed by the anchor and set the scale, so the
        # antisense channel was compressed onto the axis line and EVERY species
        # looked like a single spike. Checked on the first full run --
        # M.musculus-GCB (antisense peak at -137) and S.cerevisiae-Ino80ctl (no
        # localized peak at all) were indistinguishable by eye.
        #
        # Sense is kept as faint context on a twin axis, unlabelled, because it
        # confirms the anchor worked and nothing more.
        axes[2].plot(x, s_anti, color="tab:orange", lw=1, label="antisense")
        # Shade the band the statistic uses, so the printed ratio and the
        # picture cannot disagree about what was measured.
        for sign in (-1, 1):
            axes[2].axvspan(sign * SUMMIT_ANTI_MIN_OFFSET, sign * flank_meta,
                            color="grey", alpha=0.07)
        axes[2].axvline(0, color="k", lw=0.8, ls="--")
        axes[2].set_xlabel("offset from peak summit (bp)")
        axes[2].set_ylabel("mean fraction of site signal (antisense)")
        ctx = axes[2].twinx()
        ctx.plot(x, s_sense, color="tab:blue", lw=0.8, alpha=0.3,
                 label="sense (anchor, own scale)")
        ctx.set_yticks([])
        h1, l1 = axes[2].get_legend_handles_labels()
        h2, l2 = ctx.get_legend_handles_labels()
        axes[2].legend(h1 + h2, l1 + l2, fontsize=8)
        # The title says what to look at, because the obvious feature is the
        # uninformative one: sense peaks at 0 by construction.
        axes[2].set_title(f"summit-anchored, {n_summit:,} peaks (annotation-free)\n"
                          "read the ANTISENSE trace: upstream = OK, downstream = strands swapped")
    else:
        axes[2].text(.5, .5, "no summit metaplot", ha="center"); axes[2].axis("off")

    fig.suptitle(exp_id)
    fig.tight_layout()
    out = outdir / f"{exp_id}.orientation_qc.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def verdict(pwm, sense, anti, flank_pwm, flank_meta, tss_anchored=True):
    """Cheap automated read of the two panels.

    `tss_anchored` (config/genomes.yaml `annotation_tss_anchored`) gates whether
    the METAPLOT lines may raise a flag. Where the annotated gene start is not
    the TSS -- the ATG in S. cerevisiae, the post-trans-splicing 5' end in
    C. elegans, predicted CDS models in S. moellendorffii -- the metaplot
    measures the annotation's reference point, not the pipeline, and its flags
    are false. In the first full run the metaplot was the ONLY source of flags
    and all 11 were false positives, split exactly along species lines: 6/6
    S. cerevisiae, 4/4 C. elegans, 1/1 S. moellendorffii, 0/27 elsewhere. The
    flagged set also had HIGHER mean initiator information (0.74 vs 0.66 bits)
    than the clean set, which is impossible if the metaplot tracked data quality.

    The initiator PWM check is never gated: peak_maxima + pwm_around use only
    peaks, bigWigs and the FASTA, so it is annotation-free and is the orientation
    test that survives bad annotation.
    """
    import numpy as np
    notes = []
    if pwm is not None:
        info, bg = information_content(pwm, flank_pwm)
        peak_bits = float(info.max())
        peak_at = int(info.argmax()) - flank_pwm
        placed = peak_at in INITIATOR_OFFSETS
        # LOW AMPLITUDE ALONE IS NOT A FAULT, and treating it as one produced a
        # false positive on the first run that used this measure.
        # D.melanogaster-S2_5GROcap scored 0.13 bits and was flagged, but its
        # logo is a textbook C at -1 and A at 0 and its metaplot is sharp: the
        # motif is present and correctly placed, just diluted across 42,853 peak
        # maxima, many from low-confidence peaks whose maxima are random.
        # Amplitude measures peak-set quality; POSITION is what says whether the
        # 5' assignment is right. So the flag needs both, and where the motif is
        # correctly placed the note says so instead of raising.
        #
        # This is deliberately still a position/threshold check and not a score.
        # A numeric TSS-enrichment score was added to the metaplot once and
        # removed again -- see the metaplot notes below. Do not grow this either.
        notes.append(
            f"max information {peak_bits:.2f} bits vs local background "
            f"(A/C/G/T {'/'.join(f'{x:.2f}' for x in bg)}) at offset {peak_at:+d}"
            + ("" if peak_bits >= FLAT_BITS or placed else
               "  <-- FLAT and MISPLACED, suspect 5' assignment")
            + ("  (low amplitude but correctly placed: dilution by a large or "
               "noisy peak set, not a 5' error)"
               if peak_bits < FLAT_BITS and placed else ""))
    if sense is not None:
        # These notes are a coarse aid, not the point. THE PLOT IS THE ARTIFACT:
        # the metaplot exists so a human can see whether signal sits where it
        # should relative to the annotated TSS. Do not grow this into a scoring
        # system -- a numeric enrichment score was added here and removed again,
        # because deciding placement by eye is the actual requirement.
        def metanote(text, suspicious, hint):
            """A suspicious reading is ALWAYS surfaced. `tss_anchored` sets its
            severity, not its visibility.

            This used to drop the "<--" marker entirely where the annotation is not
            TSS-anchored, which meant such a reading was invisible to n_flagged and to
            anything grepping for "<--": present in the prose, absent from triage. The
            gate exists because that check went 0-for-11 on those species, so it must not
            raise a hard flag -- but a silent downgrade is the wrong way to say so. Emit
            REVIEW instead: it counts, it greps, and it says why it might be wrong.
            """
            if not suspicious:
                return text
            if tss_anchored:
                return f"{text}  <-- {hint}"
            return (f"{text}  <-- REVIEW: {hint} -- but this metaplot may be measuring the "
                    "ANNOTATION here (annotation_tss_anchored: false), where the check has "
                    "a history of false positives. Confirm against the initiator PWM and "
                    "the summit-anchored panel, both of which are annotation-free.")

        half = len(sense) // 2
        down, up = sense[half:].sum(), sense[:half].sum()
        notes.append(metanote(
            f"sense downstream/upstream {down / max(up, 1e-9):.2f}",
            down <= up, "signal is UPSTREAM, suspect strand or mate"))
        # The strand-swap test specifically. NOT a divergence measure -- absent
        # antisense is expected where promoters are unidirectional (C. elegans)
        # and is never flagged.
        notes.append(metanote(
            f"antisense/sense {anti.sum() / max(sense.sum(), 1e-9):.2f}",
            anti.sum() > sense.sum(), "suspect reverse_strand")
            if anti.sum() > sense.sum() else
            f"antisense/sense {anti.sum() / max(sense.sum(), 1e-9):.2f}")
        if not tss_anchored:
            notes.append("metaplot is ADVISORY for this species "
                         "(annotation_tss_anchored: false); any line above marked REVIEW "
                         "is suspicious but may reflect the annotation, not the pipeline")
    return notes


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-e", "--experiments", nargs="+", action="extend", default=[])
    ap.add_argument("--all", action="store_true", help="every experiment with data")
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "qc")
    ap.add_argument("--flank-pwm", type=int, default=10)
    ap.add_argument("--flank-meta", type=int, default=500)
    # Default to no cap. Both former caps took a prefix of a sorted file rather
    # than a sample; pass a value only to subsample deliberately.
    ap.add_argument("--max-peaks", type=int, default=None,
                    help="cap peaks used for the logo (default: all)")
    ap.add_argument("--max-tss", type=int, default=None,
                    help="cap annotated TSSs used for the metaplot (default: all)")
    ap.add_argument("--annotation", type=Path, default=None,
                    help="pre-fetched annotation (the Snakemake rule supplies this)")
    ap.add_argument("--tsv", type=Path, default=None,
                    help="write the verdict lines as TSV as well as printing them")
    args = ap.parse_args()

    from experiments import Experiment, experiment_ids
    import yaml
    with open(REPO_ROOT / "config" / "genomes.yaml") as f:
        genomes = yaml.safe_load(f)["species"]

    selected = args.experiments or (list(experiment_ids()) if args.all else [])
    if not selected:
        print("Nothing selected; pass -e EXPERIMENT or --all", file=sys.stderr)
        sys.exit(1)
    require_deps()

    args.outdir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    ran, skipped = 0, []
    for exp_id in selected:
        exp = Experiment.load(exp_id)
        # QC_INPUTS, not `.missing`: the latter also requires gc_negatives,
        # which src/make_negatives.py produces OUTSIDE the Snakemake DAG. Gating
        # on it meant every experiment was skipped during a pipeline run --
        # while still exiting 0, so Snakemake saw a successful job with no
        # outputs and reported a confusing MissingOutputException instead of the
        # real reason.
        gaps = exp.missing_paths(Experiment.QC_INPUTS)
        if gaps:
            print(f"SKIP {exp_id}: missing {gaps[0]}")
            skipped.append(exp_id)
            continue
        print(f"\n=== {exp_id} ({exp.species})")
        peaks = pd.read_csv(exp.peaks, sep="\t", usecols=[0, 1, 2], header=None,
                            names=["chrom", "start", "end"], dtype={"chrom": str},
                            comment="#")
        sites = peak_maxima(list(peaks.itertuples(index=False, name=None)),
                            exp.signals[0], exp.signals[1], args.max_peaks)
        pwm, n_pwm = pwm_around(sites, exp.sequences, args.flank_pwm)
        print(f"  peak maxima used: {n_pwm:,}")

        tss = load_annotation(exp.species, genomes[exp.species],
                              REPO_ROOT / "data" / "annotation", args.annotation)
        sense, anti, n_tss = metaplot(tss, exp.signals[0], exp.signals[1],
                                      args.flank_meta, args.max_tss)
        print(f"  annotated TSSs used: {n_tss:,}")
        s_sense, s_anti, n_summit = summit_metaplot(
            sites, exp.signals[0], exp.signals[1], args.flank_meta)
        lines = summit_notes(s_sense, s_anti, args.flank_meta)
        lines += verdict(pwm, sense, anti, args.flank_pwm, args.flank_meta,
                        tss_anchored=genomes[exp.species].get(
                            "annotation_tss_anchored", True))
        for line in lines:
            print(f"  {line}")
        out = render(exp_id, pwm, n_pwm, sense, anti,
                     n_tss, args.flank_pwm, args.flank_meta, args.outdir,
                     s_sense=s_sense, s_anti=s_anti, n_summit=n_summit)
        print(f"  wrote {out}")
        if args.tsv:
            args.tsv.parent.mkdir(parents=True, exist_ok=True)
            # Two tallies, not one. A REVIEW item is surfaced and greppable like any
            # other "<--" line, but it must not inflate n_flagged: the whole point of
            # separating them is that the flagged count stays a count of things believed
            # to be real.
            review = [l for l in lines if "<-- REVIEW:" in l]
            flagged = [l for l in lines if "<--" in l and "<-- REVIEW:" not in l]
            with open(args.tsv, "w") as f:
                f.write("experiment\tspecies\tn_peak_maxima\tn_tss\t"
                        "n_flagged\tn_review\tnotes\n")
                f.write(f"{exp_id}\t{exp.species}\t{n_pwm}\t{n_tss}\t"
                        f"{len(flagged)}\t{len(review)}\t{'; '.join(lines)}\n")
            print(f"  wrote {args.tsv}")
        ran += 1
    print(f"\n{ran} experiment(s) plotted into {args.outdir}")
    # Exiting 0 having produced nothing is the worst outcome: the caller cannot
    # tell success from silent failure. When specific experiments were REQUESTED
    # (-e, i.e. how the DAG invokes this) and none could run, that is an error.
    # Under --all, skipping not-yet-processed experiments is expected.
    if skipped and not args.all and ran == 0:
        print(f"ERROR: nothing produced for {', '.join(skipped)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
