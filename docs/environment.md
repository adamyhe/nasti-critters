# Environment and dependencies

Detailed record for the two-environment split, the Sherlock wheel ceiling and the package-name traps.
`CLAUDE.md` carries the short rules; this file carries the measurements behind them. Do not re-derive a
pin or re-litigate a rejected option without re-running the check named beside it.


## The two environments

Training and benchmarking require a CUDA GPU (`fit_bpnet.py` hard-codes `.to("cuda")`). Local development is
fine for reading/editing config-parsing logic; anything that imports torch and loads loci needs the cluster.

- **Two environments, split by job — not by language.** `environment.yml` (+ committed
  `conda-lock.yml`, 198 packages, linux-64) owns the **whole preprocessing DAG**, fetch through peaks:
  the external binaries,
  `snakemake`, and the only three libraries those scripts import (`pyyaml`, `pandas`, `numpy`). Nothing in
  `src/data_preprocessing/`, `src/experiments.py` or `config/write_split_csvs.py` imports `torch`,
  `bpnet-lite`, `cherimoya` or `tangermeme` — verified by loading each entry point and inspecting
  `sys.modules` — so label generation needs no GPU stack and no venv, and runs on a CPU or transfer node.
  `pyproject.toml` (+ `uv.lock`, 85 packages) owns **model work**: training, benchmarking, attribution.
  `snakemake` is deliberately NOT a uv dependency; do not add preprocessing-only deps there.
  `numpy`/`pandas`/`pyyaml` appear in both on purpose, because scripts on both sides import them.
- **GC-matched negatives are model prep, not preprocessing, and are deliberately outside the DAG.**
  They are a training input rather than a label, and `src/make_negatives.py` imports
  **`tangermeme`, which is PyPI-only** — checked 2026-09-02, HTTP 404 on both `bioconda/tangermeme` and
  `conda-forge/tangermeme` — so it can never move into `environment.yml` the way `umi_tools` and
  `pypints` did.
  **The package behind that argument changed and the conclusion did not.** It used to be bpnet-lite,
  because the script shelled out to `bpnet negatives`; it now calls
  `tangermeme.match.extract_matching_loci` directly and does not import bpnet-lite at all. Both are
  PyPI-only, so the DAG boundary is exactly where it was — do not read "no more bpnet-lite" as
  permission to move this into `workflow/Snakefile`.
  Keeping it out of `workflow/Snakefile` is what makes the claim above true: the entire pipeline runs
  from the mamba env with no venv and no GPU stack. Run it separately, from the venv, before training —
  `uv run python src/make_negatives.py`. Do not add it back as a rule.
- **The venv side must not shell out to a conda binary, and `make_negatives.py` did — three times.**
  It failed with `No such file or directory: 'bigWigToBedGraph'`, because that binary is an
  `environment.yml` package while this script runs from the uv venv. The other two would have been the
  next two failures: `bgzip` in `filter_peaks`, which runs unconditionally, and a `samtools faidx`
  fallback. All three are now in-process — `pybigtools` for the strand merge, the `gzip` module for the
  BED, `pyfaidx` for the index. A fourth call, `bpnet negatives` itself, went the same way for a
  different reason — see the negatives bullet under "Data conventions to preserve" in `CLAUDE.md` — so **the script now makes no
  subprocess calls at all.** Keep it that way: when adding to it, check which environment provides what
  you are calling.
- **When auditing that boundary, grep for `subprocess`, `run([` and bare command names in `shell:`
  blocks — not just imports.** An earlier version of this file claimed preprocessing needed no uv,
  verified by loading each script and checking `sys.modules`. That covers what a script *imports* and
  misses what it *shells out to*, which is exactly what `make_negatives.py` does. `conda-lock.yml` is `linux-64` only
  and that is deliberate — `star 2.7.11a` has no `osx-arm64` build, so adding that platform fails to
  solve, and preprocessing needs the cluster anyway. `environment.yml` stays the source of truth;
  regenerate with
  `uvx --from conda-lock conda-lock lock -f environment.yml -p linux-64 --lockfile conda-lock.yml`
  and install with `conda-lock install --name nasti-critters conda-lock.yml`. `requirements.txt` is gone. Activate the venv last so its interpreter wins:
  `mamba env create -f environment.yml && mamba activate nasti-critters && uv sync --extra torch && source .venv/bin/activate` — drop `--extra torch` on a CPU-only machine such as Sherlock.
- Version pins trace the ENCODE spec: `fastp=0.23.4`, `star=2.7.11a`, `samtools=1.18` (conda);
  `pyPINTS==1.1.10`, `umi-tools==1.1.5` (uv).
- **TORCH IS NOT A BASE DEPENDENCY — the base set is torch-free and there is ONE extra.**

      uv sync                 # torch-free: tfmodisco, analysis, every launcher
      uv sync --extra torch   # adds training, benchmarking, attribution

  Verified after the split: the default resolution contains **0** torch-carrying packages and
  `--extra torch` adds **21** (torch, triton, the `nvidia-*` set, bpnet-lite, tangermeme, cherimoya).
  **The split is by DEPENDENCY, not by task**, which is the part that trips people up: `tangermeme`
  requires `torch>=2.0` and `bpnet-lite` `torch>=1.9.0`, so anything importing either is on the torch
  side even where the science does not obviously involve a model — `fit_*`, `benchmark_*`,
  `attribute.py`, `make_negatives.py`, and `filter_nonACGT_regions.py --save-ohe` (the filtering is
  torch-free; only the one-hot encoding, which goes through `extract_loci`, is not). Before adding a
  base dependency, run `pip download <pkg> --no-deps` and read its `Requires-Dist`.
- **Sherlock takes the base set and runs the CPU-demanding jobs; GPU work happens elsewhere.** Its glibc
  tier has no manylinux wheel for torch above **2.6.0**, newer versions fall back to an sdist build that
  fails, and 2.6.0 is too old for cherimoya's `torch.optim.Muon` (2.10) — so there is no torch version
  that both installs there and satisfies this repo. Rather than pin a crippled one, the base set simply
  has no torch and installs cleanly, which is all tfmodisco needs.
  This replaced a briefly-lived pair of mutually exclusive `sherlock` (torch==2.6.0) and `cherimoya`
  (torch>=2.10) extras with a `[tool.uv] conflicts` declaration, copied from procap-atlas. That works
  and is more machinery than the actual requirement: **Sherlock is not used for GPU jobs here.**
- **Sherlock cannot build ANYTHING from source with its default toolchain, so the base set must install
  from wheels alone.** Two facts, both measured rather than inferred. Its glibc is below 2.28 — from the
  torch step change, `torch-2.6.0-...-manylinux1_x86_64.whl` against
  `torch-2.7.0-...-manylinux_2_28_x86_64.whl` — so any package whose newest wheel is `manylinux_2_26` or
  later falls back to an sdist. And its default compiler is **gcc 4.8.5 with binutils 2.27**, 2015
  vintage, reported by meson as `c++ (GCC) 4.8.5` / `ld.bfd 2.27`. That toolchain has **no C++17** and
  no AVX512-VNNI, so the fallback does not merely run slowly, it fails:

      contourpy 1.3.3   ERROR: C++ Compiler does not support -std=c++17
      pybigtools 0.3.0  Error: no such instruction: `vpdpbusd %ymm12,%ymm3,%ymm4`

  **An earlier version of this note said a missing wheel means a source build, "not automatically a
  failure". On this cluster it is.** That framing survived two rounds of pinning one package at a time,
  each of which just moved the failure to the next package in the graph. The base set is now capped so
  that **zero** of its 34 packages need a build — verify with the audit below after any dependency
  change.

  A newer `gcc` module would also work and would need no caps, but then every source-built extension
  links against that module's libstdc++ and the module has to be loaded in each job too. Caps keep the
  environment self-contained.

  | pin | why |
  | --- | --- |

  | pin | why |
  | --- | --- |
  | `leidenalg==0.10.2` | 0.11.0 moved from `manylinux_2_17` to `2_26/2_28`; the sdist build fails. Arrives via **modisco**, so a tfmodisco-only install is affected |
  | `igraph<1.0` | leidenalg's own dependency, same jump — 1.0.0 is 2_28-only, 0.11.9 is the last 2_17 |
  | `pillow<12.3.0` | 12.3.0 dropped `manylinux_2_17` |
  | `extra-build-variables` `HDF5PLUGIN_NATIVE=False` | hdf5plugin has no 2_17 wheel at any version, so it always builds; its `-march=native` probe emits AVX512-VPOPCNTDQ that Sherlock's assembler cannot assemble |
  | `pybigtools` **moved out of base** | not pinned — moved into the `torch` extra, so Sherlock never builds it at all. See below |
  | `contourpy<1.3.3`, `h5py<3.15.0`, `hdf5plugin<6.0.0`, `numpy<2.3.0`, `pandas<2.3.3`, `scikit-learn<1.8.0`, `scipy<1.17.0` | the first version of each whose linux x86_64 wheels moved past `manylinux_2_17` |

  The first four are procap-atlas's, which is the right authority because it runs on the same cluster;
  the caps are derived here. They are upper bounds, not exact pins, so patch releases on the
  wheel-having line still resolve. **Do not raise one without re-running the audit** — the cap is the
  wheel boundary, not a guess.
- **`pybigtools` is in the `torch` extra, NOT in base, and that placement is load-bearing.** It is
  `manylinux_2_28`-only at every version, so it always builds from source on a pre-2.28 glibc, and
  0.3.0's build dies in `libdeflate-sys` with `no such instruction: vpdpbusd` — GCC emitting AVX512-VNNI
  that Sherlock's assembler cannot assemble, the same class of failure as hdf5plugin's `-march=native`.
  Pinning to 0.2.5 (procap-atlas's choice) was the first fix and is the weaker one, because 0.2.5 also
  has to build. **Nothing on the base side needs it**: the only venv-side importer is
  `src/make_negatives.py`, which is torch-side anyway since it imports tangermeme, and `orientation_qc.py`
  reads bigWigs with it from the MAMBA env, where it is a conda package. So Sherlock never builds it.
  Do not move it back to base to "declare what we import" — the declaration lives in the `torch` extra,
  which is where the importer lives.
- **Zero base packages now need a source build.** Re-audit after ANY dependency change: resolve the base
  set, then check each wheel's platform tags against `manylinux_2_17`. A single uncapped transitive
  dependency is enough to break `uv sync` on Sherlock, and it will surface as a compiler error deep in a
  build log rather than as a resolution failure. **No 2_17 wheel means a
  source build, not automatically a failure** — but do not lean on that the way an earlier version of
  this note did. It cited `pybigtools` as proof, reasoning that upstream installs it on Sherlock despite
  its being 2_28-only at every version. That had the example exactly backwards: upstream pins
  **0.2.5** precisely *because* newer releases fail to build there. The general point survives (a
  missing wheel is not by itself a problem); the evidence for it did not.
  So the rule is empirical, not deductive: pin when a build actually fails, and take a pin from
  procap-atlas as evidence that one does. Re-audit with:

      uv export --no-emit-project --no-hashes   # then check each wheel's tags on PyPI

- **Do NOT try to get a newer torch from conda-forge. Tried, rejected.** conda-forge ships pytorch up to
  **2.13.0** for linux-64, and conda packages carry no manylinux tag, so it looks like the obvious way
  round the wheel ceiling — a modern torch there would also let cherimoya run natively instead of
  through Apptainer. It does not work: **conda's torch build does not reliably use CUDA.** The version
  list is not the problem, so re-checking it proves nothing; this was established by experience on the
  cluster.
- **`modisco` cannot move to `environment.yml`, though by this repo's own rule it belongs there.**
  Nothing imports `modiscolite`; `src/bpnet/modisco/` only ever shells out to the `modisco` CLI, exactly
  like `umi_tools` and `pints_caller`. But it is PyPI-only — checked 2026-09-05, HTTP 404 for
  `modisco`, `modisco-lite`, `modiscolite`, `tfmodisco-lite` and `tfmodisco` on **both** bioconda and
  conda-forge, and its `memelite` dependency is 404 there too. Same situation as `tangermeme`. Do not
  re-litigate this without re-checking those names.
- Two package-name traps, both already handled — do not "fix" them back: the peak caller is **`pyPINTS`**
  (PyPI `pints` is unrelated time-series inference), and PyPI **`muon` is a multi-omics framework**, not the
  optimizer. `fit_cherimoya.py` imports `torch.optim.Muon` and raises a pointed error rather than falling
  back to it.
- Sherlock: `mamba activate torch` also works; install the Python side with
  `uv pip install --python "$CONDA_PREFIX/bin/python" -r pyproject.toml`. Site-specific module loads
  go in a `launch.py --setup-file`; no tracked script carries them.
- Preprocessing shells out to: fastp, STAR, samtools, bedtools, GNU coreutils `sort`, `bgzip`,
  `bedGraphToBigWig`, `pints_caller` and `umi_tools`. **`bigWigToBedGraph` and `bigWigMerge` are gone**,
  and so is the `bpnet` CLI from this list (it was replaced by a direct
  `tangermeme.match.extract_matching_loci` call — see the negatives bullet in `CLAUDE.md`). The pipeline pulls FASTQs straight from ENA over HTTPS, so SRA Toolkit is not
  needed; homerTools, `fasterq-dump` and `proseq2.0` are no longer used anywhere.
- **Site-specific values: PARTITIONS AND THE GPU CONSTRAINT ARE NOW DEFAULTED; everything else is not.**
  This reverses part of an earlier rule, deliberately and on request (2026-09-05) — typing
  `--partition`/`-C` on every submission is its own error source. `src/launcher.py` holds
  `GPU_PARTITION = "akundaje,owners"`, `CPU_PARTITION = "normal,akundaje,owners"` and `GPU_CONSTRAINT`,
  the `|`-joined GPU SKU list copied from procap-atlas's cherimoya launchers (`|` is SLURM's OR, so any
  one card satisfies it). Both stay overridable, so another site needs a flag rather than a patch.
  **Which set a launcher gets is derived from ONE argument**, `_add_common_args(..., gpu=)`, so the
  partition and the constraint cannot drift apart from each other or from `_emit`'s `gpus`. `-C` is
  additionally gated on `gpus` at emission, so a global `--constraint` cannot leak onto a CPU job and
  narrow it to GPU nodes for nothing.
  **The `owners` partition caps jobs at 48:00:00** and is in every default here, so a longer `--time`
  will simply not schedule there; `modisco motifs` sits exactly on that cap.
  Everything else is unchanged: container use is opt-in via `APPTAINER_IMAGE` / `APPTAINER_BIND`; job
  environment setup defaults to activating this repo's mamba env + uv venv, with `launch.py
  --setup-file` for site-specific module loads. **Do not reintroduce cluster PATHS into tracked files.**
  The four hand-written `.sh` launchers, which emitted no partition or constraint of their own, are
  **deleted**: both `fit/slurm.sh` and cherimoya's `benchmark/slurm.sh` and `benchmark/cmd.sh`. So
  `src/launcher.py` is now the only thing in the repo that writes an sbatch script — do not add another.
- **Do not add an Apptainer definition to this repo.** Container images are maintained externally and
  authoritatively at [adamyhe/sherlock](https://github.com/adamyhe/sherlock) (`cherimoya/cherimoya.def`, base
  `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime`). A project-local `src/cherimoya/apptainer/` existed and
  was deleted deliberately. Containers are optional and opt-in via `APPTAINER_IMAGE`/`APPTAINER_BIND`; the
  launchers run natively when unset.
- **`cherimoya` is pinned by git commit**, not version, in `[tool.uv.sources]` — commit `8e4283fe` = version
  0.2.1, which was never published to PyPI (PyPI stops at 0.2.0). Verified by installing both: the only
  difference is four lines adding `__version__` to `__init__.py`. `cheri.py`, `cherimoya.py`, `io.py`,
  `losses.py`, `performance.py`, `wrappers.py` are byte-identical.
  **The EMA / `valid_count_corr` rewrite is already in 0.1.0** — `class EMA` (shadow weights, decay 0.999,
  used unconditionally in `fit()` at lines 395/445/458/520/526) and `valid_count_corr` checkpoint selection
  (`if valid_count_corr > best_corr`) appear identically in 0.1.0, 0.2.0 and 0.2.1. The comment in the
  upstream `cherimoya.def` attributing that rewrite to this commit is wrong by two releases; do not repeat
  it. The pin buys byte agreement with the image and immunity to tag movement (upstream's `v0.2.0` tag has
  been force-moved), not different training behaviour. `uv.lock` resolves `torch==2.13.0`.
- **`macs3` is blocked from installing, deliberately — do not "restore" it.** It is a transitive dep of
  both `bpnet-lite` and `cherimoya`, has no wheels at any version, and nothing here imports it, so it was
  pure sdist build cost (plus `cykhash`, also sdist-only, via macs3). `[tool.uv] override-dependencies`
  gates `macs3` and `cykhash` behind `sys_platform == 'nonexistent'`, an unsatisfiable marker — that is
  how uv drops a transitive dependency, since overrides replace a requirement rather than delete it. The
  packages still appear in `uv.lock` (it is a universal lock) but the dependency edges carry the false
  marker, so they never resolve on any platform; verified with `uv sync --dry-run`. The sherlock image
  drops it the same way via `pip install --no-deps`.
  Blocking it also strands `cykhash` and `hmmlearn`, which nothing else wanted.
- **`umi_tools` and `pypints` are conda deps, not uv ones — do not move them back.** Neither is ever
  imported: the repo shells out to `umi_tools dedup` and `pints_caller` and probes with `shutil.which`,
  so both are pipeline binaries like fastp/STAR/samtools. bioconda has `umi_tools` 1.1.5
  (linux-64/aarch64, osx-64/arm64) and `pypints` 1.1.10 (`noarch`). `umi_tools` also had no wheel at any
  version, so it was the last sdist build in the tree. Moving `pypints` out drops its exclusive tail too
  (`biopython`, `pybedtools`, `statsmodels` → `formulaic`/`patsy`/`interface-meta`); `pysam` and
  `pybigwig` stay, since `bam2bw` and `biodatatools` want them. Version pins in `environment.yml` are the
  ENCODE spec's, but the parameters that actually determine the peaks are in `config/procap_pipeline.yaml`
  — that is what to check for reproducibility, not the package version.
- **Therefore `environment.yml` carries no `c-compiler`/`cxx-compiler`.** Nothing reachable in the
  resolution compiles: `triton` has manylinux wheels, and the only source installs left are `cherimoya`
  (git pin) and `connection-pool`, both pure Python. If you add a Python dep that needs building, put the
  compilers back in `environment.yml`, not `pyproject.toml`.


## TF-MoDISco: the package is `modisco`, NOT `modisco-lite`

`modisco-lite` is the **deprecated name for the same project**, and both are on PyPI — `modisco` 2.5.2
against `modisco-lite` 2.4.0. They are not two packages: `modisco` installs the same top-level
`modiscolite/` package and the same `modisco` script, so anything installing both gets whichever landed
second, silently.

That is a live risk here rather than a hypothetical, because the two upstreams disagree:
**cherimoya 0.2.0 requires `modisco>=2.0.0`, bpnet-lite 1.0.0 on PyPI still requires
`modisco-lite>=2.0.0`** — so `uv sync --extra torch` would pull both. `[tool.uv] override-dependencies`
drops `modisco-lite` behind the same unsatisfiable marker used for `macs3`; verified with
`uv export`, where it appears as `modisco-lite==2.4.0 ; sys_platform == 'nonexistent'` and never
resolves, while `modisco==2.5.2` installs.

**The CLI surface is identical across the rename**, checked rather than assumed: subcommands
`motifs`/`report`/`convert`/`meme`, and `-n/--max_seqlets`, `-l/--n_leiden` (still defaulting to 2),
`-w/--window`, `-m/--meme_db`, `-l/--lite` all unchanged. So `src/bpnet/modisco/` needed no edit. The
only dependency difference is that `modisco` adds `jinja2`.


