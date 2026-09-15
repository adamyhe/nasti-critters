#!/usr/bin/env python3
"""Run the `modisco` CLI with this repo's modiscolite fixes applied.

A thin front end, not a fork: it patches modiscolite in-process and then hands
control to the real `modisco` console script with argv untouched, so every
subcommand, flag and default is whatever the installed version provides. There
is no argument parsing here to drift out of step.

It exists because `modisco` is a CLI. Nothing in this repo imports modiscolite
-- src/bpnet/modisco/ only ever shells out, exactly like umi_tools and
pints_caller -- so unlike tangermeme_compat.py there is no in-process call site
to patch, and a shim can only reach the library by owning the process.

What it fixes: `modisco motifs` crashes on an attribution track with no
negative windowed sums, which took out the counts head of
`S.cerevisiae_PROcap` and `S.pombe_PROcap`. See src/modiscolite_compat.py for
the mechanism, the measurement across the corpus, and why the patch is
self-retiring.

Usage -- identical to `modisco` itself:
    python src/bpnet/modisco/run_modisco.py motifs -s OHE -a ATTR -o OUT -n 1000000
    python src/bpnet/modisco/run_modisco.py report -i OUT.h5 -o REPORTDIR
"""

import runpy
import shutil
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from modiscolite_compat import patch_one_signed_attributions  # noqa: E402


def find_modisco() -> str | None:
    """The `modisco` script, preferring the one beside THIS interpreter.

    `shutil.which` alone is not enough: run as `.venv/bin/python
    run_modisco.py` -- without the venv activated, which is how a one-off
    invocation usually looks -- PATH does not contain the venv's bin and the
    script reports modisco missing while it sits next to the interpreter
    actually running. The launcher activates the environment, so its emitted
    command was always fine; this is for everything else.
    """
    # sysconfig rather than `Path(sys.executable).parent`, because resolving
    # the interpreter follows the venv's symlink out to the base Python, whose
    # bin directory holds no modisco. This reports the ENVIRONMENT's script
    # directory, which is the one that matches the modiscolite just imported.
    sibling = Path(sysconfig.get_path("scripts")) / "modisco"
    if sibling.is_file():
        return str(sibling)
    return shutil.which("modisco")


def main() -> int:
    script = find_modisco()
    if script is None:
        print("modisco not found beside this interpreter or on PATH. It is a "
              "uv dependency -- activate the venv (`source .venv/bin/activate`) "
              "or run via `uv run`.\n"
              "Note the package is `modisco`, NOT `modisco-lite`; see CLAUDE.md.",
              file=sys.stderr)
        return 1

    # Import modiscolite and patch it BEFORE the CLI runs, so the patched
    # function is in place by the time TFMoDISco calls extract_seqlets.
    patch_one_signed_attributions(verbose=True)

    # argv[0] should look like the tool being run, so its --help and errors
    # read the way `modisco`'s own do.
    sys.argv[0] = script
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
