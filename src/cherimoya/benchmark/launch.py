#!/usr/bin/env python3
"""Enumerate Cherimoya benchmark jobs for all experiments.

A thin wrapper over src/launcher.py, sharing the selection rule with
src/bpnet/benchmark/launch.py. The job unit is the EXPERIMENT, for the same
reason it is there: benchmark_cherimoya.py scores every fold in one run and
pools their predictions for the genome-wide block.

This replaces the deleted src/cherimoya/benchmark/cmd.sh as the way to benchmark
more than one experiment. cmd.sh hard-coded D.melanogaster-S2_PROcap.json as its
already-done check while forwarding "$@" through, so once fly had been
benchmarked every other experiment printed "Skipping" and exited 0 without
running. The check here comes from experiments.metrics_path, the same function
the script writes through.

Note torch.compile is OFF by default in the benchmark -- one inference pass does
not amortise its warmup -- so pass --benchmark-args '--compile' to enable it.

    python src/cherimoya/benchmark/launch.py --dry-run
    python src/cherimoya/benchmark/launch.py -e D.melanogaster-S2_PROcap
    python src/cherimoya/benchmark/launch.py --print-commands | bash
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from launcher import run_benchmark  # noqa: E402

if __name__ == "__main__":
    run_benchmark(
        "cherimoya",
        REPO_ROOT / "src" / "cherimoya" / "benchmark" / "benchmark_cherimoya.py",
    )
