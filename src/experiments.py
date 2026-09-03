"""Shared resolution of experiments, folds, hyperparameters and model paths.

Single source of truth for the conventions every script in this repo follows.
Before this module there were two incompatible regimes: `fit_bpnet.py` and
`launch.py` read experiment IDs from config/experiment_config.yaml, while
`fit_cherimoya.py` and the benchmark/attribute scripts read a flat
config/data_paths.json describing exactly one dm3 dataset. They disagreed on the
genome build, on the model filename, and on hyperparameter values. Everything now
goes through here.

Conventions
-----------
Experiment      An ID in config/experiment_config.yaml, e.g.
                "S.cerevisiae-Ino80ctl_PROcap". Carries its own species, which
                selects the FASTA, the fold set, and the exclusion list.

Folds           test = fold i, validation = (i + 1) % n_folds, train = the rest.
                Chromosome-level from config/chrom_splits.yaml, or peak-level
                when config/splits/{species}_random_fold_assignments.csv exists
                (S. pombe, which has too few chromosomes).

Hyperparameters config/{family}_params.json, one file per model family, read by
                training AND downstream scripts so they cannot drift.

Model paths     models/{family}/{experiment}/{experiment}.fold{f}.torch

                Both bpnet-lite and cherimoya write "{name}.torch" every time
                validation improves and "{name}.final.torch" exactly once at the
                end of fit(). Completion must therefore be tested against
                `.final.torch`; `.torch` can exist after a single epoch.

Usage:
    import sys; sys.path.insert(0, str(REPO_ROOT / "src"))
    from experiments import Experiment, load_params, model_path
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "config"
EXPERIMENTS_PATH = CONFIG / "experiment_config.yaml"
CHROM_SPLITS_PATH = CONFIG / "chrom_splits.yaml"
SPLITS_DIR = CONFIG / "splits"
MODELS_DIR = REPO_ROOT / "models"

# Bases that extract_loci should treat as unknown. Passed on every call in this
# repo; kept here so the list cannot drift between scripts.
IGNORE = list("QWERYUIOPSDFHJKLZXVBNM")


def _load(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def experiment_ids() -> list[str]:
    return list(_load(EXPERIMENTS_PATH)["experiments"])


def load_params(family: str, overrides: dict | None = None) -> dict:
    """Hyperparameters for a model family, from config/{family}_params.json."""
    path = CONFIG / f"{family}_params.json"
    if not path.exists():
        raise FileNotFoundError(f"no hyperparameter file for family {family!r}: {path}")
    with path.open() as f:
        params = json.load(f)
    for k, v in (overrides or {}).items():
        if v is not None:
            params[k] = v
    return params


def model_dir(family: str, experiment: str, suffix: str = "") -> Path:
    return MODELS_DIR / family / f"{experiment}{suffix}"


def model_path(family: str, experiment: str, fold: int, *, final: bool = False,
               suffix: str = "") -> Path:
    """Checkpoint path. `final=True` gives the artifact written once at the end
    of training, which is the only safe test of completion."""
    ext = ".final.torch" if final else ".torch"
    return model_dir(family, experiment, suffix) / f"{experiment}.fold{fold}{ext}"


def model_name(family: str, experiment: str, fold: int, suffix: str = "") -> str:
    """Value to pass as a model's `name`; the library appends .torch/.final.torch."""
    return str(model_dir(family, experiment, suffix) / f"{experiment}.fold{fold}")


@dataclass
class Experiment:
    """One experiment from config/experiment_config.yaml, with paths resolved."""

    id: str
    species: str
    entry: dict
    peaks: Path
    signals: list[Path]
    sequences: Path
    negatives: Path
    controls: list[Path] | None = None
    blacklist: list[str] | None = None
    missing: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, experiment: str, *, use_controls: bool = False) -> "Experiment":
        experiments = _load(EXPERIMENTS_PATH)["experiments"]
        if experiment not in experiments:
            raise KeyError(
                f"unknown experiment {experiment!r}; "
                f"see config/experiment_config.yaml"
            )
        entry = experiments[experiment]
        p = entry.get("processed", {})

        def abspath(key):
            return REPO_ROOT / p[key] if key in p else None

        controls = None
        if use_controls:
            if "pl_control" not in p or "mn_control" not in p:
                raise KeyError(
                    f"{experiment} has no pl_control/mn_control in "
                    "config/experiment_config.yaml"
                )
            controls = [REPO_ROOT / p["pl_control"], REPO_ROOT / p["mn_control"]]

        blacklist = [str(REPO_ROOT / p["blacklist"])] if p.get("blacklist") else None

        obj = cls(
            id=experiment,
            species=entry["species"],
            entry=entry,
            peaks=abspath("peaks"),
            signals=[REPO_ROOT / p["pl_bigwig"], REPO_ROOT / p["mn_bigwig"]],
            sequences=abspath("sequences"),
            negatives=abspath("gc_negatives"),
            controls=controls,
            blacklist=blacklist,
        )
        obj.missing = obj._missing()
        return obj

    #: Everything a TRAINING run needs. `.missing` checks all of these.
    ALL_INPUTS = ("peaks", "sequences", "gc_negatives", "signals",
                  "controls", "blacklist")
    #: What a QC or reporting step needs. Notably NOT gc_negatives: those are
    #: model prep, produced by src/make_negatives.py outside the Snakemake DAG,
    #: so they are absent for the whole of a pipeline run. A QC step that gated
    #: on `.missing` would therefore skip every experiment forever -- which is
    #: exactly what orientation_qc did, silently and with exit status 0.
    QC_INPUTS = ("peaks", "sequences", "signals")

    def missing_paths(self, kinds=None) -> list[str]:
        """Absent inputs among `kinds` (default: everything training needs)."""
        kinds = tuple(kinds) if kinds is not None else self.ALL_INPUTS
        checks = []
        if "peaks" in kinds:
            checks.append(("peaks", self.peaks))
        if "sequences" in kinds:
            checks.append(("sequences", self.sequences))
        if "gc_negatives" in kinds:
            checks.append(("gc_negatives", self.negatives))
        if "signals" in kinds:
            checks += [(f"signals[{i}]", s) for i, s in enumerate(self.signals)]
        if "controls" in kinds and self.controls:
            checks += [(f"controls[{i}]", c) for i, c in enumerate(self.controls)]
        if "blacklist" in kinds and self.blacklist:
            checks += [(f"blacklist[{i}]", Path(b))
                       for i, b in enumerate(self.blacklist)]
        return [f"{label}: {path}" for label, path in checks
                if path is None or not Path(path).exists()]

    def _missing(self) -> list[str]:
        return self.missing_paths()

    # ---------------------------------------------------------------- folds
    @property
    def random_splits_path(self) -> Path:
        return SPLITS_DIR / f"{self.species}_random_fold_assignments.csv"

    @property
    def uses_peak_level_splits(self) -> bool:
        return self.random_splits_path.exists()

    def n_folds(self) -> int:
        if self.uses_peak_level_splits:
            return pd.read_csv(self.random_splits_path, usecols=["fold"],
                               comment="#")["fold"].nunique()
        splits = _load(CHROM_SPLITS_PATH)
        if self.species not in splits:
            raise KeyError(
                f"no fold assignment for {self.species}: not in "
                f"config/chrom_splits.yaml, and {self.random_splits_path.name} "
                "does not exist. Build peak-level folds with "
                "make_random_splits.py, or add a peak-matched entry to "
                "config/chrom_splits.yaml -- see that file's notes for which "
                "applies to this species."
            )
        return len(splits[self.species])

    def fold_loci(self, peaks: pd.DataFrame, fold: int) -> dict:
        """Training/validation loci and chromosome filters for one fold.

        Unifies the two split mechanisms behind one return shape, so callers do
        not branch on which is in use:

        * chromosome-level -- all peaks are passed through and the held-out
          chromosomes do the filtering, via `chroms`;
        * peak-level -- `chroms` is None and the peak table itself is filtered,
          by merging against the fold assignments on (chrom, start, end).

        Returns train/valid loci frames plus the matching `train_chroms` /
        `valid_chroms` (None under peak-level splits).
        """
        split = self.fold_split(fold)
        if not split["peak_level"]:
            return {"train_loci": peaks, "valid_loci": peaks,
                    "train_chroms": split["train_chroms"],
                    "valid_chroms": split["valid_chroms"],
                    "n_test": None, **split}

        cols = ["chrom", "start", "end"]
        table = split["fold_table"].copy()
        table["chrom"] = table["chrom"].astype(str)

        # Assign folds POSITIONALLY. make_random_splits.py appends a
        # `fold` column to the peak table and writes it without reordering, so
        # row i of the CSV is row i of the peaks file; anything else is an
        # error rather than something to paper over.
        #
        # A coordinate merge would be wrong here: peak sets are the CONCATENATION
        # of PINTS unidirectional and bidirectional calls (never interval-merged,
        # see config/procap_pipeline.yaml), so the same interval can legitimately
        # appear twice. Merging on (chrom, start, end) would fan those into a
        # cross product and silently oversample them in training.
        aligned = (
            len(table) == len(peaks)
            and table[cols].reset_index(drop=True).equals(
                peaks[cols].reset_index(drop=True)
            )
        )
        if aligned:
            merged = peaks.reset_index(drop=True).assign(
                fold=table["fold"].to_numpy()
            )
        else:
            raise ValueError(
                f"{self.random_splits_path.name} is not row-aligned with the "
                f"peak set for {self.id} ({len(table)} vs {len(peaks)} rows). "
                "The split file is written in its input's row order and joined "
                "positionally, so a mismatch means the folds no longer describe "
                "these peaks -- training on them risks train/test leakage. "
                "Regenerate with "
                "`python src/data_preprocessing/make_random_splits.py`, "
                "or check drift with its --check flag."
            )

        unassigned = int(merged["fold"].isna().sum())
        if unassigned:
            warnings.warn(
                f"{unassigned} peaks for {self.id} have no fold assignment and "
                "will be dropped from train and validation",
                stacklevel=2,
            )

        return {
            "train_loci": merged[merged["fold"].isin(split["train_folds"])][cols]
                          .reset_index(drop=True),
            "valid_loci": merged[merged["fold"] == split["valid_fold"]][cols]
                          .reset_index(drop=True),
            "train_chroms": None,
            "valid_chroms": None,
            "n_test": int((table["fold"] == split["test_fold"]).sum()),
            **split,
        }

    def all_folds(self, family: str, *, models_dir: Path | None = None,
                  suffix: str = "") -> list[dict]:
        """Every fold with its test chromosomes and checkpoint path.

        What the benchmark/attribution scripts iterate over: each fold's model is
        evaluated on the chromosomes held out for that fold.
        """
        out = []
        for fold in range(self.n_folds()):
            split = self.fold_split(fold)
            if models_dir is not None:
                path = Path(models_dir) / f"{self.id}.fold{fold}.torch"
            else:
                path = model_path(family, self.id, fold, suffix=suffix)
            out.append({"fold": fold, "test_chroms": split["test_chroms"],
                        "model": path, "peak_level": split["peak_level"],
                        "fold_table": split["fold_table"]})
        return out

    def fold_split(self, fold: int) -> dict:
        """Resolve one fold. Chromosome-level unless peak-level splits exist.

        Returns train/valid/test chromosome lists (None under peak-level splits,
        where filtering happens on loci instead) plus the fold indices.
        """
        n = self.n_folds()
        if not 0 <= fold < n:
            raise ValueError(f"fold {fold} out of range for {n} folds")
        test, valid = fold, (fold + 1) % n
        train = [f for f in range(n) if f not in (test, valid)]
        out = {"n_folds": n, "test_fold": test, "valid_fold": valid,
               "train_folds": train, "peak_level": self.uses_peak_level_splits}

        if self.uses_peak_level_splits:
            out.update(train_chroms=None, valid_chroms=None, test_chroms=None,
                       fold_table=pd.read_csv(self.random_splits_path,
                                              dtype={"chrom": str},
                                              comment="#"))
        else:
            splits = {int(k): [str(c) for c in v]
                      for k, v in _load(CHROM_SPLITS_PATH)[self.species].items()}
            out.update(
                train_chroms=[c for f in train for c in splits[f]],
                valid_chroms=splits[valid],
                test_chroms=splits[test],
                fold_table=None,
            )
        return out
