"""Summarize every experiment run against one dataset version into a comparison table.

A dataset version folder looks like

    output/<version>/info.json
                    /dataset/
                    /experiment_<datetime>/                 # single fixed-config run
                    /experiment_<datetime>_trial_<n>/       # one folder per Optuna trial

Each run folder holds params.json (written by training.train.run_training) and
metrics.csv (written by RF-DETR's CSVLogger). This module reads both into one
DataFrame — one row per run, across every experiment in the version — and renders it
as the Markdown table that goes into the GitHub release notes.

Runs from before trials were flattened (experiment_<datetime>/trial_<n>/, nested) are
still read, so older dataset versions stay releasable.
"""

import json
import re
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Artifacts worth publishing for a released model. checkpoint.pth (last epoch, with
# optimizer state) and the tensorboard/wandb logs are deliberately excluded — see
# release_files() for the --full escape hatch.
RELEASE_FILES = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "metrics.csv",
    "params.json",
    "training_config.json",
)

# Columns kept out of the "which hyperparameters differ" scan.
_METRIC_COLUMNS = ("mAP50", "ema_mAP50", "recall", "F1")
_ID_COLUMNS = ("index", "experiment", "trial", "name", "path", "epoch", "epochs_run", "has_checkpoint")


def identify_run(run_dir: Path) -> tuple[str, int | None]:
    """Split a run folder into the experiment it belongs to and its trial number.

    Handles both layouts:
        experiment_20260730_100000_trial_3  -> ("experiment_20260730_100000", 3)
        experiment_20260730_100000/trial_3  -> ("experiment_20260730_100000", 3)   # legacy
        experiment_20260730_100000          -> ("experiment_20260730_100000", None)

    The trailing number of a single run's timestamp is never mistaken for a trial
    index because the split keys off the literal "_trial_" separator.

    Args:
        run_dir: a run folder
    Returns:
        (experiment name, trial number or None for a single fixed-config run)
    """
    if run_dir.name.startswith("trial_"):
        return run_dir.parent.name, int(run_dir.name.split("_")[1])

    match = re.fullmatch(r"(.+)_trial_(\d+)", run_dir.name)
    return (match.group(1), int(match.group(2))) if match else (run_dir.name, None)


def find_experiments(version_dir: str) -> list[Path]:
    """List every run under a dataset version, grouped by experiment and ordered by
    trial number within each.

    Args:
        version_dir: path to output/<version>/
    Returns:
        one path per run: an experiment_<datetime>_trial_<n> folder per Optuna trial,
        or the experiment_<datetime> folder itself for a single fixed-config run
    """
    root = Path(version_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset version folder not found: {version_dir}")

    runs: list[Path] = []
    for experiment in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("experiment_")):
        # Legacy layout: trials nested inside the experiment folder.
        trials = [p for p in experiment.iterdir() if p.is_dir() and p.name.startswith("trial_")]
        runs.extend(trials or [experiment])

    if not runs:
        raise FileNotFoundError(f"No experiment_* folders under {version_dir} — has anything trained yet?")

    def sort_key(run_dir: Path) -> tuple[str, int]:
        # Sort on the parsed trial number, so trial_10 follows trial_9, not trial_1.
        experiment, trial = identify_run(run_dir)
        return experiment, -1 if trial is None else trial

    return sorted(runs, key=sort_key)


def _read_params(run_dir: Path) -> dict:
    """Read a run's hyperparameters, preferring our params.json over RF-DETR's
    training_config.json (the latter is only written by model.train()).
    """
    params_path = run_dir / "params.json"
    if params_path.exists():
        with open(params_path) as f:
            return json.load(f).get("params", {})

    config_path = run_dir / "training_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f).get("train_config", {})

    logger.warning(f"No params.json or training_config.json in {run_dir}")
    return {}


def _read_metrics(run_dir: Path) -> dict:
    """Read the best epoch's validation metrics from metrics.csv.

    "Best" is the epoch with the highest val/F1 — the same criterion
    training.train._best_epoch_objectives optimizes on.
    """
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        logger.warning(f"No metrics.csv in {run_dir} — training may not have finished")
        return {}

    df = pd.read_csv(metrics_path)
    df = df[df["val/mAP_50"].notna()] if "val/mAP_50" in df else df.iloc[0:0]
    if df.empty:
        logger.warning(f"metrics.csv in {run_dir} has no validated epochs")
        return {}

    best = df.loc[df["val/F1"].idxmax()]
    return {
        "epoch": int(best["epoch"]) if "epoch" in best else None,
        "epochs_run": int(df["epoch"].max()) + 1 if "epoch" in df else len(df),
        "mAP50": float(best["val/mAP_50"]),
        "ema_mAP50": float(best.get("val/ema_mAP_50", float("nan"))),
        "recall": float(best["val/recall"]),
        "F1": float(best["val/F1"]),
    }


def build_table(version_dir: str) -> pd.DataFrame:
    """Build the comparison table across all experiments of one dataset version.

    Args:
        version_dir: path to output/<version>/
    Returns:
        one row per run: index, experiment, trial, path, hyperparameters,
        best-epoch metrics
    """
    root = Path(version_dir)
    rows = []
    for index, run_dir in enumerate(find_experiments(version_dir)):
        experiment, trial = identify_run(run_dir)
        rows.append(
            {
                "index": index,
                "experiment": experiment,
                "trial": trial,
                "name": str(run_dir.relative_to(root)),
                "path": str(run_dir),
                **_read_params(run_dir),
                **_read_metrics(run_dir),
                "has_checkpoint": (run_dir / "checkpoint_best_total.pth").exists(),
            }
        )

    df = pd.DataFrame(rows)
    logger.info(
        f"Summarized {len(df)} run(s) across {df['experiment'].nunique()} experiment(s) under {version_dir}"
    )
    return df


def select_best(df: pd.DataFrame, metric: str = "F1") -> pd.Series:
    """Pick the release candidate: the highest-scoring run that actually produced a
    checkpoint, across every experiment of this dataset version.

    Args:
        df: table from build_table
        metric: column to maximize (F1, mAP50, ema_mAP50, recall)
    Returns:
        the winning row
    """
    if metric not in df:
        raise ValueError(f"No '{metric}' column in the experiment table — did any run finish?")

    candidates = df[df["has_checkpoint"] & df[metric].notna()]
    if candidates.empty:
        raise ValueError(f"No run under this dataset version has both a checkpoint and a {metric} score")

    best = candidates.loc[candidates[metric].idxmax()]
    logger.info(f"Best run by {metric}: {best['name']} ({metric}={best[metric]:.4f})")
    return best


def release_files(run_dir: str, full: bool = False) -> list[str]:
    """List the files to publish for one run.

    Args:
        run_dir: the winning run's folder
        full: publish everything in the folder, including the last-epoch
            checkpoint.pth (which carries optimizer state and is much larger)
    Returns:
        existing file paths, in publish order
    """
    root = Path(run_dir)
    if full:
        return sorted(str(p) for p in root.rglob("*") if p.is_file())
    return [str(root / name) for name in RELEASE_FILES if (root / name).exists()]


def varying_columns(df: pd.DataFrame) -> list[str]:
    """Hyperparameter columns that differ across runs — the ones worth showing in the
    release table. Falls back to all hyperparameters when there is only one run.
    """
    params = [c for c in df.columns if c not in _METRIC_COLUMNS + _ID_COLUMNS]
    if len(df) <= 1:
        return params
    return [c for c in params if df[c].astype(str).nunique() > 1]


def to_markdown(df: pd.DataFrame, best_index: int | None = None, metric: str = "F1") -> str:
    """Render the comparison table as Markdown for the release notes.

    Args:
        df: table from build_table
        best_index: index of the released run, marked with a star
        metric: metric the release was selected on, noted under the table
    Returns:
        a Markdown table
    """
    columns = ["index", "experiment", "trial"] + varying_columns(df) + [
        c for c in ("epoch", "mAP50", "ema_mAP50", "recall", "F1") if c in df
    ]
    view = df[columns].copy()
    view["trial"] = view["trial"].map(lambda t: "—" if pd.isna(t) else str(int(t)))

    if best_index is not None:
        view.insert(0, "", ["**★**" if i == best_index else "" for i in df["index"]])

    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: "—" if pd.isna(v) else f"{v:.4g}")

    table = view.to_markdown(index=False)
    if best_index is not None:
        table += f"\n\n★ = released model (selected by highest `{metric}`)."
    return table


def dataset_markdown(info: dict) -> str:
    """Render a dataset version's info.json as a per-split distribution table.

    Args:
        info: the manifest from data.build_dataset.load_info
    Returns:
        a Markdown table: one row per split, one column per defect class
    """
    classes = list(info["totals"]["class_distribution"])
    rows = []
    for split, stats in info["splits"].items():
        rows.append({
            "split": split,
            "images": stats["images"],
            "annotations": stats["annotations"],
            **stats["class_distribution"],
        })
    rows.append({
        "split": "**total**",
        "images": info["totals"]["images"],
        "annotations": info["totals"]["annotations"],
        **info["totals"]["class_distribution"],
    })
    return pd.DataFrame(rows)[["split", "images", "annotations"] + classes].to_markdown(index=False)
