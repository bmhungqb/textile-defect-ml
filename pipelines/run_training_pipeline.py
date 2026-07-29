"""Phase 2 entrypoint: Run the training pipeline for a single run or an Optuna hyperparameter search.

Takes a dataset version folder built by pipelines/run_data_pipeline.py and writes
this experiment inside it, so every experiment stays attached to the data it was
trained on:

    output/<version>/info.json
                    /dataset/                        # what this experiment trains on
                    /experiment_<datetime>/          # single run
                    /experiment_<datetime>/trial_<n> # Optuna search
"""

import argparse
import datetime
from pathlib import Path

import yaml

from src.data.build_dataset import load_info
from src.utils.logger import get_logger
from src.training import train

logger = get_logger(__name__)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "data_version_dir",
        help="Path to a dataset version folder, e.g. output/v20260729_145230 — it holds "
        "info.json and dataset/ (see pipelines/run_data_pipeline.py)",
    )
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument(
        "--optuna",
        action="store_true",
        help="Run an Optuna hyperparameter search instead of a single fixed-config run "
        "(overrides use_optuna in the config file)",
    )
    args = parser.parse_args()

    info = load_info(args.data_version_dir)
    dataset_dir = Path(args.data_version_dir) / "dataset"
    if not dataset_dir.is_dir():
        raise SystemExit(f"No dataset/ folder in {args.data_version_dir} — rerun the data pipeline")

    config = load_yaml(args.config)
    config["dataset_dir"] = str(dataset_dir)
    config["data_version"] = info["version"]
    # Experiments live inside the dataset version they were trained on.
    config["output_dir"] = args.data_version_dir
    if args.optuna:
        config["use_optuna"] = True

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(
        f"Experiment {run_id} on dataset version {info['version']} "
        f"({info['totals']['images']} images) -> {config['output_dir']}"
    )

    if config.get("use_optuna", False):
        model_dir = train.train_with_optuna(config, run_id)
    else:
        model_dir = train.train(config, run_id)

    logger.info(f"Done. Results in {model_dir}")


if __name__ == "__main__":
    main()
