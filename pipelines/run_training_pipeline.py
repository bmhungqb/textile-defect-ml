"""Phase 2 entrypoint: Run the training pipeline for a single run or an Optuna hyperparameter search.
"""

import argparse
import datetime

import yaml

from src.utils.logger import get_logger
from src.training import train

logger = get_logger(__name__)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        help="Path to a materialized dataset folder with train/valid/test "
        "subdirs, each containing _annotations.coco.json (see "
        "pipelines/run_data_pipeline.py)",
    )
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument(
        "--optuna",
        action="store_true",
        help="Run an Optuna hyperparameter search instead of a single fixed-config run "
        "(overrides use_optuna in the config file)",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    config["dataset_dir"] = args.dataset_dir
    if args.optuna:
        config["use_optuna"] = True

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Run {run_id} -> {config['output_dir']}")

    if config.get("use_optuna", False):
        model_dir = train.train_with_optuna(config, run_id)
    else:
        model_dir = train.train(config, run_id)

    logger.info(f"Done. Results in {model_dir}")


if __name__ == "__main__":
    main()
