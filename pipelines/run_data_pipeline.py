"""Data-only entrypoint: pull annotations from Label Studio, split, and
materialize a COCO-format dataset folder for later training.

Steps:
1. Clone reviewed tasks from Label Studio -> raw COCO annotations JSON.
   Each image is already stamped with its train/val/test split (from the
   task's Label Studio meta).
2. Group images by that split field into train.json/val.json/test.json.
3. Download images from GCS and lay them out as <dataset_dir>/{train,valid,test}/
   with _annotations.coco.json, ready for pipelines/run_training_pipeline.py.
"""

import argparse

from src.data.build_dataset import build_dataset_from_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    dataset_dir = build_dataset_from_config(args.config)

    logger.info(f"Done. Dataset ready under {dataset_dir}")


if __name__ == "__main__":
    main()
