"""Data-only entrypoint: pull annotations from Label Studio, split, and
materialize a COCO-format dataset folder for later training.

Steps:
1. Clone reviewed tasks from Label Studio -> raw COCO annotations JSON.
   Each image is already stamped with its train/val/test split (from the
   task's Label Studio meta).
2. Group images by that split field into train.json/val.json/test.json.
3. Download images from GCS and lay them out as
   output/<version>/dataset/{train,valid,test}/ with _annotations.coco.json.
4. Write output/<version>/info.json — the version label, provenance and class
   distribution per split.

The output/<version>/ folder is the unit everything downstream keys off: training
writes its experiments inside it (pipelines/run_training_pipeline.py) and the
release pipeline compares them (pipelines/run_release_pipeline.py).
"""

import argparse

from src.data.build_dataset import build_dataset_from_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--version", help="Dataset version label (default: v<timestamp>)")
    args = parser.parse_args()

    version_dir = build_dataset_from_config(args.config, version=args.version)

    logger.info(f"Done. Dataset version ready under {version_dir}")


if __name__ == "__main__":
    main()
