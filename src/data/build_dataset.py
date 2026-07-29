"""Build a training-ready COCO dataset folder from Label Studio + GCS.
Includes the following steps:

1. pull_from_label_studio  — raw tasks -> COCO annotations JSON.
2. group_by_split          — one COCO JSON per train/val/test, using each
                             image's own "split" field (stamped upstream in the
                             Label Studio task meta).
3. materialize_dataset     — download the image bytes from GCS into
                             <dataset_dir>/{train,valid,test}/ next to an
                             _annotations.coco.json, the layout RF-DETR expects.
4. write_info              — record the version and class distribution in
                             info.json, the manifest every later stage reads.

Everything for one dataset version lands under a single folder:

    output/<version>/info.json                       # this dataset version
                    /raw_annotations.json            # what Label Studio returned
                    /dataset/{train,valid,test}/     # what RF-DETR trains on
                    /experiment_<datetime>/          # added by the training pipeline
"""

import argparse
import base64
import copy
import datetime
import json
import os
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.gcs import download_blobs
from src.utils.label_studio import pull_tasks
from src.utils.logger import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "val", "test")
_SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "test": "test"}
_RFDETR_DIR_NAME = {"train": "train", "val": "valid", "test": "test"}

def _load_coco(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _save_coco(data: dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path

def _extract_image_url(raw_image: str) -> str:
    if "fileuri=" in raw_image:
        b64_str = raw_image.split("fileuri=")[-1].split("&")[0]
        return base64.b64decode(b64_str).decode("utf-8")
    return raw_image


def _process_task(task: dict, label_to_id: dict[str, int]) -> dict[str, Any] | None:
    """Convert a single Label Studio task into a COCO image + annotations dict."""
    task_id = task["id"]

    raw_split = task.get("data", {}).get("meta_info", {}).get("split")
    split = _SPLIT_ALIASES.get(raw_split)
    if split is None:
        logger.warning(f"Task {task_id} has missing/invalid split {raw_split!r}, skipping")
        return None

    annotations = task.get("annotations", [])
    if not annotations:
        logger.warning(f"Task {task_id} has no annotations, skipping")
        return None

    final_annotation = annotations[-1]
    if final_annotation.get("was_cancelled", False):
        logger.warning(f"Task {task_id} was cancelled, skipping")
        return None

    results = final_annotation.get("result", [])
    orig_w = orig_h = None
    if results:
        orig_w = results[0].get("original_width")
        orig_h = results[0].get("original_height")

    annos = []
    for res in results:
        if "rectanglelabels" not in res["value"]:
            continue
        label_name = res["value"]["rectanglelabels"][0]
        if label_name not in label_to_id:
            logger.warning(f"'{label_name}' not in predefined classes, skipping")
            continue

        x_pct, y_pct = res["value"]["x"], res["value"]["y"]
        w_pct, h_pct = res["value"]["width"], res["value"]["height"]
        bbox = [
            (x_pct / 100) * orig_w,
            (y_pct / 100) * orig_h,
            (w_pct / 100) * orig_w,
            (h_pct / 100) * orig_h,
        ]
        annos.append({
            "category_id": label_to_id[label_name],
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
        })

    return {
        "task_id": task_id,
        "image_url": _extract_image_url(task["data"]["image"]),
        "width": orig_w,
        "height": orig_h,
        "split": split,
        "annos": annos,
    }


def tasks_to_coco(tasks: list[dict], defect_classes: dict[int, str]) -> dict:
    """Convert raw Label Studio tasks into a COCO dict."""
    label_to_id = {v: k for k, v in defect_classes.items()}

    coco = {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": []}
    for task in tasks:
        sample = _process_task(task, label_to_id)
        if sample is None:
            continue
        image_id = len(coco["images"]) + 1
        coco["images"].append({
            "id": image_id,
            "task_id": sample["task_id"],
            "file_name": sample["image_url"],
            "width": sample["width"],
            "height": sample["height"],
            "split": sample["split"],
        })
        for anno in sample["annos"]:
            coco["annotations"].append({
                "id": len(coco["annotations"]) + 1,
                "image_id": image_id,
                **anno,
            })

    coco["categories"] = [
        {"id": cat_id, "name": name, "supercategory": "defect"}
        for cat_id, name in defect_classes.items()
    ]
    return coco


def pull_from_label_studio(
    url: str,
    api_key: str,
    project_id: int,
    defect_classes: dict[int, str],
    output_path: str,
) -> str:
    """Pull reviewed tasks from Label Studio and write them as a COCO JSON."""
    tasks = pull_tasks(url, api_key, project_id)
    coco = tasks_to_coco(tasks, defect_classes)
    _save_coco(coco, output_path)
    logger.info(
        f"Saved {len(coco['images'])} images / {len(coco['annotations'])} annotations "
        f"to {output_path}"
    )
    return output_path

def _build_coco_subset(
    images: list[dict],
    annotations: list[dict],
    categories: list[dict],
    info: dict,
) -> dict:
    """Re-index a subset of images and their annotations into a fresh COCO dict."""
    selected_ids = {img["id"] for img in images}

    old_to_new: dict[int, int] = {}
    new_images = []
    for idx, img in enumerate(images, start=1):
        old_to_new[img["id"]] = idx
        new_img = copy.deepcopy(img)
        new_img["id"] = idx
        new_images.append(new_img)

    new_annotations = []
    for anno_id, anno in enumerate(
        (a for a in annotations if a["image_id"] in selected_ids), start=1
    ):
        new_anno = copy.deepcopy(anno)
        new_anno["id"] = anno_id
        new_anno["image_id"] = old_to_new[anno["image_id"]]
        new_annotations.append(new_anno)

    return {
        "info": info,
        "licenses": [],
        "images": new_images,
        "annotations": new_annotations,
        "categories": copy.deepcopy(categories),
    }

def group_by_split(annotation_file_path: str, output_dir: str) -> tuple[str, str, str]:
    """Split a COCO file into train/val/test using each image's own "split" field.
    """
    coco = _load_coco(annotation_file_path)
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco.get("categories", [])
    info = coco.get("info", {})

    images_by_split: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for img in images:
        images_by_split[img["split"]].append(img)

    paths = []
    for split in SPLITS:
        subset = _build_coco_subset(images_by_split[split], annotations, categories, info)
        path = _save_coco(subset, os.path.join(output_dir, f"{split}.json"))
        paths.append(path)
        logger.info(
            f"{split}: {len(subset['images'])} images, {len(subset['annotations'])} annotations"
        )
    return tuple(paths)

def materialize_dataset(
    split_paths: dict[str, str],
    dataset_dir: str,
    workers: int = 16,
) -> dict[str, str]:
    """Download each split's images from GCS into <dataset_dir>/<rfdetr_dir>/.

    Returns:
        {split: path to the split's _annotations.coco.json}, which lists only the
        images that actually downloaded — the ground truth for the distribution
        recorded in info.json
    """
    materialized: dict[str, str] = {}
    for split, path in split_paths.items():
        coco = _load_coco(path)
        split_dir = os.path.join(dataset_dir, _RFDETR_DIR_NAME[split])

        uris = list(dict.fromkeys(img["file_name"] for img in coco["images"]))
        downloaded = download_blobs(uris, split_dir, workers=workers)

        kept_images = []
        kept_ids = set()
        for img in coco["images"]:
            local_name = downloaded.get(img["file_name"])
            if local_name is None:
                continue
            new_img = copy.deepcopy(img)
            new_img["file_name"] = local_name
            kept_images.append(new_img)
            kept_ids.add(img["id"])

        coco["images"] = kept_images
        coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in kept_ids]
        materialized[split] = _save_coco(coco, os.path.join(split_dir, "_annotations.coco.json"))

    return materialized


def summarize_splits(split_paths: dict[str, str], defect_classes: dict[int, str]) -> dict[str, Any]:
    """Count images, annotations and per-class boxes in each materialized split.

    Args:
        split_paths: {split: path to that split's _annotations.coco.json}
        defect_classes: {category_id: class_name}
    Returns:
        {split: {"images": n, "annotations": m, "class_distribution": {name: count}}}
    """
    names = {int(cat_id): name for cat_id, name in defect_classes.items()}

    summary: dict[str, Any] = {}
    for split, path in split_paths.items():
        coco = _load_coco(path)
        counts = dict.fromkeys(names.values(), 0)
        for anno in coco["annotations"]:
            name = names.get(anno["category_id"])
            if name is None:
                logger.warning(f"{split}: annotation {anno['id']} has unknown category {anno['category_id']}")
                continue
            counts[name] += 1
        summary[split] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "class_distribution": counts,
        }
    return summary


def write_info(
    version_dir: str,
    version: str,
    dataset_dir: str,
    defect_classes: dict[int, str],
    splits: dict[str, Any],
    source: dict[str, Any],
) -> str:
    """Write info.json — the manifest identifying this dataset version.

    Read later by the training pipeline (to locate the dataset) and by the release
    pipeline (to report what the model was trained on).

    Args:
        version_dir: output/<version>/
        version: the version label, e.g. v20260729_145230
        dataset_dir: the RF-DETR dataset folder inside version_dir
        defect_classes: {category_id: class_name}
        splits: per-split counts from summarize_splits
        source: provenance, e.g. {"label_studio_url": ..., "project_id": 27}
    Returns:
        path to info.json
    """
    totals = dict.fromkeys((c for c in defect_classes.values()), 0)
    for stats in splits.values():
        for name, count in stats["class_distribution"].items():
            totals[name] += count

    info = {
        "version": version,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "dataset_dir": dataset_dir,
        "defect_classes": defect_classes,
        "totals": {
            "images": sum(s["images"] for s in splits.values()),
            "annotations": sum(s["annotations"] for s in splits.values()),
            "class_distribution": totals,
        },
        "splits": splits,
    }

    path = os.path.join(version_dir, "info.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(
        f"Dataset version {version}: {info['totals']['images']} images, "
        f"{info['totals']['annotations']} annotations, distribution {totals}"
    )
    return path

def build_dataset(
    url: str,
    api_key: str,
    project_id: int,
    defect_classes: dict[int, str],
    output_dir: str,
    version: str | None = None,
) -> str:
    """Run the full pipeline: Label Studio -> COCO JSONs -> downloaded images -> info.json.
    Args:
        url: Label Studio URL
        api_key: Label Studio API key
        project_id: Label Studio project ID
        defect_classes: {category_id: class_name} for the defect classes
        output_dir: root holding every dataset version (e.g. "output")
        version: version label for this pull; defaults to v<timestamp>
    Returns:
        path to this dataset version's folder, output/<version>/ — the input to
        pipelines/run_training_pipeline.py
    """
    version = version or datetime.datetime.now().strftime("v%Y%m%d_%H%M%S")
    version_dir = os.path.join(output_dir, version)
    dataset_dir = os.path.join(version_dir, "dataset")
    os.makedirs(version_dir, exist_ok=True)

    raw_annotations_path = os.path.join(version_dir, "raw_annotations.json")
    logger.info(f"Pulling tasks from Label Studio project {project_id} -> {raw_annotations_path}")

    pull_from_label_studio(
        url=url,
        api_key=api_key,
        project_id=project_id,
        defect_classes=defect_classes,
        output_path=raw_annotations_path,
    )

    train_path, val_path, test_path = group_by_split(
        annotation_file_path=raw_annotations_path,
        output_dir=dataset_dir,
    )
    split_paths = {"train": train_path, "val": val_path, "test": test_path}

    materialized = materialize_dataset(split_paths, dataset_dir=dataset_dir)
    write_info(
        version_dir=version_dir,
        version=version,
        dataset_dir=dataset_dir,
        defect_classes=defect_classes,
        splits=summarize_splits(materialized, defect_classes),
        source={"label_studio_url": url, "project_id": project_id},
    )

    logger.info(f"Dataset version {version} built at {version_dir}")
    return version_dir

def build_dataset_from_config(config_path: str, version: str | None = None) -> str:
    """Run build_dataset() with the settings in a data.yaml config file.

    Reads .env first so the "${VAR}" placeholders in the config (and
    GOOGLE_APPLICATION_CREDENTIALS, used by the GCS client) resolve.

    Args:
        config_path: path to data.yaml
        version: version label, overriding the config's own (which defaults to a
            timestamp when blank)
    """
    load_dotenv()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    label_studio = cfg["label_studio"]
    return build_dataset(
        url=os.path.expandvars(label_studio["url"]),
        api_key=os.path.expandvars(label_studio["api_key"]),
        project_id=label_studio["project_id"],
        defect_classes=cfg["defect_classes"],
        output_dir=cfg["output_dir"],
        version=version or cfg.get("version") or None,
    )


def load_info(version_dir: str) -> dict[str, Any]:
    """Read a dataset version's info.json.
    Args:
        version_dir: output/<version>/
    Returns:
        the manifest written by write_info
    """
    path = os.path.join(version_dir, "info.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No info.json in {version_dir} — is this a dataset version folder built by "
            "pipelines/run_data_pipeline.py?"
        )
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a COCO dataset from Label Studio + GCS")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to the YAML config file")
    parser.add_argument("--version", help="Dataset version label (default: v<timestamp>)")
    args = parser.parse_args()

    build_dataset_from_config(args.config, version=args.version)
