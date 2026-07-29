"""Build a training-ready COCO dataset folder from Label Studio + GCS.
Includes the following steps:

1. pull_from_label_studio  — raw tasks -> COCO annotations JSON.
2. group_by_split          — one COCO JSON per train/val/test, using each
                             image's own "split" field (stamped upstream in the
                             Label Studio task meta).
3. materialize_dataset     — download the image bytes from GCS into
                             <dataset_dir>/{train,valid,test}/ next to an
                             _annotations.coco.json, the layout RF-DETR expects.
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
) -> None:
    """Download each split's images from GCS into <dataset_dir>/<rfdetr_dir>/.
    """
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
        _save_coco(coco, os.path.join(split_dir, "_annotations.coco.json"))

def build_dataset(
    url: str,
    api_key: str,
    project_id: int,
    defect_classes: dict[int, str],
    dataset_dir: str,
) -> str:
    """Run the full pipeline: Label Studio -> COCO JSONs -> downloaded images.
    Args:
        url: Label Studio URL
        api_key: Label Studio API key
        project_id: Label Studio project ID
        defect_classes: {category_id: class_name} for the defect classes
        dataset_dir: Directory to save the dataset
    Returns:
        path to the dataset_dir
    """

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_annotations_path = f"tmp/annotations_{run_id}.json"

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

    materialize_dataset(split_paths, dataset_dir=dataset_dir)
    logger.info(f"Dataset built at {dataset_dir} with splits: {split_paths}")
    return dataset_dir

def build_dataset_from_config(config_path: str) -> str:
    """Run build_dataset() with the settings in a data.yaml config file.

    Reads .env first so the "${VAR}" placeholders in the config (and
    GOOGLE_APPLICATION_CREDENTIALS, used by the GCS client) resolve.
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
        dataset_dir=cfg["dataset_dir"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a COCO dataset from Label Studio + GCS")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to the YAML config file")
    args = parser.parse_args()

    build_dataset_from_config(args.config)
