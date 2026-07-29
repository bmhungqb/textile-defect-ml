"""Pull reviewed annotations from Label Studio into COCO format.
"""

import base64
from typing import Any

from label_studio_sdk import Client

from utils.logger import get_logger

logger = get_logger(__name__)

_SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "test": "test"}


def _extract_image_url(raw_image: str) -> str:
    if "fileuri=" in raw_image:
        b64_str = raw_image.split("fileuri=")[-1].split("&")[0]
        return base64.b64decode(b64_str).decode("utf-8")
    return raw_image


def _process_task(task: dict, label_to_id: dict[str, int]) -> dict[str, Any] | None:
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


def pull_tasks(url: str, api_key: str, project_id: int, defect_classes: dict[int, str]) -> dict:
    """Pull all reviewed tasks from a Label Studio project as a COCO dict.
    """
    label_to_id = {v: k for k, v in defect_classes.items()}

    client = Client(url, api_key)
    project = client.get_project(project_id)
    tasks = project.get_tasks()
    logger.info(f"Pulled {len(tasks)} tasks from Label Studio project {project_id}")

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
