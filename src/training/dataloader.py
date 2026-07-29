"""Class-imbalance-aware training dataloader for RF-DETR.
"""

import math
from collections import defaultdict
from typing import Any

from torch.utils.data import DataLoader, WeightedRandomSampler

from rfdetr import RFDETRDataModule

try:
    from rfdetr.training.module_data import _worker_init_fn
except ImportError:
    _worker_init_fn = None


class WeightedRFDETRDataModule(RFDETRDataModule):
    """RFDETRDataModule with per-image, inverse-class-frequency sampling on train only."""

    def _compute_image_weights(self, dataset) -> list[float]:
        coco = dataset.coco
        image_ids = dataset.ids

        cat_counts: dict[str, int] = defaultdict(int)
        for ann in coco.anns.values():
            cat_counts[coco.cats[ann["category_id"]]["name"]] += 1

        class_weight = {name: 1.0 / count for name, count in cat_counts.items()}
        min_weight = min(class_weight.values())

        image_weights = []
        for image_id in image_ids:
            names = {coco.cats[a["category_id"]]["name"] for a in coco.imgToAnns.get(image_id, [])}
            weights_present = [class_weight[n] for n in names if n in class_weight]
            image_weights.append(max(weights_present) if weights_present else min_weight)
        return image_weights

    def train_dataloader(self) -> DataLoader[Any]:
        dataset = self._dataset_train
        batch_size = self.train_config.batch_size
        effective_batch_size = batch_size * self.train_config.grad_accum_steps

        weights = self._compute_image_weights(dataset)

        min_batches = 5
        target_samples = max(len(dataset), effective_batch_size * min_batches)
        num_samples = math.ceil(target_samples / effective_batch_size) * effective_batch_size

        sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
            **({"worker_init_fn": _worker_init_fn} if _worker_init_fn is not None else {}),
        )
