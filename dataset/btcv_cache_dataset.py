from __future__ import annotations

import os
import sys

from collections.abc import Callable, Sequence
from multiprocessing.managers import ListProxy

import numpy as np
from PIL import Image

import torch

from monai.data import CacheDataset
from monai.data.utils import pickle_hashing
from monai import transforms
from monai.transforms import (
    LoadImaged,
    RandomizableTrait,
    Transform,
    convert_to_contiguous,
)

from .label_smoothing import LabelSmoothingHub

class BTCVVisaulizer:
    def __init__(self):
        self.color_map = {
            0: [0, 0, 0],
            1: [255, 0, 0],
            2: [0, 255, 0],
            3: [0, 0, 255],
            4: [255, 255, 0],
            5: [0, 255, 255],
            6: [255, 0, 255],
            7: [0, 255, 127],
            8: [128, 128, 0],
            9: [128, 0, 128],
            10: [255, 165, 0],
            11: [255, 192, 203],
            12: [75, 0, 130],
            13: [0, 128, 0]
        }

    def visualize(self, org: torch.Tensor, labels: torch.Tensor, save_path: str = "figures"):
        os.makedirs(save_path, exist_ok=True)
        self.convert_to_color(org, labels, self.color_map, save_path)

    def __call__(self, org: torch.Tensor, labels: torch.Tensor, color_map, save_path):
        C, W, H, D = labels.shape

        for d in range(D):
            idx = len(os.listdir(save_path))
            org_img = org[0, :, :, d]
            org_color_img = np.zeros((W, H, 3), dtype=np.uint8)

            for i in range(C):
                mask = (org_img == i)
                org_color_img[mask] = color_map[i]

            org_color_img = Image.fromarray(org_color_img)
            org_color_img.save(os.path.join(save_path, f"org_{idx}.png"))

            labels_img = labels[0, :, :, d, :]
            labels_color_img = np.zeros((W, H, 4), dtype=np.float32)

            for i in range(C):
                alpha = labels_img[i, :, :]
                color = np.array(color_map[i], dtype=np.float32) / 255.0
                mask = alpha > 0

                labels_color_img[mask, :3] = color
                labels_color_img[mask, 3] = alpha[mask]

            labels_color_img = (labels_color_img * 255).astype(np.uint8)
            labels_color_img = Image.fromarray(labels_color_img, "RGBA")
            labels_color_img.save(os.path.join(save_path, f"labels_{idx}.png"))

        return labels

class BTCVLabelSmoothingCacheDataset(CacheDataset):
    def __init__(
        self,
        data: Sequence,
        transform: Sequence[Callable] | Callable | None = None,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int | None = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
        hash_as_key: bool = False,
        hash_func: Callable[..., bytes] = pickle_hashing,
        runtime_cache: bool | str | list | ListProxy = False,
        num_classes: int = 14,
        label_smoothing: str = "k-nls",
        smoothing_alpha: float = 0.1,
        smoothing_order: float = 1.0,
        lambda_decay: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        self.num_classes = num_classes
        self.smoothing_alpha = smoothing_alpha
        self.smoothing_order = smoothing_order
        self.lambda_decay = lambda_decay
        self.epsilon = epsilon
        self.visualize = BTCVVisaulizer()
        self.image_loader = transforms.Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ])
        self.label_smoothing = LabelSmoothingHub(name=label_smoothing)(
            alpha=smoothing_alpha,
            num_classes=num_classes,
            smoothing_order=smoothing_order,
            lambda_decay=lambda_decay,
            epsilon=epsilon,
        )
        super().__init__(
            data=data,
            transform=transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress,
            copy_cache=copy_cache,
            as_contiguous=as_contiguous,
            hash_as_key=hash_as_key,
            hash_func=hash_func,
            runtime_cache=runtime_cache,
        )

    def _load_cache_item(self, idx: int):
        """
        Args:
            idx: the index of the input data sequence.
        """
        item = self.data[idx]
        item = self.image_loader(item)
        item['label'] = self.label_smoothing(item['label'])

        first_random = self.transform.get_index_of_first(
            lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
        )
        item = self.transform(item, end=first_random, threading=True)

        if self.as_contiguous:
            item = convert_to_contiguous(item, memory_format=torch.contiguous_format)
        return item