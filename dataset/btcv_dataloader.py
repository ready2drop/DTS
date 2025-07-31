from __future__ import annotations

import os
import glob
import json
from pathlib import Path

from monai.apps import DecathlonDataset
from monai.config import PathLike
from monai.data.decathlon_datalist import _append_paths
from monai.data import (
    CacheDataset,
    Dataset,
    ThreadDataLoader,
)

from .btcv_cache_dataset import BTCVLabelSmoothingCacheDataset
from .btcv_transforms import Transforms


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val

def load_decathlon_datalist(
    data_list_file_path: PathLike,
    is_segmentation: bool = True,
    data_list_key: str = "training",
    base_dir: PathLike | None = None,
) -> list[dict]:
    """Load image/label paths of decathlon challenge from JSON file

    Json file is similar to what you get from http://medicaldecathlon.com/
    Those dataset.json files

    Args:
        data_list_file_path: the path to the json file of datalist.
        is_segmentation: whether the datalist is for segmentation task, default is True.
        data_list_key: the key to get a list of dictionary to be used, default is "training".
        base_dir: the base directory of the dataset, if None, use the datalist directory.

    Raises:
        ValueError: When ``data_list_file_path`` does not point to a file.
        ValueError: When ``data_list_key`` is not specified in the data list file.

    Returns a list of data items, each of which is a dict keyed by element names, for example:

    .. code-block::

        [
            {'image': '/workspace/data/chest_19.nii.gz',  'label': 0},
            {'image': '/workspace/data/chest_31.nii.gz',  'label': 1}
        ]

    """
    data_list_file_path = Path(data_list_file_path)
    if not data_list_file_path.is_file():
        raise ValueError(f"Data list file {data_list_file_path} does not exist.")
    with open(data_list_file_path) as json_file:
        json_data = json.load(json_file)
    if data_list_key not in json_data:
        raise ValueError(f'Data list {data_list_key} not specified in "{data_list_file_path}".')
    expected_data = json_data[data_list_key]
    if data_list_key == "test" and not isinstance(expected_data[0], dict):
        # decathlon datalist may save the test images in a list directly instead of dict
        expected_data = [{"image": i} for i in expected_data]

    if base_dir is None:
        base_dir = data_list_file_path.parent

    return _append_paths(base_dir, is_segmentation, expected_data)


class Dataloader:
    def __init__(
        self,
        data_name: str,
        data_path: str,
        image_size: int = 256,
        spatial_size: int = 96,
        num_classes: int = 14,
        num_samples: int = 1,
        num_workers: int = 4,
        batch_size: int = 1,
        cache_rate: float = 0.1,
        use_cache: bool = False,
        label_smoothing: str = "k-nls",
        smoothing_alpha: float = 0.3,
        smoothing_order: float = 1.0,
        lambda_decay: float = 1.0,
    ):
        self.data_name = data_name
        self.data_path = data_path
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.cache_rate = cache_rate
        self.use_cache = use_cache
        self.label_smoothing = label_smoothing
        self.smoothing_alpha = smoothing_alpha
        self.smoothing_order = smoothing_order
        self.lambda_decay = lambda_decay
        self.transforms = Transforms(
            data_name=data_name,
            spatial_size=spatial_size,
            image_size=image_size,
            num_samples=num_samples,
            label_smoothing=label_smoothing,
        ).generate()

    def generate(self, mode):
        if mode == "train":
            phase = ["train", "val", "test"]
        elif mode == "test":
            phase = ["val"]

        dataloader = {}
        for p in phase:
            if mode == "train" and p == "test": continue
            elif mode == "test" and p == "train": continue
            data = load_decathlon_datalist(os.path.join(self.data_path, "dataset.json"), True, self.parse_type(p))

            if self.label_smoothing:
                assert self.num_classes > 2, "Label smoothing requires num_classes > 2"
            if p == "train" and self.label_smoothing:
                dataset = BTCVLabelSmoothingCacheDataset(
                    data=data,
                    transform=self.transforms[p],
                    cache_num=len(data),
                    cache_rate=self.cache_rate,
                    num_workers=max(self.num_workers, 20),
                    num_classes=self.num_classes,
                    label_smoothing=self.label_smoothing,
                    smoothing_alpha=self.smoothing_alpha,
                    smoothing_order=self.smoothing_order,
                )
            else:
                if self.use_cache:
                    dataset = CacheDataset(
                        data=data,
                        transform=self.transforms[p],
                        cache_num=len(data),
                        cache_rate=self.cache_rate,
                        num_workers=max(self.num_workers, 20),
                    )
                else:
                    dataset = Dataset(
                        data=data,
                        transform=self.transforms[p],
                    )

            dataloader[p] = ThreadDataLoader(
                dataset=dataset,
                num_workers=self.num_workers,
                batch_size=self.batch_size if p == "train" else 1,
                shuffle=False # True if p == "train" else False
            )

        return dataloader

    def parse_type(self, p):
        if p == "train":
            return "training"
        elif p == "val":
            return "validation"
        else:
            return p