from __future__ import annotations
from typing import Dict, List

import math
import yaml
import argparse
import warnings
from pathlib import Path
from prettytable import PrettyTable
from collections import OrderedDict

from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from models.utils.model_hub import ModelHub
from models.utils.model_type import ModelType


def model_hub(model_name: str, **kwargs):
    return ModelHub().__call__(model_name, **kwargs)

def get_model_type(model_name: str):
    if model_name == "DTS":
        return ModelType.Diffusion
    elif model_name == "swin_unetr":
        return ModelType.SwinUNETR
    else:
        return ModelType.Network

def get_class_names(classes: Dict[int, str], include_background: bool = False, bg_index: int = 0):
     with open(classes, "r") as f:
        classes = OrderedDict(yaml.safe_load(f))
        if not include_background: del classes[0]
        return classes

def get_dataloader(
    data_name: str,
    data_path: str,
    image_size: int = 256,
    spatial_size: int = 96,
    num_classes: int = 14,
    num_samples: int = 1,
    num_workers: int = 8,
    batch_size: int = 1,
    cache_rate: float = 1.0,
    use_cache: bool = False,
    label_smoothing: str = "k-nls",
    smoothing_alpha: float = 0.3,
    smoothing_order: float = 1.0,
    lambda_decay: float = 1.0,
    mode: str = "train",
):
    if data_name == "btcv":
        from dataset.btcv_dataloader import Dataloader
        return Dataloader(
            data_name=data_name,
            data_path=data_path,
            image_size=image_size,
            spatial_size=spatial_size,
            num_classes=num_classes,
            num_samples=num_samples,
            num_workers=num_workers,
            batch_size=batch_size,
            cache_rate=cache_rate,
            use_cache=use_cache,
            label_smoothing=label_smoothing,
            smoothing_alpha=smoothing_alpha,
            smoothing_order=smoothing_order,
            lambda_decay=lambda_decay,
        ).generate(mode=mode)
    else:
        raise ValueError(f"Unsupported data name: {data_name}.")

class LinearWarmupCosineAnnealingLR(_LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                1 +
                math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        args = argparse.Namespace(**config)

    table = PrettyTable(["Argument", "Value"])
    for arg, value in args.__dict__.items():
        table.add_row([arg, value])

    print(table)

    return args

if __name__ == "__main__":
    args = parse_args()
