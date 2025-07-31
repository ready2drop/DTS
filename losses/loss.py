from typing import Sequence

import torch
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss

from monai.losses import FocalLoss, DiceLoss
from .utils import dist_map_transform

class Loss:
    def __init__(
        self,
        losses: Sequence[str],
        num_classes: int,
        loss_combine: str,
        one_hot: bool,
        include_background: bool
    ) -> None:
        self.losses = []
        self.num_classes = num_classes
        self.loss_combine = loss_combine
        self.one_hot = one_hot
        self.include_background = include_background
        self.dist_transform = dist_map_transform()
        self.dist_matrix = torch.ones(num_classes, num_classes, dtype=torch.float32)

        loss_types = {
            "mse": MSELoss(),
            "ce": CrossEntropyLoss(),
            "bce": BCEWithLogitsLoss(),
            "dice": DiceLoss(sigmoid=True),
            "focal": FocalLoss(),
        }

        for name in losses.split(','):
            if name in loss_types.keys():
                self.losses.append(loss_types[name])
            else:
                raise NotImplementedError(f"Loss ({name}) is not listed yet")

        print(f"loss: {self.losses}")

    def __call__(self, preds: torch.Tensor, labels: torch.Tensor):
        losses = []

        for loss in self.losses:
            if isinstance(loss, MSELoss):
                losses.append(loss(torch.sigmoid(preds), labels))
            else:
                losses.append(loss(preds, labels))

        if len(losses) == 1: return losses[0]

        if self.loss_combine == 'sum':
            return torch.stack(losses).sum()
        elif self.loss_combine == 'mean':
            return torch.stack(losses).mean()
        elif self.loss_combine == 'log':
            return torch.log(1 + torch.stack(losses).sum())
        else:
            raise NotImplementedError("Unsupported value for loss_combine. Please choose from 'sum', 'mean', or 'log'.")
