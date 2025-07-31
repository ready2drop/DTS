from typing import Sequence, Tuple, Dict

import os
import wandb

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

from monai.inferers import SlidingWindowInferer, sliding_window_inference
from monai.metrics import DiceMetric

from models.utils.model_type import ModelType
from losses.loss import Loss
from utils import model_hub, get_model_type, get_class_names

class Engine:
    def __init__(
        self,
        model_name: str = "DTS",
        data_name: str = "btcv",
        data_path: str = None,
        batch_size: int = 1,
        sw_batch_size: int = 4,
        overlap: float = 0.25,
        image_size: int = 256,
        spatial_size: int = 96,
        sliced: bool = False,
        noise_ratio: float = 0.5,
        timesteps: int = 1000,
        classes: str = None,
        device: str = "cpu",
        num_workers: int = 2,
        losses: str = "mse,cse,dice",
        loss_combine: str = 'sum',
        model_path: str = None,
        project_name: str = None,
        wandb_name: str = None,
        include_background: bool = False,
        label_smoothing: str = "k-nls",
        extract_features: bool = True,
        freeze: bool = False,
        use_amp: bool = True,
        use_cache: bool = True,
        use_wandb: bool = True,
        mode: str = "train",
    ):
        self.model_name = model_name
        self.model_type = get_model_type(model_name)
        self.data_name = data_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.sw_batch_size = sw_batch_size
        self.overlap = float(overlap)
        self.noise_ratio = noise_ratio
        self.image_size = image_size
        self.spatial_size = spatial_size
        self.timesteps = timesteps
        self.class_names = get_class_names(classes, include_background)
        self.num_classes = len(self.class_names)
        self.device = torch.device(device)
        self.num_workers = num_workers
        self.losses = losses
        self.loss_combine = loss_combine
        self.model_path = model_path
        self.project_name = project_name
        self.wandb_name = wandb_name
        self.include_background = include_background
        self.label_smoothing = label_smoothing
        self.extract_features = extract_features
        self.freeze = freeze
        self.use_amp = use_amp
        self.use_cache = use_cache
        self.use_wandb = use_wandb
        self.one_hot = True
        self.mode = mode
        self.global_step = 0
        self.best_mean_dice = 0
        self.loss = 0

        log_msg = f"number of classes: {self.num_classes} "
        log_msg += "(including background)" if include_background else "(excluding background)"

        print(log_msg)

        if self.mode == "train":
            self.criterion = Loss(self.losses,
                                  self.num_classes,
                                  self.loss_combine,
                                  self.one_hot,
                                  self.include_background)

        self.scaler = torch.cuda.amp.GradScaler()
        self.tensor2pil = transforms.ToPILImage()
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    def load_checkpoint(self, model_path: str):
        pass # to be implemented

    def load_model(self):
        return model_hub(
            model_name=self.model_name,
            timesteps=self.timesteps,
            in_channels=1,
            out_channels=self.num_classes,
            image_size=self.image_size,
            spatial_size=self.spatial_size,
            noise_ratio=self.noise_ratio,
            extract_features=self.extract_features,
            freeze=self.freeze,
            mode=self.mode
        ).to(self.device)

    def save_model(
        self,
        model,
        optimizer=None,
        scheduler=None,
        epoch=None,
        save_path=None,
    ):
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        if isinstance(model, nn.DataParallel):
            model = model.module

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'epoch': epoch+1,
            'loss': self.loss,
            'noise_ratio': self.noise_ratio,
            'global_step': self.global_step,
            'best_mean_dice': self.best_mean_dice,
            'project_name': self.project_name,
            'id': wandb.run.id if self.use_wandb else 0,
        }

        torch.save(state, save_path)
        print(f"Model is saved in {save_path}")

    def set_dataloader(self):
        pass

    def set_losses(self):
        pass

    def get_input(self, batch: dict, phase: str = "train"):
        image = batch["image"].to(self.device)
        label = batch["label"].to(self.device)
        label = self.convert_labels(label, phase).float()

        return image, label

    def convert_labels(self, labels: torch.Tensor, phase: str = "train"):
        if not self.include_background:
            if self.label_smoothing and phase == "train":
                return labels[:, 1:, ...]
            else:
                new_labels = [labels == i for i in sorted(self.class_names.keys())]
                return torch.cat(new_labels, dim=1)

        return labels

    def infer(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, labels = self.get_input(batch, phase="val")
        imgsz = (self.spatial_size, self.image_size, self.image_size)

        if self.model_type == ModelType.Diffusion:
            if isinstance(self.model, nn.DataParallel):
                outputs = sliding_window_inference(image, imgsz, self.sw_batch_size, self.model.module, self.overlap, pred_type="ddim_sample")
            else:
                outputs = sliding_window_inference(image, imgsz, self.sw_batch_size, self.model, self.overlap, pred_type="ddim_sample")
        else:
            outputs = sliding_window_inference(image, imgsz, self.sw_batch_size, self.model, self.overlap)

        outputs = torch.sigmoid(outputs)
        outputs = (outputs > 0.5).float()

        return image, outputs, labels

    def get_numpy_image(self, t: torch.Tensor, shape: tuple, is_label: bool = False):
        _, _, d, w, h = shape
        index = int(d * 0.75) # to be customed
        if is_label: t = torch.argmax(t, dim=1)
        else: t = t.squeeze(0) * 255
        t = t[:, index, ...].to(torch.uint8)
        t = t.cpu().numpy()
        t = np.transpose(t, (1, 2, 0))
        if is_label:
            t = t[:, :, 0]
            # t = cv2.resize(t, (w, h))

        return t

    def tensor2images(self,
                      image: torch.Tensor,
                      outputs: torch.Tensor,
                      labels: torch.Tensor,
                      shape: tuple):
        return {
            "image" : self.get_numpy_image(image, shape),
            "output" : self.get_numpy_image(outputs, shape, is_label=True),
            "label" : self.get_numpy_image(labels, shape, is_label=True),
        }

    def log(self, k, v, step=None, resume=False):
        if self.use_wandb:
            wandb.log({k: v}, step=step if step is not None else self.global_step, commit=resume)
