import os
import wandb
import warnings
import yaml
from tqdm import tqdm

import torch
from torch.nn.parallel import DataParallel
import numpy as np

from engine import Engine
from models.utils.model_type import ModelType
from models.diffusion import Diffusion
from models import DTS
from metric import dice_coeff
from utils import parse_args, get_dataloader, LinearWarmupCosineAnnealingLR

warnings.filterwarnings("ignore")

class Trainer(Engine):
    def __init__(
        self,
        model_name: str = "DTS",
        data_name: str = "btcv",
        data_path: str = None,
        max_epochs: int = 5000,
        batch_size: int = 10,
        sw_batch_size: int = 4,
        overlap: float = 0.25,
        image_size: int = 256,
        spatial_size: int = 96,
        sliced: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        noise_ratio: float = 0.5,
        scheduler: str = None,
        warmup_epochs: int = 100,
        timesteps: int = 1000,
        classes: str = None,
        val_freq: int = 1,
        save_freq: int = 5,
        device: str = "cpu",
        device_ids: str = "0,1,2,3",
        num_workers: int = 2,
        losses: str = "mse,bce,dice",
        loss_combine: str = 'sum',
        log_dir: str = "logs",
        model_path: str = None,
        pretrained_path: str = None,
        project_name: str = "DTS",
        wandb_name: str = None,
        include_background: str = False,
        label_smoothing: str = "k-nls",
        smoothing_alpha: float = 0.3,
        smoothing_order: float = 1.0,
        lambda_decay: float = 1.0,
        extract_features: bool = True,
        freeze: bool = False,
        use_amp: bool = True,
        use_cache: bool = True,
        use_wandb: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            data_name=data_name,
            data_path=data_path,
            batch_size=batch_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            image_size=image_size,
            spatial_size=spatial_size,
            sliced=sliced,
            noise_ratio=noise_ratio,
            timesteps=timesteps,
            classes=classes,
            device=device,
            num_workers=num_workers,
            losses=losses,
            loss_combine=loss_combine,
            model_path=model_path,
            project_name=project_name,
            wandb_name=wandb_name,
            include_background=include_background,
            label_smoothing=label_smoothing,
            extract_features=extract_features,
            freeze=freeze,
            use_amp=use_amp,
            use_cache=use_cache,
            use_wandb=use_wandb,
            mode="train",
        )
        self.max_epochs = max_epochs
        self.lr = float(lr)
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.val_freq = val_freq
        self.save_freq = save_freq
        self.num_workers = num_workers
        self.log_dir = os.path.join("logs", log_dir)
        self.pretrained = pretrained_path is not None
        self.use_cache = use_cache
        self.smothing_alpha = smoothing_alpha
        self.smoothing_order = smoothing_order
        self.lambda_decay = lambda_decay

        self.local_rank = 0
        self.start_epoch = 0
        self.wandb_id = None
        self.weights_path = os.path.join(self.log_dir, "weights")
        self.auto_optim = True
        self.resume = model_path is not None

        self.vis_path = "figures"

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.weights_path, exist_ok=True)

        self.dataloader = self.get_dataloader()
        self.model = self.load_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        if scheduler is not None:
            print("Training with scheduler...")
            self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                           warmup_epochs=warmup_epochs,
                                                           max_epochs=max_epochs)

        if model_path is not None:
            self.load_checkpoint(model_path)
        elif self.pretrained:
            self.load_pretrained_weights(pretrained_path)

        if device_ids:
            self.model = DataParallel(self.model, device_ids=list(map(int, device_ids.split(','))))
            self.label_smoother = DataParallel(self.label_smoother, device_ids=list(map(int, device_ids.split(','))))

        if use_wandb:
            if model_path is None:
                if wandb_name is None: wandb_name = log_dir
                wandb.init(project=self.project_name,
                           name=f"{wandb_name}",
                           config=self.__dict__)
            else:
                assert self.wandb_id != 0
                wandb.init(project=self.project_name,
                           id=self.wandb_id,
                           resume=True)

    def load_checkpoint(self, model_path):
        state_dict = torch.load(model_path)
        for k in ['model', 'optimizer', 'scheduler']:
            if state_dict[k] is not None:
                getattr(self, k).load_state_dict(state_dict[k])
        self.start_epoch = state_dict['epoch']
        self.noise_ratio = state_dict['noise_ratio']
        self.project_name = state_dict['project_name']
        self.global_step = state_dict['global_step']
        self.best_mean_dice = state_dict['best_mean_dice']
        self.wandb_id = state_dict['id']

        print(f"Checkpoint loaded from {model_path}")

    def load_pretrained_weights(self, pretrained_path):
        if self.model_type == ModelType.Diffusion and isinstance(self.model, Diffusion):
            if isinstance(self.model, DTS) and os.path.basename(pretrained_path) == 'swinvit.pt':
                self.model.embed_model.swinViT.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
                print(f"Load pretrained weights from {pretrained_path} to swinViT layer")
            else:
                self.model.embed_model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
                print(f"Load pretrained weights from {pretrained_path}")

        if self.freeze:
            if isinstance(self.model, DTS):
                self.model.embed_model.requires_grad_(False)

    def get_dataloader(self):
        return get_dataloader(
            data_name=self.data_name,
            data_path=self.data_path,
            image_size=self.image_size,
            spatial_size=self.spatial_size,
            num_classes=self.num_classes if self.include_background else self.num_classes + 1,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            use_cache=self.use_cache,
            label_smoothing=self.label_smoothing,
            smoothing_alpha=self.smothing_alpha,
            smoothing_order=self.smoothing_order,
            lambda_decay=self.lambda_decay,
            mode="train"
        )

    def train(self):
        os.makedirs(self.log_dir, exist_ok=True)

        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch
            self.train_epoch(epoch)

            if (epoch + 1) % self.val_freq == 0:
                self.model.eval()
                dices = []
                for batch in tqdm(self.dataloader["val"], total=len(self.dataloader["val"])):
                    with torch.no_grad():
                        dices.append(self.validation_step(batch))

                self.validation_end(dices, epoch)

    def train_epoch(self, epoch):
        running_loss = 0
        self.model.train()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        patient_id = 0

        with tqdm(total=len(self.dataloader["train"])) as t:
            for batch in self.dataloader["train"]:
                self.global_step += 1
                self.optimizer.zero_grad()
                t.set_description('Epoch %i' % epoch)

                for param in self.model.parameters(): param.grad = None
                with torch.cuda.amp.autocast(self.use_amp):
                    loss = self.training_step(batch, epoch, patient_id).float()

                patient_id += 1

                t.set_postfix(loss=loss.item(), lr=lr)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if torch.isnan(loss).item():
                    raise Exception("Training stopped due to the loss being NaN")

                running_loss += loss.item()
                t.update(1)

            if self.scheduler is not None: self.scheduler.step()

            self.loss = running_loss / len(self.dataloader["train"])
            self.log("loss", self.loss, step=epoch, resume=self.resume)

        if (epoch + 1) % self.save_freq == 0:
            self.save_model(model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=self.epoch,
                            save_path=os.path.join(self.weights_path, f"epoch_{epoch+1}.pt"))

    def training_step(self, batch, epoch, patient_id):
        images, labels = self.get_input(batch)

        if self.model_type == ModelType.Diffusion:
            x_start = (labels) * 2 - 1
            x_t, t, _ = self.model(x=x_start, pred_type="q_sample")
            preds = self.model(x=x_t, step=t, image=images, pred_type="denoise")
        else:
            preds = self.model(images)

        print(f"Images shape: {images.shape}, Labels shape: {labels.shape}, Preds shape: {preds.shape}")
        return self.compute_loss(preds, labels)

    def compute_loss(self, preds, labels, distances=None):
        return self.criterion(preds, labels)

    def validation_step(self, batch):
        with torch.cuda.amp.autocast(self.use_amp):
            _, outputs, labels = self.infer(batch)

        dices = []
        for i in range(self.num_classes):
            output = outputs[:, i]
            label = labels[:, i]
            if output.sum() > 0 and label.sum() == 0:
                dice = torch.Tensor([1.0]).squeeze().to(outputs.device)
            else:
                dice = dice_coeff(output, label).to(outputs.device)

            dices.append(dice)

        return torch.mean(torch.stack(dices))

    def validation_end(self, dices, epoch):
        mean_dice = torch.mean(torch.stack(dices)).item()
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            if mean_dice > 0.5:
                self.save_model(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=self.epoch,
                    save_path=os.path.join(self.weights_path, f"best_{mean_dice:.4f}.pt")
                )

        print(f"mean_dice: {mean_dice:.4f}")
        self.log("mean_dice", mean_dice, epoch)


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(**vars(args))
    trainer.train()
