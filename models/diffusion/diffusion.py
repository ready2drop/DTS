from typing import Sequence

import torch
import torch.nn as nn

from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler


class Diffusion(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 3,
        out_channels: int = 1,
        image_size: int = 96,
        spatial_size: int = 96,
        features: Sequence[int] = [32, 64, 128, 256, 512], # [64, 128, 256, 512, 1024],
        dropout: float = 0.2,
        timesteps: int = 1000,
        extract_features: bool = True,
        mode: str = "train"
    ):
        super().__init__()
        self.num_classes = out_channels
        self.extract_features = extract_features
        self.mode = mode

        self.embed_model: nn.Module = None
        self.model: nn.Module = None

        betas = get_named_beta_schedule("linear", timesteps)

        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(timesteps, [timesteps]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.RESCALED_KL,
        )

        self.sample_diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(timesteps, [10]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.RESCALED_KL,
        )

        self.sampler = UniformSampler(timesteps)

    def forward(
        self,
        image: torch.Tensor = None,
        x: torch.Tensor = None,
        step: torch.Tensor = None,
        pred_type: str = None
    ):
        if image is not None and x is not None: assert image.device == x.device

        if pred_type == "q_sample":
            return self.q_sample(x)
        elif pred_type == "denoise":
            return self.denoise(image, x, step)
        elif pred_type == "ddim_sample":
            return self.ddim_sample(image)
        else:
            raise NotImplementedError(f"No such prediction type : {pred_type}")

    def q_sample(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        noise = torch.randn_like(x).to(x.device)
        t, _ = self.sampler.sample(x.shape[0], x.device)
        sample = self.diffusion.q_sample(x, t, noise)
        return sample, t, noise

    def denoise(self, image: torch.Tensor, x: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Denoise the image

        Args:
            image (torch.Tensor): image
            x (torch.Tensor): noise
            step (torch.Tensor): timestep

        Returns:
            torch.Tensor: denoised image
        """
        assert image.size(0) == x.size(0) == step.size(0)

        embeddings = self.embed_model(image) if self.extract_features else None
        return self.model(x=x, t=step, embeddings=embeddings, image=image)

    def ddim_sample(self, image: torch.Tensor) -> torch.Tensor:
        """Sample from the model using the DDIM method

        Args:
            image (torch.Tensor): image

        Returns:
            torch.Tensor: results
        """
        res = []
        for i in range(len(image)):
            batch = image[i, ...].unsqueeze(0)
            embeddings = self.embed_model(batch) if self.extract_features else None
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model,
                                                                (1, self.num_classes, *image.shape[2:]),
                                                                model_kwargs={"image": batch, "embeddings": embeddings})
            sample_return = torch.zeros((1, self.num_classes, *image.shape[2:])).to(image.device)
            all_samples = sample_out["all_samples"]

            for sample in all_samples:
                sample_return += sample.to(image.device)

            res.append(sample_return)

        return torch.cat(res, dim=0)