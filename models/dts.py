from typing import Sequence

import torch

from .blocks import DTSEncoder, DTSDenoiser
from .diffusion import Diffusion


class DTS(Diffusion):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        image_size: int = 96,
        spatial_size: int = 96,
        features: Sequence[int] = [64, 64, 128, 256, 512, 64],
        feature_size: int = None,
        noise_ratio: float = 0.5,
        dropout: float = 0.2,
        timesteps: int = 1000,
        extract_features: bool = True,
        freeze: bool = False,
        mode: str = "train"
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            spatial_size=spatial_size,
            features=features,
            dropout=dropout,
            timesteps=timesteps,
            extract_features=extract_features,
            mode=mode,
        )
        self.embed_model = DTSEncoder(
            image_size,
            in_channels,
            spatial_dims=spatial_dims,
            feature_size=feature_size,
            drop_rate=dropout,
            freeze=freeze,
        ) if extract_features else None
        self.model = DTSDenoiser(
            image_size,
            out_channels + 1,
            out_channels,
            spatial_dims=spatial_dims,
            feature_size=feature_size,
            noise_ratio=noise_ratio,
            drop_rate=dropout,
            extract_features=extract_features,
            freeze=freeze,
        )