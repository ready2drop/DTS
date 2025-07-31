# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import os
import uuid
import time
from tqdm import tqdm

import math
import numpy as np
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

from .blocks import UnetOutBlock, UnetrUpBlock, UnetrBasicBlock, UnetBasicBlock
from .patch import MERGING_MODE
from .transformer import SwinTransformer
from ..diffusion import get_timestep_embedding, nonlinearity, TimeStepEmbedder

rearrange, _ = optional_import("einops", name="rearrange")


def euclidean_distance_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Euclidean distance transform of a binary image using PyTorch,
    with modifications to replace 'inf' values with 0.
    """
    return torch.tensor(distance_transform_edt(x.detach().cpu().numpy())).to(device=x.device, dtype=x.dtype)

class ReverseAttentionUpsampleBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        embedding_size: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.conv_block = UnetBasicBlock(  # type: ignore
            spatial_dims,
            out_channels,
            out_channels,
            embedding_size=embedding_size,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = self.conv_block(out)
        return out

class ReverseAttentionConvBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int = 3,
        dropout: float = 0.0,
        norm_name: tuple | str = "instance",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            norm_name: feature normalization type and arguments.

        """

        super().__init__()
        self.layer0 = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            strides=1,
            dropout=dropout,
            norm=norm_name,
        )

        self.layer1 = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            strides=1,
            dropout=dropout,
            norm=norm_name,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = torch.relu(x)
        x = self.layer1(x)
        return x

class DTSDenoiser(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        image_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        embedding_size: int = 512,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        embedding_dim: int = 128, # for time embedding
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        mask_ratio: float = None,
        noise_ratio: float = 0.5,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        extract_features: bool = True,
        freeze: bool = False,
        use_v2=False,
    ) -> None:
        """
        Args:
            image_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(image_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(image_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(image_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        image_size = ensure_tuple_rep(image_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = out_channels
        self.extract_features = extract_features

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(image_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (image_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        # timesteps & noise
        self.noise_ratio = noise_ratio
        self.t_embedder = TimeStepEmbedder(embedding_dim)

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            embedding_size=embedding_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            embedding_size=embedding_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            embedding_size=embedding_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        if not extract_features and freeze:
            self.freeze()

        # Reverse Attention branch
        self.reverse_conv5 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=4 * feature_size,
            strides=8,
            kernel_size=3,
            norm=norm_name,
            dropout=drop_rate,
        )
        self.reverse_conv4 = ReverseAttentionConvBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            norm_name=norm_name,
            dropout=drop_rate,
        )
        self.reverse_decoder3 = ReverseAttentionUpsampleBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )
        self.reverse_conv3 = ReverseAttentionConvBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            norm_name=norm_name,
            dropout=drop_rate,
        )
        self.reverse_decoder2 = ReverseAttentionUpsampleBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )
        self.reverse_conv2 = ReverseAttentionConvBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            norm_name=norm_name,
            dropout=drop_rate,
        )
        self.reverse_decoder1 = ReverseAttentionUpsampleBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )
        self.reverse_conv1 = ReverseAttentionConvBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            norm_name=norm_name,
            dropout=drop_rate,
        )

        self.reverse_out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        image: torch.Tensor = None,
        embeddings: Any = None,
    ):
        t = self.t_embedder(t)
        x = torch.cat([image, x], dim=1)

        hidden_states_out = self.swinViT(x, t, self.normalize)

        for i in range(len(hidden_states_out)):
            if embeddings is not None:
                hidden_states_out[i] = hidden_states_out[i] + embeddings[0][i]
            else:
                hidden_states_out[i] = hidden_states_out[i]

        if embeddings is not None:
            enc0 = self.encoder1(x, t) + embeddings[1]
            enc1 = self.encoder2(hidden_states_out[0], t) + embeddings[2]
            enc2 = self.encoder3(hidden_states_out[1], t) + embeddings[3]
            enc3 = self.encoder4(hidden_states_out[2], t) + embeddings[4]
        else:
            enc0 = self.encoder1(x, t)
            enc1 = self.encoder2(hidden_states_out[0], t)
            enc2 = self.encoder3(hidden_states_out[1], t)
            enc3 = self.encoder4(hidden_states_out[2], t)

        dec4 = self.encoder10(hidden_states_out[4], t)
        dec3 = self.decoder5(dec4, hidden_states_out[3], t)
        dec2 = self.decoder4(dec3, enc3, t)
        dec1 = self.decoder3(dec2, enc2, t)
        dec0 = self.decoder2(dec1, enc1, t)
        out = self.decoder1(dec0, enc0, t)
        logits = self.out(out)

        x = self.reverse_conv5(logits)
        ra, ba = self.boundary_attention(x, enc3)
        c = self.reverse_conv4(ra)
        x = x + c # + ba

        x = self.reverse_decoder3(x)
        ra, ba = self.boundary_attention(x, enc2)
        c = self.reverse_conv3(ra)
        x = x + c # + ba

        x = self.reverse_decoder2(x)
        ra, ba = self.boundary_attention(x, enc1)
        c = self.reverse_conv2(ra)
        x = x + c # + ba

        x = self.reverse_decoder1(x)
        ra, ba = self.boundary_attention(x, enc0)
        c = self.reverse_conv1(ra)
        x = x + c # + ba

        logits = self.reverse_out(x)

        return logits

    def freeze(self):
        print("Freezing model....")
        encoders: Sequence[nn.Module] = [self.swinViT, self.encoder1, self.encoder2, self.encoder3, self.encoder4]
        for encoder in encoders:
            for e in encoder._modules:
                for p in encoder._modules[e].parameters():
                    p.requires_grad = False

    def boundary_attention(self, x: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        r = -torch.sigmoid(x) + 1
        ra = torch.mul(enc, r)
        r_binary = self.threshold_values(r)

        dt_x = euclidean_distance_transform(x)
        dt_r = euclidean_distance_transform(r_binary)
        dt_sum = dt_x + dt_r

        boundary = x * torch.sigmoid(1 - dt_sum)
        output = x + torch.mul(x, r_binary)
        ba = torch.mul(output, boundary)

        return ra, ba

    def threshold_values(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (torch.sigmoid(x) > threshold).to(x.dtype)
