from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    return int(value[0]), int(value[1])


def _same_padding(size: int, kernel: int, stride: int, dilation: int = 1) -> tuple[int, int]:
    """TensorFlow/MATLAB-style SAME padding for one spatial dimension."""
    out = math.ceil(size / stride)
    effective = dilation * (kernel - 1) + 1
    total = max((out - 1) * stride + effective - size, 0)
    before = total // 2
    return before, total - before


class SamePadConv2d(nn.Module):
    """Conv2d with dynamic asymmetric SAME padding, including stride > 1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top, bottom = _same_padding(x.shape[-2], self.kernel_size[0], self.stride[0])
        left, right = _same_padding(x.shape[-1], self.kernel_size[1], self.stride[1])
        if top or bottom or left or right:
            x = F.pad(x, (left, right, top, bottom))
        return self.conv(x)


class SamePadMaxPool2d(nn.Module):
    """MATLAB-style maxPooling2dLayer(..., Padding='same')."""

    def __init__(self, kernel_size: int | Sequence[int], stride: int | Sequence[int]) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top, bottom = _same_padding(x.shape[-2], self.kernel_size[0], self.stride[0])
        left, right = _same_padding(x.shape[-1], self.kernel_size[1], self.stride[1])
        if top or bottom or left or right:
            # MATLAB pads max pooling so padded values cannot become maxima.
            x = F.pad(x, (left, right, top, bottom), value=float("-inf"))
        return F.max_pool2d(x, self.kernel_size, self.stride)


class PaperResidualBlock(nn.Module):
    """Exact residual stage from CIR_CNN_CIRVar_Tst.m."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            SamePadConv2d(in_channels, out_channels, (4, 1), stride=(2, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SamePadConv2d(out_channels, out_channels, (4, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Sequential(
            SamePadConv2d(in_channels, out_channels, (1, 1), stride=(2, 1), bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.main(x) + self.skip(x))


class PaperResidualCNN(nn.Module):
    """Faithful PyTorch port of the official MATLAB residual CNN.

    Official graph:
      imageInput(500x2x1, zero-center)
      Conv(10x1,8)-BN-ReLU-MaxPool(10x1,stride 5x1)
      Conv(4x2,16)-BN-ReLU-MaxPool(4x2,stride 2x2)
      residual stages 32, 64, 128 with 4x1 kernels and 2x1 downsampling
      global max pool -> FC10 -> FC1 regression.

    The scalar output is the MATLAB-style one-based delay index. No sigmoid or
    output clipping is used, matching the source implementation.
    """

    def __init__(self, input_length: int = 500, input_mean: torch.Tensor | None = None) -> None:
        super().__init__()
        self.input_length = int(input_length)
        if input_mean is None:
            input_mean = torch.zeros(1, 1, self.input_length, 2, dtype=torch.float32)
        self.register_buffer("input_mean", input_mean.detach().clone().float(), persistent=True)
        self.stem = nn.Sequential(
            SamePadConv2d(1, 8, (10, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            SamePadMaxPool2d((10, 1), stride=(5, 1)),
            SamePadConv2d(8, 16, (4, 2), stride=(1, 1), bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            SamePadMaxPool2d((4, 2), stride=(2, 2)),
        )
        self.residual = nn.Sequential(
            PaperResidualBlock(16, 32),
            PaperResidualBlock(32, 64),
            PaperResidualBlock(64, 128),
        )
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        # The MATLAB graph has no activation between FC-10 and FC-1.
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 10), nn.Linear(10, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x - self.input_mean
        h = self.pool(self.residual(self.stem(x)))
        mean_index = self.head(h).squeeze(-1)
        return {
            "mean_index": mean_index,
            # Convenience only; training/prediction of the reproduction path
            # uses mean_index directly.
            "mean_fraction": mean_index / max(float(self.input_length), 1.0),
        }
