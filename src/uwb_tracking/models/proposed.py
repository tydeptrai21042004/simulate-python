from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DepthwiseSeparable1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 16, 9, padding=4, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            DepthwiseSeparable1D(16, 24, 7, stride=2),
            DepthwiseSeparable1D(24, 32, 5, stride=2),
            DepthwiseSeparable1D(32, 32, 3, stride=2),
        )
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        self.aux = nn.Sequential(nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor, min_scale_fraction: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.features(x)
        embedding = torch.cat([self.avg(h).squeeze(-1), self.max(h).squeeze(-1)], dim=1)
        aux = self.aux(embedding)
        mean = torch.sigmoid(aux[:, 0])
        scale = F.softplus(aux[:, 1]) + min_scale_fraction
        return embedding, mean, scale


class UncertaintyFusionNet(nn.Module):
    """Confidence-aware CIR/variance fusion with heteroscedastic Student-t output.

    Scientific mechanism:
    1) modality-specific lightweight separable encoders;
    2) inverse-uncertainty reliability gating;
    3) fused ToF mean and aleatoric scale;
    4) the predicted scale is propagated directly to the Particle Filter.
    """

    def __init__(self, min_scale_fraction: float = 0.004, ablation: str = "full") -> None:
        super().__init__()
        self.min_scale_fraction = min_scale_fraction
        self.ablation = ablation
        self.cir_encoder = ModalityEncoder()
        self.var_encoder = ModalityEncoder()
        self.fusion_head = nn.Sequential(
            nn.Linear(64 * 3 + 4, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(96, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        cir_x, var_x = x[:, :3], x[:, 3:]
        cir_h, cir_mean, cir_scale = self.cir_encoder(cir_x, self.min_scale_fraction)
        var_h, var_mean, var_scale = self.var_encoder(var_x, self.min_scale_fraction)

        if self.ablation == "cir_only":
            weights = torch.stack([torch.ones_like(cir_scale), torch.zeros_like(var_scale)], dim=1)
        elif self.ablation == "var_only":
            weights = torch.stack([torch.zeros_like(cir_scale), torch.ones_like(var_scale)], dim=1)
        elif self.ablation == "fixed_fusion":
            weights = torch.full((x.shape[0], 2), 0.5, dtype=x.dtype, device=x.device)
        else:
            reliability_logits = torch.stack([-torch.log(cir_scale), -torch.log(var_scale)], dim=1)
            weights = torch.softmax(reliability_logits, dim=1)

        fused = weights[:, :1] * cir_h + weights[:, 1:] * var_h
        head_input = torch.cat(
            [fused, torch.abs(cir_h - var_h), cir_h * var_h, cir_mean[:, None], var_mean[:, None], cir_scale[:, None], var_scale[:, None]],
            dim=1,
        )
        out = self.fusion_head(head_input)
        base_mean = weights[:, 0] * cir_mean + weights[:, 1] * var_mean
        mean = torch.clamp(base_mean + 0.20 * torch.tanh(out[:, 0]), 0.0, 1.0)
        branch_scale = weights[:, 0] * cir_scale + weights[:, 1] * var_scale
        scale = 0.5 * branch_scale + 0.5 * (F.softplus(out[:, 1]) + self.min_scale_fraction)
        if self.ablation == "no_uncertainty":
            scale = torch.full_like(scale, 0.03)
        return {
            "mean_fraction": mean,
            "scale_fraction": scale,
            "cir_mean_fraction": cir_mean,
            "var_mean_fraction": var_mean,
            "cir_scale_fraction": cir_scale,
            "var_scale_fraction": var_scale,
            "gate_cir": weights[:, 0],
            "gate_var": weights[:, 1],
        }
