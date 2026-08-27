from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LiteArchitecture:
    """Physical channel widths for the deployment-friendly fusion network."""

    channels: tuple[int, int, int] = (8, 12, 16)
    aux_hidden: int = 12
    fusion_hidden: int = 24


class DenseLiteModalityEncoder(nn.Module):
    """Small dense Conv1D encoder tuned for generic CPU inference.

    For tiny channel counts, standard dense convolutions can be faster than
    depthwise/grouped convolutions on CPUs without optimized depthwise kernels.
    The three stride-2 stages reduce the temporal resolution by about 8x.
    """

    def __init__(self, arch: LiteArchitecture) -> None:
        super().__init__()
        c1, c2, c3 = arch.channels
        self.conv1 = nn.Conv1d(3, c1, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, 5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, c3, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(c3)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        self.aux = nn.Sequential(
            nn.Linear(2 * c3, arch.aux_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(arch.aux_hidden, 2),
        )

    def forward(
        self, x: torch.Tensor, min_scale_fraction: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        embedding = torch.cat(
            [self.avg(x).squeeze(-1), self.max(x).squeeze(-1)], dim=1
        )
        aux = self.aux(embedding)
        mean = torch.sigmoid(aux[:, 0])
        scale = F.softplus(aux[:, 1]) + min_scale_fraction
        return embedding, mean, scale


class LiteUncertaintyFusionNet(nn.Module):
    """Deployment-first uncertainty fusion network.

    The probabilistic interface intentionally matches UncertaintyFusionNet:
    it predicts a normalized ToF mean and positive Student-t scale, while
    retaining CIR/variance branch estimates and reliability gates.

    Unlike the research model, it removes the elementwise-product feature and
    uses three small dense stride-2 Conv1D stages per modality. This makes the
    model physically compact and avoids relying on grouped/depthwise kernels.
    """

    def __init__(
        self,
        min_scale_fraction: float = 0.004,
        ablation: str = "full",
        arch: LiteArchitecture | None = None,
    ) -> None:
        super().__init__()
        self.min_scale_fraction = float(min_scale_fraction)
        self.ablation = ablation
        self.arch = arch or LiteArchitecture()
        self.cir_encoder = DenseLiteModalityEncoder(self.arch)
        self.var_encoder = DenseLiteModalityEncoder(self.arch)
        embedding_dim = 2 * self.arch.channels[-1]
        # fused embedding + absolute disagreement + four branch scalars
        self.fusion_head = nn.Sequential(
            nn.Linear(2 * embedding_dim + 4, self.arch.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.arch.fusion_hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        cir_x, var_x = x[:, :3], x[:, 3:]
        cir_h, cir_mean, cir_scale = self.cir_encoder(
            cir_x, self.min_scale_fraction
        )
        var_h, var_mean, var_scale = self.var_encoder(
            var_x, self.min_scale_fraction
        )

        if self.ablation == "cir_only":
            weights = torch.stack(
                [torch.ones_like(cir_scale), torch.zeros_like(var_scale)], dim=1
            )
        elif self.ablation == "var_only":
            weights = torch.stack(
                [torch.zeros_like(cir_scale), torch.ones_like(var_scale)], dim=1
            )
        elif self.ablation == "fixed_fusion":
            weights = torch.full(
                (x.shape[0], 2), 0.5, dtype=x.dtype, device=x.device
            )
        else:
            reliability_logits = torch.stack(
                [-torch.log(cir_scale), -torch.log(var_scale)], dim=1
            )
            weights = torch.softmax(reliability_logits, dim=1)

        fused = weights[:, :1] * cir_h + weights[:, 1:] * var_h
        head_input = torch.cat(
            [
                fused,
                torch.abs(cir_h - var_h),
                cir_mean[:, None],
                var_mean[:, None],
                cir_scale[:, None],
                var_scale[:, None],
            ],
            dim=1,
        )
        out = self.fusion_head(head_input)
        base_mean = weights[:, 0] * cir_mean + weights[:, 1] * var_mean
        mean = torch.clamp(base_mean + 0.20 * torch.tanh(out[:, 0]), 0.0, 1.0)
        branch_scale = weights[:, 0] * cir_scale + weights[:, 1] * var_scale
        scale = 0.5 * branch_scale + 0.5 * (
            F.softplus(out[:, 1]) + self.min_scale_fraction
        )
        outlier_logit = out[:, 2]
        outlier_probability = torch.sigmoid(outlier_logit)
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
            "outlier_logit": outlier_logit,
            "outlier_probability": outlier_probability,
        }
