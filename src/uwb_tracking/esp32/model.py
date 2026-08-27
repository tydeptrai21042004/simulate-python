from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ESP32Architecture:
    """Physical widths of the ESP32 deployment student."""

    channels: tuple[int, int, int] = (8, 12, 12)
    hidden: int = 16

    def __post_init__(self) -> None:
        if len(self.channels) != 3:
            raise ValueError("channels must contain exactly three Conv1D widths")
        if any(int(v) <= 0 for v in self.channels):
            raise ValueError("all channel widths must be positive")
        if int(self.hidden) <= 0:
            raise ValueError("hidden must be positive")


class ESP32StudentNet(nn.Module):
    """Tiny early-fusion Conv1D network for ESP32 deployment.

    Input channels are the existing six normalized UWB features:
    CIR dynamic/background/difference and variance dynamic/background/difference.
    The network emits three *raw* values. Nonlinear decoding of ToF, scale and
    outlier probability is kept outside the exported network where possible.
    """

    def __init__(
        self,
        arch: ESP32Architecture | None = None,
        min_scale_fraction: float = 0.004,
    ) -> None:
        super().__init__()
        self.arch = arch or ESP32Architecture()
        self.min_scale_fraction = float(min_scale_fraction)
        c1, c2, c3 = self.arch.channels
        self.conv1 = nn.Conv1d(6, c1, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, 5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, c3, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(c3)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(c3, self.arch.hidden)
        self.fc2 = nn.Linear(self.arch.hidden, 3)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 6:
            raise ValueError("ESP32StudentNet expects input shape [batch, 6, length]")
        if x.shape[-1] < 1:
            raise ValueError("input length must be >= 1")
        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        x = F.relu(self.bn2(self.conv2(x)), inplace=False)
        x = F.relu(self.bn3(self.conv3(x)), inplace=False)
        x = self.avg(x).squeeze(-1)
        x = F.relu(self.fc1(x), inplace=False)
        return self.fc2(x)

    def decode(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = torch.sigmoid(raw[:, 0])
        scale = F.softplus(raw[:, 1]) + self.min_scale_fraction
        outlier_logit = raw[:, 2]
        return {
            "raw": raw,
            "mean_fraction": mean,
            "scale_fraction": scale,
            "outlier_logit": outlier_logit,
            "outlier_probability": torch.sigmoid(outlier_logit),
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decode(self.forward_raw(x))


@dataclass(frozen=True)
class ESP32TicketSelection:
    """Physical channel/neuron identities retained by a structured ticket."""

    c1: torch.Tensor
    c2: torch.Tensor
    c3: torch.Tensor
    hidden: torch.Tensor


def _top_channels_from_score(score: torch.Tensor, count: int) -> torch.Tensor:
    if count > score.numel():
        raise ValueError("ticket width cannot exceed source width")
    return torch.topk(score, k=count, largest=True, sorted=True).indices.sort().values


def select_structured_ticket(
    trained_supernet: ESP32StudentNet,
    target_arch: ESP32Architecture,
) -> ESP32TicketSelection:
    """Select a dependency-aware structured LTH ticket.

    Ranking is hierarchical: conv2 is scored only through retained conv1 inputs,
    conv3 only through retained conv2 inputs, and hidden FC neurons use both their
    incoming magnitude and their contribution to all three output heads. This is
    more faithful to the *physical* compact graph than ranking every layer in
    isolation against channels that will later be removed.
    """

    c1_count, c2_count, c3_count = target_arch.channels
    source_arch = trained_supernet.arch
    if any(small > large for small, large in zip(target_arch.channels, source_arch.channels)):
        raise ValueError("target channels must not exceed supernet channels")
    if target_arch.hidden > source_arch.hidden:
        raise ValueError("target hidden width must not exceed supernet hidden width")

    w1 = trained_supernet.conv1.weight.detach().abs()
    c1 = _top_channels_from_score(w1.flatten(1).sum(dim=1), c1_count)

    w2 = trained_supernet.conv2.weight.detach().abs()[:, c1, :]
    c2 = _top_channels_from_score(w2.flatten(1).sum(dim=1), c2_count)

    w3 = trained_supernet.conv3.weight.detach().abs()[:, c2, :]
    c3 = _top_channels_from_score(w3.flatten(1).sum(dim=1), c3_count)

    fc1 = trained_supernet.fc1.weight.detach().abs()[:, c3]
    fc2 = trained_supernet.fc2.weight.detach().abs()
    hidden_score = fc1.sum(dim=1) + fc2.sum(dim=0)
    hidden = _top_channels_from_score(hidden_score, target_arch.hidden)
    return ESP32TicketSelection(c1=c1, c2=c2, c3=c3, hidden=hidden)


def _copy_bn(
    dst: nn.BatchNorm1d,
    src: Mapping[str, torch.Tensor],
    prefix: str,
    idx: torch.Tensor,
) -> None:
    with torch.no_grad():
        dst.weight.copy_(src[f"{prefix}.weight"][idx])
        dst.bias.copy_(src[f"{prefix}.bias"][idx])
        dst.running_mean.copy_(src[f"{prefix}.running_mean"][idx])
        dst.running_var.copy_(src[f"{prefix}.running_var"][idx])
        dst.num_batches_tracked.copy_(src[f"{prefix}.num_batches_tracked"])


def build_rewound_structured_ticket(
    trained_supernet: ESP32StudentNet,
    initial_supernet_state: Mapping[str, torch.Tensor],
    target_arch: ESP32Architecture,
) -> tuple[ESP32StudentNet, ESP32TicketSelection]:
    """Create a physically smaller winning-ticket candidate and rewind it.

    Channel/neuron identities are discovered from the *trained* supernet, while
    the retained weights are copied from the original initialization. Whole
    channels and FC neurons are removed, so the exported dense INT8 model becomes
    genuinely smaller; no sparse mask needs to be stored by the MCU.
    """

    source_arch = trained_supernet.arch
    if any(small > large for small, large in zip(target_arch.channels, source_arch.channels)):
        raise ValueError("target channels must not exceed supernet channels")
    if target_arch.hidden > source_arch.hidden:
        raise ValueError("target hidden width must not exceed supernet hidden width")

    sel = select_structured_ticket(trained_supernet, target_arch)
    ticket = ESP32StudentNet(target_arch, trained_supernet.min_scale_fraction)

    with torch.no_grad():
        ticket.conv1.weight.copy_(initial_supernet_state["conv1.weight"][sel.c1])
        ticket.conv2.weight.copy_(initial_supernet_state["conv2.weight"][sel.c2][:, sel.c1])
        ticket.conv3.weight.copy_(initial_supernet_state["conv3.weight"][sel.c3][:, sel.c2])
    _copy_bn(ticket.bn1, initial_supernet_state, "bn1", sel.c1)
    _copy_bn(ticket.bn2, initial_supernet_state, "bn2", sel.c2)
    _copy_bn(ticket.bn3, initial_supernet_state, "bn3", sel.c3)

    with torch.no_grad():
        ticket.fc1.weight.copy_(initial_supernet_state["fc1.weight"][sel.hidden][:, sel.c3])
        ticket.fc1.bias.copy_(initial_supernet_state["fc1.bias"][sel.hidden])
        ticket.fc2.weight.copy_(initial_supernet_state["fc2.weight"][:, sel.hidden])
        ticket.fc2.bias.copy_(initial_supernet_state["fc2.bias"])
    return ticket, sel
