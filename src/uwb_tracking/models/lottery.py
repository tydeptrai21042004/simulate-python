from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn.utils import prune

from .lite import LiteArchitecture, LiteUncertaintyFusionNet


@dataclass(frozen=True)
class TicketSelection:
    cir_c1: torch.Tensor
    cir_c2: torch.Tensor
    var_c1: torch.Tensor
    var_c2: torch.Tensor
    shared_c3: torch.Tensor


def _top_channels(weight: torch.Tensor, count: int) -> torch.Tensor:
    if count > weight.shape[0]:
        raise ValueError("ticket width cannot exceed source width")
    score = weight.detach().abs().flatten(1).sum(dim=1)
    return torch.topk(score, k=count, largest=True, sorted=True).indices.sort().values


def select_structured_ticket(
    trained_supernet: LiteUncertaintyFusionNet,
    target_arch: LiteArchitecture,
) -> TicketSelection:
    """Select a physically compact channel ticket by trained magnitude.

    The final embedding channels are shared across modalities because the
    fusion head compares CIR and variance embeddings elementwise.
    """

    c1, c2, c3 = target_arch.channels
    cir = trained_supernet.cir_encoder
    var = trained_supernet.var_encoder
    cir_c1 = _top_channels(cir.conv1.weight, c1)
    cir_c2 = _top_channels(cir.conv2.weight, c2)
    var_c1 = _top_channels(var.conv1.weight, c1)
    var_c2 = _top_channels(var.conv2.weight, c2)
    score_c3 = (
        cir.conv3.weight.detach().abs().flatten(1).sum(dim=1)
        + var.conv3.weight.detach().abs().flatten(1).sum(dim=1)
    )
    shared_c3 = torch.topk(score_c3, k=c3, largest=True, sorted=True).indices.sort().values
    return TicketSelection(cir_c1, cir_c2, var_c1, var_c2, shared_c3)


def _copy_bn(dst: nn.BatchNorm1d, src_state: Mapping[str, torch.Tensor], prefix: str, idx: torch.Tensor) -> None:
    with torch.no_grad():
        dst.weight.copy_(src_state[f"{prefix}.weight"][idx])
        dst.bias.copy_(src_state[f"{prefix}.bias"][idx])
        dst.running_mean.copy_(src_state[f"{prefix}.running_mean"][idx])
        dst.running_var.copy_(src_state[f"{prefix}.running_var"][idx])
        dst.num_batches_tracked.copy_(src_state[f"{prefix}.num_batches_tracked"])


def _copy_encoder_from_initial(
    dst: nn.Module,
    initial: Mapping[str, torch.Tensor],
    prefix: str,
    i1: torch.Tensor,
    i2: torch.Tensor,
    i3: torch.Tensor,
) -> None:
    with torch.no_grad():
        dst.conv1.weight.copy_(initial[f"{prefix}.conv1.weight"][i1])
        dst.conv2.weight.copy_(initial[f"{prefix}.conv2.weight"][i2][:, i1])
        dst.conv3.weight.copy_(initial[f"{prefix}.conv3.weight"][i3][:, i2])
    _copy_bn(dst.bn1, initial, f"{prefix}.bn1", i1)
    _copy_bn(dst.bn2, initial, f"{prefix}.bn2", i2)
    _copy_bn(dst.bn3, initial, f"{prefix}.bn3", i3)

    old_c3 = initial[f"{prefix}.conv3.weight"].shape[0]
    emb_idx = torch.cat([i3, i3 + old_c3])
    with torch.no_grad():
        dst.aux[0].weight.copy_(initial[f"{prefix}.aux.0.weight"][:, emb_idx])
        dst.aux[0].bias.copy_(initial[f"{prefix}.aux.0.bias"])
        dst.aux[2].weight.copy_(initial[f"{prefix}.aux.2.weight"])
        dst.aux[2].bias.copy_(initial[f"{prefix}.aux.2.bias"])


def build_rewound_structured_ticket(
    trained_supernet: LiteUncertaintyFusionNet,
    initial_supernet_state: Mapping[str, torch.Tensor],
    target_arch: LiteArchitecture,
) -> tuple[LiteUncertaintyFusionNet, TicketSelection]:
    """Build a structured lottery-ticket-inspired compact network.

    Channel identities are selected from the trained supernet by magnitude,
    but the surviving weights are rewound to their values at initialization.
    This follows the key lottery-ticket idea while producing a truly smaller
    dense network that generic CPUs can accelerate without sparse kernels.
    """

    source_arch = trained_supernet.arch
    if target_arch.aux_hidden != source_arch.aux_hidden:
        raise ValueError("aux_hidden must match between supernet and ticket")
    if target_arch.fusion_hidden != source_arch.fusion_hidden:
        raise ValueError("fusion_hidden must match between supernet and ticket")
    for small, large in zip(target_arch.channels, source_arch.channels):
        if small > large:
            raise ValueError("target channels must not exceed supernet channels")

    sel = select_structured_ticket(trained_supernet, target_arch)
    ticket = LiteUncertaintyFusionNet(
        min_scale_fraction=trained_supernet.min_scale_fraction,
        ablation=trained_supernet.ablation,
        arch=target_arch,
    )
    _copy_encoder_from_initial(
        ticket.cir_encoder,
        initial_supernet_state,
        "cir_encoder",
        sel.cir_c1,
        sel.cir_c2,
        sel.shared_c3,
    )
    _copy_encoder_from_initial(
        ticket.var_encoder,
        initial_supernet_state,
        "var_encoder",
        sel.var_c1,
        sel.var_c2,
        sel.shared_c3,
    )

    old_c3 = source_arch.channels[-1]
    old_e = 2 * old_c3
    emb_idx = torch.cat([sel.shared_c3, sel.shared_c3 + old_c3])
    head_idx = torch.cat(
        [emb_idx, old_e + emb_idx, torch.arange(2 * old_e, 2 * old_e + 4)]
    )
    with torch.no_grad():
        ticket.fusion_head[0].weight.copy_(
            initial_supernet_state["fusion_head.0.weight"][:, head_idx]
        )
        ticket.fusion_head[0].bias.copy_(initial_supernet_state["fusion_head.0.bias"])
        ticket.fusion_head[2].weight.copy_(initial_supernet_state["fusion_head.2.weight"])
        ticket.fusion_head[2].bias.copy_(initial_supernet_state["fusion_head.2.bias"])
    return ticket, sel


def apply_global_lottery_pruning(
    model: nn.Module,
    amount: float,
    protect_output_heads: bool = True,
) -> dict[str, float]:
    """Apply global unstructured L1 pruning for canonical IMP experiments.

    This is useful to study the Lottery Ticket Hypothesis or to target a sparse
    runtime. It does *not* by itself guarantee latency reduction on dense CPUs.
    """

    targets: list[tuple[nn.Module, str]] = []
    for module in model.modules():
        if not isinstance(module, (nn.Conv1d, nn.Linear)):
            continue
        if protect_output_heads and isinstance(module, nn.Linear) and module.out_features == 2:
            continue
        targets.append((module, "weight"))
    if not targets:
        return {"sparsity": 0.0, "nonzero": 0.0, "total": 0.0}
    prune.global_unstructured(
        targets,
        pruning_method=prune.L1Unstructured,
        amount=float(amount),
    )
    total = 0
    nonzero = 0
    for module, _ in targets:
        w = module.weight.detach()
        total += w.numel()
        nonzero += int(torch.count_nonzero(w))
    return {
        "sparsity": 1.0 - nonzero / max(total, 1),
        "nonzero": float(nonzero),
        "total": float(total),
    }


def rewind_pruned_model_(model: nn.Module, initial_state: Mapping[str, torch.Tensor]) -> None:
    """Rewind surviving parameters while keeping pruning masks fixed."""

    with torch.no_grad():
        for name, param in model.named_parameters():
            source_name = name.replace(".weight_orig", ".weight")
            if source_name in initial_state and initial_state[source_name].shape == param.shape:
                param.copy_(initial_state[source_name])
        for name, buffer in model.named_buffers():
            if name.endswith("weight_mask"):
                continue
            if name in initial_state and initial_state[name].shape == buffer.shape:
                buffer.copy_(initial_state[name])


def materialize_pruning_(model: nn.Module) -> None:
    """Remove pruning reparameterizations while keeping zeroed weights."""

    for module in model.modules():
        if hasattr(module, "weight_orig") and hasattr(module, "weight_mask"):
            prune.remove(module, "weight")
