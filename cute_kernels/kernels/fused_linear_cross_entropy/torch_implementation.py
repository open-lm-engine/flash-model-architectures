import torch
import torch.nn.functional as F

from ..cross_entropy import cross_entropy_torch


def fused_linear_cross_entropy_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    logits_multiplier: float | None = None,
) -> torch.Tensor:
    x = F.linear(x, weight)
    return cross_entropy_torch(x=x, labels=labels, reduction=reduction, logits_multiplier=logits_multiplier)
