import torch
import torch.nn.functional as F


def fused_linear_cross_entropy_torch(
    x: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, reduction: str = "mean", logits_multiplier: float = 1
) -> torch.Tensor:
    x = F.linear(x, weight)
    x = x.float()
    x = x * logits_multiplier
    return F.cross_entropy(x, labels, reduction=reduction)
