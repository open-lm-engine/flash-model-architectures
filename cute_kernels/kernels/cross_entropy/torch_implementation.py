import torch
import torch.nn.functional as F


def cross_entropy_torch(
    x: torch.Tensor, labels: torch.Tensor, reduction: str = "mean", logits_multiplier: float = 1
) -> torch.Tensor:
    x = x.float()
    if logits_multiplier is not None:
        x = x * logits_multiplier
    return F.cross_entropy(x, labels, reduction=reduction)
