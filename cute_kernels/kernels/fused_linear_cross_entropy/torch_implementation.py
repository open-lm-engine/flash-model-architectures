import torch
import torch.nn.functional as F


def cross_entropy_torch(
    x: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    x = F.linear(x, weight)
    return F.cross_entropy(x.float(), labels, reduction=reduction)
