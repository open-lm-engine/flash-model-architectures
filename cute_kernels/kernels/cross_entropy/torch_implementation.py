import torch
import torch.nn.functional as F


def cross_entropy_torch(x: torch.Tensor, labels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.cross_entropy(x, labels, reduction=reduction)
