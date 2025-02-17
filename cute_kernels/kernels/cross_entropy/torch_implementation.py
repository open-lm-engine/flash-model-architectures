import torch
import torch.nn.functional as F


def cross_entropy_torch(x: torch.Tensor, labels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    x = x.flatten()
    labels = labels.reshape(-1, labels.size(-1))

    return F.cross_entropy(x, labels, reduction=reduction)
