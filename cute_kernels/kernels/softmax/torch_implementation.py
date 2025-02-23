import torch
import torch.nn.functional as F


def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=-1)
