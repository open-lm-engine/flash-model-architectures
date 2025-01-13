import torch
import torch.nn.functional as F


def linear_torch(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    return F.linear(input, weight, bias)
