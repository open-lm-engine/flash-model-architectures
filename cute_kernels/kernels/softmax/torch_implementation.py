import torch
import torch.nn.functional as F


def softmax_torch(x: torch.Tensor, logits_multiplier: float = 1) -> torch.Tensor:
    x = x.float()
    x /= logits_multiplier
    return F.softmax(x, dim=-1)
