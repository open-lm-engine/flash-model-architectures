import torch
import torch.nn.functional as F


def softmax_torch(x: torch.Tensor, logits_multiplier: float = 1) -> torch.Tensor:
    dtype = x.dtype

    x = x.float()
    if logits_multiplier is not None:
        x = x * logits_multiplier
    x = F.softmax(x, dim=-1)

    return x.to(dtype)
