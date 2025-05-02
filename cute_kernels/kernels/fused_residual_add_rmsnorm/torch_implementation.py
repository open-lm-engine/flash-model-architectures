import torch
import torch.nn.functional as F


def fused_residual_add_rmsnorm_torch(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor | None, eps: float, multiplier: float | None = None
) -> tuple[torch.Tensor]:
    if multiplier not in [None, 1]:
        x = x * multiplier

    x = x + residual
    residual = x
    x = F.rms_norm(x, (x.size(-1),), weight=weight, eps=eps)

    return x, residual
