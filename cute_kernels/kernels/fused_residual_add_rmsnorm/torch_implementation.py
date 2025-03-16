import torch
import torch.nn.functional as F


def fused_residual_add_rmsnorm_torch(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor | None, eps: float, multiplier: float | None = None
) -> torch.Tensor:
    if multiplier is not None:
        x = x * multiplier

    x = x + residual
    x = F.rms_norm(x, (x.size(-1),), weight=weight, eps=eps)

    return x
