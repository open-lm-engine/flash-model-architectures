import torch

from ...cutotune import CutoTuneConfig, cutotune
from .triton_implementation import softmax_forward_triton


@cutotune(
    configs=[CutoTuneConfig(dict(kernel_backend="triton"))],
    default_config=CutoTuneConfig(dict(kernel_backend="triton")),
    triggers={"x.dtype"},
)
def _forward(
    x: torch.Tensor,
    logits_multiplier: float,
    kernel_backend: str,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> torch.Tensor:
    output = torch.empty_like(x)

    if kernel_backend == "triton":
        softmax_forward_triton(
            x=x,
            output=output,
            logits_multiplier=logits_multiplier,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
