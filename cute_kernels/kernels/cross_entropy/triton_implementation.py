import torch
import triton
import triton.language as tl

from ...constants import LIBRARY_NAME, MAX_TRITON_BLOCK_SIZE
from ...cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ...math import ceil_divide, get_powers_of_2
from ...utils import cute_op
from ..softmax.triton_implementation.forward import _load_x


_KERNEL_NAME = "cross_entropy_forward_triton"


@triton.jit
def _cross_entropy_forward_triton_kernel(
    x_ptr,
    labels_ptr,
    loss_ptr,
    logits_multiplier,
    B,
    V,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    reduction: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    indices_b = pid * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B

    Z = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    M = tl.full((BLOCK_SIZE_B, 1), -float("inf"), dtype=tl.float32)

    num_blocks_v = tl.cdiv(V, BLOCK_SIZE_V)

    for v in range(num_blocks_v):
        x, indices, mask_bv = _load_x(
            x_ptr=x_ptr, h=v, H=V, BLOCK_SIZE_H=BLOCK_SIZE_V, indices_b=indices_b, mask_b=mask_b, other=-float("inf")
        )

        x = x.to(tl.float32)
        x *= logits_multiplier

        prev_m = M
        m = tl.max(x, axis=1, keep_dims=True)
        M = max(M, m)

        x -= M
        x = tl.exp(x)
        Z = Z * tl.exp(prev_m - M) + tl.sum(x, axis=1, keep_dims=True)

    labels_ptrs = labels_ptr + indices_b
    labels = tl.load(labels_ptrs, mask=mask_b)

    x_ptrs = x_ptr + indices_b * V + labels
    x = tl.load(x_ptrs, mask=mask_b)
    x = x.to(tl.float32)
    x *= logits_multiplier

    loss = M + tl.log(Z) - x[:, None]
    loss = tl.where(mask_b[:, None], loss, 0)
    loss = tl.sum(loss, axis=0)

    if reduction == "mean":
        loss /= B

    tl.atomic_add(loss_ptr + tl.arange(0, 1), loss)


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        BLOCK_SIZE_B=get_powers_of_2(1, MAX_TRITON_BLOCK_SIZE),
        BLOCK_SIZE_V=get_powers_of_2(1, MAX_TRITON_BLOCK_SIZE),
        condition=lambda **kwargs: 1024 <= kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_V"] <= 8192,
    ),
    default_config=CutoTuneConfig({"BLOCK_SIZE_B": 64, "BLOCK_SIZE_V": 64}),
    triggers={"x.dtype"},
    reset_to_zero={"loss": lambda **kwargs: True},
)
@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"loss"})
def cross_entropy_forward_triton(
    x: torch.Tensor,
    labels: torch.Tensor,
    loss: torch.Tensor,
    logits_multiplier: float,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_V: int,
    reduction: str,
) -> None:
    num_elements, vocab_size = x.size()

    with torch.device(x.device):
        _cross_entropy_forward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE_B),)](
            x_ptr=x,
            labels_ptr=labels,
            loss_ptr=loss,
            logits_multiplier=logits_multiplier,
            B=num_elements,
            V=vocab_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
            reduction=reduction,
        )
