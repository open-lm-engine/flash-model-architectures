import torch
import triton
import triton.language as tl

from ....constants import COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2, LIBRARY_NAME, MAX_TRITON_BLOCK_SIZE
from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ....math import ceil_divide, get_powers_of_2
from ....utils import cute_op, get_num_elements_and_hidden_size


_KERNEL_NAME = "online_softmax_forward_triton"


@triton.jit
def _softmax_forward_triton_kernel(
    x_ptr,
    output_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    indices_b = pid * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B

    Z = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    M = tl.full((BLOCK_SIZE_B, 1), -float("inf"), dtype=tl.float32)

    for h in range(tl.cdiv(H, BLOCK_SIZE_H)):
        indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
        mask_h = indices_h < H

        indices = indices_b[:, None] * H + indices_h[None, :]
        mask_bh = mask_b[:, None] & mask_h[None, :]

        x_ptrs = x_ptr + indices
        x = tl.load(x_ptrs, mask=mask_bh, other=-float("inf"))

        prev_m = M
        m = tl.max(x, axis=1, keep_dims=True)

        M = max(M, m)

        x -= M
        x = x.to(tl.float32)
        x = tl.exp(x)

        Z = Z * tl.exp(prev_m - M) + tl.sum(x, axis=1, keep_dims=True)

    for h in range(tl.cdiv(H, BLOCK_SIZE_H)):
        indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
        mask_h = indices_h < H

        indices = indices_b[:, None] * H + indices_h[None, :]
        mask_bh = mask_b[:, None] & mask_h[None, :]

        x_ptrs = x_ptr + indices
        x = tl.load(x_ptrs, mask=mask_bh)

        x -= M
        x = x.to(tl.float32)
        x = tl.exp(x)
        x /= Z

        output_ptrs = output_ptr + indices
        tl.store(output_ptrs, x, mask=mask_bh)


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        BLOCK_SIZE_B=get_powers_of_2(1, 32) + COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
        BLOCK_SIZE_H=get_powers_of_2(1, 32) + COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
        condition=lambda **kwargs: 1024 <= kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_H"] <= MAX_TRITON_BLOCK_SIZE,
    ),
    default_config=CutoTuneConfig({"BLOCK_SIZE_B": 64, "BLOCK_SIZE_H": 64}),
    triggers={"x.dtype"},
)
@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def softmax_forward_triton(x: torch.Tensor, output: torch.Tensor, BLOCK_SIZE_B: int, BLOCK_SIZE_H: int) -> None:
    num_elements, hidden_size = get_num_elements_and_hidden_size(x)

    with torch.device(x.device):
        _softmax_forward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE_B),)](
            x_ptr=x,
            output_ptr=output,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
