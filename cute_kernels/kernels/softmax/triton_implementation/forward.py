import torch
import triton
import triton.language as tl

from ....constants import COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2, LIBRARY_NAME, MAX_TRITON_BLOCK_SIZE
from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ....enums import KernelBackend
from ....math import ceil_divide, get_next_power_of_2, get_powers_of_2
from ....utils import cute_op, get_num_elements_and_hidden_size
from ..enums import TritonKernelAlgorithm


_KERNEL_NAME = "softmax_forward_triton"


@triton.jit
def _softmax_forward_triton_kernel_full_row(
    x_ptr,
    output_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    indices_b = pid * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_h = indices_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    indices = indices_b[:, None] * H + indices_h[None, :]

    x_ptrs = x_ptr + indices
    x = tl.load(x_ptrs, mask=mask_bh, other=-float("inf"))

    max = tl.max(x, axis=1).to(tl.float32)
    x = x.to(tl.float32)

    x -= max
    x = tl.exp(x)
    x /= tl.sum(x, axis=1)

    output_ptrs = output_ptr + indices
    tl.store(output_ptrs, x, mask=mask_bh)


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        BLOCK_SIZE_B=get_powers_of_2(1, 32) + COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
        condition=lambda **kwargs: 1024 <= kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_H"] <= MAX_TRITON_BLOCK_SIZE,
    ),
    default_config=CutoTuneConfig({"BLOCK_SIZE_B": 1}),
    triggers={"x.dtype", "BLOCK_SIZE_H"},
)
@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def softmax_forward_full_row_triton(
    x: torch.Tensor, output: torch.Tensor, BLOCK_SIZE_B: int, BLOCK_SIZE_H: int
) -> None:
    num_elements, hidden_size = get_num_elements_and_hidden_size(x)

    with torch.device(x.device):
        if BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE:
            _softmax_forward_triton_kernel_full_row[(ceil_divide(num_elements, BLOCK_SIZE_B),)](
                x_ptr=x,
                output_ptr=output,
                B=num_elements,
                H=hidden_size,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )
