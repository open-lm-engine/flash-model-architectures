import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME, TORCH_TO_TRITON_DTYPE
from ....cutotune import cutotune
from ....math import ceil_divide
from ....utils import cute_op, get_num_elements_and_hidden_size, get_sm_count
from .parameters import get_cutotune_parameters


_KERNEL_NAME = "rmsnorm_forward_triton"


@triton.jit
def _rmsnorm_forward_triton_kernel(
    x_ptr,
    x_dtype: tl.constexpr,
    has_weight: tl.constexpr,
    weight_ptr,
    output_ptr,
    eps,
    memory_efficient: tl.constexpr,
    rmsnorm_denominator_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_h = indices_h < H

    mask_bh = mask_b[:, None] & mask_h[None, :]

    x_ptrs = x_ptr + indices_b[:, None] * H + indices_h[None, :]
    x = tl.load(x_ptrs, mask=mask_bh).to(tl.float32)

    squared_sum = tl.sum(x * x, axis=1)
    inverse_rms = tl.rsqrt((squared_sum / H) + eps)

    if not memory_efficient:
        tl.store(rmsnorm_denominator_ptr + indices_b, inverse_rms, mask=mask_b)

    x *= inverse_rms[:, None]

    if has_weight:
        weight = tl.load(weight_ptr + indices_h, mask=mask_h)
        x = x.to(x_dtype) * weight[None, :]

    output_ptrs = output_ptr + indices_b[:, None] * H + indices_h[None, :]
    tl.store(output_ptrs, x, mask=mask_bh)


@cutotune(**get_cutotune_parameters())
@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output", "rmsnorm_denominator"})
def rmsnorm_forward_triton(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    output: torch.Tensor,
    eps: float,
    rmsnorm_denominator: torch.Tensor | None,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    num_elements, hidden_size = get_num_elements_and_hidden_size(x)

    if BLOCK_SIZE_H < hidden_size:
        raise ValueError(f"hidden_size should be more than the BLOCK_SIZE_H")

    with torch.device(x.device):
        _rmsnorm_forward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE_B),)](
            x_ptr=x,
            x_dtype=TORCH_TO_TRITON_DTYPE[x.dtype],
            has_weight=weight is not None,
            weight_ptr=weight,
            output_ptr=output,
            eps=eps,
            memory_efficient=rmsnorm_denominator is None,
            rmsnorm_denominator_ptr=rmsnorm_denominator,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
