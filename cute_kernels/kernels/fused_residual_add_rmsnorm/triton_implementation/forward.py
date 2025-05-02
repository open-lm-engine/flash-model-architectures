import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME, MAX_TRITON_BLOCK_SIZE
from ....math import ceil_divide, get_next_power_of_2
from ....utils import cute_op, get_num_elements_and_hidden_size


@triton.jit
def fused_residual_add_rmsnorm_forward_triton_kernel(
    x_ptr,
    residual_ptr,
    has_weight: tl.constexpr,
    weight_ptr,
    output_ptr,
    eps,
    has_multiplier: tl.constexpr,
    multiplier,
    added_x_residual_ptr,
    has_rmsnorm_denominator: tl.constexpr,
    rmsnorm_denominator_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = tl.arange(0, BLOCK_SIZE_H)
    indices_bh = indices_b[:, None] * H + indices_h[None, :]

    mask_b = indices_b < B
    mask_h = indices_h < H

    mask_bh = mask_b[:, None] & mask_h[None, :]

    x_ptrs = x_ptr + indices_bh
    x = tl.load(x_ptrs, mask=mask_bh).to(tl.float32)

    if has_multiplier:
        x *= multiplier

    residual_ptrs = residual_ptr + indices_bh
    residual = tl.load(residual_ptrs, mask=mask_bh)

    x += residual

    added_x_residual_ptrs = added_x_residual_ptr + indices_bh
    tl.store(added_x_residual_ptrs, x, mask=mask_bh)

    squared_sum = tl.sum(x * x, axis=1)
    inverse_rms = tl.rsqrt((squared_sum / H) + eps)

    if has_rmsnorm_denominator:
        tl.store(rmsnorm_denominator_ptr + indices_b, inverse_rms, mask=mask_b)

    x *= inverse_rms[:, None]

    if has_weight:
        weight = tl.load(weight_ptr + indices_h, mask=mask_h)
        x = x.to(x_ptr.dtype.element_ty) * weight[None, :]

    output_ptrs = output_ptr + indices_bh
    tl.store(output_ptrs, x, mask=mask_bh)


@cute_op(
    f"{LIBRARY_NAME}::fused_residual_add_rmsnorm_forward_triton",
    mutates_args={"output", "added_x_residual", "rmsnorm_denominator"},
)
def fused_residual_add_rmsnorm_forward_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    output: torch.Tensor,
    eps: float,
    multiplier: float | None,
    added_x_residual: torch.Tensor,
    rmsnorm_denominator: torch.Tensor | None,
    BLOCK_SIZE_B: int,
) -> None:
    B, H = get_num_elements_and_hidden_size(x)

    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H < MAX_TRITON_BLOCK_SIZE

    with torch.device(x.device):
        fused_residual_add_rmsnorm_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](
            x_ptr=x,
            residual_ptr=residual,
            has_weight=weight is not None,
            weight_ptr=weight,
            output_ptr=output,
            eps=eps,
            has_multiplier=multiplier not in [None, 1],
            multiplier=multiplier,
            added_x_residual_ptr=added_x_residual,
            has_rmsnorm_denominator=rmsnorm_denominator is not None,
            rmsnorm_denominator_ptr=rmsnorm_denominator,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
