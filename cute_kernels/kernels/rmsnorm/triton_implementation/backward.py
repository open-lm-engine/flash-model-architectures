import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....cutotune import cutotune
from ....math import ceil_divide
from ....utils import cute_op, get_num_elements_and_hidden_size, get_sm_count
from .parameters import get_cutotune_parameters


_KERNEL_NO_WEIGHT_NAME = "rmsnorm_backward_no_weight_triton"
_KERNEL_WEIGHTED_NAME = "rmsnorm_backward_triton"


@triton.jit
def _rmsnorm_backward_triton_kernel(
    x_ptr,
    weight_ptr,
    output_grad_ptr,
    x_grad_ptr,
    weight_grad_ptr,
    eps,
    memory_efficient: tl.constexpr,
    rmsnorm_denominator_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    num_elements_per_program = tl.cdiv(B, num_programs)

    indices_h = tl.arange(0, BLOCK_SIZE_H)
    mask_h = indices_h < H

    program_start = pid * num_elements_per_program
    program_end = min(program_start + num_elements_per_program, B)
    num_elements_in_current_program = program_end - program_start

    num_loops = tl.cdiv(num_elements_in_current_program, BLOCK_SIZE_B)

    x_dtype = x_ptr.dtype.element_ty
    has_weight = weight_ptr is not None

    if has_weight:
        weight = tl.load(weight_ptr + indices_h, mask=mask_h)[None, :]
        weight_grad = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    else:
        weight = 1
        weight_grad = 0

    for i in range(num_loops):
        indices_b = program_start + i * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        mask_b = indices_b < program_end

        mask_bh = mask_b[:, None] & mask_h[None, :]

        x_ptrs = x_ptr + indices_b[:, None] * H + indices_h[None, :]
        x = tl.load(x_ptrs, mask=mask_bh).to(tl.float32)

        if memory_efficient:
            squared_sum = tl.sum(x * x, axis=1)
            inverse_rms = tl.rsqrt(squared_sum / H + eps)
        else:
            inverse_rms = tl.load(rmsnorm_denominator_ptr + indices_b, mask=mask_b)

        output_grad_ptrs = output_grad_ptr + indices_b[:, None] * H + indices_h[None, :]
        output_grad = tl.load(output_grad_ptrs, mask=mask_bh)

        output_grad_weight = (output_grad * weight).to(tl.float32)

        x_grad = inverse_rms[:, None] * output_grad_weight
        x_grad -= (
            (1 / H)
            * inverse_rms[:, None]
            * inverse_rms[:, None]
            * inverse_rms[:, None]
            * x
            * tl.sum(output_grad_weight * x, axis=1, keep_dims=True)
        )
        x_grad = x_grad.to(x_dtype)

        x_grad_ptrs = x_grad_ptr + indices_b[:, None] * H + indices_h[None, :]
        tl.store(x_grad_ptrs, x_grad, mask=mask_bh)

        if has_weight:
            weight_grad += tl.sum(output_grad * (x * inverse_rms[:, None]).to(x_dtype), axis=0)

    if has_weight:
        tl.atomic_add(weight_grad_ptr + indices_h, weight_grad, mask=mask_h)


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NO_WEIGHT_NAME}", mutates_args={"x_grad"})
def _rmsnorm_backward_no_weight_triton(
    x: torch.Tensor,
    output_grad: torch.Tensor,
    rmsnorm_denominator: torch.Tensor | None,
    x_grad: torch.Tensor,
    eps: float,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    num_elements, hidden_size = get_num_elements_and_hidden_size(x)

    if BLOCK_SIZE_H < hidden_size:
        raise ValueError(f"hidden_size should be more than the BLOCK_SIZE_H")

    sm_count = get_sm_count(x.device)
    num_programs = min(sm_count, ceil_divide(num_elements, BLOCK_SIZE_B))

    with torch.device(x.device):
        _rmsnorm_backward_triton_kernel[(num_programs,)](
            x_ptr=x,
            weight_ptr=None,
            output_grad_ptr=output_grad,
            x_grad_ptr=x_grad,
            weight_grad_ptr=None,
            eps=eps,
            memory_efficient=rmsnorm_denominator is None,
            rmsnorm_denominator_ptr=rmsnorm_denominator,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_WEIGHTED_NAME}", mutates_args={"x_grad", "weight_grad"})
def _rmsnorm_backward_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    output_grad: torch.Tensor,
    rmsnorm_denominator: torch.Tensor,
    x_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    eps: float,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    num_elements, hidden_size = get_num_elements_and_hidden_size(x)

    if BLOCK_SIZE_H < hidden_size:
        raise ValueError(f"hidden_size should be more than the BLOCK_SIZE_H")

    sm_count = get_sm_count(x.device)
    num_programs = min(sm_count, ceil_divide(num_elements, BLOCK_SIZE_B))

    with torch.device(x.device):
        _rmsnorm_backward_triton_kernel[(num_programs,)](
            x_ptr=x,
            weight_ptr=weight,
            output_grad_ptr=output_grad,
            x_grad_ptr=x_grad,
            weight_grad_ptr=weight_grad,
            eps=eps,
            memory_efficient=rmsnorm_denominator is None,
            rmsnorm_denominator_ptr=rmsnorm_denominator,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )


@cutotune(**get_cutotune_parameters())
def rmsnorm_backward_triton(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    output_grad: torch.Tensor,
    rmsnorm_denominator: torch.Tensor,
    x_grad: torch.Tensor,
    eps: float,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> torch.Tensor | None:
    if weight is None:
        weight_grad = None
        _rmsnorm_backward_no_weight_triton(
            x=x,
            output_grad=output_grad,
            rmsnorm_denominator=rmsnorm_denominator,
            x_grad=x_grad,
            eps=eps,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
    else:
        weight_grad = torch.zeros_like(weight, dtype=torch.float32)
        _rmsnorm_backward_triton(
            x=x,
            weight=weight,
            output_grad=output_grad,
            rmsnorm_denominator=rmsnorm_denominator,
            x_grad=x_grad,
            weight_grad=weight_grad,
            eps=eps,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )

        weight_grad = weight_grad.type_as(weight)

    return weight_grad
