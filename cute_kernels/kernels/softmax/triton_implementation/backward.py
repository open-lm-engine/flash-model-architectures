import torch
import triton
import triton.language as tl

from ....constants import COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2, LIBRARY_NAME, MAX_TRITON_BLOCK_SIZE
from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ....math import ceil_divide, get_powers_of_2
from ....utils import cute_op, get_num_elements_and_hidden_size


_KERNEL_NAME = "softmax_backward_triton"


@triton.jit
def _softmax_backward_triton_kernel_full_row(
    output_ptr,
    output_grad_ptr,
    x_grad_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    indices_b = pid * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B

    accumulator = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)

    for h in range(tl.cdiv(H, BLOCK_SIZE_H)):
        indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
        mask_h = indices_h < H

        indices = indices_b[:, None] * H + indices_h[None, :]
        mask_bh = mask_b[:, None] & mask_h[None, :]

        output_ptrs = output_ptr + indices
        output = tl.load(output_ptrs, mask=mask_bh)

        output_grad_ptrs = output_grad_ptr + indices
        output_grad = tl.load(output_grad_ptrs, mask=mask_bh)

        acc = output_grad * output
        acc = acc.to(tl.float32)
        accumulator += tl.sum(acc, axis=1, keep_dims=True)

    for h in range(tl.cdiv(H, BLOCK_SIZE_H)):
        indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
        mask_h = indices_h < H

        indices = indices_b[:, None] * H + indices_h[None, :]
        mask_bh = mask_b[:, None] & mask_h[None, :]

        output_ptrs = output_ptr + indices
        output = tl.load(output_ptrs, mask=mask_bh)

        output_grad_ptrs = output_grad_ptr + indices
        output_grad = tl.load(output_grad_ptrs, mask=mask_bh)

        output_grad -= accumulator
        output *= output_grad

        x_grad_ptrs = x_grad_ptr + indices
        tl.store(x_grad_ptrs, output, mask=mask_bh)


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        BLOCK_SIZE_B=get_powers_of_2(1, 32) + COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
        BLOCK_SIZE_H=get_powers_of_2(1, 32) + COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
        condition=lambda **kwargs: 1024 <= kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_H"] <= MAX_TRITON_BLOCK_SIZE,
    ),
    default_config=CutoTuneConfig({"BLOCK_SIZE_B": 64, "BLOCK_SIZE_H": 64}),
    triggers={"output.dtype"},
)
@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"x_grad"})
def softmax_backward_full_row_triton(
    output: torch.Tensor, output_grad: torch.Tensor, x_grad: torch.Tensor, BLOCK_SIZE_B: int, BLOCK_SIZE_H: int
) -> None:
    num_elements, hidden_size = get_num_elements_and_hidden_size(x_grad)

    with torch.device(x_grad.device):
        _softmax_backward_triton_kernel_full_row[(ceil_divide(num_elements, BLOCK_SIZE_B),)](
            output_ptr=output,
            output_grad_ptr=output_grad,
            x_grad_ptr=x_grad,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
