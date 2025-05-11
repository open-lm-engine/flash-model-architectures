import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op


@triton.jit
def embedding_forward_triton_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_h = indices_h < H

    x = tl.load(x_ptr + indices_b, mask=mask_b)
    word_embeddings = tl.load(weight_ptr + x[:, None] * H + indices_h[None, :], mask=mask_h[None, :])

    tl.store(
        output_ptr + indices_b[:, None] * H + indices_h[None, :],
        word_embeddings,
        mask=mask_b[:, None] & mask_h[None, :],
    )


@cute_op(f"{LIBRARY_NAME}::embedding_forward_triton", mutates_args={"output"})
def embedding_forward_triton(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    B = input_ids.numel()
    H = weight.size(-1)

    with torch.device(input_ids.device):
        embedding_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)](
            x_ptr=input_ids,
            weight_ptr=weight,
            output_ptr=output,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
