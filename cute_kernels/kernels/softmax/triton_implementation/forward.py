import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op, get_num_elements_and_hidden_size


@triton.jit
def _load_x(x_ptr, h, H, BLOCK_SIZE_H, indices_b, mask_b, other=None):
    indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask_h = indices_h < H

    indices = indices_b[:, None] * H + indices_h[None, :]
    mask_bh = mask_b[:, None] & mask_h[None, :]

    x_ptrs = x_ptr + indices
    x = tl.load(x_ptrs, mask=mask_bh, other=other)

    return x, indices, mask_bh


@triton.jit
def softmax_forward_triton_kernel(
    x_ptr,
    output_ptr,
    has_logits_multiplier: tl.constexpr,
    logits_multiplier,
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

    num_blocks_h = tl.cdiv(H, BLOCK_SIZE_H)

    for h in range(num_blocks_h):
        x, indices, mask_bh = _load_x(
            x_ptr=x_ptr, h=h, H=H, BLOCK_SIZE_H=BLOCK_SIZE_H, indices_b=indices_b, mask_b=mask_b, other=-float("inf")
        )

        x = x.to(tl.float32)
        if has_logits_multiplier:
            x *= logits_multiplier

        prev_m = M
        m = tl.max(x, axis=1, keep_dims=True)
        M = max(M, m)

        x -= M
        x = tl.exp(x)
        Z = Z * tl.exp(prev_m - M) + tl.sum(x, axis=1, keep_dims=True)

    for h in range(num_blocks_h):
        x, indices, mask_bh = _load_x(
            x_ptr=x_ptr, h=h, H=H, BLOCK_SIZE_H=BLOCK_SIZE_H, indices_b=indices_b, mask_b=mask_b
        )

        x = x.to(tl.float32)
        if has_logits_multiplier:
            x *= logits_multiplier

        x -= M
        x = tl.exp(x)
        x /= Z

        output_ptrs = output_ptr + indices
        tl.store(output_ptrs, x, mask=mask_bh)


@cute_op(f"{LIBRARY_NAME}::softmax_forward_triton", mutates_args={"output"})
def softmax_forward_triton(
    x: torch.Tensor, output: torch.Tensor, logits_multiplier: float | None, BLOCK_SIZE_B: int, BLOCK_SIZE_H: int
) -> None:
    if x.dim() == 1:
        B = 1
        H = x.size(-1)
    else:
        B, H = get_num_elements_and_hidden_size(x)

    with torch.device(x.device):
        softmax_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](
            x_ptr=x,
            output_ptr=output,
            has_logits_multiplier=logits_multiplier not in [None, 1],
            logits_multiplier=logits_multiplier,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
