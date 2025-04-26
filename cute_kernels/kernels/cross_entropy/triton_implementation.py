import torch
import triton
import triton.language as tl

from ...constants import LIBRARY_NAME
from ...math import ceil_divide
from ...utils import cute_op


@triton.jit
def cross_entropy_forward_backward_triton_kernel(
    x_ptr,
    labels_ptr,
    loss_ptr,
    x_grad_ptr,
    has_logits_multiplier: tl.constexpr,
    logits_multiplier,
    B,
    V,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    reduction: tl.constexpr,
):
    BLOCK_ID = tl.program_id(axis=0)

    indices_b = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B

    Z = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    M = tl.full((BLOCK_SIZE_B, 1), -float("inf"), dtype=tl.float32)

    NUM_BLOCKS_V = tl.cdiv(V, BLOCK_SIZE_V)

    for v in range(NUM_BLOCKS_V):
        indices_v = v * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
        mask_v = indices_v < V

        indices = indices_b[:, None] * V + indices_v[None, :]
        mask_bv = mask_b[:, None] & mask_v[None, :]

        x_ptrs = x_ptr + indices
        x = tl.load(x_ptrs, mask=mask_bv, other=-float("inf"))

        x = x.to(tl.float32)
        if has_logits_multiplier:
            x *= logits_multiplier

        prev_m = M
        m = tl.max(x, axis=1, keep_dims=True)
        M = max(M, m)

        x -= M
        x = tl.exp(x)
        Z = Z * tl.exp(prev_m - M) + tl.sum(x, axis=1, keep_dims=True)

    labels_ptrs = labels_ptr + indices_b
    labels = tl.load(labels_ptrs, mask=mask_b)

    for v in range(NUM_BLOCKS_V):
        indices_v = v * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
        mask_v = indices_v < V

        indices = indices_b[:, None] * V + indices_v[None, :]
        mask_bv = mask_b[:, None] & mask_v[None, :]

        x_ptrs = x_ptr + indices
        x = tl.load(x_ptrs, mask=mask_bv)

        x = x.to(tl.float32)
        if has_logits_multiplier:
            x *= logits_multiplier
        x -= M
        x = tl.exp(x)
        x /= Z

        x -= tl.where(indices_v[None, :] == labels[:, None], 1, 0)
        if has_logits_multiplier:
            x *= logits_multiplier
        if reduction == "mean":
            x /= B

        x_grad_ptrs = x_grad_ptr + indices
        tl.store(x_grad_ptrs, x, mask=mask_bv)

    x_ptrs = x_ptr + indices_b * V + labels
    x = tl.load(x_ptrs, mask=mask_b)
    x = x.to(tl.float32)
    if has_logits_multiplier:
        x *= logits_multiplier

    loss = M + tl.log(Z) - x[:, None]
    loss = tl.where(mask_b[:, None], loss, 0)
    loss = tl.sum(loss, axis=0)

    if reduction == "mean":
        loss /= B

    tl.atomic_add(loss_ptr + tl.arange(0, 1), loss)


@cute_op(f"{LIBRARY_NAME}::cross_entropy_forward_backward_triton", mutates_args={"loss", "x_grad"})
def cross_entropy_forward_backward_triton(
    x: torch.Tensor,
    labels: torch.Tensor,
    loss: torch.Tensor,
    x_grad: torch.Tensor,
    logits_multiplier: float | None,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_V: int,
    reduction: str,
) -> None:
    B, V = x.size()

    with torch.device(x.device):
        cross_entropy_forward_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](
            x_ptr=x,
            labels_ptr=labels,
            loss_ptr=loss,
            x_grad_ptr=x_grad,
            has_logits_multiplier=logits_multiplier not in [None, 1],
            logits_multiplier=logits_multiplier,
            B=B,
            V=V,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
            reduction=reduction,
        )
