# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ...constants import LIBRARY_NAME
from ...math import ceil_divide, get_next_power_of_2, get_powers_of_2


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_SIZE_B in get_powers_of_2(1, 8):
        for num_warps in get_powers_of_2(4, 8):
            configs.append(triton.Config({"BLOCK_SIZE_B": BLOCK_SIZE_B}, num_warps=num_warps))

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_V"], reset_to_zero=["loss_ptr"])
@triton.jit
def cross_entropy_forward_backward_triton_kernel(
    x_ptr,
    x_stride,
    labels_ptr,
    labels_stride,
    loss_ptr,
    dx_ptr,
    dx_stride,
    logits_multiplier,
    B,
    V,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    reduction: tl.constexpr,
):
    BLOCK_ID = tl.program_id(axis=0)

    BLOCK_B = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    MASK_B = BLOCK_B < B

    Z = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    M = tl.full((BLOCK_SIZE_B, 1), -float("inf"), dtype=tl.float32)

    NUM_BLOCKS_V = tl.cdiv(V, BLOCK_SIZE_V)
    BLOCK_V = tl.arange(0, BLOCK_SIZE_V)
    x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_V[None, :] * x_stride[1]

    for _ in range(NUM_BLOCKS_V):
        MASK_V = BLOCK_V < V
        MASK_BV = MASK_B[:, None] & MASK_V[None, :]

        x = tl.load(x_ptrs, mask=MASK_BV, other=-float("inf")).to(tl.float32)
        if logits_multiplier is not None:
            x *= logits_multiplier

        prev_m = M
        m = tl.max(x, axis=1, keep_dims=True)
        M = max(M, m)

        x -= M
        x = tl.exp(x)
        Z = Z * tl.exp(prev_m - M) + tl.sum(x, axis=1, keep_dims=True)

        BLOCK_V += BLOCK_SIZE_V
        x_ptrs += BLOCK_SIZE_V * x_stride[1]

    labels = tl.load(labels_ptr + BLOCK_B * labels_stride[0], mask=MASK_B)

    BLOCK_V = tl.arange(0, BLOCK_SIZE_V)
    x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_V[None, :] * x_stride[1]
    dx_ptrs = dx_ptr + BLOCK_B[:, None] * dx_stride[0] + BLOCK_V[None, :] * dx_stride[1]

    for _ in range(NUM_BLOCKS_V):
        MASK_V = BLOCK_V < V
        MASK_BV = MASK_B[:, None] & MASK_V[None, :]

        x = tl.load(x_ptrs, mask=MASK_BV).to(tl.float32)
        if logits_multiplier is not None:
            x *= logits_multiplier

        x -= M
        x = tl.exp(x)
        x /= Z

        x -= tl.where(BLOCK_V[None, :] == labels[:, None], 1, 0)

        if logits_multiplier is not None:
            x *= logits_multiplier
        if reduction == "mean":
            x /= B

        tl.store(dx_ptrs, x, mask=MASK_BV)

        BLOCK_V += BLOCK_SIZE_V
        x_ptrs += BLOCK_SIZE_V * x_stride[1]
        dx_ptrs += BLOCK_SIZE_V * dx_stride[1]

    x = tl.load(x_ptr + BLOCK_B * x_stride[0] + labels * x_stride[1], mask=MASK_B).to(tl.float32)
    if logits_multiplier is not None:
        x *= logits_multiplier

    loss = M + tl.log(Z) - x[:, None]
    loss = tl.where(MASK_B[:, None], loss, 0)
    loss = tl.sum(loss, axis=0)

    if reduction == "mean":
        loss /= B

    tl.atomic_add(loss_ptr + tl.arange(0, 1), loss, sem="relaxed")


@custom_op(f"{LIBRARY_NAME}::cross_entropy_forward_backward_triton", mutates_args={"loss", "x_grad"})
def cross_entropy_forward_backward_triton(
    x: torch.Tensor,
    labels: torch.Tensor,
    loss: torch.Tensor,
    x_grad: torch.Tensor,
    logits_multiplier: float | None,
    reduction: str,
) -> None:
    B, V = x.size()

    BLOCK_SIZE_V = min(get_next_power_of_2(V), 4096 if x.dtype == torch.float32 else 8192)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]),)

    with torch.device(x.device):
        cross_entropy_forward_backward_triton_kernel[GRID](
            x_ptr=x,
            x_stride=x.stride(),
            labels_ptr=labels,
            labels_stride=labels.stride(),
            loss_ptr=loss,
            dx_ptr=x_grad,
            dx_stride=x_grad.stride(),
            logits_multiplier=logits_multiplier,
            B=B,
            V=V,
            reduction=reduction,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
        )
