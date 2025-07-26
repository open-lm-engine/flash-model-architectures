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
    labels_ptr,
    loss_ptr,
    x_grad_ptr,
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

        x = tl.load(x_ptr + indices, mask=mask_bv, other=-float("inf")).to(tl.float32)
        if logits_multiplier is not None:
            x *= logits_multiplier

        prev_m = M
        m = tl.max(x, axis=1, keep_dims=True)
        M = max(M, m)

        x -= M
        x = tl.exp(x)
        Z = Z * tl.exp(prev_m - M) + tl.sum(x, axis=1, keep_dims=True)

    labels = tl.load(labels_ptr + indices_b, mask=mask_b)

    for v in range(NUM_BLOCKS_V):
        indices_v = v * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
        mask_v = indices_v < V

        indices = indices_b[:, None] * V + indices_v[None, :]
        mask_bv = mask_b[:, None] & mask_v[None, :]

        x = tl.load(x_ptr + indices, mask=mask_bv).to(tl.float32)
        if logits_multiplier is not None:
            x *= logits_multiplier
        x -= M
        x = tl.exp(x)
        x /= Z

        x -= tl.where(indices_v[None, :] == labels[:, None], 1, 0)
        if logits_multiplier is not None:
            x *= logits_multiplier
        if reduction == "mean":
            x /= B

        tl.store(x_grad_ptr + indices, x, mask=mask_bv)

    x = tl.load(x_ptr + indices_b * V + labels, mask=mask_b).to(tl.float32)
    if logits_multiplier is not None:
        x *= logits_multiplier

    loss = M + tl.log(Z) - x[:, None]
    loss = tl.where(mask_b[:, None], loss, 0)
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
            labels_ptr=labels,
            loss_ptr=loss,
            x_grad_ptr=x_grad,
            logits_multiplier=logits_multiplier,
            B=B,
            V=V,
            reduction=reduction,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
        )
