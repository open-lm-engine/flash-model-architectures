# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import triton
import triton.language as tl

from ...math import get_powers_of_2


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
    dx_ptr,
    logits_multiplier,
    B,
    V,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    reduction: tl.constexpr,
):
    BLOCK_ID = tl.program_id(axis=0)

    BLOCK_B = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = BLOCK_B < B

    Z = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    M = tl.full((BLOCK_SIZE_B, 1), -float("inf"), dtype=tl.float32)

    NUM_BLOCKS_V = tl.cdiv(V, BLOCK_SIZE_V)

    for v in range(NUM_BLOCKS_V):
        BLOCK_V = v * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
        mask_v = BLOCK_V < V

        BLOCK = BLOCK_B[:, None] * V + BLOCK_V[None, :]
        mask_bv = mask_b[:, None] & mask_v[None, :]

        x = tl.load(x_ptr + BLOCK, mask=mask_bv, other=-float("inf")).to(tl.float32)
        if logits_multiplier is not None:
            x *= logits_multiplier

        prev_m = M
        m = tl.max(x, axis=1, keep_dims=True)
        M = max(M, m)

        x -= M
        x = tl.exp(x)
        Z = Z * tl.exp(prev_m - M) + tl.sum(x, axis=1, keep_dims=True)

    labels = tl.load(labels_ptr + BLOCK_B, mask=mask_b)

    for v in range(NUM_BLOCKS_V):
        BLOCK_V = v * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
        mask_v = BLOCK_V < V

        BLOCK = BLOCK_B[:, None] * V + BLOCK_V[None, :]
        mask_bv = mask_b[:, None] & mask_v[None, :]

        x = tl.load(x_ptr + BLOCK, mask=mask_bv).to(tl.float32)
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

        tl.store(dx_ptr + BLOCK, x, mask=mask_bv)

    x = tl.load(x_ptr + BLOCK_B * V + labels, mask=mask_b).to(tl.float32)
    if logits_multiplier is not None:
        x *= logits_multiplier

    loss = M + tl.log(Z) - x[:, None]
    loss = tl.where(mask_b[:, None], loss, 0)
    loss = tl.sum(loss, axis=0)

    if reduction == "mean":
        loss /= B

    tl.atomic_add(loss_ptr + tl.arange(0, 1), loss, sem="relaxed")
