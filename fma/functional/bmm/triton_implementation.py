# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ...constants import LIBRARY_NAME
from ...math import ceil_divide


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2
        ),
    ],
    key=[],
)
@triton.jit
def bmm_triton_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    output_ptr,
    alpha,
    beta,
    IS_A_TRANSPOSED: tl.constexpr,
    IS_B_TRANSPOSED: tl.constexpr,
    M,
    K,
    N,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # A -> K x M if is_A_transposed else M x K
    # B -> N x K if is_B_transposed else K x N
    # C -> M x N

    BLOCK_ID_L = tl.program_id(axis=0)
    BLOCK_ID = tl.program_id(axis=1)

    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_SIZE_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_SIZE_N)

    NUM_BLOCKS_IN_GROUP = GROUP_SIZE_M * NUM_BLOCKS_N
    GROUP_ID = BLOCK_ID // NUM_BLOCKS_IN_GROUP

    FIRST_BLOCK_M_IN_GROUP = GROUP_ID * GROUP_SIZE_M
    CURRENT_GROUP_SIZE_M = min(NUM_BLOCKS_M - FIRST_BLOCK_M_IN_GROUP, GROUP_SIZE_M)

    BLOCK_ID_M = FIRST_BLOCK_M_IN_GROUP + ((BLOCK_ID % NUM_BLOCKS_IN_GROUP) % CURRENT_GROUP_SIZE_M)
    BLOCK_ID_N = (BLOCK_ID % NUM_BLOCKS_IN_GROUP) // CURRENT_GROUP_SIZE_M

    if BLOCK_ID_N >= NUM_BLOCKS_N:
        return

    indices_m = BLOCK_ID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    indices_n = BLOCK_ID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = indices_m < M
    mask_n = indices_n < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    indices_k = tl.arange(0, BLOCK_SIZE_K)

    for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = indices_k < K

        if IS_A_TRANSPOSED:
            mask_A = mask_k[:, None] & mask_m[None, :]
            A_ptrs = A_ptr + BLOCK_ID_L * M * K + indices_k[:, None] * M + indices_m[None, :]
        else:
            mask_A = mask_m[:, None] & mask_k[None, :]
            A_ptrs = A_ptr + BLOCK_ID_L * M * K + indices_m[:, None] * K + indices_k[None, :]

        A = tl.load(A_ptrs, mask=mask_A)

        if IS_A_TRANSPOSED:
            A = A.T

        if IS_B_TRANSPOSED:
            mask_B = mask_n[:, None] & mask_k[None, :]
            B_ptrs = B_ptr + BLOCK_ID_L * K * N + indices_n[:, None] * K + indices_k[None, :]
        else:
            mask_B = mask_k[:, None] & mask_n[None, :]
            B_ptrs = B_ptr + BLOCK_ID_L * K * N + indices_k[:, None] * N + indices_n[None, :]

        B = tl.load(B_ptrs, mask=mask_B)

        if IS_B_TRANSPOSED:
            B = B.T

        accumulator = tl.dot(A, B, accumulator, allow_tf32=True)
        indices_k += BLOCK_SIZE_K

    accumulator = accumulator.to(A_ptr.dtype.element_ty)
    accumulator *= alpha

    indices_lmn = BLOCK_ID_L * M * N + indices_m[:, None] * N + indices_n[None, :]
    mask_mn = mask_m[:, None] & mask_n[None, :]

    if C_ptr is not None:
        C = tl.load(C_ptr + indices_lmn, mask=mask_mn)
        accumulator += beta * C

    tl.store(output_ptr + indices_lmn, accumulator, mask=mask_mn)


@custom_op(f"{LIBRARY_NAME}::bmm_triton", mutates_args={"output"})
def bmm_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    output: torch.Tensor,
    is_A_transposed: bool,
    is_B_transposed: bool,
    alpha: float,
    beta: float,
) -> None:
    L, M, K = A.size()
    if is_A_transposed:
        M, K = K, M

    N = B.size(1 if is_B_transposed else 2)

    GRID = lambda meta: (
        L,
        ceil_divide(M, meta["BLOCK_SIZE_M"]) * ceil_divide(N, meta["BLOCK_SIZE_N"]),
    )

    with torch.device(A.device):
        bmm_triton_kernel[GRID](
            A_ptr=A,
            B_ptr=B,
            C_ptr=C,
            output_ptr=output,
            alpha=alpha,
            beta=beta,
            IS_A_TRANSPOSED=is_A_transposed,
            IS_B_TRANSPOSED=is_B_transposed,
            M=M,
            K=K,
            N=N,
        )
