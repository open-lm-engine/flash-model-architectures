# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ...constants import LIBRARY_NAME
from ...math import ceil_divide, get_powers_of_2


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_SIZE_M in get_powers_of_2(32, 128):
        for BLOCK_SIZE_N in get_powers_of_2(32, 128):
            for BLOCK_SIZE_K in get_powers_of_2(16, 64):
                for NUM_WARPS in get_powers_of_2(4, 8):
                    for NUM_STAGES in range(4):
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_SIZE_M": BLOCK_SIZE_M,
                                    "BLOCK_SIZE_N": BLOCK_SIZE_N,
                                    "BLOCK_SIZE_K": BLOCK_SIZE_K,
                                },
                                num_warps=NUM_WARPS,
                                num_stages=NUM_STAGES,
                            )
                        )

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=[])
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # A -> K x M if is_A_transposed else M x K
    # B -> N x K if is_B_transposed else K x N
    # C -> M x N

    BLOCK_ID_L = tl.program_id(axis=0)

    BLOCK_ID = tl.program_id(axis=1)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_SIZE_N)

    BLOCK_ID_M = BLOCK_ID // NUM_BLOCKS_N
    BLOCK_ID_N = BLOCK_ID % NUM_BLOCKS_N

    indices_m = BLOCK_ID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    indices_n = BLOCK_ID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = indices_m < M
    mask_n = indices_n < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        indices_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
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
