# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...utils import ensure_contiguous
from .cuda_implementation import grouped_gemm_cuda


def prepare_grouped_gemm_inputs_cute(
    A: torch.Tensor, B: torch.Tensor, is_A_transposed: bool = False, is_B_transposed: bool = False
) -> tuple[torch.Tensor]:
    E = A.size(0)
    M = A.size(2 if is_A_transposed else 1)
    N = B.size(1 if is_B_transposed else 2)

    ptr_A = torch.empty(E, device=A.device, dtype=torch.uint64)
    ptr_B = torch.empty(E, device=A.device, dtype=torch.uint64)
    ptr_C = torch.empty(E, device=A.device, dtype=torch.uint64)
    ptr_D = torch.empty(E, device=A.device, dtype=torch.uint64)

    stride_A = torch.empty(E, device=A.device, dtype=torch.uint64)
    stride_B = torch.empty(E, device=A.device, dtype=torch.uint64)
    stride_C = torch.empty(E, device=A.device, dtype=torch.uint64)

    problem_sizes = torch.empty(3 * E, device=A.device, dtype=torch.uint32)
    output = torch.empty(E, M, N, device=A.device, dtype=torch.bfloat16)

    return ptr_A, ptr_B, ptr_C, ptr_D, stride_A, stride_B, stride_C, problem_sizes, output


@ensure_contiguous
def grouped_gemm_cute(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    M_array: torch.Tensor,
    N_array: torch.Tensor,
    K_array: torch.Tensor,
    ptr_A: torch.Tensor,
    ptr_B: torch.Tensor,
    ptr_C: torch.Tensor,
    ptr_D: torch.Tensor,
    stride_A: torch.Tensor,
    stride_B: torch.Tensor,
    stride_C: torch.Tensor,
    problem_sizes: torch.Tensor,
    output: torch.Tensor,
    alpha: float = 1,
    beta: float = 0,
    is_A_transposed: bool = False,
    is_B_transposed: bool = False,
    benchmark: bool = False,
) -> torch.Tensor:
    assert beta == 0
    assert C is None

    E = A.size(0)
    M = A.size(2 if is_A_transposed else 1)
    N = B.size(1 if is_B_transposed else 2)

    grouped_gemm_cuda(
        A=A,
        B=B,
        C=C,
        output=output,
        M_array=M_array,
        N_array=N_array,
        K_array=K_array,
        ptr_A=ptr_A,
        ptr_B=ptr_B,
        ptr_C=ptr_C,
        ptr_D=ptr_D,
        stride_A=stride_A,
        stride_B=stride_B,
        stride_C=stride_C,
        problem_sizes=problem_sizes,
        is_A_transposed=is_A_transposed,
        is_B_transposed=is_B_transposed,
        alpha=alpha,
        beta=beta,
        benchmark=benchmark,
    )

    return output
