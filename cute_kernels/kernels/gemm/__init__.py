import torch

from ...math import ceil_divide
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size
from .cuda_implementation import (
    cutlass_gemm_cuda,
    cutlass_tensorcore_mma_gemm_cuda,
    naive_gemm_cuda,
    shared_memory_gemm_cuda,
)
from .torch_implementation import gemm_torch
from .triton_implementation import _gemm_triton_kernel


@ensure_contiguous
def gemm_cute(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    is_A_transposed: bool = False,
    is_B_transposed: bool = False,
    alpha: float = 1,
    beta: float = 1,
    use_tf32: bool = True,
    kernel_backend: str | None = None,
) -> torch.Tensor:
    if is_A_transposed:
        assert A.dim() == 2, "only 2 dimensional a tensor is supported when a is transposed"
        K, M = A.size()
    else:
        M, K = get_num_elements_and_hidden_size(A)

    assert B.dim() == 2, "only 2 dimensional B tensor is supported"
    assert B.size(1 if is_B_transposed else 0) == K

    N = B.size(0 if is_B_transposed else 1)

    output = torch.empty(M, N, dtype=A.dtype, device=A.device)

    if beta == 0:
        assert C is None
    else:
        assert C is not None

    if kernel_backend == "cutlass_tensorcore_mma_gemm_cuda":
        cutlass_tensorcore_mma_gemm_cuda(
            A=A,
            B=B,
            C=C,
            output=output,
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
            alpha=alpha,
            beta=beta,
            M=M,
            K=K,
            N=N,
        )
    elif kernel_backend == "cutlass":
        cutlass_gemm_cuda(
            A=A,
            B=B,
            C=C,
            output=output,
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
            alpha=alpha,
            beta=beta,
            M=M,
            K=K,
            N=N,
        )
    elif kernel_backend == "shared_memory_cuda":
        BLOCK_SIZE = 32

        shared_memory_gemm_cuda(
            A=A,
            B=B,
            C=C,
            output=output,
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
            alpha=alpha,
            beta=beta,
            M=M,
            K=K,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif kernel_backend == "naive_cuda":
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16

        naive_gemm_cuda(
            A=A,
            B=B,
            C=C,
            output=output,
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
            alpha=alpha,
            beta=beta,
            M=M,
            K=K,
            N=N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    elif kernel_backend == "triton":
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_K = 64
        BLOCK_SIZE_N = 128
        num_warps = 8
        num_stages = 2

        with torch.cuda.device(A.device):
            _gemm_triton_kernel[ceil_divide(M, BLOCK_SIZE_M) * ceil_divide(N, BLOCK_SIZE_N),](
                A_ptr=A,
                B_ptr=B,
                C_ptr=C,
                output_ptr=output,
                alpha=alpha,
                beta=beta,
                is_A_transposed=is_A_transposed,
                is_B_transposed=is_B_transposed,
                use_tf32=use_tf32,
                M=M,
                K=K,
                N=N,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
