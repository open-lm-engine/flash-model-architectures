import torch

from ...cutotune import CutoTuneConfig, CutoTuneParameter, cutotune
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size
from .cuda_implementation import cutlass_gemm_cuda, naive_gemm_cuda, shared_memory_gemm_cuda
from .torch_implementation import gemm_torch
from .triton_implementation import gemm_triton


@ensure_contiguous
@cutotune(
    configs=[
        CutoTuneConfig(dict(kernel_backend="triton", cuda_kernel_algorithm=None)),
        CutoTuneConfig(dict(kernel_backend="cuda", cuda_kernel_algorithm="naive")),
        CutoTuneConfig(
            dict(kernel_backend="cuda", cuda_kernel_algorithm="shared_memory"),
            condition=lambda **kwargs: not kwargs.get("is_A_transposed", False)
            and not kwargs.get("is_B_transposed", False),
        ),
    ],
    default_config=CutoTuneConfig(dict(kernel_backend="triton")),
    triggers={"A.dtype", "is_A_transposed", "is_B_transposed"},
)
def gemm_cute(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    is_A_transposed: bool = False,
    is_B_transposed: bool = False,
    alpha: float = 1,
    beta: float = 1,
    use_tf32: bool = True,
    kernel_backend: str = CutoTuneParameter(),
    cuda_kernel_algorithm: str = CutoTuneParameter(),
    BLOCK_SIZE_M: int = CutoTuneParameter(),
    BLOCK_SIZE_K: int = CutoTuneParameter(),
    BLOCK_SIZE_N: int = CutoTuneParameter(),
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

    if kernel_backend == "cuda":
        if cuda_kernel_algorithm == "cutlass_gemm_cuda":
            assert isinstance(BLOCK_SIZE_M, CutoTuneParameter)
            assert isinstance(BLOCK_SIZE_K, CutoTuneParameter)
            assert isinstance(BLOCK_SIZE_N, CutoTuneParameter)

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
        elif cuda_kernel_algorithm == "shared_memory":
            if (
                not isinstance(BLOCK_SIZE_M, CutoTuneParameter)
                or not isinstance(BLOCK_SIZE_K, CutoTuneParameter)
                or not isinstance(BLOCK_SIZE_N, CutoTuneParameter)
            ):
                assert BLOCK_SIZE_M == BLOCK_SIZE_K == BLOCK_SIZE_N

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
                BLOCK_SIZE=BLOCK_SIZE_M,
            )
        elif cuda_kernel_algorithm == "naive":
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
        else:
            raise ValueError(f"unexpected cuda_kernel_algorithm ({cuda_kernel_algorithm})")
    elif kernel_backend == "triton":
        assert cuda_kernel_algorithm is None

        gemm_triton(
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
            use_tf32=use_tf32,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=CutoTuneParameter(),
            num_stages=CutoTuneParameter(),
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
