import torch

from ...cutotune import CutoTuneConfig, CutoTuneParameter, cutotune
from ...enums import KernelBackend
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size
from .cuda_implementation import naive_gemm_cuda, shared_memory_gemm_cuda
from .enums import CUDAKernelAlgorithm
from .torch_implementation import gemm_torch
from .triton_implementation import gemm_triton


@ensure_contiguous
@cutotune(
    configs=[
        CutoTuneConfig(dict(kernel_backend=KernelBackend.triton, cuda_kernel_algorithm=None)),
        CutoTuneConfig(dict(kernel_backend=KernelBackend.cuda, cuda_kernel_algorithm=CUDAKernelAlgorithm.naive)),
        CutoTuneConfig(
            dict(kernel_backend=KernelBackend.cuda, cuda_kernel_algorithm=CUDAKernelAlgorithm.shared_memory),
            condition=lambda **kwargs: not kwargs.get("is_a_transposed", False)
            and not kwargs.get("is_b_transposed", False),
        ),
    ],
    default_config=CutoTuneConfig(dict(kernel_backend=KernelBackend.triton)),
    triggers={"a.dtype", "is_a_transposed", "is_b_transposed"},
)
def gemm_cute(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None,
    is_a_transposed: bool = False,
    is_b_transposed: bool = False,
    alpha: float = 1,
    beta: float = 1,
    use_tf32: bool = True,
    kernel_backend: KernelBackend = CutoTuneParameter(),
    cuda_kernel_algorithm: CUDAKernelAlgorithm = CutoTuneParameter(),
    BLOCK_SIZE_M: int = CutoTuneParameter(),
    BLOCK_SIZE_K: int = CutoTuneParameter(),
    BLOCK_SIZE_N: int = CutoTuneParameter(),
) -> torch.Tensor:
    if is_a_transposed:
        assert a.dim() == 2, "only 2 dimensional a tensor is supported when a is transposed"
        K, M = a.size()
    else:
        M, K = get_num_elements_and_hidden_size(a)

    assert b.dim() == 2, "only 2 dimensional b tensor is supported"
    assert b.size(1 if is_b_transposed else 0) == K

    N = b.size(0 if is_b_transposed else 1)

    output = torch.empty(M, N, dtype=a.dtype, device=a.device)

    if beta == 0:
        assert c is None
    else:
        assert c is not None

    if kernel_backend == KernelBackend.cuda:
        if cuda_kernel_algorithm == CUDAKernelAlgorithm.shared_memory:
            if not isinstance(BLOCK_SIZE_M, CutoTuneParameter):
                assert BLOCK_SIZE_M == BLOCK_SIZE_K == BLOCK_SIZE_N

            shared_memory_gemm_cuda(
                a=a,
                b=b,
                c=c,
                output=output,
                is_a_transposed=is_a_transposed,
                is_b_transposed=is_b_transposed,
                alpha=alpha,
                beta=beta,
                M=M,
                K=K,
                N=N,
                BLOCK_SIZE=BLOCK_SIZE_M,
            )
        elif cuda_kernel_algorithm == CUDAKernelAlgorithm.naive:
            naive_gemm_cuda(
                a=a,
                b=b,
                c=c,
                output=output,
                is_a_transposed=is_a_transposed,
                is_b_transposed=is_b_transposed,
                alpha=alpha,
                beta=beta,
                M=M,
                K=K,
                N=N,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
    elif kernel_backend == KernelBackend.triton:
        assert cuda_kernel_algorithm is None

        gemm_triton(
            a=a,
            b=b,
            c=c,
            output=output,
            is_a_transposed=is_a_transposed,
            is_b_transposed=is_b_transposed,
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

    return output
