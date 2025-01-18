import torch

from ...cutotune import CutoTuneParameter
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size
from .torch_implementation import gemm_torch
from .triton_implementation import gemm_triton


@ensure_contiguous
def gemm_cute(
    a: torch.Tensor,
    b: torch.Tensor,
    is_a_transposed: bool = False,
    is_b_transposed: bool = False,
    use_tf32: bool = True,
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

    if is_b_transposed:
        N = b.size(0)
        assert b.size(1) == K
    else:
        N = b.size(1)
        assert b.size(0) == K

    output = torch.empty(M, N, dtype=a.dtype, device=a.device)

    gemm_triton(
        a=a,
        b=b,
        c=output,
        is_a_transposed=is_a_transposed,
        is_b_transposed=is_b_transposed,
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
