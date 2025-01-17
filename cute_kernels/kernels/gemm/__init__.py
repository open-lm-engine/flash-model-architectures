import torch

from ...cutotune import CutoTuneParameter
from ...utils import ensure_contiguous
from .torch_implementation import gemm_torch
from .triton_implementation import gemm_triton


@ensure_contiguous
def gemm_cute(
    a: torch.Tensor,
    b: torch.Tensor,
    use_tf32: bool = True,
    BLOCK_SIZE_M: int = CutoTuneParameter(),
    BLOCK_SIZE_K: int = CutoTuneParameter(),
    BLOCK_SIZE_N: int = CutoTuneParameter(),
) -> torch.Tensor:
    output = torch.empty(*a.size()[:-1], b.size(-1), dtype=a.dtype, device=a.device)

    gemm_triton(
        a=a,
        b=b,
        c=output,
        use_tf32=use_tf32,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=CutoTuneParameter(),
        num_stages=CutoTuneParameter(),
    )

    return output
