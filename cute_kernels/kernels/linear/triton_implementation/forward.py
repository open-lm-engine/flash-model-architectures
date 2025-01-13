import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op


_KERNEL_NAME = "linear_forward_triton"


@triton.jit
def _linear_forward_triton_kernel(
    input_ptr,
    weight_ptr,
    has_bias: tl.constexpr,
    bias_ptr,
    output_ptr,
    M,
    K,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    return


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def linear_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_K: int,
    BLOCK_SIZE_N: int,
) -> None:
    K = input.size(-1)
    M = input.numel() // K
    N = weight.size(0)

    with torch.device(input.device):
        _linear_forward_triton_kernel[
            (ceil_divide(M, BLOCK_SIZE_M), ceil_divide(N, BLOCK_SIZE_N), ceil_divide(K, BLOCK_SIZE_K))
        ](
            input_ptr=input,
            weight_ptr=weight,
            has_bias=bias is not None,
            bias_ptr=bias,
            output_ptr=output,
            M=M,
            K=K,
            N=N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
