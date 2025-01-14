import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
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
    # NOTE pytorch linear layer has following layout
    # input -> M x K
    # weight -> N x K
    # bias -> N

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    indices_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    indices_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = indices_m < M
    mask_n = indices_n < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        indices_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = indices_k < K

        mask_mk = mask_m[:, None] & mask_k[None, :]
        mask_nk = mask_n[:, None] & mask_k[None, :]

        input_ptrs = input_ptr + indices_m[:, None] * K + indices_k[None, :]
        input = tl.load(input_ptrs, mask=mask_mk)

        weight_ptrs = weight_ptr + indices_n[:, None] * K + indices_k[None, :]
        weight = tl.load(weight_ptrs, mask=mask_nk)

        accumulator = tl.dot(input, weight.T, accumulator)

    if has_bias:
        bias = tl.load(bias_ptr + indices_n, mask=mask_n)
        accumulator += bias[None, :]

    output_ptrs = output_ptr + indices_m[:, None] * N + indices_n[None, :]
    tl.store(output_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])


@cutotune(
    get_cartesian_product_cutotune_configs(
        BLOCK_SIZE_M=[32, 64, 128, 256], BLOCK_SIZE_K=[32, 64, 128], BLOCK_SIZE_N=[32, 64, 128, 256]
    ),
    default_config=CutoTuneConfig(dict(BLOCK_SIZE_M=64, BLOCK_SIZE_K=64, BLOCK_SIZE_N=64)),
)
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
        _linear_forward_triton_kernel[(ceil_divide(M, BLOCK_SIZE_M), ceil_divide(N, BLOCK_SIZE_N))](
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
