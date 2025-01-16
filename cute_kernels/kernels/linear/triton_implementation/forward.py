import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ....math import ceil_divide
from ....utils import cute_op, get_num_elements_and_hidden_size


_KERNEL_NAME = "linear_forward_triton"


@triton.jit
def _linear_forward_triton_kernel(
    input_ptr,
    weight_ptr,
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
    # output -> M x N

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
        input = tl.load(input_ptrs, mask=mask_mk, other=0)

        weight_ptrs = weight_ptr + indices_n[:, None] * K + indices_k[None, :]
        weight = tl.load(weight_ptrs, mask=mask_nk, other=0)

        accumulator = tl.dot(input, weight.T, accumulator, allow_tf32=True)

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + indices_n, mask=mask_n, other=0)
        accumulator += bias[None, :]

    output_ptrs = output_ptr + indices_m[:, None] * N + indices_n[None, :]
    tl.store(output_ptrs, accumulator.to(input_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@cutotune(
    get_cartesian_product_cutotune_configs(
        BLOCK_SIZE_M=[16, 32, 64, 128],
        BLOCK_SIZE_K=[16, 32, 64, 128],
        BLOCK_SIZE_N=[16, 32, 64, 128],
        num_warps=[1, 2, 4, 8, 16, 32],
        num_stages=[1, 2],
    ),
    default_config=CutoTuneConfig(dict(BLOCK_SIZE_M=128, BLOCK_SIZE_K=32, BLOCK_SIZE_N=128, num_warps=8)),
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
    num_warps: int,
    num_stages: int,
) -> None:
    M, K = get_num_elements_and_hidden_size(input)
    N = weight.size(0)

    with torch.device(input.device):
        _linear_forward_triton_kernel[(ceil_divide(M, BLOCK_SIZE_M), ceil_divide(N, BLOCK_SIZE_N))](
            input_ptr=input,
            weight_ptr=weight,
            bias_ptr=bias,
            output_ptr=output,
            M=M,
            K=K,
            N=N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
