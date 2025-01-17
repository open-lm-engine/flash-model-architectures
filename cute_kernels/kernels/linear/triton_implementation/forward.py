import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ....math import ceil_divide, get_powers_of_2
from ....utils import cute_op, get_num_elements_and_hidden_size


_KERNEL_NAME = "linear_forward_triton"


def _get_triton_autotune_configs() -> list[triton.Config]:
    configs = []
    for num_stages in [1, 2]:
        for num_warps in get_powers_of_2(1, 32):
            configs.append(triton.Config({}, num_stages=num_stages, num_warps=num_warps))

    return configs


@triton.autotune(configs=_get_triton_autotune_configs(), key=["BLOCK_SIZE_M", "BLOCK_SIZE_K", "BLOCK_SIZE_K"])
@triton.jit
def _linear_forward_triton_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    use_tf32: tl.constexpr,
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

    pid = tl.program_id(axis=0)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_programs_n
    pid_n = pid % num_programs_n

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

        accumulator = tl.dot(input, weight.T, accumulator, allow_tf32=use_tf32)

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + indices_n, mask=mask_n, other=0)
        accumulator += bias[None, :]

    output_ptrs = output_ptr + indices_m[:, None] * N + indices_n[None, :]
    tl.store(output_ptrs, accumulator.to(input_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def _condition(input: torch.Tensor, BLOCK_SIZE_M: int, BLOCK_SIZE_K: int, BLOCK_SIZE_N: int, **kwargs) -> bool:
    if input.dtype == torch.float32 and BLOCK_SIZE_M == 128 and BLOCK_SIZE_K == 128 and BLOCK_SIZE_N == 128:
        return False

    return True


@cutotune(
    get_cartesian_product_cutotune_configs(
        GROUP_SIZE_M=[8],
        BLOCK_SIZE_M=[16, 32, 64, 128],
        BLOCK_SIZE_K=[16, 32, 64, 128],
        BLOCK_SIZE_N=[16, 32, 64, 128],
        condition=_condition,
    ),
    default_config=CutoTuneConfig(
        dict(BLOCK_SIZE_M=128, BLOCK_SIZE_K=64, BLOCK_SIZE_N=128, num_warps=8, num_stages=2)
    ),
    triggers={"input.dtype"},
)
@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def linear_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    use_tf32: bool,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_K: int,
    BLOCK_SIZE_N: int,
) -> None:
    M, K = get_num_elements_and_hidden_size(input)
    N = weight.size(0)

    with torch.device(input.device):
        _linear_forward_triton_kernel[(ceil_divide(M, BLOCK_SIZE_M) * ceil_divide(N, BLOCK_SIZE_N),)](
            input_ptr=input,
            weight_ptr=weight,
            bias_ptr=bias,
            output_ptr=output,
            use_tf32=use_tf32,
            M=M,
            K=K,
            N=N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
