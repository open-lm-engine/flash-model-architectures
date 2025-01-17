import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ....math import ceil_divide, get_powers_of_2
from ....utils import cute_op, get_num_elements_and_hidden_size


_KERNEL_NAME = "linear_backward_triton"


@triton.jit
def _linear_backward_triton_kernel(
    input_ptr,
    output_grad_ptr,
    weight_grad_ptr,
    bias_grad_ptr,
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
    # output_grad -> M x N

    pid = tl.program_id(axis=0)
    num_programs_k = tl.cdiv(K, BLOCK_SIZE_K)

    pid_n = pid // num_programs_k
    pid_k = pid % num_programs_k

    indices_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    indices_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    mask_n = indices_n < N
    mask_k = indices_k < K

    accumulator_weight_grad = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    if bias_grad_ptr is not None and pid_k == 0:
        accumulator_bias_grad = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for m in range(tl.cdiv(M, BLOCK_SIZE_M)):
        indices_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = indices_m < M

        mask_mn = mask_m[:, None] & mask_n[None, :]
        mask_mk = mask_m[:, None] & mask_k[None, :]

        output_grad_ptrs = output_grad_ptr + indices_m[:, None] * N + indices_n[None, :]
        output_grad = tl.load(output_grad_ptrs, mask=mask_mn, other=0)

        input_ptrs = input_ptr + indices_m[:, None] * K + indices_k[None, :]
        input = tl.load(input_ptrs, mask=mask_mk, other=0)

        accumulator_weight_grad = tl.dot(output_grad.T, input, accumulator_weight_grad, allow_tf32=use_tf32)

        if bias_grad_ptr is not None and pid_k == 0:
            accumulator_bias_grad += tl.sum(output_grad, axis=0)

    tl.store(
        weight_grad_ptr + indices_m[:, None] * N + indices_n[None, :],
        accumulator_weight_grad.to(input_ptr.dtype.element_ty),
        mask=mask_n[:, None] & mask_k[None, :],
    )


def _condition(input: torch.Tensor, BLOCK_SIZE_M: int, BLOCK_SIZE_K: int, BLOCK_SIZE_N: int, **kwargs) -> bool:
    if input.dtype == torch.float32 and BLOCK_SIZE_M == 128 and BLOCK_SIZE_K == 128 and BLOCK_SIZE_N == 128:
        return False

    return True


@cutotune(
    get_cartesian_product_cutotune_configs(
        BLOCK_SIZE_M=get_powers_of_2(16, 128),
        BLOCK_SIZE_K=get_powers_of_2(16, 128),
        BLOCK_SIZE_N=get_powers_of_2(16, 128),
        num_warps=get_powers_of_2(1, 32),
        num_stages=[1, 2],
        condition=_condition,
    ),
    default_config=CutoTuneConfig(
        dict(BLOCK_SIZE_M=128, BLOCK_SIZE_K=64, BLOCK_SIZE_N=128, num_warps=8, num_stages=2)
    ),
    triggers={"input.dtype"},
)
@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"weight_grad", "bias_grad"})
def linear_backward_triton(
    input: torch.Tensor,
    output_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    bias_grad: torch.Tensor | None,
    use_tf32: bool,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_K: int,
    BLOCK_SIZE_N: int,
    num_warps: int,
    num_stages: int,
) -> None:
    M, K = get_num_elements_and_hidden_size(input)
    N = weight_grad.size(0)

    with torch.device(input.device):
        _linear_backward_triton_kernel[(ceil_divide(M, BLOCK_SIZE_M) * ceil_divide(N, BLOCK_SIZE_N),)](
            input_ptr=input,
            output_grad_ptr=output_grad,
            weight_grad_ptr=weight_grad,
            bias_grad_ptr=bias_grad,
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
