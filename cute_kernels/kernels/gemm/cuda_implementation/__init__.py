import torch

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit
from ....utils import cute_op


_NAIVE_KERNEL_NAME = "naive_gemm_cuda"
_SHARED_MEMORY_KERNEL_NAME = "shared_memory_gemm_cuda"
_CUTLASS_KERNEL_NAME = "cutlass_gemm_cuda"
_CUTLASS_TENSORCORE_MMA_KERNEL_NAME = "cutlass_tensorcore_mma_gemm_cuda"


@cute_op(f"{LIBRARY_NAME}::{_NAIVE_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit()
def naive_gemm_cuda(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    output: torch.Tensor,
    is_A_transposed: bool,
    is_B_transposed: bool,
    alpha: float,
    beta: float,
    M: int,
    K: int,
    N: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
) -> None: ...


@cute_op(f"{LIBRARY_NAME}::{_SHARED_MEMORY_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit()
def shared_memory_gemm_cuda(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    output: torch.Tensor,
    is_A_transposed: bool,
    is_B_transposed: bool,
    alpha: float,
    beta: float,
    M: int,
    K: int,
    N: int,
    BLOCK_SIZE: int,
) -> None: ...


@cute_op(f"{LIBRARY_NAME}::{_CUTLASS_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit()
def cutlass_gemm_cuda(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    output: torch.Tensor,
    is_A_transposed: bool,
    is_B_transposed: bool,
    alpha: float,
    beta: float,
    M: int,
    K: int,
    N: int,
) -> None: ...


@cute_op(f"{LIBRARY_NAME}::{_CUTLASS_TENSORCORE_MMA_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit()
def cutlass_tensorcore_mma_gemm_cuda(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    output: torch.Tensor,
    is_A_transposed: bool,
    is_B_transposed: bool,
    alpha: float,
    beta: float,
    M: int,
    K: int,
    N: int,
) -> None: ...
