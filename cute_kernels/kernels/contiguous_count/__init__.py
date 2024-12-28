import torch

from ...constants import (
    COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2,
    COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
    MAX_CUDA_BLOCK_SIZE,
)
from ...cutotune import CutoTuneConfig, CutoTuneParameter, cutotune, get_cartesian_product_cutotune_configs
from ...enums import KernelBackend
from ...math import get_next_power_of_2
from ...utils import get_sm_count
from .cuda_implementation import contiguous_count_cuda
from .torch_implementation import contiguous_count_torch
from .triton_implementation import contiguous_count_triton


def _contiguous_count_cute(
    x: torch.Tensor,
    size: int,
    kernel_backend: KernelBackend = KernelBackend.triton,
    BLOCK_SIZE: int = 64,
) -> torch.Tensor:
    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    output = torch.zeros(size, dtype=torch.uint32, device=x.device)

    if kernel_backend == KernelBackend.cuda:
        contiguous_count_cuda(
            x=x,
            output=output,
            sm_count=get_sm_count(x.device),
            thread_block_cluster_size=4,
            size=size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif kernel_backend == KernelBackend.triton:
        contiguous_count_triton(x=x, output=output, size=size, BLOCK_SIZE=BLOCK_SIZE)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output


# @torch.no_grad()
# @cutotune(
#     configs=(
#         get_cartesian_product_cutotune_configs(
#             kernel_backend=[KernelBackend.cuda], BLOCK_SIZE=COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2
#         )
#         if torch.cuda.is_available()
#         else []
#     )
#     + (
#         get_cartesian_product_cutotune_configs(
#             kernel_backend=[KernelBackend.cuda],
#             BLOCK_SIZE=COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2,
#             condition=lambda **kwargs: kwargs["x"].dtype in [torch.float16, torch.bfloat16],
#         )
#         if torch.cuda.is_available()
#         else []
#     )
#     + get_cartesian_product_cutotune_configs(
#         kernel_backend=[KernelBackend.triton], BLOCK_SIZE=COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2
#     ),
#     default_config=CutoTuneConfig({"kernel_backend": KernelBackend.triton, "BLOCK_SIZE": MAX_CUDA_BLOCK_SIZE}),
#     functional_triggers={"size_next_power_of_2": lambda **kwargs: get_next_power_of_2(kwargs["size"])},
# )
def contiguous_count_cute(
    x: torch.Tensor,
    size: int,
    kernel_backend: KernelBackend = CutoTuneParameter(),
    BLOCK_SIZE: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _contiguous_count_cute(x=x, size=size, kernel_backend=kernel_backend, BLOCK_SIZE=BLOCK_SIZE)
