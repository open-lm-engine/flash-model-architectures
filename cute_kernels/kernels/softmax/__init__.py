import torch

from ...cutotune import CutoTuneParameter
from ...enums import KernelBackend
from .enums import TritonKernelAlgorithm
from .forward import _forward
from .torch_implementation import softmax_torch


class _Softmax_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        kernel_backend_forward: KernelBackend,
        triton_kernel_algorithm_forward: TritonKernelAlgorithm,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_H_forward: int,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_backward: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(x)

        is_x_1d = x.dim() == 1
        if is_x_1d:
            x = x.unsqueeze(0)

        output = _forward(
            x=x,
            kernel_backend=kernel_backend_forward,
            triton_kernel_algorithm=triton_kernel_algorithm_forward,
            BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
            BLOCK_SIZE_H=BLOCK_SIZE_H_forward,
        )

        if is_x_1d:
            output = output.squeeze(0)

        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        output = ctx.saved_tensors[0]
        return output * (1 - output), *[None] * 6


def softmax_cute(
    x: torch.Tensor,
    kernel_backend_forward: KernelBackend = CutoTuneParameter(),
    triton_kernel_algorithm_forward: TritonKernelAlgorithm = CutoTuneParameter(),
    BLOCK_SIZE_B_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_B_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_backward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _Softmax_Cute.apply(
        x,
        kernel_backend_forward,
        triton_kernel_algorithm_forward,
        BLOCK_SIZE_B_forward,
        BLOCK_SIZE_H_forward,
        BLOCK_SIZE_B_backward,
        BLOCK_SIZE_H_backward,
    )
