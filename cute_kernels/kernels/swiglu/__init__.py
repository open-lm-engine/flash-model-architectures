import torch

from ...cutotune import CutoTuneParameter
from ...utils import ensure_contiguous
from .backward import _backward
from .forward import _forward
from .torch_implementation import swiglu_torch


class _Swiglu_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        kernel_backend_forward: str,
        kernel_backend_backward: str,
        BLOCK_SIZE_forward: int,
        BLOCK_SIZE_backward: int,
    ) -> torch.Tensor:
        assert gate.size() == up.size(), "tensors gate and up should have same shape"
        assert gate.dtype == up.dtype, "tensors gate and up should have same dtype"
        assert gate.dtype in [torch.float32, torch.float16, torch.bfloat16]

        ctx.save_for_backward(gate, up)
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.BLOCK_SIZE_backward = BLOCK_SIZE_backward

        return _forward(gate=gate, up=up, kernel_backend=kernel_backend_forward, BLOCK_SIZE=BLOCK_SIZE_forward)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors

        gate_grad, up_grad = _backward(
            gate=gate,
            up=up,
            output_grad=output_grad,
            kernel_backend=ctx.kernel_backend_backward,
            BLOCK_SIZE=ctx.BLOCK_SIZE_backward,
        )

        return gate_grad, up_grad, *[None] * 6


def swiglu_cute(
    gate: torch.Tensor,
    up: torch.Tensor,
    kernel_backend_forward: str = CutoTuneParameter(),
    kernel_backend_backward: str = CutoTuneParameter(),
    BLOCK_SIZE_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_backward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _Swiglu_Cute.apply(
        gate,
        up,
        kernel_backend_forward,
        kernel_backend_backward,
        BLOCK_SIZE_forward,
        BLOCK_SIZE_backward,
    )
