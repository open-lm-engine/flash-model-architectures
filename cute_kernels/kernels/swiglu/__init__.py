import torch

from ...kernel_backend import is_cuda_kernel_backend_allowed, is_triton_kernel_backend_allowed
from ...math import ceil_divide
from ...utils import ensure_contiguous, is_nvidia_gpu
from .cuda_implementation import swiglu_backward_cuda, swiglu_forward_cuda
from .torch_implementation import swiglu_torch
from .triton_implementation import _swiglu_backward_triton_kernel, _swiglu_forward_triton_kernel


class _Swiglu_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        assert gate.size() == up.size(), "tensors gate and up should have same shape"
        assert gate.type() == up.type(), "tensors gate and up should have same dtype"

        ctx.save_for_backward(gate, up)
        ctx.use_cuda_backend = is_cuda_kernel_backend_allowed() and is_nvidia_gpu() and gate.is_cuda and up.is_cuda
        ctx.use_triton_backend = is_triton_kernel_backend_allowed()

        output = torch.empty_like(gate)

        if ctx.use_cuda_backend:
            BLOCK_SIZE = 1024
            swiglu_forward_cuda(gate=gate, up=up, output=output, BLOCK_SIZE=BLOCK_SIZE)
        elif ctx.use_triton_backend:
            num_elements = gate.numel()
            BLOCK_SIZE = 1024

            with torch.cuda.device(gate.device):
                _swiglu_forward_triton_kernel[ceil_divide(num_elements, BLOCK_SIZE),](
                    gate_ptr=gate,
                    up_ptr=up,
                    output_ptr=output,
                    num_elements=num_elements,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
        else:
            raise ValueError("unexpected kernel_backend")

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors
        gate_grad = torch.empty_like(gate)
        up_grad = torch.empty_like(up)

        if ctx.use_cuda_backend:
            BLOCK_SIZE = 1024
            swiglu_backward_cuda(
                gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad, BLOCK_SIZE=BLOCK_SIZE
            )
        elif ctx.use_triton_backend:
            num_elements = gate.numel()
            BLOCK_SIZE = 1024

            with torch.cuda.device(gate.device):
                _swiglu_backward_triton_kernel[ceil_divide(num_elements, BLOCK_SIZE),](
                    gate_ptr=gate,
                    up_ptr=up,
                    output_grad_ptr=output_grad,
                    gate_grad_ptr=gate_grad,
                    up_grad_ptr=up_grad,
                    num_elements=num_elements,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
        else:
            raise ValueError("unexpected kernel_backend")

        return gate_grad, up_grad


def swiglu_cute(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """computes swiglu activation as `up` * `gate` * sigmoid(`gate`)

    Args:
        gate (torch.Tensor): `gate` activation tensor
        up (torch.Tensor): `up` activation tensor

    Returns:
        torch.Tensor: output tensor
    """

    return _Swiglu_Cute.apply(gate, up)
