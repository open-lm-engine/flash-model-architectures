import torch

from ...kernel_backend import KernelBackend, is_cuda_kernel_backend_allowed, is_triton_kernel_backend_allowed
from ...math import ceil_divide
from ...utils import ensure_contiguous, is_nvidia_gpu
from .cuda_implementation import swiglu_backward_cuda, swiglu_forward_cuda
from .torch_implementation import swiglu_torch
from .triton_implementation import _swiglu_backward_triton_kernel, _swiglu_forward_triton_kernel


class _Swiglu_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        kernel_backend: KernelBackend,
        BLOCK_SIZE_CUDA: int,
        BLOCK_SIZE_TRITON: int,
        NUM_WARPS_TRITON: int,
    ) -> torch.Tensor:
        assert gate.size() == up.size(), "tensors gate and up should have same shape"
        assert gate.type() == up.type(), "tensors gate and up should have same dtype"

        ctx.save_for_backward(gate, up)
        output = torch.empty_like(gate)

        if is_cuda_kernel_backend_allowed(kernel_backend) and is_nvidia_gpu() and gate.is_cuda and up.is_cuda:
            swiglu_forward_cuda(gate=gate, up=up, output=output, BLOCK_SIZE=BLOCK_SIZE_CUDA)
        elif is_triton_kernel_backend_allowed(kernel_backend):
            B = gate.numel()

            with torch.cuda.device(gate.device):
                _swiglu_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_TRITON),](
                    gate_ptr=gate,
                    up_ptr=up,
                    output_ptr=output,
                    B=B,
                    BLOCK_SIZE=BLOCK_SIZE_TRITON,
                    num_warps=NUM_WARPS_TRITON,
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

        return gate_grad, up_grad, *[None] * 3


def swiglu_cute(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    kernel_backend: KernelBackend = KernelBackend.cuda,
    # cuda
    BLOCK_SIZE_CUDA: int = 1024,
    # triton
    BLOCK_SIZE_TRITON: int = 4096,
    NUM_WARPS_TRITON: int = 32,
) -> torch.Tensor:
    """computes swiglu activation as `up` * `gate` * sigmoid(`gate`)

    Args:
        gate (torch.Tensor): `gate` activation tensor
        up (torch.Tensor): `up` activation tensor
        kernel_backend (KernelBackend, optional): kernel backend to prioritize. Defaults to KernelBackend.cuda.
        BLOCK_SIZE_CUDA (int, optional): block size for CUDA backend. Defaults to 1024.
        BLOCK_SIZE_TRITON (int, optional): block size for triton backend. Defaults to 4096.
        NUM_WARPS_TRITON (int, optional): warps for triton backend. Defaults to 32.

    Returns:
        torch.Tensor: output tensor
    """

    return _Swiglu_Cute.apply(gate, up, BLOCK_SIZE_CUDA, BLOCK_SIZE_TRITON, NUM_WARPS_TRITON)
