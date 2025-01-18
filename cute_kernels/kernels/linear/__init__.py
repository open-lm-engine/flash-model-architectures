import torch

from ...cutotune import CutoTuneParameter
from ...enums import KernelBackend
from ...utils import ensure_contiguous
from ..gemm import gemm_cute
from .torch_implementation import linear_torch
from .triton_implementation import linear_backward_triton, linear_forward_triton


class _Linear_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        use_tf32: bool,
        kernel_backend_forward: KernelBackend,
        BLOCK_SIZE_M_forward: int,
        BLOCK_SIZE_K_forward: int,
        BLOCK_SIZE_N_forward: int,
        kernel_backend_backward: KernelBackend,
        BLOCK_SIZE_M_backward: int,
        BLOCK_SIZE_K_backward: int,
        BLOCK_SIZE_N_backward: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        ctx.use_tf32 = use_tf32
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.BLOCK_SIZE_M_backward = BLOCK_SIZE_M_backward
        ctx.BLOCK_SIZE_K_backward = BLOCK_SIZE_K_backward
        ctx.BLOCK_SIZE_N_backward = BLOCK_SIZE_N_backward

        output = torch.empty(*input.size()[:-1], weight.size(0), dtype=input.dtype, device=input.device)

        if kernel_backend_forward == KernelBackend.triton:
            linear_forward_triton(
                input=input,
                weight=weight,
                bias=bias,
                output=output,
                use_tf32=use_tf32,
                BLOCK_SIZE_M=BLOCK_SIZE_M_forward,
                BLOCK_SIZE_K=BLOCK_SIZE_K_forward,
                BLOCK_SIZE_N=BLOCK_SIZE_N_forward,
                num_warps=CutoTuneParameter(),
                num_stages=CutoTuneParameter(),
            )
        else:
            raise ValueError(f"unexpected kernel_backend_forward ({kernel_backend_forward})")

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input, weight = ctx.saved_tensors

        use_tf32 = ctx.use_tf32
        kernel_backend_backward = ctx.kernel_backend_backward
        BLOCK_SIZE_M_backward = ctx.BLOCK_SIZE_M_backward
        BLOCK_SIZE_K_backward = ctx.BLOCK_SIZE_K_backward
        BLOCK_SIZE_N_backward = ctx.BLOCK_SIZE_N_backward

        if kernel_backend_backward == KernelBackend.triton:
            input_grad = gemm_cute(
                a=output_grad,
                b=weight,
                is_a_transposed=False,
                is_b_transposed=False,
                use_tf32=use_tf32,
                BLOCK_SIZE_M=BLOCK_SIZE_M_backward,
                BLOCK_SIZE_K=BLOCK_SIZE_K_backward,
                BLOCK_SIZE_N=BLOCK_SIZE_N_backward,
            )

            weight_grad = torch.empty_like(weight)
            bias_grad = None
            if ctx.has_bias:
                bias_grad = torch.empty(weight.size(0), device=weight.device, dtype=weight.dtype)

            # input_grad = output_grad @ weight
            # weight_grad = output_grad.T @ input
            # if bias_grad is not None:
            #     bias_grad = output_grad.sum(dim=0)

            linear_backward_triton(
                input=input,
                output_grad=output_grad,
                weight_grad=weight_grad,
                bias_grad=bias_grad,
                use_tf32=ctx.use_tf32,
                BLOCK_SIZE_M=ctx.BLOCK_SIZE_M_backward,
                BLOCK_SIZE_K=ctx.BLOCK_SIZE_K_backward,
                BLOCK_SIZE_N=ctx.BLOCK_SIZE_N_backward,
                num_warps=CutoTuneParameter(),
                num_stages=CutoTuneParameter(),
            )
        else:
            raise ValueError(f"unexpected kernel_backend_backward ({kernel_backend_backward})")

        return input_grad, weight_grad, bias_grad, *[None] * 7


def linear_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    use_tf32: bool = True,
    kernel_backend_forward: KernelBackend = KernelBackend.triton,
    BLOCK_SIZE_M_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_K_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_N_forward: int = CutoTuneParameter(),
    kernel_backend_backward: KernelBackend = KernelBackend.triton,
    BLOCK_SIZE_M_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_K_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_N_backward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _Linear_Cute.apply(
        input,
        weight,
        bias,
        use_tf32,
        kernel_backend_forward,
        BLOCK_SIZE_M_forward,
        BLOCK_SIZE_K_forward,
        BLOCK_SIZE_N_forward,
        kernel_backend_backward,
        BLOCK_SIZE_M_backward,
        BLOCK_SIZE_K_backward,
        BLOCK_SIZE_N_backward,
    )
