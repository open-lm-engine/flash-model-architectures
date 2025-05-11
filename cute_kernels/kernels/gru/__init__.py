import torch

from ...utils import ensure_contiguous
from .torch_implementation import gru_torch
from .triton_implementation import (
    gru_backward_triton,
    gru_forward_triton,
    gru_varlen_backward_triton,
    gru_varlen_forward_triton,
)


class _GRU_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        forget_input: torch.Tensor,
        forget_weight: torch.Tensor,
        reset_input: torch.Tensor,
        reset_weight: torch.Tensor,
        input_state: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_B_backward: int,
    ) -> torch.Tensor:
        if gradient_clipping is not None and gradient_clipping < 0:
            gradient_clipping = -gradient_clipping

        output = torch.empty_like(input)

        if cu_seqlens is None:
            assert max_seqlen is None

            gru_forward_triton(
                input=input,
                weight=weight,
                forget_input=forget_input,
                forget_weight=forget_weight,
                reset_input=reset_input,
                reset_weight=reset_weight,
                input_state=input_state,
                output=output,
                BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
            )
        else:
            assert max_seqlen is not None
            is_max_seqlen_tensor = isinstance(max_seqlen, torch.Tensor)

            gru_varlen_forward_triton(
                input=input,
                weight=weight,
                input_state=input_state,
                output=output,
                cu_seqlens=cu_seqlens,
                max_seqlen_tensor=max_seqlen if is_max_seqlen_tensor else None,
                max_seqlen=None if is_max_seqlen_tensor else max_seqlen,
                BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
            )

        ctx.save_for_backward(weight, output, input_state, cu_seqlens, max_seqlen)
        ctx.gradient_clipping = gradient_clipping
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        weight, output, input_state, cu_seqlens, max_seqlen = ctx.saved_tensors
        input_grad = torch.empty_like(output)
        weight_grad = torch.empty_like(weight)

        BLOCK_SIZE_B = ctx.BLOCK_SIZE_B_backward
        gradient_clipping = ctx.gradient_clipping

        if cu_seqlens is None:
            gru_backward_triton(
                output=output,
                weight=weight,
                forget_input=forget_input,
                forget_weight=forget_weight,
                reset_input=reset_input,
                reset_weight=reset_weight,
                input_state=input_state,
                output_grad=output_grad,
                input_grad=input_grad,
                weight_grad=weight_grad,
                gradient_clipping=gradient_clipping,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
            )
        else:
            is_max_seqlen_tensor = isinstance(max_seqlen, torch.Tensor)

            gru_varlen_backward_triton(
                weight=weight,
                output=output,
                input_state=input_state,
                output_grad=output_grad,
                input_grad=input_grad,
                weight_grad=weight_grad,
                cu_seqlens=cu_seqlens,
                max_seqlen_tensor=max_seqlen if is_max_seqlen_tensor else None,
                max_seqlen=None if is_max_seqlen_tensor else max_seqlen,
                gradient_clipping=gradient_clipping,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
            )

        return input_grad, weight_grad, *[None] * 8


def gru_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    forget_weight: torch.Tensor,
    reset_input: torch.Tensor,
    reset_weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
    *,
    BLOCK_SIZE_B_forward: int = 32,
    BLOCK_SIZE_B_backward: int = 32,
) -> torch.Tensor:
    """computes multihead RNN: tanh(`input_state` @ `weight` + `input`)

    Args:
        input (torch.Tensor): input tensor of shape (B, S, N, H) where N is the number of heads and H is the head
            dimension. Should have shape (T, N, H) and `cu_seqlens` should be passed.
        weight (torch.Tensor): weight tensor of shape (N, H, H)
        input_state (torch.Tensor | None, optional): starting state of shape (B, N, H), None means starting state
            is 0 tensor. Defaults to None.
        gradient_clipping (float | None, optional): gradient clipping for the state gradient in backward, None
            implies no clipping. Defaults to None.
        cu_seqlens (torch.Tensor | None, optional): cumulative sequence length (must contain 0 as first element). Defaults to None.
        max_seqlen (torch.Tensor | int | None, optional): max sequence length in the batch. Defaults to None.
        BLOCK_SIZE_B_forward (int, optional): block size for forward along batch dimension for forward. Defaults to
            32.
        BLOCK_SIZE_B_backward (int, optional): block size for backward along batch dimension for backward. Defaults to
            32.

    Returns:
        torch.Tensor: output tensor of shape (B, S, N, H)
    """

    return _GRU_Cute.apply(
        input,
        weight,
        forget_input,
        forget_weight,
        reset_input,
        reset_weight,
        input_state,
        gradient_clipping,
        cu_seqlens,
        max_seqlen,
        BLOCK_SIZE_B_forward,
        BLOCK_SIZE_B_backward,
    )
