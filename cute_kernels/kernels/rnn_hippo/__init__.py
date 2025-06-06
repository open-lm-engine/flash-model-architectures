# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...utils import ensure_contiguous
from .triton_implementation import rnn_hippo_backward_triton, rnn_hippo_forward_triton


class _RNN_HiPPO_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_hippo: torch.Tensor,
        weight_hippo_projection: torch.Tensor,
        hippo_A: torch.Tensor,
        hippo_B: torch.Tensor,
        starting_timestep: int,
        input_state: torch.Tensor | None,
        hippo_state: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
    ) -> torch.Tensor:
        assert input.dim() in [3, 4]
        assert weight.dim() == 3
        assert cu_seqlens is None
        assert max_seqlen is None
        assert hippo_A.dim() == 2
        assert hippo_B.dim() == 2

        N, H = input.size()[-2:]
        assert weight.size() == (N, H, H)

        if gradient_clipping is not None and gradient_clipping < 0:
            gradient_clipping = -gradient_clipping

        output = torch.empty_like(input)

        rnn_hippo_forward_triton(
            input=input,
            weight=weight,
            weight_hippo=weight_hippo,
            weight_hippo_projection=weight_hippo_projection,
            hippo_A=hippo_A,
            hippo_B=hippo_B,
            input_state=input_state,
            hippo_state=hippo_state,
            output=output,
            starting_timestep=starting_timestep + 1,  # HiPPO indexing starts at 1
        )

        ctx.save_for_backward(
            weight,
            weight_hippo,
            weight_hippo_projection,
            hippo_A,
            hippo_B,
            output,
            input_state,
            cu_seqlens,
            max_seqlen,
        )
        ctx.gradient_clipping = gradient_clipping
        ctx.starting_timestep = starting_timestep

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        (
            weight,
            weight_hippo,
            weight_hippo_projection,
            hippo_A,
            hippo_B,
            output,
            input_state,
            cu_seqlens,
            max_seqlen,
        ) = ctx.saved_tensors

        input_grad = torch.empty_like(output)
        weight_grad = torch.zeros_like(weight, dtype=torch.float32)

        kwargs = {
            "weight": weight,
            "output": output,
            "input_state": input_state,
            "output_grad": output_grad,
            "input_grad": input_grad,
            "weight_grad": weight_grad,
            "gradient_clipping": ctx.gradient_clipping,
        }

        rnn_hippo_backward_triton(**kwargs)

        return input_grad, weight_grad, *[None] * 10


def rnn_hippo_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_hippo: torch.Tensor,
    weight_hippo_projection: torch.Tensor,
    hippo_A: torch.Tensor,
    hippo_B: torch.Tensor,
    starting_timestep: int,
    input_state: torch.Tensor | None = None,
    hippo_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """computes multihead RNN recurrent update over the sequence length: tanh(`input_state` @ `weight` + `input`)

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
        hippo_A (torch.Tensor | None, optional): A matrix for HiPPO. Defaults to None.
        hippo_B (torch.Tensor | None, optional): B matrix for HiPPO. Defaults to None.

    Returns:
        torch.Tensor: output tensor of shape (B, S, N, H)
    """

    return _RNN_HiPPO_Cute.apply(
        input,
        weight,
        weight_hippo,
        weight_hippo_projection,
        hippo_A,
        hippo_B,
        input_state,
        hippo_state,
        gradient_clipping,
        cu_seqlens,
        max_seqlen,
        starting_timestep,
    )
