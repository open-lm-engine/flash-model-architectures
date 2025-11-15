# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...custom_op import CustomOp, ctx_needs_gradients, ctx_save_for_backward
from ...enums import KernelBackend
from ...torch_utils import clip_gradients, sigmoid, tanh
from ...utils import (
    empty_like_contiguous,
    get_max_seqlen_and_max_seqlen_tensor,
    is_triton_available,
    zeros_like_contiguous,
)


if is_triton_available():
    from .triton_implementation import gru_backward_triton, gru_forward_triton


class _GRU(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor,
        W: torch.Tensor,
        xf: torch.Tensor,
        Wf: torch.Tensor,
        xr: torch.Tensor,
        Wr: torch.Tensor,
        h0: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
    ) -> torch.Tensor:
        y = torch.empty_like(x)

        if cu_seqlens is None:
            assert max_seqlen is None
            B, S, N, H = x.size()

            h0 = torch.zeros(B, N, H, device=x.device, dtype=x.dtype) if h0 is None else h0

            # input -> (B, S, N, H)
            # weight -> (N, H, H)
            # input_state -> (B, N, H)

            for s in range(S):
                # (B, N, 1, H) = (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
                f = h0.unsqueeze(-2) @ Wf.unsqueeze(0) + xf[:, s].unsqueeze(-2)
                f = sigmoid(f)

                # (B, N, 1, H) = (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
                r = h0.unsqueeze(-2) @ Wr.unsqueeze(0) + xr[:, s].unsqueeze(-2)
                r = sigmoid(r)

                # (B, N, 1, H) = [(B, N, 1, H) * (B, N, 1, H)] @ (1, N, H, H) + (B, N, 1, H)
                possible_new_state = (h0.unsqueeze(-2) * r) @ W.unsqueeze(0) + x[:, s].unsqueeze(-2)
                possible_new_state = tanh(possible_new_state)

                h = f * h0.unsqueeze(-2) + (1 - f) * possible_new_state
                h = h.squeeze(-2)
                h = clip_gradients(h, gradient_clipping)

                y[:, s] = h
                h0 = h
        else:
            assert max_seqlen is not None
            B = cu_seqlens.numel() - 1
            _, N, H = input.size()

            h0 = torch.zeros(B, N, H, device=x.device, dtype=x.dtype) if h0 is None else h0.clone()

            # input -> (cu_seqlens[-1], N, H)
            # weight -> (N, H, H)
            # input_state -> (B, N, H)

            start = cu_seqlens[:-1]
            end = cu_seqlens[1:]

            for s in range(max_seqlen):
                offset = start + s
                unfinished = offset < end

                new_state = h0[unfinished].unsqueeze(-2)
                offset_unfinished = offset[unfinished]

                # don't update the finished sequences
                # (B, N, 1, H) = (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
                f = new_state @ Wf.unsqueeze(0) + xf[offset_unfinished].unsqueeze(-2)
                f = sigmoid(f)

                # (B, N, 1, H) = (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
                r = new_state @ Wr.unsqueeze(0) + xr[offset_unfinished].unsqueeze(-2)
                r = sigmoid(r)

                # (B, N, 1, H) = [(B, N, 1, H) * (B, N, 1, H)] @ (1, N, H, H) + (B, N, 1, H)
                possible_new_state = (new_state * r) @ W.unsqueeze(0) + x[offset_unfinished].unsqueeze(-2)
                possible_new_state = tanh(possible_new_state)

                new_state = f * new_state + (1 - f) * possible_new_state

                if gradient_clipping is not None:
                    new_state = clip_gradients(new_state, gradient_clipping)

                new_state = new_state.squeeze(-2)

                y[offset_unfinished] = new_state
                h0[unfinished] = new_state

        return y

    @staticmethod
    def forward_triton(
        ctx,
        x: torch.Tensor,
        W: torch.Tensor,
        xf: torch.Tensor,
        Wf: torch.Tensor,
        xr: torch.Tensor,
        Wr: torch.Tensor,
        h0: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
    ) -> torch.Tensor:
        needs_grad = ctx_needs_gradients(ctx)

        y = empty_like_contiguous(x)
        max_seqlen_tensor, max_seqlen = get_max_seqlen_and_max_seqlen_tensor(max_seqlen)
        f = empty_like_contiguous(x) if needs_grad else None
        r = empty_like_contiguous(x) if needs_grad else None
        z = empty_like_contiguous(x) if needs_grad else None

        gru_forward_triton(
            x=x,
            W=W,
            xf=xf,
            Wf=Wf,
            f=f,
            xr=xr,
            Wr=Wr,
            r=r,
            z=z,
            h0=h0,
            y=y,
            cu_seqlens=cu_seqlens,
            max_seqlen_tensor=max_seqlen_tensor,
            max_seqlen=max_seqlen,
        )

        ctx_save_for_backward(ctx, W, Wf, f, Wr, r, z, y, h0, cu_seqlens, max_seqlen_tensor)

        ctx.max_seqlen = max_seqlen
        ctx.gradient_clipping = gradient_clipping

        return y

    @staticmethod
    def backward_triton(ctx, dy: torch.Tensor) -> tuple[torch.Tensor | None]:
        W, Wf, f, Wr, r, z, y, h0, cu_seqlens, max_seqlen_tensor = ctx.saved_tensors

        dx = empty_like_contiguous(y)
        dxf = empty_like_contiguous(y)
        dxr = empty_like_contiguous(y)
        dW = zeros_like_contiguous(W, dtype=torch.float32)
        dWf = zeros_like_contiguous(W, dtype=torch.float32)
        dWr = zeros_like_contiguous(W, dtype=torch.float32)

        gru_backward_triton(
            W=W,
            y=y,
            Wf=Wf,
            f=f,
            dxf=dxf,
            dWf=dWf,
            Wr=Wr,
            r=r,
            dxr=dxr,
            dWr=dWr,
            z=z,
            h0=h0,
            dy=dy,
            dx=dx,
            dW=dW,
            cu_seqlens=cu_seqlens,
            max_seqlen_tensor=max_seqlen_tensor,
            max_seqlen=ctx.max_seqlen,
            gradient_clipping=ctx.gradient_clipping,
        )

        dW = dW.type_as(W)
        dWf = dWf.type_as(Wf)
        dWr = dWr.type_as(Wr)

        return dx, dW, dxf, dWf, dxr, dWr, *[None] * 5


def gru(
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
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    Returns:
        tuple[torch.Tensor, torch.Tensor]: output tensor of shape (B, S, N, H) and output state tensor of shape (B, N, H)
    """

    assert input.dim() in [3, 4]
    assert weight.dim() == 3

    N, H = input.size()[-2:]
    assert weight.size() == (N, H, H)

    if gradient_clipping is not None and gradient_clipping < 0:
        gradient_clipping = -gradient_clipping

    input = _GRU.run(
        x=input,
        W=weight,
        xf=forget_input,
        Wf=forget_weight,
        xr=reset_input,
        Wr=reset_weight,
        h0=input_state,
        gradient_clipping=gradient_clipping,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )

    output_state = input[:, -1] if cu_seqlens is None else input[cu_seqlens[1:] - 1]

    return input, output_state
