# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ....math import ceil_divide
from .backward import _backward_core
from .forward import _forward_core
from .state_passing import _state_passing_core


def _depthwise_causal_convolution_forward_pallas(
    x: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None, h0: torch.Tensor | None, BLOCK_SIZE_S: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    W = W.transpose(1, 0)
    b = None if b is None else b[None, :]
    h0 = None if h0 is None else h0.transpose(1, 2).to(x.dtype)

    state_size = W.shape[0] - 1

    if h0 is None:
        ht = F.pad(x, (0, 0, state_size - x.shape[1], 0)) if x.shape[1] < state_size else x[:, -state_size:, :]
    else:
        ht = torch.cat((h0, x), dim=1)[:, -state_size:, :]

    if h0 is not None:
        pad = ceil_divide(state_size, 8) * 8
        h0 = F.pad(h0, (0, 0, pad - state_size, 0))

    y = _forward_core(x=x, W=W, b=b, h0=h0, BLOCK_SIZE_S=BLOCK_SIZE_S)

    return y, ht


def _depthwise_causal_convolution_backward_pallas(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None,
    h0: torch.Tensor | None,
    dy: torch.Tensor,
    dht: torch.Tensor | None,
    BLOCK_SIZE_S: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    K = W.shape[-1]

    W = W.transpose(1, 0)

    if h0 is not None:
        h0 = h0.transpose(1, 2).to(x.dtype)

        state_size = K - 1
        pad = ceil_divide(state_size, 8) * 8
        h0 = F.pad(h0, (0, 0, pad - state_size, 0))

    h = _state_passing_core(x=x, h0=h0, BLOCK_SIZE_S=BLOCK_SIZE_S, K=K)

    dx, dW, db, dh0 = _backward_core(x=x, W=W, h=h, dy=dy, dht=dht, BLOCK_SIZE_S=BLOCK_SIZE_S, K=K)

    dW = dW.transpose(1, 0)
    db = None if b is None else db[0]
    dh0 = None if h0 is None else dh0[:, 1 - K :, :].transpose(1, 2)

    return dx, dW, db, dh0
