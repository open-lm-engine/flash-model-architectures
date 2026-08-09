# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Callable

import torch
import torch.nn.functional as F

from ....layers_jax.depthwise_causal_convolution.pallas_implementation import _backward_core as _backward_core_jax
from ....layers_jax.depthwise_causal_convolution.pallas_implementation import (
    _state_passing_core as _depthwise_causal_convolution_state_passing_core_jax,
)
from ....math import ceil_divide


def _make_state_passing_output_shape_dtype_fn(BLOCK_SIZE_S: int, K: int) -> Callable:
    # make_kernel_from_pallas calls this with only the non-static tensor args (x, h0), so BLOCK_SIZE_S/K
    # (needed for the output shape) must be captured via closure instead of taken as parameters.
    def _output_shape_dtype_fn(x: torch.Tensor, h0: torch.Tensor | None) -> list[tuple[tuple[int, ...], torch.dtype]]:
        B, S, H = x.shape
        NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
        PAD = ceil_divide(K - 1, 8) * 8

        return [((B, NUM_BLOCKS_S, PAD, H), torch.float32)]

    return _output_shape_dtype_fn


def _backward_output_shape_dtype_fn(
    x: torch.Tensor, W: torch.Tensor, h: torch.Tensor, dy: torch.Tensor, dht: torch.Tensor | None
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    B, S, H = x.shape
    K = W.shape[0]
    pad = ceil_divide(K - 1, 8) * 8

    return [
        ((B, S, H), x.dtype),
        ((K, H), torch.float32),
        ((1, H), torch.float32),
        ((B, pad, H), torch.float32),
    ]


_STATE_PASSING_CACHE = {}
_BACKWARD_CACHE = {}


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

    cache_key = (h0 is None, BLOCK_SIZE_S, K)

    if cache_key not in _STATE_PASSING_CACHE:
        from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

        _STATE_PASSING_CACHE[cache_key] = make_kernel_from_pallas(
            _depthwise_causal_convolution_state_passing_core_jax,
            _make_state_passing_output_shape_dtype_fn(BLOCK_SIZE_S, K),
        )

    h = _STATE_PASSING_CACHE[cache_key](x, h0, static_argnames=("BLOCK_SIZE_S", "K"), BLOCK_SIZE_S=BLOCK_SIZE_S, K=K)

    cache_key = dht is None

    if cache_key not in _BACKWARD_CACHE:
        from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

        _BACKWARD_CACHE[cache_key] = make_kernel_from_pallas(_backward_core_jax, _backward_output_shape_dtype_fn)

    dx, dW, db, dh0 = _BACKWARD_CACHE[cache_key](
        x, W, h, dy, dht, static_argnames=("BLOCK_SIZE_S", "K"), BLOCK_SIZE_S=BLOCK_SIZE_S, K=K
    )

    dW = dW.transpose(1, 0)
    db = None if b is None else db[0]
    dh0 = None if h0 is None else dh0[:, 1 - K :, :].transpose(1, 2)

    return dx, dW, db, dh0
