# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Callable

import torch

from ....layers_jax.depthwise_causal_convolution.pallas_implementation.backward import (
    _backward_core as _backward_core_jax,
)
from ....layers_jax.depthwise_causal_convolution.pallas_implementation.backward import (
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


_STATE_PASSING_CACHE = {}


def _depthwise_causal_convolution_state_passing_core(
    x: torch.Tensor, h0: torch.Tensor | None, BLOCK_SIZE_S: int, K: int
) -> torch.Tensor:
    cache_key = (h0 is None, BLOCK_SIZE_S, K)

    if cache_key not in _STATE_PASSING_CACHE:
        from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

        _STATE_PASSING_CACHE[cache_key] = make_kernel_from_pallas(
            _depthwise_causal_convolution_state_passing_core_jax,
            _make_state_passing_output_shape_dtype_fn(BLOCK_SIZE_S, K),
        )

    return _STATE_PASSING_CACHE[cache_key](
        x, h0, static_argnames=("BLOCK_SIZE_S", "K"), BLOCK_SIZE_S=BLOCK_SIZE_S, K=K
    )


def _backward_output_shape_dtype_fn(
    x: torch.Tensor, W: torch.Tensor, h: torch.Tensor, dy: torch.Tensor, dht: torch.Tensor
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    B, S, H = x.shape
    K = W.shape[0]

    return [
        ((B, S, H), x.dtype),
        ((K, H), torch.float32),
        ((1, H), torch.float32),
        ((B, K - 1, H), torch.float32),
    ]


_BACKWARD_CACHE = None


def _depthwise_causal_convolution_backward_core(
    x: torch.Tensor, W: torch.Tensor, h: torch.Tensor, dy: torch.Tensor, dht: torch.Tensor, BLOCK_SIZE_S: int, K: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global _BACKWARD_CACHE

    if _BACKWARD_CACHE is None:
        from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

        _BACKWARD_CACHE = make_kernel_from_pallas(_backward_core_jax, _backward_output_shape_dtype_fn)

    return _BACKWARD_CACHE(x, W, h, dy, dht, static_argnames=("BLOCK_SIZE_S", "K"), BLOCK_SIZE_S=BLOCK_SIZE_S, K=K)


def _depthwise_causal_convolution_backward_pallas(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None,
    h0: torch.Tensor | None,
    dy: torch.Tensor,
    dht: torch.Tensor | None,
    BLOCK_SIZE_S: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    # x, W, b: the forward's original (un-transposed) inputs, saved as residuals - mirrors the jax-side outer
    # wrapper (pallas_implementation/__init__.py::_depthwise_causal_convolution_backward). h0: (B, H, K - 1),
    # matching the (B, H, K - 1) input_state/final_state contract - just transpose, no slicing needed.
    B, _, H = x.shape
    K = W.shape[-1]

    W = W.transpose(1, 0)

    h0_in = None if h0 is None else h0.transpose(1, 2).to(x.dtype)

    h = _depthwise_causal_convolution_state_passing_core(x, h0_in, BLOCK_SIZE_S=BLOCK_SIZE_S, K=K)

    dht = torch.zeros(B, K - 1, H, dtype=torch.float32, device=x.device) if dht is None else dht.float()

    dx, dW, db, dh0 = _depthwise_causal_convolution_backward_core(x, W, h, dy, dht, BLOCK_SIZE_S=BLOCK_SIZE_S, K=K)

    dW = dW.transpose(1, 0)
    db = None if b is None else db[0]
    dh0 = None if h0 is None else dh0.transpose(1, 2)

    return dx, dW, db, dh0
