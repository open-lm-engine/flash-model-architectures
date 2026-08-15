# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ....layers_jax.depthwise_causal_convolution.pallas_implementation import _backward_core as _backward_core_jax
from ....math import ceil_divide


def _backward_output_shape_dtype_fn(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None,
    h: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor | None,
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


def _backward_core(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None,
    h: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor | None,
    BLOCK_SIZE_S: int,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(_backward_core, "cache"):
        _backward_core.cache = {}

    cache_key = (b is None, dht is None)
    kernel = _backward_core.cache.get(cache_key)

    if kernel is None:
        from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

        kernel = make_kernel_from_pallas(_backward_core_jax, _backward_output_shape_dtype_fn)
        _backward_core.cache[cache_key] = kernel

    return kernel(x, W, b, h, dy, dht, static_argnames=("BLOCK_SIZE_S", "K"), BLOCK_SIZE_S=BLOCK_SIZE_S, K=K)


def _depthwise_causal_convolution_backward_pallas(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None,
    h0: torch.Tensor | None,
    dy: torch.Tensor,
    dht: torch.Tensor | None,
    BLOCK_SIZE_S: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    K = W.shape[-1]

    W = W.transpose(1, 0)

    if h0 is not None:
        h0 = h0.transpose(1, 2).to(x.dtype)

        state_size = K - 1
        pad = ceil_divide(state_size, 8) * 8
        h0 = F.pad(h0, (0, 0, pad - state_size, 0))

    h = _state_passing_core(x=x, h0=h0, BLOCK_SIZE_S=BLOCK_SIZE_S, K=K)

    dx, dW, db, dh0 = _backward_core(
        x=x,
        W=W,
        b=None if b is None else b[None, :],
        h=h,
        dy=dy,
        dht=dht,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        K=K,
    )

    dW = dW.transpose(1, 0)
    db = None if b is None else db[0]
    dh0 = None if h0 is None else dh0[:, 1 - K :, :].transpose(1, 2)

    return dx, dW, db, dh0
