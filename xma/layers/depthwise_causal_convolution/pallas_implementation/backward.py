# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************


import torch

from ....layers_jax.depthwise_causal_convolution.pallas_implementation.backward import (
    _backward_core as _backward_core_jax,
)
from ....math import ceil_divide


def _metadata(
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


_BACKWARD_CACHE = {}


def _backward_core(
    x: torch.Tensor,
    W: torch.Tensor,
    h: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor | None,
    BLOCK_SIZE_S: int,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_key = dht is None

    if cache_key not in _BACKWARD_CACHE:
        from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

        _BACKWARD_CACHE[cache_key] = make_kernel_from_pallas(_backward_core_jax, _metadata)

    return _BACKWARD_CACHE[cache_key](
        x, W, h, dy, dht, static_argnames=("BLOCK_SIZE_S", "K"), BLOCK_SIZE_S=BLOCK_SIZE_S, K=K
    )
