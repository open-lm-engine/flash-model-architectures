# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ....layers_jax.depthwise_causal_convolution.pallas_implementation import _forward_core as _forward_core_jax
from ....math import ceil_divide


def _metadata(
    x: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None, h0: torch.Tensor | None
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    return [(x.shape, x.dtype)]


def _forward_core(
    x: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None, h0: torch.Tensor | None, BLOCK_SIZE_S: int
) -> torch.Tensor:
    if not hasattr(_forward_core, "cache"):
        _forward_core.cache = {}

    cache_key = (b is None, h0 is None)
    kernel = _forward_core.cache.get(cache_key)

    if kernel is None:
        from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

        kernel = make_kernel_from_pallas(_forward_core_jax, _metadata)
        _forward_core.cache[cache_key] = kernel

    return kernel(x, W, b, h0, static_argnames=("BLOCK_SIZE_S",), BLOCK_SIZE_S=BLOCK_SIZE_S)


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
