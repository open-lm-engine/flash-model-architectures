# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.library import custom_op
from torch_xla.experimental.custom_kernel import jax_import_guard, make_kernel_from_pallas

from ....constants import LIBRARY_NAME
from ....math import ceil_divide


jax_import_guard()

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.nn import sigmoid


def swiglu_forward_pallas_kernel(g_ref, u_ref, y_ref):
    g = g_ref[...]
    u = u_ref[...]

    y = u * g * sigmoid(g)

    y_ref[...] = y


@jax.jit
def swiglu_forward_pallas_jit(g: jax.Array, u: jax.Array) -> jax.Array:
    B, H = g.shape

    kernel = pl.pallas_call(
        swiglu_forward_pallas_kernel,
        out_shape=jax.ShapeDtypeStruct(shape=g.shape, dtype=g.dtype),
        grid=(ceil_divide(B, 8), ceil_divide(H, 1024)),
        in_specs=[
            pl.BlockSpec(block_shape=(8, 1024), index_map=lambda x, y: (x, y)),
            pl.BlockSpec(block_shape=(8, 1024), index_map=lambda x, y: (x, y)),
        ],
        out_specs=pl.BlockSpec(block_shape=(8, 1024), index_map=lambda x, y: (x, y)),
    )

    return kernel(g, u)


@custom_op(f"{LIBRARY_NAME}::swiglu_forward_pallas", mutates_args={})
def swiglu_forward_pallas(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    assert g.is_contiguous()
    assert u.is_contiguous()

    if swiglu_forward_pallas.cache is None:
        swiglu_forward_pallas.cache = make_kernel_from_pallas(
            swiglu_forward_pallas_jit, lambda g, u: [(g.shape, g.dtype)]
        )

    return swiglu_forward_pallas.cache(g, u)


@swiglu_forward_pallas.register_fake
def _(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    assert g.is_contiguous()
    assert u.is_contiguous()

    return torch.empty_like(g)


swiglu_forward_pallas.cache = None
