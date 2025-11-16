# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.library import custom_op
from torch_xla.experimental.custom_kernel import jax_import_guard, make_kernel_from_pallas

from ....constants import LIBRARY_NAME


jax_import_guard()

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp


def swiglu_forward_pallas_kernel(g_ref, u_ref, y_ref):
    g = g_ref[...]
    u = u_ref[...]

    y = u * g * sigmoid(g)

    y_ref[...] = y


@jax.jit
def swiglu_forward_pallas_jit(g: jax.Array, u: jax.Array) -> jax.Array:
    return pl.pallas_call(swiglu_forward_pallas_kernel, out_shape=jax.ShapeDtypeStruct(shape=g.shape, dtype=g.dtype))(
        g, u
    )


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
    return torch.empty_like(g)


swiglu_forward_pallas.cache = None
