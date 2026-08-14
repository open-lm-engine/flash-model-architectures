# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""Pallas forward kernel for linear attention.

Design (register-transpose layout):
  * tensors stay in the native host layout (B, S, N, K) end to end — the host never
    materializes transposed copies (a module-level `jnp.swapaxes` round-trip for q/k/v
    would cost ~0.74 ms per forward at B8/S4096/N16/K128/V128 on v6e-1).
  * each program owns one (batch, S-tile, V-tile) cell and loads a (BLOCK_SIZE_S, N, K)
    tile; the tile is transposed ONCE per cell in registers to (N, BLOCK_SIZE_S, K), so
    every per-head slice is a contiguous row extract (cheap on the vector unit) instead
    of a strided gather.
  * heads are looped inside the kernel, which also keeps BlockSpec legal when the
    last-two-dims rule forbids indexing the head axis.
  * kernels assume S % BLOCK_SIZE_S == 0 and K, V >= 128 lanes; ragged S and short
    feature dims are zero-padded at the host level in op.py (mathematically exact:
    padded rows are causally isolated, padded feature columns are column-isolated,
    both are sliced off on the way out). There is deliberately no in-kernel masking —
    Mosaic does not implement the required broadcasts for small shapes, and dead
    compile paths are worse than explicit host-side pads.
  * the running state `ht_ref` is carried across S-tiles by read-modify-write on the
    same output BlockSpec location (b, 0, 0, v_tile). The grid is ordered
    (batch, V-tile, S-tile) so that every revisit of a given `ht` block comes from
    consecutive grid cells (S axis innermost): Pallas only keeps an output buffer
    live across *consecutive* cells mapping to the same block, so the V-tile axis
    must NOT be the innermost one, and the "arbitrary" semantics on the S axis
    forces sequential execution along the chain. The backward / state-passing
    state-passing and backward kernels avoid revisit hazards by folding all
    V-tiles into a static unrolled loop inside one cell instead: state_passing
    does it so each cell fetches/stores all heads' state in one block (matching
    the backward's fetch layout), and the backward does it because dq and dk
    accumulate across V-tiles — parallel V-tile cells would race on those
    output accumulators.

Measured on v6e-1 (B8, S4096, Nq=Nk=Nv=16, K=V=128, bf16, BLOCK_SIZE_S=256, fixed
batch, no input state): 0.81 ms per call against an analytical memory-bound
roofline at 1.6 TB/s (670 MB of traffic) of 0.419 ms, i.e. 52% of the bound.
BLOCK_SIZE_S < 256 is rejected up front in op.py: the kernels are only validated
at 256+ (support envelope, not a numerical requirement).
"""

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _linear_attention_forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    h0_ref,
    y_ref,
    ht_ref,
    *,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    N: int,
    Gq: int,
    Gk: int,
    Gv: int,
) -> None:
    # grid is (batch, V-tile, S-tile); the S axis is innermost (see module docstring)
    @pl.when(pl.program_id(2) == 0)
    def _():
        if h0_ref is None:
            ht_ref[...] = jnp.zeros_like(ht_ref)
        else:
            ht_ref[...] = h0_ref[...].astype(jnp.float32)

    dtype = q_ref.dtype

    # one register-level transpose per cell: (S, N, K) -> (N, S, K), see the module docstring
    q = q_ref[...].transpose(1, 0, 2)
    k = k_ref[...].transpose(1, 0, 2)
    v = v_ref[...].transpose(1, 0, 2)

    row = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 1)
    causal_mask = row >= col

    for n in range(N):
        q_n = q[n // Gq].astype(dtype)  # contiguous (BLOCK_SIZE_S, K)
        k_n = k[n // Gk].astype(dtype)
        v_n = v[n // Gv].astype(dtype)
        h_n = ht_ref[n]

        qk = jax.lax.dot_general(q_n, k_n, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
        qk = jnp.where(causal_mask, qk, 0).astype(dtype)

        y_n = jnp.dot(qk, v_n, preferred_element_type=jnp.float32)
        y_n += jnp.dot(q_n, h_n.astype(dtype), preferred_element_type=jnp.float32)
        y_ref[:, n, :] = (y_n * attention_multiplier).astype(dtype)

        ht_ref[n] = h_n + jax.lax.dot_general(k_n, v_n, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_forward_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array]:
    B, S, Nq, K = q.shape
    Nk = k.shape[2]
    Nv = v.shape[2]
    V = v.shape[-1]
    N = max(Nq, Nk, Nv)
    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    assert S % BLOCK_SIZE_S == 0
    assert V % BLOCK_SIZE_V == 0

    # (b, 0, 0, v_tile): the running state block is independent of the S index, and the
    # (batch, V-tile, S-tile) grid order guarantees consecutive accesses per block
    h_spec = pl.BlockSpec(
        block_shape=(None, N, K, BLOCK_SIZE_V),
        index_map=lambda BLOCK_ID_B, BLOCK_ID_V, BLOCK_ID_S: (BLOCK_ID_B, 0, 0, BLOCK_ID_V),
    )

    kernel = pl.pallas_call(
        partial(
            _linear_attention_forward_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            N=N,
            Gq=Gq,
            Gk=Gk,
            Gv=Gv,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, S, N, V), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, ceil_divide(V, BLOCK_SIZE_V), ceil_divide(S, BLOCK_SIZE_S)),
        in_specs=(
            pl.BlockSpec(block_shape=(None, BLOCK_SIZE_S, Nq, K), index_map=lambda B, V, S: (B, S, 0, 0)),
            pl.BlockSpec(block_shape=(None, BLOCK_SIZE_S, Nk, K), index_map=lambda B, V, S: (B, S, 0, 0)),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nv, BLOCK_SIZE_V), index_map=lambda B, V, S: (B, S, 0, V)
            ),
            None if h0 is None else h_spec,
        ),
        out_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, BLOCK_SIZE_V), index_map=lambda B, V, S: (B, S, 0, V)
            ),
            h_spec,
        ),
        # all block shapes are built from host-validated dimensions (op.py guarantees
        # S % BLOCK_SIZE_S == 0, 128-lane-multiple K/V, V % BLOCK_SIZE_V == 0), so
        # bounds checks are provably redundant
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )

    y, ht = kernel(q, k, v, h0)
    return y, ht
