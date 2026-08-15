# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose


jax = pytest.importorskip("jax")

import jax.numpy as jnp

from xma import KernelBackend
from xma.layers_jax import LinearAttentionJAX, linear_attention_jax


_ATTENTION_MULTIPLIER = 0.3
_TOLERANCES = {jnp.float32: {"atol": 8e-4, "rtol": 0}, jnp.bfloat16: {"atol": 8e-4, "rtol": 0}}


def _get_problem_shapes() -> list[tuple[int, int, int, int, int]]:
    # (K, V, Nq, Nk, Nv)
    return [
        (16, 16, 1, 1, 1),
        (32, 24, 4, 4, 4),
        (16, 16, 4, 2, 1),
        (4, 16, 1, 1, 1),  # K smaller than the minimum Pallas tile size (8)
        (10, 24, 4, 2, 1),  # K not a power of 2
        (128, 128, 2, 2, 2),  # the unpadded / no-host-pad path at the production tile width
    ]


def _generate_args() -> list:
    # the pallas kernels require BLOCK_SIZE_S >= 256 (op.py raises below that); sequence lengths cover
    # shorter than, equal to, and not a multiple of BLOCK_SIZE_S (ragged host-side padding path), and
    # NUM_BLOCKS_S = 1 and 2
    args = list(
        product(
            [37, 130, 256, 512],  # sequence length
            [256],  # BLOCK_SIZE_S
            [128],  # BLOCK_SIZE_V
            _get_problem_shapes(),
            [jnp.float32, jnp.bfloat16],
            [False, True],  # has_input_state
        )
    )
    args += list(
        product(
            [300, 1024],
            [512],  # BLOCK_SIZE_S = 512 coverage
            [128],
            _get_problem_shapes(),
            [jnp.float32, jnp.bfloat16],
            [False, True],
        )
    )
    # NUM_BLOCKS_S = 3 (768 / 256): the reversed dh chain and state checkpoints have a genuinely
    # different shape at >= 3 cells (middle cells that neither seed nor publish)
    args += list(
        product(
            [768],
            [256],
            [128],
            [(16, 16, 4, 2, 1)],
            [jnp.float32, jnp.bfloat16],
            [False, True],
        )
    )
    # BLOCK_SIZE_V < V below: genuinely exercises multiple V-tiles (256 / 128 = 2, 384 / 128 = 3),
    # across sequence lengths that span both single-cell (S <= BLOCK_SIZE_S) and multi-cell
    # (S > BLOCK_SIZE_S) cases — the latter pins the state-checkpoint chain against the former
    args += list(
        product(
            [37, 130, 256, 512],
            [256],
            [128],
            [(16, 256, 2, 2, 2), (16, 384, 2, 2, 1)],  # (K, V, Nq, Nk, Nv)
            [jnp.float32, jnp.bfloat16],
            [False, True],
        )
    )
    # N = 32 > _MAX_HEADS_PER_PALLAS_CELL (16): host-level head-chunked dispatch in op.py, in both a
    # plain and a grouped-qk layout (group sizes still divide the chunk size)
    args += list(
        product(
            [128, 300],
            [256],
            [128],
            [(16, 16, 32, 32, 32), (16, 16, 32, 16, 16)],
            [jnp.float32, jnp.bfloat16],
            [False, True],
        )
    )
    return args


@pytest.mark.parametrize("S,BLOCK_SIZE_S,BLOCK_SIZE_V,problem_shape,dtype,has_input_state", _generate_args())
def test_linear_attention_pallas(
    S: int,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    problem_shape: tuple[int, int, int, int, int],
    dtype: str,
    has_input_state: bool,
) -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("KernelBackend.pallas is only supported on TPU")

    K, V, Nq, Nk, Nv = problem_shape
    N = max(Nq, Nk, Nv)
    B = 2

    tolerance = _TOLERANCES[dtype]

    key_q, key_k, key_v, key_h0, key_dy, key_dht = jax.random.split(jax.random.PRNGKey(0), 6)
    std = 0.01

    q = jax.random.normal(key_q, (B, S, Nq, K), dtype=jnp.float32).astype(dtype) * std
    k = jax.random.normal(key_k, (B, S, Nk, K), dtype=jnp.float32).astype(dtype) * std
    v = jax.random.normal(key_v, (B, S, Nv, V), dtype=jnp.float32).astype(dtype) * std
    h0 = jax.random.normal(key_h0, (B, N, K, V), dtype=jnp.float32) * std if has_input_state else None

    def _run(kernel_backend: KernelBackend, q: jax.Array, k: jax.Array, v: jax.Array, h0: jax.Array | None):
        return linear_attention_jax(
            q,
            k,
            v,
            h0,
            attention_multiplier=_ATTENTION_MULTIPLIER,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
            kernel_backend=kernel_backend,
        )

    (y_kernel, ht_kernel), vjp_kernel = jax.vjp(
        lambda q, k, v, h0: _run(KernelBackend.pallas, q, k, v, h0), q, k, v, h0
    )
    (y_expected, ht_expected), vjp_expected = jax.vjp(
        lambda q, k, v, h0: _run(KernelBackend.jax, q, k, v, h0), q, k, v, h0
    )

    assert_allclose(np.asarray(y_kernel, dtype=np.float32), np.asarray(y_expected, dtype=np.float32), **tolerance)
    assert_allclose(np.asarray(ht_kernel, dtype=np.float32), np.asarray(ht_expected, dtype=np.float32), **tolerance)

    dy = jax.random.normal(key_dy, y_kernel.shape, dtype=jnp.float32).astype(dtype) * std
    dht = jax.random.normal(key_dht, ht_kernel.shape, dtype=jnp.float32) * std

    dq_kernel, dk_kernel, dv_kernel, dh0_kernel = vjp_kernel((dy, dht))
    dq_expected, dk_expected, dv_expected, dh0_expected = vjp_expected((dy, dht))

    assert_allclose(np.asarray(dq_kernel, dtype=np.float32), np.asarray(dq_expected, dtype=np.float32), **tolerance)
    assert_allclose(np.asarray(dk_kernel, dtype=np.float32), np.asarray(dk_expected, dtype=np.float32), **tolerance)
    assert_allclose(np.asarray(dv_kernel, dtype=np.float32), np.asarray(dv_expected, dtype=np.float32), **tolerance)

    if has_input_state:
        assert_allclose(
            np.asarray(dh0_kernel, dtype=np.float32), np.asarray(dh0_expected, dtype=np.float32), **tolerance
        )
    else:
        assert dh0_kernel is None
        assert dh0_expected is None


def test_linear_attention_pallas_block_size_v_above_pad_floor() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("KernelBackend.pallas is only supported on TPU")

    # V = 100 is below BLOCK_SIZE_V = 256: op.py must round V up to BLOCK_SIZE_V (not just to
    # the 128-lane floor) so the core's "V % BLOCK_SIZE_V == 0" invariant holds.
    B, S, K, V = 2, 130, 128, 100
    N = 2
    key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(0), 3)
    std = 0.01
    q = jax.random.normal(key_q, (B, S, N, K)) * std
    k = jax.random.normal(key_k, (B, S, N, K)) * std
    v = jax.random.normal(key_v, (B, S, N, V)) * std

    def _run(kernel_backend: KernelBackend):
        return linear_attention_jax(
            q,
            k,
            v,
            attention_multiplier=_ATTENTION_MULTIPLIER,
            BLOCK_SIZE_S=256,
            BLOCK_SIZE_V=256,
            kernel_backend=kernel_backend,
        )

    y_kernel, ht_kernel = _run(KernelBackend.pallas)
    y_expected, ht_expected = _run(KernelBackend.jax)

    assert y_kernel.shape == (B, S, N, V)
    assert_allclose(np.asarray(y_kernel), np.asarray(y_expected), **_TOLERANCES[jnp.float32])
    assert_allclose(np.asarray(ht_kernel), np.asarray(ht_expected), **_TOLERANCES[jnp.float32])


def test_linear_attention_pallas_block_size_v_guard() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("KernelBackend.pallas is only supported on TPU")

    q = jnp.zeros((1, 256, 1, 16))
    for BLOCK_SIZE_V in (0, 24):
        with pytest.raises(ValueError, match="positive multiple of 128"):
            linear_attention_jax(q, q, q, BLOCK_SIZE_V=BLOCK_SIZE_V, kernel_backend=KernelBackend.pallas)


@pytest.mark.parametrize("kernel_backend", [KernelBackend.jax, KernelBackend.pallas])
def test_linear_attention_inclusive_recurrence_semantics(kernel_backend: KernelBackend) -> None:
    """Pin the operator semantics: y[s] = q[s] @ h[s] with the diagonal INCLUDED.

    Every kernel in this repo (triton, pallas) implements the inclusive recurrence; an
    earlier revision of the eager references implemented the exclusive one (update h after
    reading y), which differed per element by attention_multiplier * (q_s . k_s) * v_s --
    far below the historical test scale, so the tolerance gate could not see it. Unit
    inputs make the convention discriminating at machine precision.
    """
    if kernel_backend == KernelBackend.pallas and jax.default_backend() != "tpu":
        pytest.skip("KernelBackend.pallas is only supported on TPU")

    B, S, K, V, N = 2, 130, 16, 16, 1
    q = jnp.ones((B, S, N, K))
    k = jnp.ones((B, S, N, K))
    v = jnp.ones((B, S, N, V))

    y, ht = linear_attention_jax(
        q,
        k,
        v,
        attention_multiplier=_ATTENTION_MULTIPLIER,
        BLOCK_SIZE_S=256,
        BLOCK_SIZE_V=128,
        kernel_backend=kernel_backend,
    )

    # h[s] = (s + 1) * ones(K, V); y[s] = attention_multiplier * (s + 1) * K * ones(V)
    s = jnp.arange(S, dtype=jnp.float32)
    y_expected = _ATTENTION_MULTIPLIER * (s[:, None, None] + 1) * K * jnp.ones((S, N, V))
    ht_expected = S * jnp.ones((B, N, K, V))

    assert_allclose(np.asarray(y), np.asarray(y_expected[None, ...].repeat(B, axis=0)), atol=1e-4, rtol=0)
    assert_allclose(np.asarray(ht), np.asarray(ht_expected), atol=1e-4, rtol=0)


def test_linear_attention_pallas_chunked_grouped_head_guard() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("KernelBackend.pallas is only supported on TPU")

    # N = 32 heads > _MAX_HEADS_PER_PALLAS_CELL (16) forces the host-level head-chunked dispatch,
    # but the single query/key/value head (group size 32) does not divide the 16-head chunks, so
    # the chunk-local group mapping cannot reproduce the global one and op.py must refuse
    B, S, K, V = 1, 256, 16, 16
    q = jnp.zeros((B, S, 32, K), dtype=jnp.float32)
    k = jnp.zeros((B, S, 1, K), dtype=jnp.float32)
    v = jnp.zeros((B, S, 1, V), dtype=jnp.float32)

    with pytest.raises(ValueError, match="group size"):
        linear_attention_jax(q, k, v, kernel_backend=KernelBackend.pallas)


@pytest.mark.parametrize("has_input_state", [False, True])
def test_linear_attention_module_works(has_input_state: bool) -> None:
    embed_size = 32
    output_size = 24
    B = 2
    S = 16

    key_init, key_input, key_state = jax.random.split(jax.random.PRNGKey(0), 3)

    module = LinearAttentionJAX.init(
        embed_size,
        output_size,
        key_head_dim=8,
        value_head_dim=8,
        num_query_heads=4,
        num_key_heads=2,
        num_value_heads=1,
        add_bias=True,
        key=key_init,
    )

    input = jax.random.normal(key_input, (B, S, embed_size))
    input_state = (
        jax.random.normal(key_state, (B, module.num_heads, module.key_head_dim, module.value_head_dim))
        if has_input_state
        else None
    )

    output, output_state, conv_state = module(input, input_state, kernel_backend=KernelBackend.jax)

    assert output.shape == (B, S, output_size)
    assert output_state.shape == (B, module.num_heads, module.key_head_dim, module.value_head_dim)
    assert conv_state is None
