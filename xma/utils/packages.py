# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

try:
    import cutlass.cute

    _IS_CUTE_DSL_AVAILABLE = True
except:
    _IS_CUTE_DSL_AVAILABLE = False


def is_cute_dsl_available() -> bool:
    return _IS_CUTE_DSL_AVAILABLE


try:
    import torch

    _IS_TORCH_AVAILABLE = True
except ImportError:
    _IS_TORCH_AVAILABLE = False


def is_torch_available() -> bool:
    return _IS_TORCH_AVAILABLE


try:
    import torch_neuronx

    _IS_TORCH_NEURONX_AVAILABLE = True
except:
    _IS_TORCH_NEURONX_AVAILABLE = False


def is_torch_neuronx_available() -> bool:
    return _IS_TORCH_NEURONX_AVAILABLE


try:
    import torch_xla

    _IS_TORCH_XLA_AVAILABLE = True
except ImportError:
    _IS_TORCH_XLA_AVAILABLE = False


def is_torch_xla_available() -> bool:
    return _IS_TORCH_XLA_AVAILABLE


_IS_JAX_AVAILABLE = None


def is_jax_available() -> bool:
    # jax_import_guard() must run before JAX is imported anywhere in the process, so JAX attaches to the same
    # libtpu runtime torch_xla is using instead of racing it for the TPU chip. Both are deferred to the first
    # call (instead of running at `import xma` time) so that merely importing xma never touches the XLA/PJRT
    # runtime in a process that hasn't forked its per-device workers yet (e.g. under torch_xla's xmp.spawn).
    global _IS_JAX_AVAILABLE

    if _IS_JAX_AVAILABLE is None:
        if is_torch_xla_available():
            from torch_xla.experimental.custom_kernel import jax_import_guard

            jax_import_guard()

        try:
            import jax

            _IS_JAX_AVAILABLE = True
        except ImportError:
            _IS_JAX_AVAILABLE = False

    return _IS_JAX_AVAILABLE


try:
    import haliax

    _IS_HALIAX_AVAILABLE = True
except ImportError:
    _IS_HALIAX_AVAILABLE = False


def is_haliax_available() -> bool:
    return _IS_HALIAX_AVAILABLE


try:
    import triton

    _IS_TRITON_AVAILABLE = True
except:
    _IS_TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _IS_TRITON_AVAILABLE


try:
    import causal_conv1d

    _IS_CAUSAL_CONV1D_AVAILABLE = True
except ImportError:
    _IS_CAUSAL_CONV1D_AVAILABLE = False


def is_causal_conv1d_available() -> bool:
    return _IS_CAUSAL_CONV1D_AVAILABLE
