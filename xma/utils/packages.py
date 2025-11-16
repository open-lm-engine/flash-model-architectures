# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************


try:
    import cutlass.cute

    _IS_CUTE_DSL_AVAILABLE = True
except:
    _IS_CUTE_DSL_AVAILABLE = False


def is_cute_dsl_available() -> bool:
    return _IS_CUTE_DSL_AVAILABLE


try:
    import torch_xla

    _IS_TORCH_XLA_AVAILABLE = True
except ImportError:
    _IS_TORCH_XLA_AVAILABLE = False


def is_torch_xla_available() -> bool:
    return _IS_TORCH_XLA_AVAILABLE


try:
    import triton

    _IS_TRITON_AVAILABLE = True
except:
    _IS_TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _IS_TRITON_AVAILABLE
