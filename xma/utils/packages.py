# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

try:
    import triton

    _IS_TRITON_AVAILABLE = True
except:
    _IS_TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _IS_TRITON_AVAILABLE
