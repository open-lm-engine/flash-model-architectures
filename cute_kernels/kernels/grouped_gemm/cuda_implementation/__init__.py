# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit
from ....utils import cute_op


_KERNEL_NAME = "main"


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={})
@cpp_jit()
def main() -> None: ...
