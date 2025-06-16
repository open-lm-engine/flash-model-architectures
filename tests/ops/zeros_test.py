# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import KernelBackend, zeros_cute

from ..test_commons import TestCommons


class ZerosTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [KernelBackend.cuda, KernelBackend.triton],  # kernel_backend
            [zeros_cute, torch.compile(zeros_cute, fullgraph=True)],  # function
        )
    )
    def test_zeros(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        kernel_backend: KernelBackend,
        function: Callable,
    ) -> None:
        x = function(shape=size, dtype=dtype, device=device, kernel_backend=kernel_backend)
        assert (x == 0).all()
