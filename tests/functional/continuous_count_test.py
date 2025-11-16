# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from parameterized import parameterized

from xma import KernelBackend, continuous_count, set_seed

from ..test_commons import TestCommons


_MAX_EXPERTS = 72
_SEED = 42


class ContiguousCountTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_1d_tensor_sizes(),  # size
            [KernelBackend.cuda],  # device
            [torch.long, torch.int],  # dtype
            [continuous_count, torch.compile(continuous_count, fullgraph=True)],  # function
        )
    )
    def test_continuous_count(
        self,
        bins: int,
        kernel_backend: KernelBackend,
        dtype: torch.dtype,
        function: Callable,
    ) -> None:
        self.skip_if_incompatible_kernel_backend(kernel_backend)
        device = kernel_backend.get_current_device()

        set_seed(_SEED)
        x = torch.randint(0, _MAX_EXPERTS, (bins,), device=device, dtype=dtype)

        z_kernel = function(x=x, bins=_MAX_EXPERTS, kernel_backend=KernelBackend.cuda)
        z_expected = continuous_count(x.view(-1), bins=_MAX_EXPERTS, kernel_backend=KernelBackend.torch)

        self.assert_equal_tensors(z_kernel, z_expected, True)
