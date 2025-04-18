from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import continuous_count_cute, continuous_count_torch, set_seed

from ..test_commons import TestCommons


_MAX_EXPERTS = 72
_SEED = 42


class ContiguousCountTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_1d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.long, torch.int],  # dtype
            [continuous_count_cute, torch.compile(continuous_count_cute, fullgraph=True)],  # function
        )
    )
    def test_continuous_count(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
        function: Callable,
    ) -> None:
        set_seed(_SEED)
        x = torch.randint(0, _MAX_EXPERTS, (size,), device=device, dtype=dtype)

        z_kernel = function(x=x, size=_MAX_EXPERTS)
        z_expected = continuous_count_torch(x.view(-1), size=_MAX_EXPERTS)

        self.assert_equal_tensors(z_kernel, z_expected, True)
