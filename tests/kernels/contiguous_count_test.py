from typing import Callable

import torch
from parameterized import parameterized
from transformers import set_seed

from cute_kernels import KernelBackend, contiguous_count_cute

from ..test_commons import TestCommons


MAX_EXPERTS = 4


class ContiguousCountTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [8200],  # size
            [torch.device("cuda")],  # device
            [KernelBackend.cuda, KernelBackend.triton][:1],  # kernel_backend
            [32],  # BLOCK_SIZE_B
            [contiguous_count_cute],  # , torch.compile(contiguous_count_cute, fullgraph=True)],  # function
        )
    )
    def test_contiguous_count(
        self,
        size: int,
        device: torch.device,
        kernel_backend: KernelBackend,
        BLOCK_SIZE_B: int,
        function: Callable,
    ) -> None:
        x = torch.randint(0, MAX_EXPERTS, (size,), device=device, dtype=torch.long)

        z_kernel = function(x=x, size=MAX_EXPERTS, kernel_backend=kernel_backend, BLOCK_SIZE_B=BLOCK_SIZE_B)
        z_expected = x.view(-1).bincount(minlength=MAX_EXPERTS)

        self.assert_equal_tensors(z_kernel, z_expected, True)
