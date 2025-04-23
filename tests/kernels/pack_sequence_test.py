from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import KernelBackend, pack_sequence_cute, pack_sequence_torch

from ..test_commons import TestCommons


class PackSequenceTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [(7, 1000, 12, 14)],  # size
            [[0, 70, 170, 295, 393, 412, 515, 691]],  # cu_seqlens
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [pack_sequence_cute],  # , torch.compile(pack_sequence_cute, fullgraph=True)],  # function
        )
    )
    def test_pack_sequence(
        self,
        size: tuple[int],
        cu_seqlens: list[int],
        device: torch.device,
        dtype: torch.dtype,
        function: Callable,
    ) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.uint32)
        max_seqlen = (cu_seqlens.to(torch.int)[1:] - cu_seqlens.to(torch.int)[:-1]).max().to(torch.uint32)

        z_kernel = function(x_kernel, cu_seqlens, max_seqlen)
        assert False
        # z_expected = pack_sequence_torch(x_expected, cu_seqlens, max_seqlen)

        # self.assert_equal_tensors(z_kernel, z_expected, True)
