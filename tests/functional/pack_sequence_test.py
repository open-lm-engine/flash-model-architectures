# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from parameterized import parameterized

from xma import KernelBackend, pack_sequence, unpack_sequence

from ..test_commons import TestCommons


class PackSequenceTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [(7, 1000, 12, 14)],  # size
            [[0, 70, 170, 295, 393, 412, 515, 691]],  # cu_seqlens
            TestCommons.get_dtypes(),  # dtype
            ["left", "right"],  # padding_side
            [KernelBackend.cuda, KernelBackend.triton],  # kernel_backend
            [pack_sequence, torch.compile(pack_sequence, fullgraph=True)],  # function
        )
    )
    def test_pack_sequence(
        self,
        size: tuple[int],
        cu_seqlens: list[int],
        dtype: torch.dtype,
        padding_side: str,
        kernel_backend: KernelBackend,
        function: Callable,
    ) -> None:
        self.skip_if_incompatible_kernel_backend(kernel_backend)
        device = kernel_backend.get_accelerator().get_current_device()

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.uint32)

        z_kernel = function(
            [x_kernel],
            cu_seqlens=cu_seqlens,
            total_tokens=cu_seqlens[-1].item(),
            padding_side=padding_side,
            kernel_backend=kernel_backend,
        )[0]

        z_expected = pack_sequence(
            [x_expected],
            cu_seqlens=cu_seqlens.to(torch.int),
            total_tokens=cu_seqlens[-1].item(),
            padding_side=padding_side,
            kernel_backend=KernelBackend.torch,
        )[0]

        z_expected.sum().backward()
        z_kernel.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [(691, 12, 14)],  # size
            [[0, 70, 170, 295, 393, 412, 515, 691]],  # cu_seqlens
            [1000],  # sequence_length
            TestCommons.get_dtypes(),  # dtype
            ["left", "right"],  # padding_side
            [KernelBackend.cuda, KernelBackend.triton],  # kernel_backend
            [unpack_sequence, torch.compile(unpack_sequence, fullgraph=True)],  # function
        )
    )
    def test_unpack_sequence(
        self,
        size: tuple[int],
        cu_seqlens: list[int],
        sequence_length: tuple[int],
        dtype: torch.dtype,
        padding_side: str,
        kernel_backend: KernelBackend,
        function: Callable,
    ) -> None:
        self.skip_if_incompatible_kernel_backend(kernel_backend)
        device = kernel_backend.get_accelerator().get_current_device()

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.uint32)

        z_kernel = function(
            [x_kernel],
            cu_seqlens=cu_seqlens,
            batch_size=cu_seqlens.size(0) - 1,
            sequence_length=sequence_length,
            padding_side=padding_side,
            kernel_backend=kernel_backend,
        )[0]

        z_expected = unpack_sequence(
            [x_expected],
            cu_seqlens=cu_seqlens.to(torch.int),
            batch_size=cu_seqlens.size(0) - 1,
            sequence_length=sequence_length,
            padding_side=padding_side,
            kernel_backend=KernelBackend.torch,
        )[0]

        z_expected.sum().backward()
        z_kernel.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
