# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from contextlib import nullcontext
from typing import Callable

import torch
from parameterized import parameterized

from xma import KernelBackend, cross_entropy, force_kernel_backend, set_seed

from ..test_commons import TestCommons


_SEED = 42


class CrossEntropyTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.float32, torch.bfloat16],  # dtype
            [None, 0.7],  # logits_multiplier
            [cross_entropy, torch.compile(cross_entropy, fullgraph=True)],  # function
            [False],  # no_grad
        )
        + TestCommons.make_args_matrix(
            [(4, 17)],  # size
            [torch.device("cuda")],  # device
            [torch.float32, torch.bfloat16],  # dtype
            [None, 0.7],  # logits_multiplier
            [cross_entropy, torch.compile(cross_entropy, fullgraph=True)],  # function
            [True],  # no_grad
        )
    )
    def test_cross_entropy(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        logits_multiplier: float | None,
        function: Callable,
        no_grad: bool,
    ) -> None:
        set_seed(_SEED)

        if isinstance(size, int):
            size = (size,)

        context = torch.no_grad if no_grad else nullcontext

        with context():
            x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype, std=0.02)
            labels = torch.randint(0, x_kernel.size(-1), (x_kernel.size(0),), device=x_kernel.device)

            loss_kernel = function(
                x=x_kernel, labels=labels, logits_multiplier=logits_multiplier, kernel_backend=KernelBackend.triton
            )

            loss_expected = cross_entropy(
                x=x_expected, labels=labels, logits_multiplier=logits_multiplier, kernel_backend=KernelBackend.torch
            )

            self.assert_equal_tensors(loss_kernel, loss_expected, False, atol_float32=6.3e-5, rtol_float32=0)

            if not no_grad:
                loss_kernel.backward()
                loss_expected.backward()

                self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False)
