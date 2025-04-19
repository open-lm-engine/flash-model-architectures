from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import cross_entropy_cute, cross_entropy_torch, set_seed

from ..test_commons import TestCommons


_SEED = 42


class CrossEntropyTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.float32, torch.bfloat16],  # dtype
            [None, 0.7],  # logits_multiplier
            [cross_entropy_cute, torch.compile(cross_entropy_cute, fullgraph=True)],  # function
        )
    )
    def test_cross_entropy(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        logits_multiplier: float | None,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        if isinstance(size, int):
            size = (size,)

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype, std=0.02)
        labels = torch.randint(0, x_kernel.size(-1), (x_kernel.size(0),), device=x_kernel.device)

        loss_kernel = function(x=x_kernel, labels=labels, logits_multiplier=logits_multiplier)
        loss_expected = cross_entropy_torch(x=x_expected, labels=labels, logits_multiplier=logits_multiplier)

        loss_kernel.backward()
        loss_expected.backward()

        self.assert_equal_tensors(loss_kernel, loss_expected, False, atol_float32=6.3e-5, rtol_float32=0)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False)
