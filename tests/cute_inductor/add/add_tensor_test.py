from typing import Callable

import torch
import torch.nn as nn

from cute_kernels import add_scalar_torch, add_tensor_replacement_config

from ...test_commons import TestCommons


class CuteInductorAddTensorReplacementTest(TestCommons):
    def test_cute_inductor_add_tensor_replacement(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        kernel_backend: str,
        BLOCK_SIZE: int,
        function: Callable,
    ) -> None:
        def _forward(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            x1 = x * 3
            x1 = x1 + y
            x1 += x1
            z = z + x1
            return z

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        y = 0.42

        z_kernel = function(
            x_kernel,
            y,
            kernel_backend=kernel_backend,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        z_expected = add_scalar_torch(x_expected, y)

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
