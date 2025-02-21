import torch

from cute_kernels import CuteInductor, swiglu_unchunked_cute, swiglu_unchunked_torch
from cute_kernels.cute_inductor.compiler import GraphCapture

from ..test_commons import TestCommons


class SwigluUnchunkedTest(TestCommons):
    def test_swiglu_unchunked(self) -> None:
        def forward(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor]:
            x = swiglu_unchunked_torch(x)
            y = swiglu_unchunked_torch(y)

            x1 = swiglu_unchunked_torch(x + y)
            y1 = swiglu_unchunked_torch(x * y)

            x = x1
            y = y1

            del x1, y1

            return swiglu_unchunked_torch(x + y)

        x_expected, x_kernel = self.get_random_duplicated_tensors(
            (3, 4, 12), device=torch.cuda.current_device(), dtype=torch.float32
        )
        y_expected, y_kernel = self.get_random_duplicated_tensors(
            (3, 4, 12), device=torch.cuda.current_device(), dtype=torch.float32
        )

        cute_inductor = CuteInductor(use_torch_inductor_after_cute_inductor=True)
        compiled_forward = torch.compile(forward, backend=cute_inductor.compiler)

        output_expected = forward(x_expected, y_expected)
        output_kernel = compiled_forward(x_kernel, y_kernel)

        self.assert_equal_tensors(output_kernel, output_expected, False)
