import torch
import torch.nn as nn
import torch.nn.functional as F

from cute_kernels import CuteInductor

from ..test_commons import TestCommons


class RMSNormTest(TestCommons):
    def test_rmsnorm(self) -> None:
        class Model(nn.Module):
            def __init__(self, shape: int):
                super().__init__()
                self.shape = shape

                self.l1 = nn.Linear(shape, shape)
                self.l2 = nn.Linear(shape, shape)
                self.norm1 = nn.RMSNorm(shape, eps=0.3, elementwise_affine=True)
                self.norm2 = nn.RMSNorm(shape, eps=0.1, elementwise_affine=False)

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                x = self.l1(x)
                y = self.l2(y)

                x = self.norm1(x)
                y = self.norm2(y)

                x1 = F.rms_norm(x + y, normalized_shape=(x.size(-1),), weight=self.norm1.weight, eps=0.5)
                y1 = torch.rms_norm(x * y, normalized_shape=(y.size(-1),))

                x = x1
                y = y1

                del x1, y1

                return self.norm1(x + y)

        model = Model(5).to(torch.cuda.current_device())

        x_expected, x_kernel = self.get_random_duplicated_tensors(
            (3, 4, 5), device=torch.cuda.current_device(), dtype=torch.float32
        )
        y_expected, y_kernel = self.get_random_duplicated_tensors(
            (3, 4, 5), device=torch.cuda.current_device(), dtype=torch.float32
        )

        cute_inductor = CuteInductor(use_torch_inductor_after_cute_inductor=True)
        compiled_model = torch.compile(model, backend=cute_inductor.compiler)

        output_expected = model(x_expected, y_expected)
        output_kernel = compiled_model(x_kernel, y_kernel)

        self.assert_equal_tensors(output_kernel, output_expected, False)
