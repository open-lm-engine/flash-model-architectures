import torch

from cute_kernels import CuteInductor, add_tensor_replacement_config

from ...test_commons import TestCommons


class CuteInductorAddTensorReplacementTest(TestCommons):
    def test_cute_inductor_add_tensor_replacement(self) -> None:
        size = (4, 4)
        device = torch.cuda.current_device()
        dtype = torch.float32

        def _forward(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            x1 = x * 3
            x1 = x1 + y
            z = z + x1
            return z

        x = torch.randn(size, device=device, dtype=dtype, requires_grad=True)
        y = torch.randn(size, device=device, dtype=dtype, requires_grad=True)
        z = torch.randn(size, device=device, dtype=dtype, requires_grad=True)

        _compiled_forward = torch.compile(
            _forward,
            backend=CuteInductor(
                replacement_configs=[add_tensor_replacement_config],
                apply_torch_inductor_after_cute_inductor=True,
            ),
        )

        output_expected = _forward(x, y, z)
        output_compiled = _compiled_forward(x, y, z)

        self.assert_equal_tensors(output_compiled, output_expected, True)
        assert False
