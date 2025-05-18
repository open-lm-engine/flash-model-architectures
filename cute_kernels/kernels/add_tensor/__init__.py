import torch

from ...cutotune import CutoTuneConfig, CutoTuneParameter, cutotune
from ...kernel_backend import KernelBackend, is_cuda_kernel_backend_allowed, is_triton_kernel_backend_allowed
from ...utils import ensure_same_strides, is_nvidia_gpu
from .cuda_implementation import add_tensor_cuda
from .torch_implementation import add_tensor_torch
from .triton_implementation import add_tensor_triton


@cutotune(
    configs=[
        CutoTuneConfig(
            {"kernel_backend": KernelBackend.cuda},
            condition=lambda **kwargs: is_cuda_kernel_backend_allowed(kwargs["kernel_backend"])
            and is_nvidia_gpu()
            and kwargs["x"].is_cuda
            and kwargs["y"].is_cuda,
        ),
        CutoTuneConfig(
            {"kernel_backend": KernelBackend.triton},
            condition=lambda **kwargs: is_triton_kernel_backend_allowed(kwargs["kernel_backend"]),
        ),
    ],
    reset_to_zero={"output": None},
)
def _forward(
    x: torch.Tensor, y: float, output: torch.Tensor, kernel_backend: KernelBackend | CutoTuneParameter
) -> None:
    if kernel_backend == KernelBackend.cuda:
        add_tensor_cuda(x=x, y=y, output=output, BLOCK_SIZE=1024)
    elif kernel_backend == KernelBackend.triton:
        add_tensor_triton(x=x, y=y, output=output)
    else:
        raise ValueError("unexpected kernel_backend")


class _AddTensor_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, y: torch.Tensor, kernel_backend: KernelBackend | CutoTuneParameter
    ) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert isinstance(
            kernel_backend, (KernelBackend, CutoTuneParameter)
        ), f"unexpected kernel_backend ({kernel_backend})"

        assert x.size() == y.size(), "tensors x and y should have same shape"
        assert x.type() == y.type(), "tensors x and y should have same dtype"

        x, y = ensure_same_strides(x, y)
        output = torch.empty_like(x)

        _forward(x=x, y=y, output=output, kernel_backend=kernel_backend)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad, None


def add_tensor_cute(
    x: torch.Tensor, y: torch.Tensor, *, kernel_backend: KernelBackend | CutoTuneParameter = CutoTuneParameter()
) -> torch.Tensor:
    """add 2 tensors

    Args:
        x (torch.Tensor): first tensor
        y (torch.Tensor): second tensor
        kernel_backend (KernelBackend | CutoTuneParameter, optional): kernel backend to prioritize.
            Defaults to CutoTuneParameter().

    Returns:
        torch.Tensor: output tensor
    """

    return _AddTensor_Cute.apply(x, y, kernel_backend)
