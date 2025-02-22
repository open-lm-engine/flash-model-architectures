import torch
import torch.nn as nn

from cute_kernels import CuteInductor
from cute_kernels.cute_inductor.rmsnorm import replace_rmsnorm
from cute_kernels.cute_inductor.swiglu_unchunked import replace_swiglu_unchunked


# NOTE swiglu unchunked computes:
# ------------------------------------------------------------------------------
# def swiglu_unchunked_torch(x: torch.Tensor) -> torch.Tensor:
#     x = x.chunk(2, dim=-1)
#     return x[0] * F.silu(x[1])
# ------------------------------------------------------------------------------


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.norm1 = nn.RMSNorm(4)
        self.linear = nn.Linear(4, 4)
        self.norm2 = nn.RMSNorm(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = self.linear(x)
        x = self.norm2(x)
        return x


model = Model().to(torch.cuda.current_device())

use_torch_inductor_after_cute_inductor = True  # to use torch's compiler optimizations as well
replace_functions = [replace_rmsnorm]  # add other replacing functions

cute_inductor = CuteInductor(
    use_torch_inductor_after_cute_inductor=use_torch_inductor_after_cute_inductor, replace_functions=replace_functions
)

compiled_model = torch.compile(model, backend=cute_inductor.compiler, fullgraph=True)

# trigger JIT compilation
x = torch.randn(4, 4, device=torch.cuda.current_device(), requires_grad=True)
x_clone = x.clone().detach().requires_grad_()

y = compiled_model(x)
y_clone = model(x_clone)

y_clone.sum().backward()
y.sum().backward()

print(y_clone - y)
print(x.grad - x_clone.grad)
