import torch
import torch._inductor.config as config
import torch.nn.functional as F
from torch._inductor.pattern_matcher import PatternMatcherPass, fwd_only, joint_fwd_bwd, register_replacement


def search(x: torch.Tensor) -> torch.Tensor:
    x = x.chunk(2, dim=-1)
    return x[0] * F.silu(x[1])


def replace(x: torch.Tensor) -> torch.Tensor:
    x = x.chunk(2, dim=-1)
    return F.relu(x[0]) + F.relu(-x[1])


def f(x):
    x = x * 4
    x = x + 3
    x = search(x)
    return F.sigmoid(x)


class _CustomPass(PatternMatcherPass):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, g: torch.fx.graph.Graph):
        print(g)
        self.apply(g)
        print(g)


device = "cpu"


with config.patch(
    pattern_matcher=False,
    # define pattern match as custom post grad opt pass
    post_grad_custom_pre_pass=None,
    post_grad_custom_post_pass=_CustomPass(),
):
    my_args = [torch.empty([10, 10], device=device, requires_grad=True)]

    invoked = False

    def extra_check(match):
        global invoked
        invoked = True
        return True

    register_replacement(
        search,
        replace,
        my_args,
        fwd_only,
        [config.post_grad_custom_post_pass],
        extra_check=extra_check,
    )

    compiled_f = torch.compile(f, dynamic=True)

    x = torch.randn([8, 8], device=device)
    x = x.detach()  # .requires_grad_()

    x_clone = x.clone().detach().requires_grad_()

    z = compiled_f(x)
    z_clone = f(x_clone)

    print(z - z_clone)
