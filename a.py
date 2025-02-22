import torch
import torch._inductor.config as config
import torch.nn.functional as F
from torch._inductor.pattern_matcher import PatternMatcherPass, fwd_only, joint_fwd_bwd, register_replacement


def swiglu_unchunked_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.chunk(2, dim=-1)
    return x[0] * F.silu(x[1])


def f(x):
    x = x.chunk(2, dim=-1)
    return F.relu(x[0])


class _CustomPass(PatternMatcherPass):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, g: torch.fx.graph.Graph):
        print(g)
        self.apply(g)
        print(g)


with config.patch(
    pattern_matcher=False,
    # define pattern match as custom post grad opt pass
    post_grad_custom_pre_pass=None,
    post_grad_custom_post_pass=_CustomPass(),
):
    my_args = [
        torch.empty([10, 10], device=torch.cuda.current_device(), requires_grad=True),
    ]

    invoked = False

    def extra_check(match):
        global invoked
        invoked = True
        return True

    register_replacement(
        swiglu_unchunked_torch,
        f,
        my_args,
        fwd_only,
        [config.post_grad_custom_post_pass],
        extra_check=extra_check,
    )

    compiled = torch.compile(swiglu_unchunked_torch, dynamic=True)

    x = torch.randn([8, 8], device=torch.cuda.current_device())
    x = x.detach()  # .requires_grad_()

    x_clone = x.clone().detach().requires_grad_()

    z = compiled(x)
    z_clone = swiglu_unchunked_torch(x_clone)

    print(z - z_clone)

    # z.sum().backward()
    # z_clone.sum().backward()

    # print(x.grad - x_clone.grad)
