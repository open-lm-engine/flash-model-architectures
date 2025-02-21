import torch
import torch._inductor.config as config
import torch.nn.functional as F
from torch._inductor.pattern_matcher import PatternMatcherPass, fwd_only, register_replacement
from torch.testing import FileCheck


def swiglu_unchunked_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.chunk(2, dim=-1)
    return x[0] * F.silu(x[1])


def swiglu_unchunked_cute(x):
    return x.chunk(2, dim=-1)[0] + 1


saved_graph = None


class _CustomPass(PatternMatcherPass):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, g: torch.fx.graph.Graph):
        self.apply(g)
        global saved_graph
        saved_graph = g


with config.patch(
    # leave custom pass only in post_grad_passes()
    pattern_matcher=False,
    # define pattern match as custom post grad opt pass
    post_grad_custom_pre_pass=None,
    post_grad_custom_post_pass=_CustomPass(),
):

    # def add(x, y):
    #     return x + y

    # # testing that
    # def sym_minus(x, y):
    #     return (x - (-y.size(0))) - (y * -1) - y.size(0)

    device = "cpu"
    # my_args = [
    #     torch.empty([8, 1], device=device),
    #     torch.empty([10], device=device),
    # ]
    my_args = [torch.empty([10, 10], device=device)]

    invoked = False

    def extra_check(match):
        global invoked
        invoked = True
        return True

    register_replacement(
        # add,
        # sym_minus,
        swiglu_unchunked_torch,
        swiglu_unchunked_cute,
        my_args,
        fwd_only,
        [config.post_grad_custom_post_pass],
        extra_check=extra_check,
    )

    # @torch.compile(dynamic=True)
    # def foo(x, y):
    #     return x + y

    compiled = torch.compile(swiglu_unchunked_torch, dynamic=True)

    x = torch.rand([8, 8])

    z = compiled(x)
    print(z - swiglu_unchunked_torch(x))

    print(saved_graph)

    # # we trace out the y.sym_size in replacement
    # FileCheck().check("sym_size_int").check_same("num_users=2").check_same("target=torch.ops.aten.sym_size").run(
    #     str(saved_graph)
    # )
