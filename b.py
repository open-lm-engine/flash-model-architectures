import torch
import torch._inductor.fx_passes.fuse_attention
from torch._inductor.pattern_matcher import gen_register_replacement


def _sfdp_pattern_1(query, key, value, inv_scale):
    return torch.matmul(query, key.transpose(-2, -1)).div(inv_scale).softmax(dim=-1).matmul(value)


_sfdp_pattern_1_compiled = torch.compile(_sfdp_pattern_1)

device = "cpu"
q = torch.randn((2, 4, 8, 16), device=device, requires_grad=True)
k = torch.randn((2, 4, 8, 16), device=device, requires_grad=True)
v = torch.randn((2, 4, 8, 16), device=device, requires_grad=True)

o = _sfdp_pattern_1_compiled(q, k, v, 1)

print(o)
print(torch._dynamo.utils.counters)
