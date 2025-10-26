import torch


class _F(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, x):
        print(cls, x)
        print(cls.backward)
        return x

    @classmethod
    def backward(cls, x):
        print(cls, x)
        return x


x = torch.randn(4)
_F.apply(x)
