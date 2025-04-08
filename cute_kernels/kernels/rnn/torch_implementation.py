import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.compile
def rnn_torch(input: torch.Tensor, weight: torch.Tensor, input_state: torch.Tensor | None = None) -> torch.Tensor:
    # input -> (B, S, N, H)
    B, S, N, H = input.size()
    output = torch.empty_like(input)

    if input_state is None:
        input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)

    # input_state -> (B, N, H)
    input_state = input_state.unsqueeze(-2)
    # input_state -> (B, N, 1, H)

    # weight -> (N, H, H)
    weight = weight.unsqueeze(0)
    # weight -> (1, N, H, H)
    input = input.unsqueeze(-2)
    # input -> (B, S, N, 1, H)

    for s in range(S):
        # (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
        input_state = input_state @ weight + input[:, s, ...]
        input_state = F.tanh(input_state)

        output[:, s, ...] = input_state.squeeze(-2)

    return output


class RNNCute(nn.Module):
    def __init__(
        self, input_size: int, state_size: int, output_size: int, num_heads: int, add_bias: bool = True
    ) -> None:
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.num_heads = num_heads

        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_size, self.state_size))
        self.input_projection = nn.Linear(self.input_size, self.num_heads * self.state_size, bias=add_bias)
        self.output_projection = nn.Linear(self.num_heads * self.state_size, self.output_size, bias=False)

        self.reset_parameters()

    def forward(self, x: torch.Tensor, input_state: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_projection(x)
        x = rnn_torch(input=x, weight=self.state_weight, state_size=self.state_size, input_state=input_state)
        x = self.output_projection(x)
        return x
