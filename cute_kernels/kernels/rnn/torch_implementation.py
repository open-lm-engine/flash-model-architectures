import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.compile
def rnn_torch(input: torch.Tensor, weight: torch.Tensor, input_state: torch.Tensor | None = None) -> torch.Tensor:
    # input -> (B, S, N, H)
    B, S, N, H = input.size()
    output = torch.empty_like(input)

    if input_state is None:
        input_state = torch.zeros(B, N, H, device=input.device, dtype=torch.float32)
    else:
        input_state = input_state.float()

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
        input_state = input_state.type_as(weight)
        input_state = input_state @ weight + input[:, s, ...]
        input_state = input_state.float()
        input_state = F.tanh(input_state)

        output[:, s, ...] = input_state.squeeze(-2)

    return output


class RNNTorch(nn.Module):
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

    def forward(self, input: torch.Tensor, input_state: torch.Tensor | None = None) -> torch.Tensor:
        input = self.input_projection(input)
        input = rnn_torch(input=input, weight=self.state_weight, input_state=input_state)
        input = self.output_projection(input)
        return input

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight)
