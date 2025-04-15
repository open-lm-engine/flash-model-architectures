import torch
import torch.nn.functional as F


def rnn_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
) -> torch.Tensor:
    if gradient_clipping is not None:
        raise NotImplementedError("rnn_torch doesn't support gradient_clipping")

    B, S, N, H = input.size()
    output = torch.empty_like(input)

    if input_state is None:
        input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)

    weight = weight.unsqueeze(0)
    input = input.unsqueeze(-2)

    # input -> (B, S, N, 1, H)
    # weight -> (1, N, H, H)
    # input_state -> (B, N, H)

    for s in range(S):
        input_state = input_state.unsqueeze(-2)

        # (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
        input_state = input_state @ weight + input[:, s, ...]

        input_state = input_state.float()
        input_state = F.tanh(input_state)
        input_state = input_state.type_as(input)

        input_state = input_state.squeeze(-2)

        output[:, s, ...] = input_state

    return output
