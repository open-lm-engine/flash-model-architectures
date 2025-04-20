import torch
import torch.nn.functional as F


def rnn_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
) -> torch.Tensor:
    if gradient_clipping is not None:
        raise NotImplementedError("rnn_torch doesn't support gradient_clipping")

    output = torch.empty_like(input)

    if cu_seqlens is None:
        assert max_seqlen is None
        B, S, N, H = input.size()

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
    else:
        assert max_seqlen is not None
        B = cu_seqlens.numel() - 1
        _, N, H = input.size()

        if input_state is None:
            input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)

        weight = weight.unsqueeze(0)
        input = input.unsqueeze(-2)

        # input -> (cu_seqlens[-1], N, 1, H)
        # weight -> (1, N, H, H)
        # input_state -> (B, N, H)

        offset = cu_seqlens[:-1]

        for s in range(max_seqlen):
            mask = offset < cu_seqlens[1:]
            new_state = input_state.unsqueeze(-2)

            # don't update the finished sequences
            # (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
            new_state = new_state[mask] @ weight + input[offset[mask], ...]

            new_state = new_state.float()
            new_state = F.tanh(new_state)
            new_state = new_state.type_as(input)

            new_state = new_state.squeeze(-2)

            output[offset[mask], ...] = new_state
            input_state[mask] = new_state

            offset = offset + 1

    return output
