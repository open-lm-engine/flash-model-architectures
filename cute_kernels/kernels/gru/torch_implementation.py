import torch

from ...torch_math import sigmoid, tanh


def gru_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    forget_weight: torch.Tensor,
    reset_input: torch.Tensor,
    reset_weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
) -> torch.Tensor:
    if gradient_clipping is not None and gradient_clipping < 0:
        gradient_clipping = -gradient_clipping

    output = torch.empty_like(input)

    if cu_seqlens is None:
        assert max_seqlen is None
        B, S, N, H = input.size()

        if input_state is None:
            input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)

        weight = weight.unsqueeze(0)
        input = input.unsqueeze(-2)
        input_state = input_state.unsqueeze(-2)

        # input -> (B, S, N, 1, H)
        # weight -> (1, N, H, H)
        # input_state -> (B, N, 1, H)

        for s in range(S):
            # (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
            forget_gate = sigmoid(input_state @ forget_weight + forget_input[:, s])
            reset_gate = sigmoid(input_state @ reset_weight + reset_input[:, s])

            possible_new_state = tanh((input_state * reset_gate) @ weight + input[:, s])
            input_state = forget_gate * input_state + (1 - forget_gate) * possible_new_state

            output[:, s] = input_state.squeeze(-2)
    else:
        assert max_seqlen is not None
        B = cu_seqlens.numel() - 1
        _, N, H = input.size()

        if input_state is None:
            input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)
        else:
            input_state = input_state.clone()

        weight = weight.unsqueeze(0)
        forget_weight = forget_weight.unsqueeze(0)
        reset_weight = reset_weight.unsqueeze(0)

        input = input.unsqueeze(-2)
        forget_input = forget_input.unsqueeze(-2)
        reset_input = reset_input.unsqueeze(-2)

        # input -> (cu_seqlens[-1], N, 1, H)
        # weight -> (1, N, H, H)
        # input_state -> (B, N, H)

        start = cu_seqlens[:-1]
        end = cu_seqlens[1:]

        for s in range(max_seqlen):
            offset = start + s
            unfinished = offset < end
            new_state = input_state.unsqueeze(-2)

            new_state = new_state[unfinished]
            offset_unfinished = offset[unfinished]

            # don't update the finished sequences
            # (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
            forget_gate = sigmoid(new_state @ forget_weight + forget_input[offset_unfinished])
            reset_gate = sigmoid(new_state @ reset_weight + reset_input[offset_unfinished])

            possible_new_state = tanh((new_state * reset_gate) @ weight + input[offset_unfinished])
            new_state = forget_gate * new_state + (1 - forget_gate) * possible_new_state

            new_state = new_state.squeeze(-2)

            output[offset[unfinished]] = new_state
            input_state[unfinished] = new_state

    return output
