import torch
import torch.nn.functional as F


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype

    x = F.sigmoid(x)
    x = x.to(dtype)

    return x


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
    activation_function: str = "tanh",
    relu_negative_slope: float | None = None,
) -> torch.Tensor:
    if activation_function == "leaky_relu":
        assert relu_negative_slope is not None
    else:
        assert relu_negative_slope is None

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

        # input -> (B, S, N, 1, H)
        # weight -> (1, N, H, H)
        # input_state -> (B, N, H)

        for s in range(S):
            input_state = input_state.unsqueeze(-2)

            forget_gate = _sigmoid(forget_input[:, s] + input_state @ forget_weight)
            reset_gate = _sigmoid(reset_input[:, s] + input_state @ reset_weight)

            # (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
            input_state = input_state @ weight + input[:, s, ...]

            input_state = _activation_with_clipped_gradients(
                x=input_state,
                activation_function=activation_function,
                relu_negative_slope=relu_negative_slope,
                gradient_clipping=gradient_clipping,
            )

            input_state = input_state.squeeze(-2)

            output[:, s, ...] = input_state
    else:
        assert max_seqlen is not None
        B = cu_seqlens.numel() - 1
        _, N, H = input.size()

        if input_state is None:
            input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)
        else:
            input_state = input_state.clone()

        weight = weight.unsqueeze(0)
        input = input.unsqueeze(-2)

        # input -> (cu_seqlens[-1], N, 1, H)
        # weight -> (1, N, H, H)
        # input_state -> (B, N, H)

        start = cu_seqlens[:-1]
        end = cu_seqlens[1:]

        for s in range(max_seqlen):
            offset = start + s
            unfinished = offset < end
            new_state = input_state.unsqueeze(-2)

            # don't update the finished sequences
            # (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
            new_state = new_state[unfinished] @ weight + input[offset[unfinished], ...]

            new_state = _activation_with_clipped_gradients(
                x=new_state,
                activation_function=activation_function,
                relu_negative_slope=relu_negative_slope,
                gradient_clipping=gradient_clipping,
            )

            new_state = new_state.squeeze(-2)

            output[offset[unfinished], ...] = new_state
            input_state[unfinished] = new_state

    return output
