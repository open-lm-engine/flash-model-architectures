import torch
from parameterized import parameterized

from cute_kernels import RNNCute, RNNTorch, set_seed

from ..test_commons import TestCommons


_SEED = 42


class RNNTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [torch.float32, torch.bfloat16],
            [4],  # batch_size
            [1024],  # sequence_length
            [2048],  # input_size
            [64],  # state_size
            [2560],  # output_size
            [4],  # num_heads
            [False],  # is_compiling
        )
    )
    def test_rnn(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        sequence_length: int,
        input_size: int,
        state_size: int,
        output_size: int,
        num_heads: int,
        is_compiling: bool,
    ) -> None:
        set_seed(_SEED)

        with torch.device(device):
            rnn_cute = RNNCute(
                input_size=input_size,
                state_size=state_size,
                output_size=output_size,
                num_heads=num_heads,
                add_bias=False,
            ).to(dtype=dtype)

            with torch.no_grad():
                rnn_cute.state_weight.normal_(mean=0, std=0.01)

            rnn_torch = RNNTorch(
                input_size=input_size,
                state_size=state_size,
                output_size=output_size,
                num_heads=num_heads,
                add_bias=False,
            ).to(dtype=dtype)

        if is_compiling:
            rnn_cute = torch.compile(rnn_cute, fullgraph=True)

        state_dict = rnn_cute.state_dict()

        if is_compiling:
            new_state_dict = {}
            for key in state_dict:
                new_key = key.split("_orig_mod.")[1]
                new_state_dict[new_key] = state_dict[key]

            state_dict = new_state_dict
            del new_state_dict

        rnn_torch.load_state_dict(state_dict)

        x_torch = torch.randn(batch_size, sequence_length, input_size, device=device, dtype=dtype, requires_grad=True)
        x_cute = x_torch.clone().detach().requires_grad_()

        y_torch = rnn_torch(x_torch)
        y_cute = rnn_cute(x_cute)

        self.assert_equal_tensors(
            y_cute,
            y_torch,
            False,
            atol_float16=4e-3,
            rtol_float16=0,
            atol_bfloat16=2e-2,
            rtol_bfloat16=0,
            atol_float32=6e-3,
            rtol_float32=0,
        )

        y_torch.sum().backward()
        y_cute.sum().backward()

        self.assert_equal_tensors(
            x_cute.grad,
            x_torch.grad,
            False,
            atol_float16=4e-3,
            rtol_float16=0,
            atol_bfloat16=4e-2,
            rtol_bfloat16=0,
            atol_float32=6e-3,
            rtol_float32=0,
        )

        # print(x_cute.grad - x_torch.grad)
        # print((x_cute.grad - x_torch.grad).abs().max())
        # print(x_cute.grad)
        # print(x_torch.grad)

        print(rnn_cute.state_weight.grad - rnn_torch.state_weight.grad)
        print((rnn_cute.state_weight.grad - rnn_torch.state_weight.grad).abs().max())
        print(rnn_cute.state_weight.grad)
        print(rnn_torch.state_weight.grad)
        assert False
