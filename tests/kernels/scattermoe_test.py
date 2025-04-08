import torch
from parameterized import parameterized

from cute_kernels import MoE_Torch, MoE_Triton, set_seed

from ..test_commons import TestCommons


_SEED = 42


class ScatterMoETest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [2, 4, 6, 8],  # num_experts
            [2, 4],  # num_experts_per_tok
            [2048],  # hidden_size
            [8192],  # intermediate_size
            [True, False],  # is_glu
            [True, False],  # is_compiling
        )
    )
    def test_scattermoe_triton(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        is_glu: bool,
        is_compiling: bool,
    ) -> None:
        set_seed(_SEED)

        if num_experts_per_tok > num_experts:
            self.skipTest(
                f"skipping test since number of experts per token ({num_experts_per_tok}) is more than number of experts ({num_experts})"
            )

        with torch.device(device):
            moe_custom = MoE_Triton(
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation_function=self.get_activation_function(is_glu=is_glu),
                is_glu=is_glu,
                add_bias=False,
                std=0.02,
            ).to(dtype=dtype)

            moe_torch = MoE_Torch(
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation_function=self.get_activation_function(is_glu=is_glu),
                is_glu=is_glu,
                add_bias=False,
                std=0.02,
            ).to(dtype=dtype)

        if is_compiling:
            moe_custom = torch.compile(moe_custom, fullgraph=True)

        state_dict = moe_custom.state_dict()

        if is_compiling:
            new_state_dict = {}
            for key in state_dict:
                new_key = key.split("_orig_mod.")[1]
                new_state_dict[new_key] = state_dict[key]

            state_dict = new_state_dict
            del new_state_dict

        moe_torch.load_state_dict(state_dict)

        x_torch = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)
        x_custom = x_torch.clone().detach().requires_grad_()

        y_torch = moe_torch(x_torch)[0]
        y_custom = moe_custom(x_custom)[0]

        self.assert_equal_tensors(
            y_custom,
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
        y_custom.sum().backward()

        self.assert_equal_tensors(
            x_custom.grad,
            x_torch.grad,
            False,
            atol_float16=4e-3,
            rtol_float16=0,
            atol_bfloat16=4e-2,
            rtol_bfloat16=0,
            atol_float32=6e-3,
            rtol_float32=0,
        )
