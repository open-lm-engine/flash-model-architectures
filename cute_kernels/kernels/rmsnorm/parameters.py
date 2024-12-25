from ...cutotune import CutoTuneConfig
from ...enums import KernelBackend


def get_cutotune_parameters() -> dict:
    return dict(
        configs=[CutoTuneConfig({"kernel_backend": KernelBackend.triton})],
        default_config=CutoTuneConfig({"kernel_backend": KernelBackend.triton}),
        triggers={"x.dtype"},
    )
