from ...cutotune import CutoTuneConfig
from ...enums import KernelBackend


def get_cutotune_parameters() -> dict:
    return dict(
        configs=[CutoTuneConfig(dict(kernel_backend=KernelBackend.triton))],
        default_config=CutoTuneConfig(dict(kernel_backend=KernelBackend.triton)),
        triggers={"x.dtype"},
    )
