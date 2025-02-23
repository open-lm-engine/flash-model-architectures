from ...cutotune import CutoTuneConfig


def get_cutotune_parameters() -> dict:
    return dict(
        configs=[CutoTuneConfig(dict(kernel_backend="triton"))],
        default_config=CutoTuneConfig(dict(kernel_backend="triton")),
        triggers={"x.dtype"},
    )
