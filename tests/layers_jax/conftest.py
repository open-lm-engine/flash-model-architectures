# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import itertools

import pytest


try:
    import jax

    _devices = jax.devices()
except ImportError:
    jax = None
    _devices = []


# unlike CUDA_VISIBLE_DEVICES (one GPU per pytest-xdist worker process), a single JAX process
# already sees every local TPU chip, and libtpu's per-chip lock makes splitting that across
# worker processes unreliable (stale /tmp/libtpu_lockfile after a killed worker). so this runs
# tests single-process and round-robins jax.default_device() across chips instead: every test
# still runs sequentially, so this spreads chip usage across a session, it doesn't cut wall time
# the way `pytest -n` does for CUDA.
_device_cycle = itertools.cycle(_devices) if len(_devices) > 1 else None


@pytest.fixture(autouse=True)
def _round_robin_jax_device():
    if _device_cycle is None:
        yield
    else:
        with jax.default_device(next(_device_cycle)):
            yield
