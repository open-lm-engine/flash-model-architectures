import inspect
from contextlib import contextmanager
from typing import Callable, Iterable, Sequence

import torch

from ..counters import kernel_counter


_IS_CUTE_TRACING = False


@contextmanager
def enable_cute_tracing():
    global _IS_CUTE_TRACING
    _IS_CUTE_TRACING = True

    yield

    _IS_CUTE_TRACING = False


def _dispatch(func: Callable, compileable_fn: Callable, *args, **kwargs):
    if _IS_CUTE_TRACING or torch.compiler.is_compiling():
        output = compileable_fn(*args, **kwargs)
    else:
        output = func(*args, **kwargs)

    return output


def cute_op(
    name: str = None,
    mutates_args: str | Iterable[str] = None,
    device_types: str | Sequence[str] | None = None,
    schema: str | None = None,
    fake_func: Callable | None = None,
) -> Callable:
    def _inner(func: Callable):
        compileable_func = torch.library.custom_op(
            name, func, mutates_args=mutates_args, device_types=device_types, schema=schema
        )

        if fake_func is not None:
            compileable_func.register_fake(fake_func)

        def _run(*args, **kwargs):
            kernel_counter[name] += 1
            return _dispatch(func, compileable_func, *args, **kwargs)

        _run.__signature__ = inspect.signature(func)
        _run.__name__ = func.__name__

        return _run

    return _inner
