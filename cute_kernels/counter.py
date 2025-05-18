from typing import Callable


def increment_counter(func: Callable) -> Callable:
    def _run(*args, **kwargs) -> Callable:
        func._counter = getattr(func, "_counter", 0) + 1
        return func(*args, **kwargs)

    return _run
