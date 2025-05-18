from typing import Callable


_FUNCTIONS_WITH_COUNTERS = []


def increment_counter(func: Callable) -> Callable:
    _FUNCTIONS_WITH_COUNTERS.append(func)

    def _run(*args, **kwargs) -> Callable:
        func._counter = getattr(func, "_counter", 0) + 1
        return func(*args, **kwargs)

    return _run


def reset_counters() -> None:
    for func in _FUNCTIONS_WITH_COUNTERS:
        func._counter = 0
