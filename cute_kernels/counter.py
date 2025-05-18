from typing import Callable


_FUNCTIONS_WITH_COUNTERS = []


def get_counter(func: Callable) -> None:
    return getattr(func, "_counter", 0)


def reset_counters() -> None:
    for func in _FUNCTIONS_WITH_COUNTERS:
        func._counter = 0


def increment_counter(func: Callable) -> Callable:
    _FUNCTIONS_WITH_COUNTERS.append(func)

    def _run(*args, **kwargs) -> Callable:
        _run._counter = get_counter(_run) + 1
        return func(*args, **kwargs)

    return _run
