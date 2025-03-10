from collections import defaultdict


kernel_counter = defaultdict(int)


def kernel_counter_increment(*args, **kwargs):
    def _call(func):
        kernel_counter[func.__name__] += 1
        return func(*args, **kwargs)

    return _call
