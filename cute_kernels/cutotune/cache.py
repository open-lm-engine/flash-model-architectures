import os

import yaml

from ..utils import get_boolean_env_variable
from .config import CutoTuneConfig


_LOAD_CUTOTUNE_CACHE = get_boolean_env_variable("LOAD_CUTOTUNE_CACHE", True)
_CUTOTUNE_CACHE_FILENAME = os.path.dirname(os.path.dirname(__file__), "cache.yml")


class _CutoTuneCache:
    def __init__(self) -> None:
        self.cache = {}

        if _LOAD_CUTOTUNE_CACHE and os.path.exists(_CUTOTUNE_CACHE_FILENAME):
            cache = yaml.load(open(_CUTOTUNE_CACHE_FILENAME, "r"), yaml.SafeLoader)
            self.cache = self._deserialize(cache)

    def add_config(self, function_hash: str, lookup_key: str, config: CutoTuneConfig) -> None:
        if function_hash not in self.cache:
            self.cache[function_hash] = {}

        self.cache[function_hash][lookup_key] = config

    def get_config(self, function_hash: str, lookup_key: str, default: CutoTuneConfig = None) -> CutoTuneConfig:
        if function_hash in self.cache:
            function_cache = self.cache[function_hash]
            return function_cache.get(lookup_key, default)

        return default

    def save(self) -> None:
        yaml.dump(self._serialize(self.cache), open(_CUTOTUNE_CACHE_FILENAME, "w"))

    def _serialize(self, x: dict) -> dict:
        result = {}

        for function_hash in x:
            function_cache = x[function_hash]
            result[function_hash] = {}

            for lookup_key, config in function_cache.items():
                result[function_hash][lookup_key] = {key: value for key, value in config.get_key_values().items()}

        return result

    def _deserialize(self, x: dict) -> dict:
        result = {}

        for function_hash in x:
            function_cache = x[function_hash]
            result[function_hash] = {}

            for lookup_key, config in function_cache.items():
                result[function_hash][lookup_key] = CutoTuneConfig({key: value for key, value in config.items()})

        return result


_CUTOTUNE_CACHE = None


def get_cutotune_cache() -> _CutoTuneCache:
    global _CUTOTUNE_CACHE
    if _CUTOTUNE_CACHE is None:
        _CUTOTUNE_CACHE = _CutoTuneCache()

    return _CUTOTUNE_CACHE


def save_cutotune_cache() -> None:
    global _CUTOTUNE_CACHE
    _CUTOTUNE_CACHE.save()
