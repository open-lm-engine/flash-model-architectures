import os
from enum import Enum

import yaml

from ..enums import KernelBackend
from ..utils import get_boolean_env_variable
from .config import CutoTuneConfig


_CUTOTUNE_CACHE_FILENAME = os.path.join(os.path.dirname(__file__), "cutotune_cache.yml")
_LOAD_CUTOTUNE_CACHE = get_boolean_env_variable("LOAD_CUTOTUNE_CACHE", True)


class _CutoTuneCache:
    def __init__(self, function_hash: str) -> None:
        self.full_cache = {}
        self.best_cache = {}
        self.function_hash = function_hash

        if _LOAD_CUTOTUNE_CACHE and os.path.exists(_CUTOTUNE_CACHE_FILENAME):
            self.load()

    def add_config(self, lookup_key: str, config: CutoTuneConfig, time: float) -> None:
        if lookup_key not in self.full_cache:
            self.full_cache[lookup_key] = []

        self.full_cache[lookup_key].append((config, time))

        min_time = float("inf")
        if lookup_key in self.best_cache:
            min_time = self.best_cache[lookup_key][1]

        if time < min_time:
            self.best_cache[lookup_key] = (config, time)

    def save(self) -> None:
        full_cache_serialized = {
            "all_configs": self._serialize(self.full_cache, True),
            "best_configs": self._serialize(self.best_cache, False),
        }

        for lookup_key in full_cache_serialized["all_configs"]:
            full_cache_serialized["all_configs"][lookup_key].sort(key=lambda x: x["time"])

        yaml.dump(full_cache_serialized, open(_CUTOTUNE_CACHE_FILENAME, "w"))

    def load(self) -> None:
        cache = yaml.load(open(_CUTOTUNE_CACHE_FILENAME, "r"), yaml.SafeLoader)

        self.full_cache = self._deserialize(cache["all_configs"], True)
        self.best_cache = self._deserialize(cache["best_configs"], False)

    def get_best_configs(self) -> dict[str, CutoTuneConfig]:
        return self.best_cache

    def _serialize(self, x: dict, has_config_time_list: bool) -> dict:
        result = {}

        for lookup_key in x:
            config_time_list = x[lookup_key]
            if not has_config_time_list:
                config_time_list = [config_time_list]

            def _serialize(v):
                if isinstance(v, Enum):
                    v = v.value
                return v

            for i, config_time in enumerate(config_time_list):
                config, time = config_time
                config = {key: _serialize(value) for key, value in config.get_key_values().items()}

                config_time_list[i] = {"config": config, "time": time}

            if not has_config_time_list:
                config_time_list = config_time_list[0]

            result[lookup_key] = config_time_list

        return result

    def _deserialize(self, x: dict, has_config_time_list: bool) -> dict:
        result = {}

        for lookup_key in x:
            config_time_list = x[lookup_key]
            if not has_config_time_list:
                config_time_list = [config_time_list]

            def _deserialize(k, v):
                if k.startswith("kernel_backend"):
                    v = KernelBackend(v)
                return v

            for i, config_time in enumerate(config_time_list):
                config = config_time["config"]
                time = config_time["time"]
                config = CutoTuneConfig({key: _deserialize(key, value) for key, value in config.items()})

                config_time_list[i] = [config, time]

            if not has_config_time_list:
                config_time_list = config_time_list[0]

            result[lookup_key] = config_time_list

        return result


_CUTOTUNE_CACHE_MAP = {}


def get_cutotune_cache(function_hash: str) -> _CutoTuneCache:
    global _CUTOTUNE_CACHE_MAP
    cutotune_cache = _CUTOTUNE_CACHE_MAP.get(function_hash, None)

    if cutotune_cache is None:
        _CUTOTUNE_CACHE_MAP[function_hash] = _CutoTuneCache(function_hash)
        cutotune_cache = _CUTOTUNE_CACHE_MAP[function_hash]

    return cutotune_cache


def save_cutotune_cache() -> None:
    global _CUTOTUNE_CACHE
    assert _CUTOTUNE_CACHE is not None

    _CUTOTUNE_CACHE.save()
