"""Canonical compatibility metadata for Numeta native artifacts."""

from __future__ import annotations

import platform
import sys
import sysconfig
from collections.abc import Mapping

import numpy as np

from .settings import settings

NUMETA_WRAPPER_ABI_VERSION = 1
CODEGEN_ABI_SETTINGS_VERSION = 1


def numpy_c_abi_version() -> int:
    core = getattr(np, "_core", None)
    multiarray = getattr(core, "_multiarray_umath", None)
    getter = getattr(multiarray, "_get_ndarray_c_version", None)
    return int(getter()) if getter is not None else 0


def codegen_abi_settings() -> dict:
    """Return global settings that affect persisted call or ownership contracts."""
    return {
        "version": CODEGEN_ABI_SETTINGS_VERSION,
        "add_shape_descriptors": bool(settings.add_shape_descriptors),
        "ignore_fixed_shape_in_nested_calls": bool(settings.ignore_fixed_shape_in_nested_calls),
        "use_numpy_allocator": bool(settings.use_numpy_allocator),
        "reorder_kwargs": bool(settings.reorder_kwargs),
    }


def native_runtime_abi() -> dict:
    """Return runtime properties required to load a native Numeta artifact."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": [sys.version_info.major, sys.version_info.minor],
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "numpy_c_abi": numpy_c_abi_version(),
        "wrapper_abi": NUMETA_WRAPPER_ABI_VERSION,
        "codegen_settings": codegen_abi_settings(),
    }


def metadata_mismatches(saved, current, *, prefix: str = "") -> dict[str, tuple[object, object]]:
    """Return leaf mismatches as ``path: (saved, current)`` entries."""
    if isinstance(current, Mapping):
        saved_mapping = saved if isinstance(saved, Mapping) else {}
        mismatches = {}
        for key, current_value in current.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in saved_mapping:
                mismatches[path] = (None, current_value)
                continue
            mismatches.update(metadata_mismatches(saved_mapping[key], current_value, prefix=path))
        return mismatches
    if saved != current:
        return {prefix: (saved, current)}
    return {}


def format_metadata_mismatches(mismatches: Mapping[str, tuple[object, object]]) -> str:
    return ", ".join(
        f"{path}={saved!r} (current {current!r})"
        for path, (saved, current) in sorted(mismatches.items())
    )
