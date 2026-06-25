from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimdTarget:
    arch: str = "scalar"
    has_fma: bool = False
    prefer_aligned: bool = False
    strict_lanes: bool = False


def make_simd_target(
    arch: str | SimdTarget | None = None,
    *,
    features=(),
    prefer_aligned: bool = False,
    strict_lanes: bool = False,
) -> SimdTarget:
    if isinstance(arch, SimdTarget):
        return arch
    arch = "scalar" if arch is None else str(arch).lower()
    if arch not in {"scalar", "sse2", "avx", "avx2", "avx512f", "neon", "native"}:
        raise ValueError(f"Unsupported SIMD architecture: {arch}")
    if isinstance(features, str):
        features = (features,)
    normalized_features = {str(feature).lower() for feature in (features or ())}
    return SimdTarget(
        arch=arch,
        has_fma="fma" in normalized_features,
        prefer_aligned=prefer_aligned,
        strict_lanes=strict_lanes,
    )


def native_lanes(base_dtype, target: SimdTarget) -> int:
    from numeta.datatype import float32, float64, int32, int64, get_datatype

    base_dtype = get_datatype(base_dtype)
    arch = target.arch
    if arch == "native":
        arch = "avx2"
    if arch == "scalar":
        return 1
    if base_dtype is float64:
        return {"sse2": 2, "avx": 4, "avx2": 4, "avx512f": 8, "neon": 2}[arch]
    if base_dtype is float32:
        return {"sse2": 4, "avx": 8, "avx2": 8, "avx512f": 16, "neon": 4}[arch]
    if base_dtype is int64:
        return {"sse2": 2, "avx": 4, "avx2": 4, "avx512f": 8, "neon": 2}[arch]
    if base_dtype is int32:
        return {"sse2": 4, "avx": 8, "avx2": 8, "avx512f": 16, "neon": 4}[arch]
    raise NotImplementedError(f"SIMD lanes are not defined for dtype {base_dtype}")
