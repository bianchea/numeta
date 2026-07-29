from __future__ import annotations

from dataclasses import dataclass

from numeta.datatype import float32, float64, get_datatype, int32, int64

SUPPORTED_ARCHITECTURES = frozenset({"scalar", "sse2", "avx", "avx2", "avx512f", "neon", "native"})
X86_ARCHITECTURES = frozenset({"sse2", "avx", "avx2", "avx512f"})

# Ordered widest-to-narrowest. This is the single source of truth for both
# target lane queries and the C ABI's native chunk selection.
NATIVE_VECTOR_TYPES = {
    "sse2": {
        float64: ((2, "__m128d"),),
        float32: ((4, "__m128"),),
        int64: ((2, "__m128i"),),
        int32: ((4, "__m128i"),),
    },
    "avx": {
        float64: ((4, "__m256d"), (2, "__m128d")),
        float32: ((8, "__m256"), (4, "__m128")),
        int64: ((2, "__m128i"),),
        int32: ((4, "__m128i"),),
    },
    "avx2": {
        float64: ((4, "__m256d"), (2, "__m128d")),
        float32: ((8, "__m256"), (4, "__m128")),
        int64: ((4, "__m256i"), (2, "__m128i")),
        int32: ((8, "__m256i"), (4, "__m128i")),
    },
    "avx512f": {
        float64: ((8, "__m512d"), (4, "__m256d"), (2, "__m128d")),
        float32: ((16, "__m512"), (8, "__m256"), (4, "__m128")),
        int64: ((8, "__m512i"), (4, "__m256i"), (2, "__m128i")),
        int32: ((16, "__m512i"), (8, "__m256i"), (4, "__m128i")),
    },
    "neon": {
        float64: ((2, "float64x2_t"),),
        float32: ((4, "float32x4_t"),),
        int64: ((2, "int64x2_t"),),
        int32: ((4, "int32x4_t"),),
    },
}


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
    if arch not in SUPPORTED_ARCHITECTURES:
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
    base_dtype = get_datatype(base_dtype)
    arch = target.arch
    if arch == "native":
        arch = "avx2"
    if arch == "scalar":
        return 1
    return native_vector_candidates(base_dtype, arch)[0][0]


def native_vector_candidates(base_dtype, arch: str) -> tuple[tuple[int, str], ...]:
    """Return supported native vector chunks, widest first."""
    base_dtype = get_datatype(base_dtype)
    if arch == "native":
        arch = "avx2"
    try:
        return NATIVE_VECTOR_TYPES[arch][base_dtype]
    except KeyError:
        if arch not in NATIVE_VECTOR_TYPES:
            raise NotImplementedError(f"Unsupported SIMD architecture: {arch}") from None
        raise NotImplementedError(
            f"SIMD vectors are not supported for dtype {base_dtype} on {arch}"
        ) from None
