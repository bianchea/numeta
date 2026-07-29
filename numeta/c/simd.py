from __future__ import annotations

from dataclasses import dataclass

from numeta.datatype import _short_c_name, float32, float64, int32, int64
from numeta.simd.target import (
    NATIVE_VECTOR_TYPES,
    X86_ARCHITECTURES,
    SimdTarget,
    native_vector_candidates,
)

SUPPORTED_BASE_DTYPES = frozenset(
    dtype for arch_types in NATIVE_VECTOR_TYPES.values() for dtype in arch_types
)

# Intrinsic token -> generated helper operation. Both requirement collection
# and rendering consume this table, so adding a direct helper intrinsic does
# not require parallel dispatch edits in the C emitter.
INTRINSIC_HELPER_OPS = {
    "simd_broadcast": "broadcast",
    "simd_vload": "load",
    "simd_vgather": "gather",
    "simd_fma": "fma",
    "simd_reduce_sum": "reduce_sum",
    "simd_unpack_low": "unpack_low",
    "simd_unpack_high": "unpack_high",
    "simd_pairwise_halves_sum": "pairwise_halves_sum",
}

DIRECT_INTRINSIC_NAMES = frozenset(
    {
        "simd_fnma",
        "simd_cvt_f64_f32",
        "simd_cvt_f32_f64",
        "simd_cvt_f64_i32",
        "simd_cvt_i32_f64",
        "simd_floor",
        "simd_rcp",
        "simd_rsqrt",
        "simd_compare",
        "simd_blend",
        "simd_movemask",
        "simd_extract_i32",
        "simd_set4_f64",
        "simd_exp2_neg_i32",
    }
)

_PERMUTE_SECOND_HALVES_IMMEDIATE = 0x21
_BLEND_UPPER_HALF_IMMEDIATE = 0xC


def render_direct_intrinsic(
    name: str,
    args: list[str],
    *,
    target: SimdTarget,
    vector_dtype=None,
    metadata: dict | None = None,
) -> str:
    """Render a semantic SIMD intrinsic for the selected native target.

    Keeping instruction spelling here prevents the C emitter and public API
    from depending on AVX names. Additional targets can implement the same
    semantic operations without changing lowering.
    """
    target = _canonical_target(target)
    if target.arch != "avx2":
        raise NotImplementedError(f"{name} currently has no {target.arch} implementation")
    metadata = metadata or {}
    fixed = {
        "simd_cvt_f64_f32": "_mm256_cvtpd_ps",
        "simd_cvt_f32_f64": "_mm256_cvtps_pd",
        "simd_cvt_f64_i32": "_mm256_cvttpd_epi32",
        "simd_cvt_i32_f64": "_mm256_cvtepi32_pd",
        "simd_floor": "_mm256_floor_pd",
        "simd_rcp": "_mm_rcp_ps",
        "simd_rsqrt": "_mm_rsqrt_ps",
        "simd_blend": "_mm256_blendv_pd",
        "simd_movemask": "_mm256_movemask_pd",
    }
    if name in fixed:
        return f"{fixed[name]}({', '.join(args)})"
    if name == "simd_fnma":
        suffix = "pd" if vector_dtype.base_dtype() is float64 else "ps"
        prefix = "_mm256" if vector_dtype.get_nbytes() == 32 else "_mm"
        return f"{prefix}_fnmadd_{suffix}({', '.join(args)})"
    if name == "simd_compare":
        predicates = {
            "lt": "_CMP_LT_OQ",
            "le": "_CMP_LE_OQ",
            "gt": "_CMP_GT_OQ",
            "ge": "_CMP_GE_OQ",
            "eq": "_CMP_EQ_OQ",
            "neq": "_CMP_NEQ_OQ",
        }
        return f"_mm256_cmp_pd({args[0]}, {args[1]}, {predicates[metadata['predicate']]})"
    if name == "simd_extract_i32":
        lane = metadata["lane"]
        if lane == 0:
            return f"_mm_cvtsi128_si32({args[0]})"
        return f"_mm_extract_epi32({args[0]}, {lane})"
    if name == "simd_set4_f64":
        return f"_mm256_set_pd({args[3]}, {args[2]}, {args[1]}, {args[0]})"
    if name == "simd_exp2_neg_i32":
        return (
            "_mm256_castsi256_pd(_mm256_slli_epi64("
            "_mm256_sub_epi64(_mm256_set1_epi64x(1023), "
            f"_mm256_cvtepi32_epi64({args[0]})), 52))"
        )
    raise NotImplementedError(f"Unsupported SIMD intrinsic: {name}")


@dataclass(frozen=True)
class LoweredVectorABI:
    c_name: str
    base_dtype: type
    base_ctype: str
    lanes: int
    native_lanes: int
    chunks: int
    native_type: str | None
    target: SimdTarget

    @property
    def is_scalar_fallback(self) -> bool:
        return self.native_type is None

    @property
    def is_single_native(self) -> bool:
        return not self.is_scalar_fallback and self.chunks == 1


def vector_abi(vector_dtype, target: SimdTarget) -> LoweredVectorABI:
    base_dtype = vector_dtype.base_dtype()
    if base_dtype not in SUPPORTED_BASE_DTYPES:
        raise NotImplementedError(f"C SIMD vectors are not supported for dtype {base_dtype}")

    target = _canonical_target(target)
    lanes = vector_dtype.lanes()
    if target.arch == "scalar":
        return _scalar_vector_abi(vector_dtype, target)

    candidates = _native_candidates(target.arch, base_dtype)
    for native_width, native_type in candidates:
        if lanes % native_width == 0:
            return LoweredVectorABI(
                c_name=vector_dtype.get_cnumpy(),
                base_dtype=base_dtype,
                base_ctype=base_dtype.get_cnumpy(),
                lanes=lanes,
                native_lanes=native_width,
                chunks=lanes // native_width,
                native_type=native_type,
                target=target,
            )

    if target.strict_lanes:
        widths = ", ".join(str(native_width) for native_width, _native_type in candidates)
        raise NotImplementedError(
            f"Vector[{base_dtype._name}, {lanes}] does not fit native SIMD widths for "
            f"{target.arch}: {widths}"
        )

    # Zig-like logical vector semantics: odd/small lane counts still work, they
    # just lower to a scalar lane array for now.
    return _scalar_vector_abi(vector_dtype, target)


def needs_header(vector_dtypes, target: SimdTarget) -> str | None:
    target = _canonical_target(target)
    if target.arch == "scalar":
        return None
    if not any(not vector_abi(dtype, target).is_scalar_fallback for dtype in vector_dtypes):
        return None
    if target.arch in X86_ARCHITECTURES:
        return "immintrin.h"
    if target.arch == "neon":
        return "arm_neon.h"
    return None


def helper_name(op: str, vector_dtype) -> str:
    return _helper_name_from_parts(op, vector_dtype.base_dtype(), vector_dtype.lanes())


def _helper_name_from_parts(op: str, base_dtype, lanes: int) -> str:
    suffix = f"{_short_c_name(base_dtype)}_{lanes}"
    if op in {"add", "sub", "mul", "div", "fma"}:
        return f"nm_vec_{op}_{suffix}"
    if op == "sqrt":
        return f"nm_vec_sqrt_{suffix}"
    if op == "broadcast":
        return f"nm_vbroadcast_{suffix}"
    if op == "load":
        return f"nm_vload_{suffix}"
    if op == "gather":
        return f"nm_vgather_{suffix}"
    if op == "store":
        return f"nm_vstore_{suffix}"
    if op == "reduce_sum":
        return f"nm_reduce_sum_{suffix}"
    if op in {"unpack_low", "unpack_high", "pairwise_halves_sum"}:
        return f"nm_{op}_{suffix}"
    raise NotImplementedError(f"Unsupported SIMD helper op: {op}")


def render_typedefs(
    vector_dtypes,
    target: SimdTarget,
    *,
    vector_type_style: str = "intrinsic",
) -> list[str]:
    lines: list[str] = []
    for vector_dtype in sorted(vector_dtypes, key=lambda dtype: dtype.get_cnumpy()):
        abi = vector_abi(vector_dtype, target)
        if vector_type_style == "gcc_vector" and not abi.is_scalar_fallback:
            bytes_ = vector_dtype.get_nbytes()
            lines.append(
                f"typedef {abi.base_ctype} {abi.c_name} __attribute__((vector_size({bytes_})));\n"
            )
        elif abi.is_scalar_fallback:
            lines.append(
                f"typedef struct {{ {abi.base_ctype} lane[{abi.lanes}]; }} {abi.c_name};\n"
            )
        elif abi.is_single_native:
            lines.append(f"typedef {abi.native_type} {abi.c_name};\n")
        else:
            members = " ".join(f"{abi.native_type} v{i};" for i in range(abi.chunks))
            lines.append(f"typedef struct {{ {members} }} {abi.c_name};\n")
    return lines


def render_helpers(vector_dtypes, helper_ops, target: SimdTarget) -> list[str]:
    lines: list[str] = []
    ordered_types = sorted(vector_dtypes, key=lambda dtype: dtype.get_cnumpy())
    ordered_ops = [
        "broadcast",
        "load",
        "gather",
        "store",
        "add",
        "sub",
        "mul",
        "div",
        "sqrt",
        "fma",
        "reduce_sum",
        "unpack_low",
        "unpack_high",
        "pairwise_halves_sum",
    ]
    helper_ops = set(helper_ops)
    for vector_dtype in ordered_types:
        abi = vector_abi(vector_dtype, target)
        for op in ordered_ops:
            if (op, vector_dtype) not in helper_ops:
                continue
            lines.extend(_render_helper(op, abi))
            lines.append("\n")
    return lines


def _canonical_target(target: SimdTarget) -> SimdTarget:
    if target.arch != "native":
        return target
    return SimdTarget(
        arch="avx2",
        has_fma=target.has_fma,
        prefer_aligned=target.prefer_aligned,
        strict_lanes=target.strict_lanes,
    )


def _scalar_vector_abi(vector_dtype, target: SimdTarget) -> LoweredVectorABI:
    base_dtype = vector_dtype.base_dtype()
    lanes = vector_dtype.lanes()
    return LoweredVectorABI(
        c_name=vector_dtype.get_cnumpy(),
        base_dtype=base_dtype,
        base_ctype=base_dtype.get_cnumpy(),
        lanes=lanes,
        native_lanes=1,
        chunks=lanes,
        native_type=None,
        target=target,
    )


def _native_candidates(arch: str, base_dtype) -> tuple[tuple[int, str], ...]:
    return native_vector_candidates(base_dtype, arch)


def _render_helper(op: str, abi: LoweredVectorABI) -> list[str]:
    if op in {"add", "sub", "mul", "div"}:
        return _render_binary_helper(op, abi)
    if op == "sqrt":
        return _render_sqrt_helper(abi)
    if op == "fma":
        return _render_fma_helper(abi)
    if op == "broadcast":
        return _render_broadcast_helper(abi)
    if op == "load":
        return _render_load_helper(abi)
    if op == "gather":
        return _render_gather_helper(abi)
    if op == "store":
        return _render_store_helper(abi)
    if op == "reduce_sum":
        return _render_reduce_sum_helper(abi)
    if op in {"unpack_low", "unpack_high", "pairwise_halves_sum"}:
        return _render_lane_permute_helper(op, abi)
    raise NotImplementedError(f"Unsupported SIMD helper op: {op}")


def _render_binary_helper(op: str, abi: LoweredVectorABI) -> list[str]:
    name = _abi_helper_name(op, abi)
    lines = [f"static inline {abi.c_name} {name}({abi.c_name} a, {abi.c_name} b) {{\n"]
    if abi.is_scalar_fallback or not _has_native_binary(op, abi):
        lines.extend(_render_scalarized_binary_body(op, abi))
    elif abi.is_single_native:
        lines.append(f"    return {_binary_intrinsic(op, abi)}(a, b);\n")
    else:
        intrinsic = _binary_intrinsic(op, abi)
        chunks = [f"{intrinsic}(a.v{i}, b.v{i})" for i in range(abi.chunks)]
        lines.append(f"    return ({abi.c_name}){{{', '.join(chunks)}}};\n")
    lines.append("}\n")
    return lines


def _render_sqrt_helper(abi: LoweredVectorABI) -> list[str]:
    name = _abi_helper_name("sqrt", abi)
    lines = [f"static inline {abi.c_name} {name}({abi.c_name} a) {{\n"]
    if abi.is_scalar_fallback or not _has_native_sqrt(abi):
        lines.extend(_render_scalarized_sqrt_body(abi))
    elif abi.is_single_native:
        lines.append(f"    return {_sqrt_intrinsic(abi)}(a);\n")
    else:
        intrinsic = _sqrt_intrinsic(abi)
        chunks = [f"{intrinsic}(a.v{i})" for i in range(abi.chunks)]
        lines.append(f"    return ({abi.c_name}){{{', '.join(chunks)}}};\n")
    lines.append("}\n")
    return lines


def _render_fma_helper(abi: LoweredVectorABI) -> list[str]:
    name = _abi_helper_name("fma", abi)
    lines = [
        f"static inline {abi.c_name} {name}({abi.c_name} a, {abi.c_name} b, {abi.c_name} c) {{\n"
    ]
    fma_intrinsic = _fma_intrinsic(abi) if _has_native_fma(abi) else None
    if abi.is_scalar_fallback:
        lines.extend(_render_scalarized_fma_body(abi))
    elif fma_intrinsic is not None:
        if abi.is_single_native:
            lines.append(f"    return {fma_intrinsic}(a, b, c);\n")
        else:
            chunks = [f"{fma_intrinsic}(a.v{i}, b.v{i}, c.v{i})" for i in range(abi.chunks)]
            lines.append(f"    return ({abi.c_name}){{{', '.join(chunks)}}};\n")
    elif _has_native_binary("mul", abi) and _has_native_binary("add", abi):
        add_intrinsic = _binary_intrinsic("add", abi)
        mul_intrinsic = _binary_intrinsic("mul", abi)
        if abi.is_single_native:
            lines.append(f"    return {add_intrinsic}({mul_intrinsic}(a, b), c);\n")
        else:
            chunks = [
                f"{add_intrinsic}({mul_intrinsic}(a.v{i}, b.v{i}), c.v{i})"
                for i in range(abi.chunks)
            ]
            lines.append(f"    return ({abi.c_name}){{{', '.join(chunks)}}};\n")
    else:
        lines.extend(_render_scalarized_fma_body(abi))
    lines.append("}\n")
    return lines


def _render_broadcast_helper(abi: LoweredVectorABI) -> list[str]:
    name = _abi_helper_name("broadcast", abi)
    lines = [f"static inline {abi.c_name} {name}({abi.base_ctype} x) {{\n"]
    if abi.is_scalar_fallback:
        lines.append(f"    {abi.c_name} out;\n")
        lines.append(f"    for (int i = 0; i < {abi.lanes}; ++i) {{\n")
        lines.append("        out.lane[i] = x;\n")
        lines.append("    }\n")
        lines.append("    return out;\n")
    elif abi.is_single_native:
        lines.append(f"    return {_broadcast_expr('x', abi)};\n")
    else:
        chunks = [_broadcast_expr("x", abi) for _ in range(abi.chunks)]
        lines.append(f"    return ({abi.c_name}){{{', '.join(chunks)}}};\n")
    lines.append("}\n")
    return lines


def _render_load_helper(abi: LoweredVectorABI) -> list[str]:
    name = _abi_helper_name("load", abi)
    lines = [f"static inline {abi.c_name} {name}(const {abi.base_ctype} *p) {{\n"]
    if abi.is_scalar_fallback:
        lines.append(f"    {abi.c_name} out;\n")
        lines.append(f"    for (int i = 0; i < {abi.lanes}; ++i) {{\n")
        lines.append("        out.lane[i] = p[i];\n")
        lines.append("    }\n")
        lines.append("    return out;\n")
    else:
        lines.append(f"    return {_load_full_expr('p', abi)};\n")
    lines.append("}\n")
    return lines


def _render_gather_helper(abi: LoweredVectorABI) -> list[str]:
    name = _abi_helper_name("gather", abi)
    lines = [
        f"static inline {abi.c_name} {name}(const {abi.base_ctype} *p, const npy_int64 *idx) {{\n"
    ]
    if abi.is_scalar_fallback or not _has_native_gather(abi):
        lines.extend(_render_scalarized_gather_body(abi))
    elif abi.is_single_native:
        lines.extend(_render_native_gather_return("p", "idx", abi, "    "))
    else:
        chunks = []
        for i in range(abi.chunks):
            chunk_name = f"chunk{i}"
            chunk_lines = _render_native_gather_return(
                "p",
                f"idx + {i * abi.native_lanes}",
                _chunk_abi(abi),
                "    ",
                result_name=chunk_name,
            )
            lines.extend(chunk_lines)
            chunks.append(chunk_name)
        lines.append(f"    return ({abi.c_name}){{{', '.join(chunks)}}};\n")
    lines.append("}\n")
    return lines


def _render_store_helper(abi: LoweredVectorABI) -> list[str]:
    name = _abi_helper_name("store", abi)
    lines = [f"static inline void {name}({abi.base_ctype} *p, {abi.c_name} v) {{\n"]
    if abi.is_scalar_fallback:
        lines.append(f"    for (int i = 0; i < {abi.lanes}; ++i) {{\n")
        lines.append("        p[i] = v.lane[i];\n")
        lines.append("    }\n")
    else:
        lines.extend(_store_full_lines("p", "v", abi, "    "))
    lines.append("}\n")
    return lines


def _render_reduce_sum_helper(abi: LoweredVectorABI) -> list[str]:
    name = _abi_helper_name("reduce_sum", abi)
    lines = [f"static inline {abi.base_ctype} {name}({abi.c_name} v) {{\n"]
    if abi.is_scalar_fallback:
        lines.append(f"    {abi.base_ctype} acc = ({abi.base_ctype})0;\n")
        lines.append(f"    for (int i = 0; i < {abi.lanes}; ++i) {{\n")
        lines.append("        acc += v.lane[i];\n")
        lines.append("    }\n")
        lines.append("    return acc;\n")
    else:
        terms = " + ".join(f"tmp[{i}]" for i in range(abi.lanes)) or "0"
        lines.append(f"    {abi.base_ctype} tmp[{abi.lanes}];\n")
        lines.extend(_store_full_lines("tmp", "v", abi, "    "))
        lines.append(f"    return {terms};\n")
    lines.append("}\n")
    return lines


def _render_lane_permute_helper(op: str, abi: LoweredVectorABI) -> list[str]:
    name = _abi_helper_name(op, abi)
    lines = [f"static inline {abi.c_name} {name}({abi.c_name} a, {abi.c_name} b) {{\n"]
    if abi.is_single_native and abi.native_type == "__m256d" and abi.lanes == 4:
        if op == "unpack_low":
            lines.append("    return _mm256_unpacklo_pd(a, b);\n")
        elif op == "unpack_high":
            lines.append("    return _mm256_unpackhi_pd(a, b);\n")
        else:
            lines.append(
                "    // Select a's upper half followed by b's lower half.\n"
                f"    __m256d p = _mm256_permute2f128_pd("
                f"a, b, 0x{_PERMUTE_SECOND_HALVES_IMMEDIATE:X});\n"
            )
            lines.append(
                "    // Keep a's lower half and b's upper half before adding.\n"
                f"    return _mm256_add_pd("
                f"p, _mm256_blend_pd(a, b, 0x{_BLEND_UPPER_HALF_IMMEDIATE:X}));\n"
            )
    else:
        if abi.is_scalar_fallback:
            lines.append(f"    {abi.c_name} out;\n")
            a_lane = "a.lane"
            b_lane = "b.lane"
            out_lane = "out.lane"
        else:
            lines.extend(_native_inputs_to_arrays(abi, ("a", "b")))
            lines.append(f"    {abi.base_ctype} out_tmp[{abi.lanes}];\n")
            a_lane = "a_tmp"
            b_lane = "b_tmp"
            out_lane = "out_tmp"
        half = abi.lanes // 2
        if abi.lanes % 2:
            raise NotImplementedError(f"{op} requires an even SIMD lane count")
        if op in {"unpack_low", "unpack_high"}:
            offset = 0 if op == "unpack_low" else 1
            for pair in range(half):
                source = 2 * pair + offset
                lines.append(f"    {out_lane}[{2 * pair}] = {a_lane}[{source}];\n")
                lines.append(f"    {out_lane}[{2 * pair + 1}] = {b_lane}[{source}];\n")
        else:
            for lane in range(half):
                lines.append(
                    f"    {out_lane}[{lane}] = {a_lane}[{lane}] + {a_lane}[{lane + half}];\n"
                )
                lines.append(
                    f"    {out_lane}[{lane + half}] = {b_lane}[{lane}] + {b_lane}[{lane + half}];\n"
                )
        if abi.is_scalar_fallback:
            lines.append("    return out;\n")
        else:
            lines.append(f"    return {_load_full_expr('out_tmp', abi)};\n")
    lines.append("}\n")
    return lines


def _render_scalarized_binary_body(op: str, abi: LoweredVectorABI) -> list[str]:
    c_op = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
    return _render_scalarized_elementwise_body(
        abi,
        ("a", "b"),
        lambda lane: f"{lane('a')} {c_op} {lane('b')}",
    )


def _render_scalarized_sqrt_body(abi: LoweredVectorABI) -> list[str]:
    return _render_scalarized_elementwise_body(
        abi,
        ("a",),
        lambda lane: _sqrt_call(lane("a"), abi.base_dtype),
    )


def _render_scalarized_fma_body(abi: LoweredVectorABI) -> list[str]:
    return _render_scalarized_elementwise_body(
        abi,
        ("a", "b", "c"),
        lambda lane: f"{lane('a')} * {lane('b')} + {lane('c')}",
    )


def _render_scalarized_elementwise_body(
    abi: LoweredVectorABI,
    input_names: tuple[str, ...],
    expression,
) -> list[str]:
    """Render the common unpack/loop/repack shape of scalarized helpers."""
    if abi.is_scalar_fallback:
        lines = [f"    {abi.c_name} out;\n"]

        def lane(name):
            return f"{name}.lane[i]"

        output = "out.lane[i]"
        result = "out"
    else:
        lines = _native_inputs_to_arrays(abi, input_names)
        lines.append(f"    {abi.base_ctype} out_tmp[{abi.lanes}];\n")

        def lane(name):
            return f"{name}_tmp[i]"

        output = "out_tmp[i]"
        result = _load_full_expr("out_tmp", abi)

    lines.append(f"    for (int i = 0; i < {abi.lanes}; ++i) {{\n")
    lines.append(f"        {output} = {expression(lane)};\n")
    lines.append("    }\n")
    lines.append(f"    return {result};\n")
    return lines


def _render_scalarized_gather_body(abi: LoweredVectorABI) -> list[str]:
    if abi.is_scalar_fallback:
        lines = [f"    {abi.c_name} out;\n"]
        lines.append(f"    for (int i = 0; i < {abi.lanes}; ++i) {{\n")
        lines.append("        out.lane[i] = p[idx[i]];\n")
        lines.append("    }\n")
        lines.append("    return out;\n")
        return lines

    lines = [f"    {abi.base_ctype} out_tmp[{abi.lanes}];\n"]
    lines.append(f"    for (int i = 0; i < {abi.lanes}; ++i) {{\n")
    lines.append("        out_tmp[i] = p[idx[i]];\n")
    lines.append("    }\n")
    lines.append(f"    return {_load_full_expr('out_tmp', abi)};\n")
    return lines


def _native_inputs_to_arrays(abi: LoweredVectorABI, names: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for name in names:
        lines.append(f"    {abi.base_ctype} {name}_tmp[{abi.lanes}];\n")
        lines.extend(_store_full_lines(f"{name}_tmp", name, abi, "    "))
    return lines


def _abi_helper_name(op: str, abi: LoweredVectorABI) -> str:
    return _helper_name_from_parts(op, abi.base_dtype, abi.lanes)


def _chunk_abi(abi: LoweredVectorABI) -> LoweredVectorABI:
    return LoweredVectorABI(
        c_name=abi.native_type or abi.c_name,
        base_dtype=abi.base_dtype,
        base_ctype=abi.base_ctype,
        lanes=abi.native_lanes,
        native_lanes=abi.native_lanes,
        chunks=1,
        native_type=abi.native_type,
        target=abi.target,
    )


def _has_native_binary(op: str, abi: LoweredVectorABI) -> bool:
    if abi.is_scalar_fallback:
        return False
    if abi.target.arch in X86_ARCHITECTURES:
        if abi.base_dtype in {float64, float32}:
            return op in {"add", "sub", "mul", "div"}
        if op in {"add", "sub"}:
            return True
        if op == "mul" and abi.base_dtype is int32:
            return abi.target.arch in {"avx2", "avx512f"}
        return False
    if abi.target.arch == "neon":
        if abi.base_dtype in {float64, float32}:
            return op in {"add", "sub", "mul", "div"}
        if op in {"add", "sub"}:
            return True
        return op == "mul" and abi.base_dtype is int32
    return False


def _has_native_sqrt(abi: LoweredVectorABI) -> bool:
    return not abi.is_scalar_fallback and abi.base_dtype in {float64, float32}


def _has_native_fma(abi: LoweredVectorABI) -> bool:
    if abi.is_scalar_fallback or not abi.target.has_fma:
        return False
    return abi.base_dtype in {float64, float32} and abi.target.arch in {
        "avx",
        "avx2",
        "avx512f",
        "neon",
    }


def _has_native_gather(abi: LoweredVectorABI) -> bool:
    if abi.is_scalar_fallback or abi.target.arch not in {"avx2", "avx512f"}:
        return False
    return abi.base_dtype in {float64, int64}


def _x86_prefix(abi: LoweredVectorABI) -> str:
    return {
        "__m128d": "_mm",
        "__m128": "_mm",
        "__m128i": "_mm",
        "__m256d": "_mm256",
        "__m256": "_mm256",
        "__m256i": "_mm256",
        "__m512d": "_mm512",
        "__m512": "_mm512",
        "__m512i": "_mm512",
    }[abi.native_type]


def _x86_float_suffix(abi: LoweredVectorABI) -> str:
    if abi.base_dtype is float64:
        return "pd"
    if abi.base_dtype is float32:
        return "ps"
    raise NotImplementedError(f"dtype {abi.base_dtype} is not a floating SIMD dtype")


def _x86_int_suffix(abi: LoweredVectorABI) -> str:
    if abi.base_dtype is int64:
        return "epi64"
    if abi.base_dtype is int32:
        return "epi32"
    raise NotImplementedError(f"dtype {abi.base_dtype} is not an integer SIMD dtype")


def _binary_intrinsic(op: str, abi: LoweredVectorABI) -> str:
    if abi.target.arch == "neon":
        return _neon_binary_intrinsic(op, abi)
    prefix = _x86_prefix(abi)
    suffix = {"add": "add", "sub": "sub", "mul": "mul", "div": "div"}[op]
    if abi.base_dtype in {float64, float32}:
        return f"{prefix}_{suffix}_{_x86_float_suffix(abi)}"
    if op == "mul" and abi.base_dtype is int32:
        return f"{prefix}_mullo_epi32"
    return f"{prefix}_{suffix}_{_x86_int_suffix(abi)}"


def _sqrt_intrinsic(abi: LoweredVectorABI) -> str:
    if abi.target.arch == "neon":
        return "vsqrtq_f64" if abi.base_dtype is float64 else "vsqrtq_f32"
    return f"{_x86_prefix(abi)}_sqrt_{_x86_float_suffix(abi)}"


def _fma_intrinsic(abi: LoweredVectorABI) -> str:
    if abi.target.arch == "neon":
        return "vfmaq_f64" if abi.base_dtype is float64 else "vfmaq_f32"
    return f"{_x86_prefix(abi)}_fmadd_{_x86_float_suffix(abi)}"


def _broadcast_expr(value: str, abi: LoweredVectorABI) -> str:
    if abi.target.arch == "neon":
        suffix = _neon_suffix(abi.base_dtype)
        return f"vdupq_n_{suffix}({value})"
    prefix = _x86_prefix(abi)
    if abi.base_dtype in {float64, float32}:
        return f"{prefix}_set1_{_x86_float_suffix(abi)}({value})"
    if abi.base_dtype is int64:
        if abi.native_type == "__m512i":
            return f"_mm512_set1_epi64({value})"
        return f"{prefix}_set1_epi64x({value})"
    return f"{prefix}_set1_epi32({value})"


def _load_full_expr(ptr: str, abi: LoweredVectorABI) -> str:
    if abi.is_single_native:
        return _load_native_expr(ptr, abi)
    chunk = _chunk_abi(abi)
    chunks = [
        _load_native_expr(f"{ptr} + {i * abi.native_lanes}", chunk) for i in range(abi.chunks)
    ]
    return f"({abi.c_name}){{{', '.join(chunks)}}}"


def _load_native_expr(ptr: str, abi: LoweredVectorABI) -> str:
    if abi.target.arch == "neon":
        return f"vld1q_{_neon_suffix(abi.base_dtype)}({ptr})"
    if abi.base_dtype in {float64, float32}:
        return f"{_x86_prefix(abi)}_loadu_{_x86_float_suffix(abi)}({ptr})"
    if abi.native_type == "__m128i":
        return f"_mm_loadu_si128((const __m128i*)({ptr}))"
    if abi.native_type == "__m256i":
        return f"_mm256_loadu_si256((const __m256i*)({ptr}))"
    return f"_mm512_loadu_si512((const void*)({ptr}))"


def _store_full_lines(ptr: str, value: str, abi: LoweredVectorABI, indent: str) -> list[str]:
    if abi.is_single_native:
        return [_store_native_line(ptr, value, abi, indent)]
    chunk = _chunk_abi(abi)
    return [
        _store_native_line(f"{ptr} + {i * abi.native_lanes}", f"{value}.v{i}", chunk, indent)
        for i in range(abi.chunks)
    ]


def _store_native_line(ptr: str, value: str, abi: LoweredVectorABI, indent: str) -> str:
    if abi.target.arch == "neon":
        return f"{indent}vst1q_{_neon_suffix(abi.base_dtype)}({ptr}, {value});\n"
    if abi.base_dtype in {float64, float32}:
        return f"{indent}{_x86_prefix(abi)}_storeu_{_x86_float_suffix(abi)}({ptr}, {value});\n"
    if abi.native_type == "__m128i":
        return f"{indent}_mm_storeu_si128((__m128i*)({ptr}), {value});\n"
    if abi.native_type == "__m256i":
        return f"{indent}_mm256_storeu_si256((__m256i*)({ptr}), {value});\n"
    return f"{indent}_mm512_storeu_si512((void*)({ptr}), {value});\n"


def _render_native_gather_return(
    ptr: str,
    idx: str,
    abi: LoweredVectorABI,
    indent: str,
    *,
    result_name: str | None = None,
) -> list[str]:
    lines: list[str] = []
    scale = abi.base_dtype.get_nbytes()
    assign = f"{abi.native_type} {result_name} = " if result_name is not None else "return "
    vidx_name = f"vidx_{result_name}" if result_name is not None else "vidx"
    if abi.native_type == "__m128d" and abi.base_dtype is float64:
        lines.append(f"{indent}{assign}_mm_set_pd({ptr}[({idx})[1]], {ptr}[({idx})[0]]);\n")
        return lines
    if abi.native_type == "__m128i" and abi.base_dtype is int64:
        lines.append(f"{indent}{assign}_mm_set_epi64x({ptr}[({idx})[1]], {ptr}[({idx})[0]]);\n")
        return lines
    if abi.native_type == "__m256d" and abi.base_dtype is float64:
        lines.append(
            f"{indent}__m256i {vidx_name} = _mm256_loadu_si256((const __m256i*)({idx}));\n"
        )
        lines.append(f"{indent}{assign}_mm256_i64gather_pd({ptr}, {vidx_name}, {scale});\n")
        return lines
    if abi.native_type == "__m256i" and abi.base_dtype is int64:
        lines.append(
            f"{indent}__m256i {vidx_name} = _mm256_loadu_si256((const __m256i*)({idx}));\n"
        )
        lines.append(
            f"{indent}{assign}_mm256_i64gather_epi64("
            f"(const long long*)({ptr}), {vidx_name}, {scale});\n"
        )
        return lines
    if abi.native_type == "__m512d" and abi.base_dtype is float64:
        lines.append(f"{indent}__m512i {vidx_name} = _mm512_loadu_si512((const void*)({idx}));\n")
        lines.append(f"{indent}{assign}_mm512_i64gather_pd({vidx_name}, {ptr}, {scale});\n")
        return lines
    if abi.native_type == "__m512i" and abi.base_dtype is int64:
        lines.append(f"{indent}__m512i {vidx_name} = _mm512_loadu_si512((const void*)({idx}));\n")
        lines.append(
            f"{indent}{assign}_mm512_i64gather_epi64("
            f"{vidx_name}, (const long long*)({ptr}), {scale});\n"
        )
        return lines
    raise NotImplementedError(
        f"Native gather is not implemented for {abi.native_type}/{abi.base_dtype}"
    )


def _neon_binary_intrinsic(op: str, abi: LoweredVectorABI) -> str:
    suffix = _neon_suffix(abi.base_dtype)
    op_name = {"add": "add", "sub": "sub", "mul": "mul", "div": "div"}[op]
    return f"v{op_name}q_{suffix}"


def _neon_suffix(base_dtype) -> str:
    if base_dtype is float64:
        return "f64"
    if base_dtype is float32:
        return "f32"
    if base_dtype is int64:
        return "s64"
    if base_dtype is int32:
        return "s32"
    raise NotImplementedError(f"NEON dtype {base_dtype} is not supported")


def _sqrt_call(expr: str, base_dtype) -> str:
    if base_dtype is float32:
        return f"sqrtf({expr})"
    return f"sqrt({expr})"
