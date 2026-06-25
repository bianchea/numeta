from pathlib import Path

import numpy as np
import pytest

import numeta as nm
from numeta.c.emitter import CEmitter
from numeta.ir import lower_procedure


def _emit_axpy_code(simd_arch="scalar", simd_features=()):
    @nm.jit(backend="c", simd_arch=simd_arch, simd_features=simd_features)
    def axpy(a, x, y, n):
        lanes = 8
        av = nm.broadcast(a, lanes=lanes)
        for i in nm.range(0, n, lanes):
            xv = nm.vload(x, i, lanes=lanes)
            yv = nm.vload(y, i, lanes=lanes)
            nm.vstore(y, i, nm.fma(av, xv, yv))

    axpy(nm.f8, nm.f8[:], nm.f8[:], nm.i8)
    compiled = next(iter(axpy._compiled_functions.values()))
    ir_proc = lower_procedure(compiled.symbolic_function, backend="c")
    code, _requires_math = CEmitter(
        simd_arch=compiled.simd_arch,
        simd_features=compiled.simd_features,
    ).emit_procedure(ir_proc)
    return code


def _emit_vector_add_code(simd_arch="scalar", simd_features=()):
    @nm.jit(backend="c", simd_arch=simd_arch, simd_features=simd_features)
    def add_vec(x, y, z, n):
        lanes = 4
        for i in nm.range(0, n, lanes):
            xv = nm.vload(x, i, lanes=lanes)
            yv = nm.vload(y, i, lanes=lanes)
            zv = xv + yv
            nm.vstore(z, i, zv)

    add_vec(nm.f8[:], nm.f8[:], nm.f8[:], nm.i8)
    compiled = next(iter(add_vec._compiled_functions.values()))
    ir_proc = lower_procedure(compiled.symbolic_function, backend="c")
    code, _requires_math = CEmitter(
        simd_arch=compiled.simd_arch,
        simd_features=compiled.simd_features,
    ).emit_procedure(ir_proc)
    return code


def _emit_vector_sqrt_code(simd_arch="scalar", simd_features=()):
    @nm.jit(backend="c", simd_arch=simd_arch, simd_features=simd_features)
    def sqrt_vec(x, y):
        lanes = 4
        xv = nm.vload(x, 0, lanes=lanes)
        nm.vstore(y, 0, nm.sqrt(xv))

    sqrt_vec(nm.f8[:], nm.f8[:])
    compiled = next(iter(sqrt_vec._compiled_functions.values()))
    ir_proc = lower_procedure(compiled.symbolic_function, backend="c")
    code, _requires_math = CEmitter(
        simd_arch=compiled.simd_arch,
        simd_features=compiled.simd_features,
    ).emit_procedure(ir_proc)
    return code


def _emit_vector_gather_code(simd_arch="scalar", simd_features=()):
    @nm.jit(backend="c", simd_arch=simd_arch, simd_features=simd_features)
    def gather_vec(values, indices, out):
        lanes = 4
        nm.vstore(out, 0, nm.vgather(values, indices, lanes=lanes))

    gather_vec(nm.f8[:], nm.i8[:], nm.f8[:])
    compiled = next(iter(gather_vec._compiled_functions.values()))
    ir_proc = lower_procedure(compiled.symbolic_function, backend="c")
    code, _requires_math = CEmitter(
        simd_arch=compiled.simd_arch,
        simd_features=compiled.simd_features,
    ).emit_procedure(ir_proc)
    return code


def _emit_vector_array_code(simd_arch="scalar", simd_features=()):
    @nm.jit(backend="c", simd_arch=simd_arch, simd_features=simd_features)
    def vector_array_kernel(x, y):
        lanes = 4
        scratch = nm.empty(2, nm.Vector[nm.f8, lanes], name="scratch_vec", allocation="stack")
        scratch[0] = nm.vload(x, 0, lanes=lanes)
        scratch[1] = nm.fma(scratch[0], nm.broadcast(2.0, lanes=lanes), scratch[0])
        scratch[1] += scratch[0]
        nm.vstore(y, 0, scratch[1])

    vector_array_kernel(nm.f8[:], nm.f8[:])
    compiled = next(iter(vector_array_kernel._compiled_functions.values()))
    ir_proc = lower_procedure(compiled.symbolic_function, backend="c")
    code, _requires_math = CEmitter(
        simd_arch=compiled.simd_arch,
        simd_features=compiled.simd_features,
    ).emit_procedure(ir_proc)
    return code


def _emit_typed_vector_binary_code(dtype, op, lanes, simd_arch="scalar", simd_features=()):
    @nm.jit(backend="c", simd_arch=simd_arch, simd_features=simd_features)
    def binary_vec(x, y, z, n):
        for i in nm.range(0, n, lanes):
            xv = nm.vload(x, i, lanes=lanes)
            yv = nm.vload(y, i, lanes=lanes)
            if op == "add":
                zv = xv + yv
            elif op == "mul":
                zv = xv * yv
            else:
                zv = xv - yv
            nm.vstore(z, i, zv)

    binary_vec(dtype[:], dtype[:], dtype[:], nm.i8)
    compiled = next(iter(binary_vec._compiled_functions.values()))
    ir_proc = lower_procedure(compiled.symbolic_function, backend="c")
    code, _requires_math = CEmitter(
        simd_arch=compiled.simd_arch,
        simd_features=compiled.simd_features,
    ).emit_procedure(ir_proc)
    return code


def _cpu_supports(flag):
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return False
    return flag in cpuinfo.read_text().lower().split()


def test_simd_scalar_fallback_c_emit():
    code = _emit_axpy_code()

    assert "#include <immintrin.h>" not in code
    assert "typedef struct { npy_float64 lane[8]; } nm_vec_f64_8;" in code
    assert "nm_vec_fma_f64_8" in code
    assert "nm_vload_f64_8" in code
    assert "nm_vstore_f64_8" in code


def test_simd_avx2_c_emit():
    code = _emit_axpy_code(simd_arch="avx2", simd_features=("fma",))

    assert "#include <immintrin.h>" in code
    assert "typedef struct { __m256d v0; __m256d v1; } nm_vec_f64_8;" in code
    assert "_mm256_fmadd_pd" in code
    assert "_mm256_loadu_pd" in code
    assert "_mm256_storeu_pd" in code


def test_simd_vector_binary_c_emit_uses_helpers():
    code = _emit_vector_add_code(simd_arch="avx2", simd_features=("fma",))

    assert "nm_vec_add_f64_4" in code
    assert "_mm256_add_pd" in code


def test_simd_vector_sqrt_c_emit_uses_helper():
    code = _emit_vector_sqrt_code(simd_arch="avx2", simd_features=("fma",))

    assert "nm_vec_sqrt_f64_4" in code
    assert "_mm256_sqrt_pd" in code
    assert "sqrt(nm_vload" not in code


def test_simd_vector_gather_c_emit_uses_helper():
    code = _emit_vector_gather_code(simd_arch="avx2", simd_features=("fma",))

    assert "nm_vgather_f64_4" in code
    assert "_mm256_i64gather_pd" in code
    assert "_mm256_loadu_si256" in code


def test_simd_vector_stack_array_c_emit():
    code = _emit_vector_array_code(simd_arch="avx2", simd_features=("fma",))

    assert "nm_vec_f64_4 scratch_vec[2];" in code
    assert "(scratch_vec)[0] = nm_vload_f64_4" in code
    assert "(scratch_vec)[1] = nm_vec_fma_f64_4" in code
    assert "(scratch_vec)[1] = nm_vec_add_f64_4" in code


def test_simd_avx2_f64x2_c_emit_uses_sse_helpers():
    @nm.jit(backend="c", simd_arch="avx2", simd_features=("fma",))
    def pair_vec(a, x, y):
        lanes = 2
        av = nm.broadcast(a, lanes=lanes)
        xv = nm.vload(x, 0, lanes=lanes)
        yv = nm.vload(y, 0, lanes=lanes)
        nm.vstore(y, 0, nm.fma(av, xv, yv))

    pair_vec(nm.f8, nm.f8[:], nm.f8[:])
    compiled = next(iter(pair_vec._compiled_functions.values()))
    ir_proc = lower_procedure(compiled.symbolic_function, backend="c")
    code, _requires_math = CEmitter(
        simd_arch=compiled.simd_arch,
        simd_features=compiled.simd_features,
    ).emit_procedure(ir_proc)

    assert "typedef __m128d nm_vec_f64_2;" in code
    assert "nm_vec_fma_f64_2" in code
    assert "_mm_fmadd_pd" in code
    assert "_mm_loadu_pd" in code
    assert "_mm_storeu_pd" in code


def test_simd_avx2_f32_c_emit():
    code = _emit_typed_vector_binary_code(nm.f4, "add", 8, simd_arch="avx2")

    assert "typedef __m256 nm_vec_f32_8;" in code
    assert "_mm256_add_ps" in code
    assert "_mm256_loadu_ps" in code
    assert "_mm256_storeu_ps" in code


def test_simd_avx2_i32_c_emit():
    code = _emit_typed_vector_binary_code(nm.i4, "add", 8, simd_arch="avx2")

    assert "typedef __m256i nm_vec_i32_8;" in code
    assert "_mm256_add_epi32" in code
    assert "_mm256_loadu_si256" in code
    assert "_mm256_storeu_si256" in code


def test_simd_avx2_i64_mul_scalarizes_helper_body():
    code = _emit_typed_vector_binary_code(nm.i8, "mul", 4, simd_arch="avx2")

    assert "typedef __m256i nm_vec_i64_4;" in code
    assert "_mm256_mul" not in code
    assert "out_tmp[i] = a_tmp[i] * b_tmp[i];" in code


def test_simd_avx2_odd_lanes_scalar_fallback_c_emit():
    code = _emit_typed_vector_binary_code(nm.f8, "add", 3, simd_arch="avx2")

    assert "#include <immintrin.h>" not in code
    assert "typedef struct { npy_float64 lane[3]; } nm_vec_f64_3;" in code
    assert "out.lane[i] = a.lane[i] + b.lane[i];" in code


def test_simd_neon_f32_c_emit():
    code = _emit_typed_vector_binary_code(nm.f4, "add", 4, simd_arch="neon")

    assert "#include <arm_neon.h>" in code
    assert "#include <immintrin.h>" not in code
    assert "typedef float32x4_t nm_vec_f32_4;" in code
    assert "vaddq_f32" in code


def test_simd_neon_f64x4_chunks_c_emit():
    code = _emit_typed_vector_binary_code(nm.f8, "add", 4, simd_arch="neon")

    assert "typedef struct { float64x2_t v0; float64x2_t v1; } nm_vec_f64_4;" in code
    assert "vaddq_f64(a.v0, b.v0)" in code
    assert "vaddq_f64(a.v1, b.v1)" in code


def test_simd_scalar_fallback_run_axpy():
    @nm.jit(backend="c", simd_arch="scalar")
    def axpy(a, x, y, n):
        lanes = 4
        av = nm.broadcast(a, lanes=lanes)
        for i in nm.range(0, n, lanes):
            xv = nm.vload(x, i, lanes=lanes)
            yv = nm.vload(y, i, lanes=lanes)
            nm.vstore(y, i, nm.fma(av, xv, yv))

    x = np.arange(8, dtype=np.float64)
    y = np.linspace(1.0, 2.0, 8).astype(np.float64)
    expected = 2.5 * x + y

    axpy(np.float64(2.5), x, y, np.int64(x.size))

    np.testing.assert_allclose(y, expected)


def test_simd_scalar_fallback_run_vector_add_binary():
    @nm.jit(backend="c", simd_arch="scalar")
    def add_vec(x, y, z, n):
        lanes = 4
        for i in nm.range(0, n, lanes):
            xv = nm.vload(x, i, lanes=lanes)
            yv = nm.vload(y, i, lanes=lanes)
            nm.vstore(z, i, xv + yv)

    x = np.arange(8, dtype=np.float64)
    y = np.linspace(1.0, 2.0, 8).astype(np.float64)
    z = np.zeros_like(x)

    add_vec(x, y, z, np.int64(x.size))

    np.testing.assert_allclose(z, x + y)


def test_simd_scalar_fallback_run_f32_vector_add():
    @nm.jit(backend="c", simd_arch="scalar")
    def add_vec(x, y, z, n):
        lanes = 4
        for i in nm.range(0, n, lanes):
            xv = nm.vload(x, i, lanes=lanes)
            yv = nm.vload(y, i, lanes=lanes)
            nm.vstore(z, i, xv + yv)

    x = np.arange(8, dtype=np.float32)
    y = np.linspace(1.0, 2.0, 8).astype(np.float32)
    z = np.zeros_like(x)

    add_vec(x, y, z, np.int64(x.size))

    np.testing.assert_allclose(z, x + y)


def test_simd_scalar_fallback_run_i32_vector_add():
    @nm.jit(backend="c", simd_arch="scalar")
    def add_vec(x, y, z, n):
        lanes = 4
        for i in nm.range(0, n, lanes):
            xv = nm.vload(x, i, lanes=lanes)
            yv = nm.vload(y, i, lanes=lanes)
            nm.vstore(z, i, xv + yv)

    x = np.arange(8, dtype=np.int32)
    y = np.arange(8, dtype=np.int32) * 2
    z = np.zeros_like(x)

    add_vec(x, y, z, np.int64(x.size))

    np.testing.assert_array_equal(z, x + y)


def test_simd_scalar_fallback_run_vector_sqrt():
    @nm.jit(backend="c", simd_arch="scalar")
    def sqrt_vec(x, y):
        lanes = 4
        xv = nm.vload(x, 0, lanes=lanes)
        nm.vstore(y, 0, nm.sqrt(xv))

    x = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float64)
    y = np.zeros_like(x)

    sqrt_vec(x, y)

    np.testing.assert_allclose(y, np.sqrt(x))


def test_simd_scalar_fallback_run_vector_gather():
    @nm.jit(backend="c", simd_arch="scalar")
    def gather_vec(values, indices, out):
        lanes = 4
        nm.vstore(out, 0, nm.vgather(values, indices, lanes=lanes, offset=1))

    values = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)
    indices = np.array([4, 0, 2, 1], dtype=np.int64)
    out = np.zeros(4, dtype=np.float64)

    gather_vec(values, indices, out)

    np.testing.assert_allclose(out, values[indices + 1])


def test_simd_scalar_fallback_run_vector_stack_array():
    @nm.jit(backend="c", simd_arch="scalar")
    def vector_array_kernel(x, y):
        lanes = 4
        scratch = nm.empty(2, nm.Vector[nm.f8, lanes], name="scratch_vec", allocation="stack")
        scratch[0] = nm.vload(x, 0, lanes=lanes)
        scratch[1] = nm.fma(scratch[0], nm.broadcast(2.0, lanes=lanes), scratch[0])
        scratch[1] += scratch[0]
        nm.vstore(y, 0, scratch[1])

    x = np.arange(4, dtype=np.float64)
    y = np.zeros_like(x)

    vector_array_kernel(x, y)

    np.testing.assert_allclose(y, 4.0 * x)


def test_simd_reduce_sum_scalar_fallback_run():
    @nm.jit(backend="c", simd_arch="scalar")
    def sum4(x):
        v = nm.vload(x, 0, lanes=4)
        return nm.reduce_sum(v)

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    np.testing.assert_allclose(sum4(x), x.sum())


@pytest.mark.skipif(not _cpu_supports("avx2"), reason="AVX2 is not available on this CPU")
def test_simd_avx2_run_axpy():
    @nm.jit(
        backend="c",
        simd_arch="avx2",
        simd_features=("fma",),
        compile_flags="-O2 -mavx2 -mfma",
    )
    def axpy(a, x, y, n):
        lanes = 8
        av = nm.broadcast(a, lanes=lanes)
        for i in nm.range(0, n, lanes):
            xv = nm.vload(x, i, lanes=lanes)
            yv = nm.vload(y, i, lanes=lanes)
            nm.vstore(y, i, nm.fma(av, xv, yv))

    x = np.arange(16, dtype=np.float64)
    y = np.linspace(1.0, 2.0, 16).astype(np.float64)
    expected = 3.0 * x + y

    axpy(np.float64(3.0), x, y, np.int64(x.size))

    np.testing.assert_allclose(y, expected)


@pytest.mark.skipif(not _cpu_supports("avx2"), reason="AVX2 is not available on this CPU")
def test_simd_avx2_run_vector_sqrt():
    @nm.jit(
        backend="c",
        simd_arch="avx2",
        simd_features=("fma",),
        compile_flags="-O2 -mavx2 -mfma",
    )
    def sqrt_vec(x, y):
        lanes = 4
        xv = nm.vload(x, 0, lanes=lanes)
        nm.vstore(y, 0, nm.sqrt(xv))

    x = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float64)
    y = np.zeros_like(x)

    sqrt_vec(x, y)

    np.testing.assert_allclose(y, np.sqrt(x))


@pytest.mark.skipif(not _cpu_supports("avx2"), reason="AVX2 is not available on this CPU")
def test_simd_avx2_run_vector_gather():
    @nm.jit(
        backend="c",
        simd_arch="avx2",
        simd_features=("fma",),
        compile_flags="-O2 -mavx2 -mfma",
    )
    def gather_vec(values, indices, out):
        lanes = 4
        nm.vstore(out, 0, nm.vgather(values, indices, lanes=lanes, offset=1))

    values = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)
    indices = np.array([4, 0, 2, 1], dtype=np.int64)
    out = np.zeros(4, dtype=np.float64)

    gather_vec(values, indices, out)

    np.testing.assert_allclose(out, values[indices + 1])


@pytest.mark.skipif(not _cpu_supports("avx2"), reason="AVX2 is not available on this CPU")
def test_simd_avx2_run_vector_stack_array():
    @nm.jit(
        backend="c",
        simd_arch="avx2",
        simd_features=("fma",),
        compile_flags="-O2 -mavx2 -mfma",
    )
    def vector_array_kernel(x, y):
        lanes = 4
        scratch = nm.empty(2, nm.Vector[nm.f8, lanes], name="scratch_vec", allocation="stack")
        scratch[0] = nm.vload(x, 0, lanes=lanes)
        scratch[1] = nm.fma(scratch[0], nm.broadcast(2.0, lanes=lanes), scratch[0])
        scratch[1] += scratch[0]
        nm.vstore(y, 0, scratch[1])

    x = np.arange(4, dtype=np.float64)
    y = np.zeros_like(x)

    vector_array_kernel(x, y)

    np.testing.assert_allclose(y, 4.0 * x)


@pytest.mark.skipif(not _cpu_supports("avx2"), reason="AVX2 is not available on this CPU")
def test_simd_avx2_run_f32_vector_add():
    @nm.jit(
        backend="c",
        simd_arch="avx2",
        compile_flags="-O2 -mavx2",
    )
    def add_vec(x, y, z, n):
        lanes = 8
        for i in nm.range(0, n, lanes):
            xv = nm.vload(x, i, lanes=lanes)
            yv = nm.vload(y, i, lanes=lanes)
            nm.vstore(z, i, xv + yv)

    x = np.arange(16, dtype=np.float32)
    y = np.linspace(1.0, 2.0, 16).astype(np.float32)
    z = np.zeros_like(x)

    add_vec(x, y, z, np.int64(x.size))

    np.testing.assert_allclose(z, x + y)


@pytest.mark.skipif(not _cpu_supports("avx2"), reason="AVX2 is not available on this CPU")
def test_simd_avx2_run_i64_mul_scalarized():
    @nm.jit(
        backend="c",
        simd_arch="avx2",
        compile_flags="-O2 -mavx2",
    )
    def mul_vec(x, y, z, n):
        lanes = 4
        for i in nm.range(0, n, lanes):
            xv = nm.vload(x, i, lanes=lanes)
            yv = nm.vload(y, i, lanes=lanes)
            nm.vstore(z, i, xv * yv)

    x = np.arange(8, dtype=np.int64)
    y = np.arange(8, dtype=np.int64) + 3
    z = np.zeros_like(x)

    mul_vec(x, y, z, np.int64(x.size))

    np.testing.assert_array_equal(z, x * y)


@pytest.mark.skipif(not _cpu_supports("avx2"), reason="AVX2 is not available on this CPU")
def test_simd_avx2_run_f64x2_fma():
    @nm.jit(
        backend="c",
        simd_arch="avx2",
        simd_features=("fma",),
        compile_flags="-O2 -mavx2 -mfma",
    )
    def pair_vec(a, x, y):
        lanes = 2
        av = nm.broadcast(a, lanes=lanes)
        xv = nm.vload(x, 0, lanes=lanes)
        yv = nm.vload(y, 0, lanes=lanes)
        nm.vstore(y, 0, nm.fma(av, xv, yv))

    x = np.array([2.0, 5.0], dtype=np.float64)
    y = np.array([7.0, 11.0], dtype=np.float64)
    expected = 3.0 * x + y

    pair_vec(np.float64(3.0), x, y)

    np.testing.assert_allclose(y, expected)
