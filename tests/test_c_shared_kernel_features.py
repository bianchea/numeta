import subprocess
import sysconfig

import numpy as np
import pytest

import numeta as nm
from numeta.c.emitter import CEmitter
from numeta.ir import lower_procedure


def _compiled_proc(nm_function):
    return next(iter(nm_function._compiled_functions.values())).symbolic_function


def _emit(nm_function, *, simd_arch="scalar", simd_features=(), vector_type_style="intrinsic"):
    proc = _compiled_proc(nm_function)
    ir_proc = lower_procedure(proc, backend="c")
    code, _requires_math = CEmitter(
        simd_arch=simd_arch,
        simd_features=simd_features,
        vector_type_style=vector_type_style,
    ).emit_procedure(ir_proc)
    return code


def test_c_simd_vector_arguments_emit_by_value_and_compile(tmp_path):
    vec = nm.Vector[nm.f8, 4]

    @nm.jit(backend="c", simd_arch="avx2", simd_features=("fma",), name="shared_vec_args")
    def shared_vec_args(out, outer_p, outer_prefactor, outer_px, outer_py, outer_pz, outer_coeff):
        acc = nm.fma(outer_p, outer_prefactor, outer_px)
        acc = acc + outer_py + outer_pz + outer_coeff
        nm.vstore(out, 0, acc)

    shared_vec_args(nm.ptr(nm.f8), vec, vec, vec, vec, vec, vec)
    code = _emit(shared_vec_args, simd_arch="avx2", simd_features=("fma",))

    assert "nm_vec_f64_4 outer_p" in code
    assert "nm_vec_f64_4 outer_prefactor" in code
    assert "nm_vec_f64_4 outer_coeff" in code
    assert "outer_p_lanes" not in code
    assert "nm_vload_f64_4(outer_p" not in code

    gcc = "/usr/bin/gcc"
    source = tmp_path / "shared_vec_args.c"
    obj = tmp_path / "shared_vec_args.o"
    source.write_text(code)
    include_dirs = [sysconfig.get_paths()["include"], np.get_include()]
    cmd = [
        gcc,
        "-fopenmp",
        "-fPIC",
        "-mavx2",
        "-mfma",
        "-c",
        str(source),
        "-o",
        str(obj),
        *[f"-I{path}" for path in include_dirs],
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        pytest.skip("/usr/bin/gcc is not available")
    assert result.returncode == 0, result.stderr


def test_c_function_attributes_and_emit_modes():
    @nm.jit(
        backend="c",
        name="hot_hidden_aligned",
        c_attributes=["hot", "aligned(64)", 'visibility("hidden")'],
    )
    def hot_kernel(x):
        pass

    @nm.jit(backend="c", c_attributes=["noinline"])
    def noinline_kernel(x):
        pass

    @nm.jit(backend="c", emit_mode="static_inline", c_attributes=["always_inline"])
    def inline_kernel(x):
        pass

    hot_kernel(nm.i8)
    noinline_kernel(nm.i8)
    inline_kernel(nm.i8)

    hot_code = _emit(hot_kernel)
    noinline_code = _emit(noinline_kernel)
    inline_code = _emit(inline_kernel)

    assert '__attribute__((hot, aligned(64), visibility("hidden")))' in hot_code
    assert "void hot_hidden_aligned(npy_int64 x)" in hot_code
    assert "__attribute__((noinline))" in noinline_code
    assert "static inline __attribute__((always_inline)) void inline_kernel" in inline_code


def test_c_pointer_const_restrict_qualifiers():
    @nm.jit(backend="c")
    def ptr_kernel(workspace, readonly, readonly_restrict):
        workspace[0] = readonly[0] + readonly_restrict[0]

    ptr_kernel(
        nm.ptr(nm.f8, restrict=True),
        nm.ptr(nm.f8, const=True),
        nm.ptr(nm.f8, const=True, restrict=True),
    )
    code = _emit(ptr_kernel)

    assert "npy_float64 * restrict workspace" in code
    assert "const npy_float64 * readonly" in code
    assert "const npy_float64 * restrict readonly_restrict" in code


def test_c_external_helper_call_preserves_vector_arguments():
    vec = nm.Vector[nm.f8, 4]
    helper = nm.external_function(
        "build_R_L8_declared",
        [nm.ptr(nm.f8), nm.Arg(vec, pass_by_value=True), nm.Arg(vec, pass_by_value=True)],
        None,
    )

    @nm.jit(backend="c", simd_arch="avx2", simd_features=("fma",))
    def wrapper(out, inp):
        outer_p = nm.vload(inp, 0, lanes=4)
        outer_px = nm.vload(inp, 4, lanes=4)
        helper(out, outer_p, outer_px)

    wrapper(nm.ptr(nm.f8), nm.ptr(nm.f8))
    code = _emit(wrapper, simd_arch="avx2", simd_features=("fma",))

    assert "void build_R_L8_declared(npy_float64 * a0, nm_vec_f64_4 a1, nm_vec_f64_4 a2);" in code
    assert "build_R_L8_declared(out, nm_vload_f64_4(inp + (0)), nm_vload_f64_4(inp + (4)));" in code
    assert "_lanes" not in code


def test_c_translation_unit_and_header_emission_are_ordered():
    helper_decl = nm.external_function(
        "build_R_L8_unit",
        [nm.ptr(nm.f8), nm.Arg(nm.f8, pass_by_value=True)],
        None,
    )

    @nm.jit(backend="c", name="build_R_L8_unit")
    def build_R(out, x):
        out[0] = x

    @nm.jit(backend="c")
    def eval_2222(out, x):
        helper_decl(out, x)

    @nm.jit(backend="c")
    def eval_1322(out, x):
        helper_decl(out, x + 1.0)

    build_R(nm.ptr(nm.f8), nm.f8)
    eval_2222(nm.ptr(nm.f8), nm.f8)
    eval_1322(nm.ptr(nm.f8), nm.f8)

    emitter = CEmitter()
    unit = emitter.create_translation_unit("tttcc_l4v2_P4_Q4_L8.c")
    unit.add_function(_compiled_proc(build_R), linkage="static")
    unit.add_function(_compiled_proc(eval_2222))
    unit.add_function(_compiled_proc(eval_1322))

    source = unit.emit()
    header = unit.emit_header(guard="TTTCC_L4V2_P4_Q4_L8_H")

    assert source.find("static void build_R_L8_unit") < source.find("void eval_2222")
    assert source.find("void eval_2222") < source.find("void eval_1322")
    assert source.count("static void build_R_L8_unit(npy_float64 * out, npy_float64 x) {") == 1
    assert "build_R_L8_unit(out, x);" in source
    assert "build_R_L8_unit(out, (x + 1.0));" in source
    assert "#ifndef TTTCC_L4V2_P4_Q4_L8_H" in header
    assert "static void build_R_L8_unit(npy_float64 * out, npy_float64 x);" in header


def test_c_gcc_vector_typedef_style():
    vec = nm.Vector[nm.f8, 4]

    @nm.jit(backend="c", simd_arch="avx2")
    def gcc_vec(out, x):
        nm.vstore(out, 0, x)

    gcc_vec(nm.ptr(nm.f8), vec)
    code = _emit(gcc_vec, simd_arch="avx2", vector_type_style="gcc_vector")

    assert "typedef npy_float64 nm_vec_f64_4 __attribute__((vector_size(32)));" in code
