import importlib.util
import sys

import numpy as np
import pytest

import numeta as nm
from numeta.c.emitter import CEmitter
from numeta.ir import lower_procedure


def test_c_nested_slice_call_without_shape_descriptors():
    nm.settings.unset_add_shape_descriptors()
    try:

        @nm.jit(backend="c")
        def callee(a):
            a[:] = 7

        @nm.jit(backend="c")
        def caller(a):
            callee(a[2:5])

        a = np.zeros(8, dtype=np.int64)
        caller(a)

        expected = np.zeros(8, dtype=np.int64)
        expected[2:5] = 7
        np.testing.assert_equal(a, expected)
    finally:
        nm.settings.set_add_shape_descriptors()


def test_c_contiguous_slice_call_uses_direct_pointer_without_temp():
    nm.settings.unset_add_shape_descriptors()
    try:

        @nm.jit(backend="c")
        def callee(a):
            a[:] = 7

        @nm.jit(backend="c")
        def caller(a):
            callee(a[2:5])

        caller(nm.i8[:])
        compiled = next(iter(caller._compiled_functions.values()))
        ir_proc = lower_procedure(compiled.symbolic_function, backend="c")
        code, _requires_math = CEmitter().emit_procedure(ir_proc)

        assert "((a) + (2));" in code
        assert "_nm_tmp" not in code
    finally:
        nm.settings.set_add_shape_descriptors()


def test_c_nested_runtime_slice_call_passes_expected_shape_descriptor():
    @nm.jit(backend="c")
    def fill_with_length(a):
        a[:] = a.shape[0]

    @nm.jit(backend="c")
    def caller(n, a):
        fill_with_length(a[1 : n + 1])

    a = np.zeros(8, dtype=np.int64)
    caller(4, a)

    expected = np.zeros(8, dtype=np.int64)
    expected[1:5] = 4
    np.testing.assert_equal(a, expected)


def test_c_slice_assignment_uses_rhs_relative_indices():
    @nm.jit(backend="c")
    def copy_offset(dst, src):
        dst[4:5] = src[0:1]

    dst = np.zeros(6, dtype=np.float64)
    src = np.array([3.25], dtype=np.float64)
    copy_offset(dst, src)

    expected = np.zeros(6, dtype=np.float64)
    expected[4] = 3.25
    np.testing.assert_allclose(dst, expected, atol=0)


def test_c_slice_assignment_honors_rhs_step():
    @nm.jit(backend="c")
    def copy_stepped(dst, src):
        dst[:] = src[::2]

    dst = np.zeros(3, dtype=np.int64)
    src = np.array([1, 9, 2, 9, 3, 9], dtype=np.int64)
    copy_stepped(dst, src)

    np.testing.assert_equal(dst, np.array([1, 2, 3], dtype=np.int64))


def test_c_array_slice_used_as_scalar_index_errors():
    @nm.jit(backend="c")
    def gather_like(out, indices):
        table = nm.constant(np.arange(8, dtype=np.float64), dtype=nm.f8, name="table")
        out[0] = table[indices[:]]

    out = np.zeros(1, dtype=np.float64)
    indices = np.array([2, 3, 4, 5], dtype=np.int64)

    with pytest.raises(NotImplementedError, match="array-valued slice expressions"):
        gather_like(out, indices)


def test_c_array_assignment_supports_elementwise_gather_index():
    @nm.jit(backend="c")
    def gather_like(out, indices):
        table = nm.constant(
            np.arange(24, dtype=np.float64).reshape(8, 3),
            dtype=nm.f8,
            name="c_elementwise_gather_table",
        )
        out[:] = table[indices[:], 1]

    out = np.zeros(4, dtype=np.float64)
    indices = np.array([2, 3, 4, 5], dtype=np.int64)
    gather_like(out, indices)

    np.testing.assert_equal(out, np.array([7.0, 10.0, 13.0, 16.0]))


def test_c_struct_array_field_call_uses_element_pointer():
    dtype = np.dtype([("coords", np.float64, (3,))], align=True)

    @nm.jit(backend="c")
    def copy3(src, out):
        out[:] = src[:]

    @nm.jit(backend="c")
    def caller(value, out):
        copy3(value["coords"], out)

    value = np.zeros(1, dtype=dtype)
    value[0]["coords"] = [1.25, -2.5, 4.0]
    out = np.zeros(3, dtype=np.float64)
    caller(value[0], out)

    np.testing.assert_equal(out, value[0]["coords"])


def test_c_declared_global_constant_emits_weak_definition():
    table = nm.declare_global_constant(
        (2,),
        np.float64,
        value=np.array([2.0, -1.0], dtype=np.float64),
        name="c_declared_global_weak_table",
        backend="c",
    )

    @nm.jit(backend="c")
    def use_table(out):
        out[0] = table[0]
        out[1] = table[1]

    use_table(nm.f8[:])
    compiled = next(iter(use_table._compiled_functions.values()))
    ir_proc = lower_procedure(compiled.symbolic_function, backend="c")
    code, _requires_math = CEmitter().emit_procedure(ir_proc)

    assert "__attribute__((weak)) const npy_float64 c_declared_global_weak_table[2]" in code
    assert "extern const npy_float64 c_declared_global_weak_table" not in code


def test_c_static_constant_array_has_static_storage_duration():
    @nm.jit(backend="c")
    def use_table(index, out):
        table = nm.constant(
            np.array([2.0, -1.0], dtype=np.float64),
            name="c_static_local_table",
            static=True,
        )
        out[0] = table[index]

    use_table(nm.int64, nm.f8[:])
    compiled = next(iter(use_table._compiled_functions.values()))
    ir_proc = lower_procedure(compiled.symbolic_function, backend="c")
    code, _requires_math = CEmitter().emit_procedure(ir_proc)

    assert "static const npy_float64 c_static_local_table" in code
    assert "[2] = {2.0, -1.0};" in code


def test_c_integer_power_small_exponents_emit_multiplication():
    @nm.jit(backend="c")
    def square(x, y):
        y[:] = x[:] ** 2

    square(nm.f8[:], nm.f8[:])
    compiled = next(iter(square._compiled_functions.values()))
    ir_proc = lower_procedure(compiled.symbolic_function, backend="c")
    code, _requires_math = CEmitter().emit_procedure(ir_proc)

    assert "pow(" not in code
    assert "*" in code


def test_c_negative_step_slice_errors_with_source_location():
    @nm.jit(backend="c")
    def reverse_copy(dst, src):
        dst[:] = src[::-1]  # NEGATIVE_STEP_LINE

    dst = np.zeros(3, dtype=np.int64)
    src = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(NotImplementedError) as exc_info:
        reverse_copy(dst, src)

    error_msg = str(exc_info.value)
    assert "negative" in error_msg.lower() or "step" in error_msg.lower()
    assert "NEGATIVE_STEP_LINE" in error_msg or "test_c_emitter_regressions.py" in error_msg


def test_c_struct_pointer_local_passed_to_nested_helper():
    dtype = np.dtype([("a", np.float64), ("b", np.int64)], align=True)

    @nm.jit(backend="c")
    def fill_struct(s):
        s["a"][:] = 4.5
        s["b"][:] = 6

    @nm.jit(backend="c")
    def caller(raw):
        s = nm.view(raw, dtype)
        fill_struct(s)

    raw = np.zeros(dtype.itemsize, dtype=np.bool_)
    caller(raw)

    viewed = raw.view(dtype)[0]
    assert viewed["a"] == 4.5
    assert viewed["b"] == 6


def test_c_array_pointer_local_used_in_slice_call():
    @nm.jit(backend="c")
    def fill(a):
        a[:] = 8.0

    @nm.jit(backend="c")
    def caller(raw):
        a_view = nm.view(raw, np.float64, shape=(4,))
        fill(a_view[1:3])

    raw = np.zeros(4 * np.dtype(np.float64).itemsize, dtype=np.bool_)
    caller(raw)

    np.testing.assert_allclose(raw.view(np.float64), np.array([0.0, 8.0, 8.0, 0.0]))


def test_c_no_numpy_allocator_local_array_runs():
    nm.settings.unset_numpy_allocator()
    try:

        @nm.jit(backend="c")
        def fill_from_local(out):
            tmp = nm.zeros(3, np.float64)
            tmp[:] = 2.0
            out[:] = tmp[:]

        out = np.zeros(3, dtype=np.float64)
        fill_from_local(out)
        np.testing.assert_allclose(out, np.full(3, 2.0))
    finally:
        nm.settings.set_numpy_allocator()


def test_phasedint_jit_config_backend_env(monkeypatch):
    monkeypatch.setenv("PHASEDINT_NUMETA_BACKEND", "c")
    module_name = "_numeta_test_jit_config_backend_env"
    sys.modules.pop(module_name, None)
    original = {
        "use_numpy_allocator": nm.settings.use_numpy_allocator,
        "reorder_kwargs": nm.settings.reorder_kwargs,
        "add_shape_descriptors": nm.settings.add_shape_descriptors,
        "ignore_fixed_shape_in_nested_calls": nm.settings.ignore_fixed_shape_in_nested_calls,
        "use_c_dispatch": nm.settings.use_c_dispatch,
        "use_c_signature_parser": nm.settings.use_c_signature_parser,
    }
    try:
        spec = importlib.util.spec_from_file_location(module_name, "jit_config.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        assert module.JIT_DEFAULTS["backend"] == "c"
    finally:
        if original["use_numpy_allocator"]:
            nm.settings.set_numpy_allocator()
        else:
            nm.settings.unset_numpy_allocator()
        if original["reorder_kwargs"]:
            nm.settings.set_reorder_kwargs()
        else:
            nm.settings.unset_reorder_kwargs()
        if original["add_shape_descriptors"]:
            nm.settings.set_add_shape_descriptors()
        else:
            nm.settings.unset_add_shape_descriptors()
        nm.settings.ignore_fixed_shape_in_nested_calls = original[
            "ignore_fixed_shape_in_nested_calls"
        ]
        nm.settings.use_c_dispatch = original["use_c_dispatch"]
        nm.settings.use_c_signature_parser = original["use_c_signature_parser"]
        sys.modules.pop(module_name, None)
