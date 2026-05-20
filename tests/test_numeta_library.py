import numpy as np
import os
import subprocess
import sys
from pathlib import Path
import pickle
from functools import partial
from types import MethodType

import pytest
import numeta as nm

from numeta.compiler import Compiler
from numeta.pyc_extension import PyCExtension


def _rank_namer(prefix, *signature):
    return f"{prefix}_{signature[0][2]}"


def _load_saved_entries(pickle_path):
    with open(pickle_path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        return payload["entries"]
    return payload


def test_library_save_and_load(tmp_path, backend):
    lib = nm.NumetaLibrary(f"save_and_load_{backend}")

    @nm.jit(backend=backend, library=lib)
    def add(a):
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    add(array)
    assert all(array == 1)


def test_library_write_code(tmp_path, backend):
    lib = nm.NumetaLibrary(f"write_code_{backend}")

    @nm.jit(backend=backend, library=lib)
    def add(a):
        a[:] += 1

    @nm.jit(backend=backend, library=lib)
    def mul(a):
        a[:] *= 2

    array = np.ones(4, dtype=np.int64)
    lib.add(array)
    lib.mul(array)

    lib.write_code(tmp_path)

    compiled_names = []
    for nm_function in lib._entries.values():
        compiled_names.extend(
            [compiled.func_name for compiled in nm_function._compiled_functions.values()]
        )

    for name in compiled_names:
        if backend == "fortran":
            src = Path(tmp_path) / f"{name}_src.f90"
            assert src.exists()
            code = src.read_text().lower()
            assert f"subroutine {name}" in code
        elif backend == "c":
            src = Path(tmp_path) / f"{name}_src.c"
            assert src.exists()
            code = src.read_text().lower()
            assert f"void {name}" in code
        else:
            raise ValueError(f"Unsupported backend: {backend}")


def test_library_write_code_with_global_constant(tmp_path, backend):
    lib = nm.NumetaLibrary(f"write_code_global_{backend}")
    global_name = f"global_constant_var_write_code_{backend}"

    nm.declare_global_constant(
        (2, 1),
        np.float64,
        value=np.array([2.0, -1.0]),
        name=global_name,
        backend=backend,
        library=lib,
    )

    lib.write_code(tmp_path)

    namespace_name = f"{global_name}_namespace"
    if backend == "fortran":
        src = Path(tmp_path) / f"{namespace_name}_src.f90"
    elif backend == "c":
        src = Path(tmp_path) / f"{namespace_name}_src.c"
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    assert src.exists()
    code = src.read_text().lower()
    assert global_name in code


def test_library_save_load_write_code_with_struct_dtype(tmp_path, backend):
    name = f"struct_symbolic_cache_{backend}"
    lib = nm.NumetaLibrary(name)
    dtype = np.dtype([("x", np.int32), ("y", np.float64, (2,))], align=True)

    @nm.jit(backend=backend, library=lib)
    def fill_struct(a):
        a[0]["x"] = 5
        a[0]["y"][1] = -2.0

    array = np.zeros(1, dtype=dtype)
    lib.fill_struct(array)
    lib.save(tmp_path, "")

    lib_loaded = nm.NumetaLibrary.load(name, tmp_path)
    compiled = next(iter(lib_loaded.fill_struct._compiled_functions.values()))
    assert hasattr(compiled, "symbolic_function")

    code_dir = tmp_path / "struct_code"
    lib_loaded.write_code(code_dir)
    suffix = "_src.f90" if backend == "fortran" else "_src.c"
    assert (code_dir / f"{compiled.func_name}{suffix}").exists()

    lib_loaded.save(tmp_path, "")


def test_library_save_and_load_with_dep(tmp_path, backend):
    lib = nm.NumetaLibrary(f"save_and_load_with_dep_{backend}")

    @nm.jit(backend=backend, library=lib)
    def set_zero(a):
        a[:] = 0

    @nm.jit(backend=backend, library=lib)
    def add(a):
        set_zero(a)
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    lib.add(array)
    assert all(array == 1)
    lib.save(tmp_path, "")
    assert len(lib._entries) == 2

    lib_loaded = nm.NumetaLibrary.load(f"save_and_load_with_dep_{backend}", tmp_path)
    assert len(lib_loaded._entries) == 2

    array = np.zeros(4, dtype=np.int64)
    lib_loaded.add(array)
    assert all(array == 1)


def test_library_save_and_load_with_dep_2(tmp_path, backend):
    lib = nm.NumetaLibrary(f"save_and_load_with_dep_2_{backend}")

    @nm.jit(backend=backend)
    def set_zero(a):
        a[:] = 0

    @nm.jit(backend=backend, library=lib)
    def add(a):
        set_zero(a)
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    lib.add(array)
    assert all(array == 1)
    assert len(lib._entries) == 1
    lib.save(tmp_path, "")

    lib_loaded = nm.NumetaLibrary.load(f"save_and_load_with_dep_2_{backend}", tmp_path)
    assert len(lib_loaded._entries) == 1

    array = np.zeros(4, dtype=np.int64)
    lib_loaded.add(array)
    assert all(array == 1)


def test_library_save_and_load_use_dep(tmp_path, backend):
    lib = nm.NumetaLibrary(f"save_and_load_use_dep_{backend}")

    @nm.jit(backend=backend, library=lib)
    def set_zero(a):
        a[:] = 0

    @nm.jit(backend=backend, library=lib)
    def add(a):
        set_zero(a)
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    lib.add(array)
    assert all(array == 1)
    lib.save(tmp_path, "")

    lib_loaded = nm.NumetaLibrary.load(f"save_and_load_use_dep_{backend}", tmp_path)

    @nm.jit(backend=backend)
    def minus(a):
        lib_loaded.set_zero(a)
        a[:] -= 1

    array = np.zeros(4, dtype=np.int64)
    minus(array)
    assert all(array == -1)


def test_library_global_variable_dep(tmp_path, backend):
    lib = nm.NumetaLibrary(f"global_variable_dep_{backend}")

    global_constant_var = nm.declare_global_constant(
        (2, 1), np.float64, value=np.array([2.0, -1.0]), name="global_constant_var", backend=backend
    )

    @nm.jit(backend=backend, library=lib)
    def set(var):
        var[0] = global_constant_var[0, 0]
        var[1] = global_constant_var[1, 0]

    a = np.empty(2, dtype=np.float64)
    lib.set(a)
    np.testing.assert_allclose(a, np.array([2.0, -1.0]))

    lib.save(tmp_path, "")

    lib_loaded = nm.NumetaLibrary.load(f"global_variable_dep_{backend}", tmp_path)

    a = np.empty(2, dtype=np.float64)
    lib_loaded.set(a)
    np.testing.assert_allclose(a, np.array([2.0, -1.0]))


def test_library_save_load_preserves_multiple_global_entries(tmp_path, backend):
    name = f"multiple_global_entries_{backend}"
    lib = nm.NumetaLibrary(name)
    first_name = f"{name}_first"
    second_name = f"{name}_second"

    nm.declare_global_constant(
        (1,),
        np.float64,
        value=np.array([1.0]),
        name=first_name,
        backend=backend,
        library=lib,
    )
    nm.declare_global_constant(
        (1,),
        np.float64,
        value=np.array([2.0]),
        name=second_name,
        backend=backend,
        library=lib,
    )

    expected_keys = {f"{first_name}_namespace", f"{second_name}_namespace"}
    assert set(lib._global_entries) == expected_keys

    lib.save(tmp_path, "")
    lib_loaded = nm.NumetaLibrary.load(name, tmp_path)
    assert set(lib_loaded._global_entries) == expected_keys

    code_dir = tmp_path / "global_code"
    lib_loaded.write_code(code_dir)
    suffix = "_src.f90" if backend == "fortran" else "_src.c"
    for key in expected_keys:
        assert (code_dir / f"{key}{suffix}").exists()


def test_library_name_conflict(tmp_path, backend):

    lib = nm.NumetaLibrary(f"name_conflict_{backend}")

    @nm.jit(backend=backend, library=lib)
    def set_zero(a):
        a[:] = 0

    @nm.jit(backend=backend, library=lib)
    def add(a):
        set_zero(a)
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    lib.add(array)
    assert all(array == 1)
    lib.save(tmp_path, "")

    lib_loaded_1 = nm.NumetaLibrary.load(f"name_conflict_{backend}", tmp_path)
    try:
        lib_loaded_2 = nm.NumetaLibrary.load(f"name_conflict_{backend}", tmp_path)
    except ValueError:
        pass

    array = np.zeros(4, dtype=np.int64)
    lib_loaded_1.add(array)
    assert all(array == 1)


def test_library_external_dep(tmp_path, backend):
    import ctypes.util
    import os

    lib = nm.NumetaLibrary(f"external_dep_{backend}")

    if ctypes.util.find_library("blas") is None:
        pytest.skip("BLAS library not found")
    blas = nm.ExternalLibraryWrapper("blas")
    blas.add_method(
        "dgemm",
        [
            nm.char,
            nm.char,
            nm.i8,
            nm.i8,
            nm.i8,
            nm.f8,
            nm.f8[None],
            nm.i8,
            nm.f8[None],
            nm.i8,
            nm.f8,
            nm.f8[None],
            nm.i8,
        ],
        None,
        bind_c=False,
    )

    n = 100

    @nm.jit(backend=backend, library=lib)
    def matmul(a, b, c):
        blas.dgemm(
            "N",
            "N",
            b.shape[0],
            a.shape[1],
            c.shape[1],
            1.0,
            b,
            b.shape[0],
            a,
            a.shape[0],
            0.0,
            c,
            c.shape[0],
        )

    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    c = np.zeros((n, n))

    lib.matmul(a, b, c)

    np.testing.assert_allclose(c, np.dot(a, b))
    lib.save(tmp_path)

    lib_loaded = nm.NumetaLibrary.load(f"external_dep_{backend}", tmp_path)

    lib_loaded.matmul(a, b, c)
    np.testing.assert_allclose(c, np.dot(a, b))


def test_library_public_api(backend):
    lib = nm.NumetaLibrary(f"public_api_{backend}")

    @nm.jit(backend=backend)
    def add(a):
        a[:] += 1

    registered = lib.register(add)
    assert registered is add
    assert "add" in lib
    assert lib["add"] is add
    assert lib.list_functions() == ["add"]
    assert len(lib) == 1
    assert list(lib)[0] is add

    lib.remove("add")
    assert "add" not in lib
    assert len(lib) == 0


def test_library_public_introspection_api(backend):
    lib = nm.NumetaLibrary(f"public_introspection_{backend}")

    @nm.jit(backend=backend, library=lib)
    def fill(scale: nm.comptime, out):
        out[:] = scale

    array_type = nm.float64[3]
    fill(1, array_type)
    fill(2, array_type)

    first_signature = lib.signature_for_call("fill", 1, array_type)
    second_signature = lib.signature_for_call("fill", 2, array_type)

    assert lib.signatures("fill") == [first_signature, second_signature]

    symbols = lib.compiled_symbols("fill")
    assert set(symbols) == {first_signature, second_signature}
    assert all(isinstance(symbol, str) for symbol in symbols.values())

    compiled = lib.compiled_function("fill", first_signature)
    assert compiled.func_name == symbols[first_signature]

    with pytest.raises(KeyError, match="no compiled specialization"):
        lib.compiled_function("fill", (("missing", np.float64),))


def _make_selected_replace_case(name, backend):
    lib = nm.NumetaLibrary(f"{name}_{backend}")

    @nm.jit(backend=backend, library=lib)
    def fill(scale: nm.comptime, out):
        out[:] = scale

    array_type = nm.float64[3]
    fill(1, array_type)
    fill(2, array_type)

    old_func = lib["fill"]
    first_signature = fill.get_signature(1, array_type)
    second_signature = fill.get_signature(2, array_type)
    old_symbols = {
        signature: compiled.func_name
        for signature, compiled in old_func._compiled_functions.items()
    }

    @nm.jit(backend=backend)
    def replacement(scale: nm.comptime, out):
        out[:] = scale + 10

    return lib, old_func, replacement, first_signature, second_signature, old_symbols


def test_library_replace_selected_signature_preserves_unselected(backend):
    lib, old_func, replacement, first_signature, second_signature, old_symbols = (
        _make_selected_replace_case("replace_selected_preserve", backend)
    )

    construct_calls = []
    original_construct = replacement.construct_compiled_target

    def counted_construct(self, signature, *args, **kwargs):
        construct_calls.append((signature, kwargs.get("forced_name")))
        return original_construct(signature, *args, **kwargs)

    replacement.construct_compiled_target = MethodType(counted_construct, replacement)

    replaced = lib.replace(
        "fill",
        replacement,
        signatures=[first_signature],
        compile_now=False,
    )

    assert replaced is replacement
    assert construct_calls == [(first_signature, old_symbols[first_signature])]
    assert set(replaced._compiled_functions) == {first_signature, second_signature}
    assert replaced._compiled_functions[first_signature].func_name == old_symbols[first_signature]
    assert (
        replaced._compiled_functions[first_signature]
        is not old_func._compiled_functions[first_signature]
    )
    assert (
        replaced._compiled_functions[second_signature]
        is old_func._compiled_functions[second_signature]
    )


def test_library_replace_selected_signature_can_invalidate_unselected(backend):
    lib, _old_func, replacement, first_signature, second_signature, old_symbols = (
        _make_selected_replace_case("replace_selected_invalidate", backend)
    )

    replaced = lib.replace(
        "fill",
        replacement,
        signatures=[first_signature],
        compile_now=False,
        non_selected="invalidate",
    )

    assert set(replaced._compiled_functions) == {first_signature}
    assert second_signature not in replaced._compiled_functions

    lib.fill(2, nm.float64[3])

    assert second_signature in replaced._compiled_functions
    assert replaced._compiled_functions[second_signature].func_name != old_symbols[second_signature]


def test_library_replace_selected_signature_error_policy_rejects_subset(backend):
    lib, old_func, replacement, first_signature, _second_signature, _old_symbols = (
        _make_selected_replace_case("replace_selected_error", backend)
    )

    with pytest.raises(ValueError, match="non_selected='error'"):
        lib.replace(
            "fill",
            replacement,
            signatures=[first_signature],
            compile_now=False,
            non_selected="error",
        )

    assert lib["fill"] is old_func
    assert replacement._compiled_functions == {}


def test_library_register_rejects_reserved_name(backend):
    lib = nm.NumetaLibrary(f"public_api_reserved_{backend}")

    with pytest.raises(ValueError, match="reserved"):

        @nm.jit(backend=backend, library=lib)
        def register(a):
            a[:] += 1


def test_library_load_with_existing_reserved_name(tmp_path, backend):
    lib = nm.NumetaLibrary(f"reserved_name_lib_{backend}")

    @nm.jit(backend=backend, library=lib)
    def add(a):
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    lib.add(array)
    lib.save(tmp_path, "")

    lib_loaded = nm.NumetaLibrary.load(f"reserved_name_lib_{backend}", tmp_path)
    lib_loaded.add(array)
    assert all(array == 2)


def test_library_load_can_extend_existing_function(tmp_path, backend):
    lib = nm.NumetaLibrary(f"extend_existing_{backend}")

    @nm.jit(backend=backend, library=lib)
    def add(a):
        a[:] += 1

    vector = np.zeros(4, dtype=np.int64)
    lib.add(vector)
    lib.save(tmp_path, "")

    lib_loaded = nm.NumetaLibrary.load(f"extend_existing_{backend}", tmp_path)

    with pytest.raises(ValueError, match="reattach=True"):

        @nm.jit(backend=backend, library=lib_loaded)
        def add(a):
            a[:] += 1

    @nm.jit(backend=backend, library=lib_loaded, reattach=True)
    def add(a):
        a[:] += 1

    vector = np.zeros(4, dtype=np.int64)
    lib_loaded.add(vector)
    np.testing.assert_array_equal(vector, np.ones(4, dtype=np.int64))

    matrix = np.zeros((2, 2), dtype=np.int64)
    lib_loaded.add(matrix)
    np.testing.assert_array_equal(matrix, np.ones((2, 2), dtype=np.int64))
    assert len(lib_loaded.add._compiled_functions) == 2


def test_library_load_preserves_picklable_namer_for_new_signatures(tmp_path, backend):
    name = f"picklable_namer_{backend}"
    prefix = f"saved_picklable_namer_{backend}"
    lib = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, library=lib, namer=partial(_rank_namer, prefix))
    def add(a):
        a[:] += 1

    vector = np.zeros(4, dtype=np.int64)
    lib.add(vector)
    lib.save(tmp_path, "")

    lib_loaded = nm.NumetaLibrary.load(name, tmp_path)

    @nm.jit(backend=backend, library=lib_loaded, reattach=True)
    def add(a):
        a[:] += 1

    matrix = np.zeros((2, 2), dtype=np.int64)
    lib_loaded.add(matrix)

    compiled_names = {
        compiled.func_name for compiled in lib_loaded.add._compiled_functions.values()
    }
    assert f"{prefix}_2" in compiled_names


def test_library_reattach_can_restore_unpickleable_namer(tmp_path, backend):
    name = f"unpickleable_namer_{backend}"
    prefix = f"saved_lambda_namer_{backend}"
    lib = nm.NumetaLibrary(name)

    def namer(*signature):
        return f"{prefix}_{signature[0][2]}"

    @nm.jit(backend=backend, library=lib, namer=namer)
    def add(a):
        a[:] += 1

    vector = np.zeros(4, dtype=np.int64)
    lib.add(vector)
    with pytest.warns(RuntimeWarning, match="not pickleable"):
        lib.save(tmp_path, "")

    lib_loaded = nm.NumetaLibrary.load(name, tmp_path)

    @nm.jit(backend=backend, library=lib_loaded, reattach=True, namer=namer)
    def add(a):
        a[:] += 1

    matrix = np.zeros((2, 2), dtype=np.int64)
    lib_loaded.add(matrix)

    compiled_names = {
        compiled.func_name for compiled in lib_loaded.add._compiled_functions.values()
    }
    assert f"{prefix}_2" in compiled_names


def test_library_save_load_save_keeps_wrapper_specs_unique(tmp_path, backend):
    name = f"unique_wrapper_specs_{backend}"
    lib = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, library=lib)
    def add(a):
        a[:] += 1

    @nm.jit(backend=backend, library=lib)
    def mul(a):
        a[:] *= 2

    vector = np.ones(4, dtype=np.int64)
    lib.add(vector)
    lib.mul(vector)
    lib.save(tmp_path, "")

    lib_loaded = nm.NumetaLibrary.load(name, tmp_path)

    aggregate_extensions = [func._library_pyc_extension for func in lib_loaded]
    assert all(extension is not None for extension in aggregate_extensions)
    assert len({id(extension) for extension in aggregate_extensions}) == 1
    assert all(func._pyc_extensions == {} for func in lib_loaded)

    lib_loaded.save(tmp_path, "")

    saved_functions = _load_saved_entries(Path(tmp_path) / f"{name}.pkl")

    aggregate_extension = saved_functions[0]._library_pyc_extension
    wrapper_names = [wrapper_spec[0] for wrapper_spec in aggregate_extension.functions]
    assert len(wrapper_names) == len(set(wrapper_names))
    assert len(wrapper_names) == 2


def test_library_save_loaded_library_keeps_all_core_objects(tmp_path, backend):
    name = f"save_loaded_core_objects_{backend}"
    lib = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, library=lib)
    def add(a):
        a[:] += 1

    @nm.jit(backend=backend, library=lib)
    def mul(a):
        a[:] *= 2

    @nm.jit(backend=backend, library=lib)
    def sub(a):
        a[:] -= 3

    vector = np.ones(4, dtype=np.int64)
    lib.add(vector)
    lib.mul(vector)
    lib.sub(vector)
    lib.save(tmp_path, "")

    lib_loaded = nm.NumetaLibrary.load(name, tmp_path)
    lib_loaded.save(tmp_path, "")

    vector = np.ones(4, dtype=np.int64)
    lib_loaded.add(vector)
    lib_loaded.mul(vector)
    lib_loaded.sub(vector)
    np.testing.assert_array_equal(vector, np.ones(4, dtype=np.int64))


def test_library_save_persists_aggregate_wrapper_cache_info(tmp_path, backend):
    name = f"wrapper_cache_info_{backend}"
    lib = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, library=lib)
    def add(a):
        a[:] += 1

    vector = np.zeros(4, dtype=np.int64)
    lib.add(vector)
    lib.save(tmp_path, "")

    wrapper_path = Path(tmp_path) / f"lib{name}{PyCExtension.SUFFIX}.so"
    assert wrapper_path.exists()

    saved_functions = _load_saved_entries(Path(tmp_path) / f"{name}.pkl")

    extension = saved_functions[0]._library_pyc_extension
    assert extension.cache_info is not None
    assert extension.cache_info["wrapper_name"] == f"{name}{PyCExtension.SUFFIX}"
    assert extension.cache_info["backend"] == backend
    assert "functions" not in extension.cache_info


def test_library_load_reuses_compatible_aggregate_wrapper(tmp_path, backend, monkeypatch):
    name = f"reuse_wrapper_{backend}"
    lib = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, library=lib)
    def add(a):
        a[:] += 1

    vector = np.zeros(4, dtype=np.int64)
    lib.add(vector)
    lib.save(tmp_path, "")

    def fail_compile(*args, **kwargs):
        raise AssertionError("wrapper should have been reused")

    monkeypatch.setattr(PyCExtension, "compile", fail_compile)

    lib_loaded = nm.NumetaLibrary.load(name, tmp_path)
    assert lib_loaded.add._library_pyc_extension.lib_path == Path(tmp_path).absolute() / (
        f"lib{name}{PyCExtension.SUFFIX}.so"
    )

    vector = np.zeros(4, dtype=np.int64)
    lib_loaded.add(vector)
    np.testing.assert_array_equal(vector, np.ones(4, dtype=np.int64))


def test_library_load_recompiles_wrapper_on_cache_info_mismatch(tmp_path, backend, monkeypatch):
    name = f"reuse_wrapper_mismatch_{backend}"
    lib = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, library=lib)
    def add(a):
        a[:] += 1

    vector = np.zeros(4, dtype=np.int64)
    lib.add(vector)
    lib.save(tmp_path, "")

    original_build_cache_info = PyCExtension.build_cache_info
    original_compile = PyCExtension.compile
    calls = []

    def mismatched_cache_info(self, *args, **kwargs):
        cache_info = original_build_cache_info(self, *args, **kwargs)
        cache_info["python_soabi"] = "different"
        return cache_info

    def record_compile(self, *args, **kwargs):
        calls.append(self.name)
        return original_compile(self, *args, **kwargs)

    monkeypatch.setattr(PyCExtension, "build_cache_info", mismatched_cache_info)
    monkeypatch.setattr(PyCExtension, "compile", record_compile)

    lib_loaded = nm.NumetaLibrary.load(name, tmp_path)
    assert lib_loaded.add._library_pyc_extension.lib_path is None

    vector = np.zeros(4, dtype=np.int64)
    lib_loaded.add(vector)
    np.testing.assert_array_equal(vector, np.ones(4, dtype=np.int64))
    assert calls == [f"{name}{PyCExtension.SUFFIX}"]


def test_library_load_normalizes_legacy_duplicate_aggregate_wrappers(tmp_path, backend):
    from numeta.numeta_function import NumetaFunction

    name = f"legacy_duplicate_wrappers_{backend}"
    lib = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, library=lib)
    def add(a):
        a[:] += 1

    vector = np.zeros(4, dtype=np.int64)
    lib.add(vector)
    lib.save(tmp_path, "")

    pickle_path = Path(tmp_path) / f"{name}.pkl"
    saved_functions = _load_saved_entries(pickle_path)

    extension = saved_functions[0]._library_pyc_extension
    extension.functions = extension.functions + extension.functions
    legacy_state = saved_functions[0].__dict__.copy()
    legacy_state["_pyc_extensions"] = {
        signature: extension for signature in legacy_state["_compiled_functions"]
    }
    legacy_state.pop("_library_pyc_extension")
    legacy_state.pop("_wrapper_specs")

    class LegacyNumetaFunction:
        def __reduce__(self):
            return (NumetaFunction.__new__, (NumetaFunction,), legacy_state)

    with open(pickle_path, "wb") as handle:
        pickle.dump([LegacyNumetaFunction()], handle)

    lib_loaded = nm.NumetaLibrary.load(name, tmp_path)
    wrapper_names = [
        wrapper_spec[0] for wrapper_spec in lib_loaded.add._library_pyc_extension.functions
    ]
    assert len(wrapper_names) == len(set(wrapper_names)) == 1

    vector = np.zeros(4, dtype=np.int64)
    lib_loaded.add(vector)
    np.testing.assert_array_equal(vector, np.ones(4, dtype=np.int64))


def test_library_save_and_load_openmp_prange(tmp_path):
    lib = nm.NumetaLibrary("openmp_prange_cache")

    @nm.jit(
        backend="fortran",
        library=lib,
        compile_flags="-O3 -fopenmp",
    )
    def add_one(out, x):
        for i in nm.prange(x.shape[0], shared=[out, x, x.shape[0].variable]):
            out[i] = x[i] + 1.0

    x = np.asfortranarray(np.array([1.0, 2.0, 3.0]))
    out = np.zeros_like(x, order="F")
    lib.add_one(out, x)
    np.testing.assert_array_equal(out, np.array([2.0, 3.0, 4.0]))

    lib.save(tmp_path, "-O3 -fopenmp")
    lib_loaded = nm.NumetaLibrary.load("openmp_prange_cache", tmp_path)

    out = np.zeros_like(x, order="F")
    lib_loaded.add_one(out, x)
    np.testing.assert_array_equal(out, np.array([2.0, 3.0, 4.0]))


def test_library_load_openmp_prange_in_fresh_process(tmp_path):
    lib = nm.NumetaLibrary("openmp_fresh_process_cache")

    @nm.jit(
        backend="fortran",
        library=lib,
        compile_flags="-O3 -fopenmp",
    )
    def add_one(out, x):
        for i in nm.prange(x.shape[0], shared=[out, x, x.shape[0].variable]):
            out[i] = x[i] + 1.0

    x = np.asfortranarray(np.array([1.0, 2.0, 3.0]))
    out = np.zeros_like(x, order="F")
    lib.add_one(out, x)
    lib.save(tmp_path, "-O3 -fopenmp")

    script = f"""
import numpy as np
import numeta as nm
lib = nm.NumetaLibrary.load('openmp_fresh_process_cache', {str(tmp_path)!r})
x = np.asfortranarray(np.array([1.0, 2.0, 3.0]))
out = np.zeros_like(x, order='F')
lib.add_one(out, x)
np.testing.assert_array_equal(out, np.array([2.0, 3.0, 4.0]))
"""
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_library_save_is_atomic_on_failure(tmp_path, backend, monkeypatch):
    lib = nm.NumetaLibrary(f"atomic_save_{backend}")

    @nm.jit(backend=backend, library=lib)
    def add(a):
        a[:] += 1

    vector = np.zeros(4, dtype=np.int64)
    lib.add(vector)
    lib.save(tmp_path, "")

    pickle_path = Path(tmp_path) / f"atomic_save_{backend}.pkl"
    original_bytes = pickle_path.read_bytes()

    def fail_compile_to_library(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(Compiler, "compile_to_library", fail_compile_to_library)

    with pytest.raises(RuntimeError, match="boom"):
        lib.save(tmp_path, "")

    assert pickle_path.read_bytes() == original_bytes
    assert not list(Path(tmp_path).glob(f".atomic_save_{backend}.*.pkl.tmp"))


def test_library_save_skips_extra_runtime_attributes(tmp_path, backend):
    lib = nm.NumetaLibrary(f"stable_state_{backend}")

    @nm.jit(backend=backend, library=lib)
    def add(a):
        a[:] += 1

    add.extra_runtime_state = {"temporary": True}

    vector = np.zeros(4, dtype=np.int64)
    lib.add(vector)
    lib.save(tmp_path, "")

    lib_loaded = nm.NumetaLibrary.load(f"stable_state_{backend}", tmp_path)
    assert not hasattr(lib_loaded.add, "extra_runtime_state")


def test_library_safe_load_treats_corrupt_pickle_as_cache_miss(tmp_path, backend):
    name = f"corrupt_cache_{backend}"
    pickle_path = Path(tmp_path) / f"{name}.pkl"
    pickle_path.write_bytes(b"not a pickle")

    with pytest.raises(pickle.UnpicklingError):
        nm.NumetaLibrary.load(name, tmp_path)

    with pytest.warns(RuntimeWarning, match="cache miss"):
        lib = nm.NumetaLibrary.load(name, tmp_path, safe=True)

    assert isinstance(lib, nm.NumetaLibrary)
    assert len(lib) == 0


def test_library_reserved_suffix_rejected(tmp_path, backend):
    reserved_name = f"bad{PyCExtension.SUFFIX}"
    with pytest.raises(ValueError, match="reserved"):
        nm.NumetaLibrary(reserved_name)
    with pytest.raises(ValueError, match="reserved"):
        nm.NumetaLibrary.load(reserved_name, tmp_path)


def test_library_wrapper_module_collision(backend):
    wrapper_module = f"collision{PyCExtension.SUFFIX}"
    sys.modules[wrapper_module] = object()
    try:
        with pytest.raises(ValueError, match="wrapper module"):
            nm.NumetaLibrary("collision")
    finally:
        sys.modules.pop(wrapper_module, None)
