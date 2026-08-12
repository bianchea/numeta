from pathlib import Path

import numpy as np
import pytest

import numeta as nm
from numeta.compiler import Compiler
from numeta.exceptions import CompilationError
from numeta.numeta_function import NumetaCompiledFunction
from numeta.pyc_extension import PyCExtension


def test_specialize_exposes_source_and_metadata_without_compiling(tmp_path, backend):
    @nm.jit(backend=backend, directory=tmp_path, compile_flags="-O1")
    def add_one(a):
        a[:] += 1

    specialization = add_one.specialize(nm.float64[:])
    repeated = add_one.specialize(nm.float64[:])

    assert isinstance(specialization, nm.NumetaSpecialization)
    assert specialization.function is add_one
    assert specialization.signature == add_one.get_signature(np.zeros(3, dtype=np.float64))
    assert specialization.signature_id.startswith("sig-v1-")
    assert specialization.symbol == repeated.symbol
    assert specialization.backend == backend
    assert specialization.compile_flags == ("-O1",)
    assert specialization.return_signature == ()
    assert specialization.dependency_symbols == ()
    assert specialization.source_path is None
    assert specialization.object_path is None
    assert specialization.library_path is None
    assert specialization.wrapper_path is None
    assert specialization.compiled is False
    assert specialization.loaded is False

    source = specialization.source.lower()
    declaration = "subroutine" if backend == "fortran" else "void"
    source_suffix = "_src.f90" if backend == "fortran" else "_src.c"
    assert declaration in source
    assert specialization.symbol.lower() in source
    assert list(tmp_path.glob(f"*{source_suffix}")) == []

    assert len(add_one.specializations) == 1
    assert add_one.get_specialization(specialization.signature).symbol == specialization.symbol
    symbolic_signature = add_one.get_signature(nm.float64[:])
    assert add_one.get_specialization(symbolic_signature).symbol == specialization.symbol
    assert add_one.get_specialization(specialization.signature_id).symbol == specialization.symbol


def test_specialize_supports_comptime_keyword_and_return_metadata(tmp_path, backend):
    @nm.jit(backend=backend, directory=tmp_path)
    def scaled_sum(count: nm.comptime, a, *, scale):
        result = nm.float64(0)
        for i in range(count):
            result += a[i] * scale
        return result

    specialization = scaled_sum.specialize(3, nm.float64[:], scale=float)
    runtime_signature = scaled_sum.get_signature(
        3,
        np.arange(4, dtype=np.float64),
        scale=2.0,
    )

    assert specialization.signature == runtime_signature
    assert specialization.signature[0] == 3
    assert specialization.return_signature == ((nm.float64, 0),)


def test_specialization_dependency_symbols(tmp_path, backend):
    @nm.jit(backend=backend, directory=tmp_path)
    def set_zero(a):
        a[:] = 0

    @nm.jit(backend=backend, directory=tmp_path)
    def reset_and_add(a):
        set_zero(a)
        a[:] += 1

    specialization = reset_and_add.specialize(nm.float64[:])
    dependency = set_zero.specializations[0]

    assert specialization.dependency_symbols == (dependency.symbol,)


def test_specialization_compile_reuses_native_artifacts_on_first_call(
    tmp_path, backend, monkeypatch
):
    @nm.jit(backend=backend, directory=tmp_path)
    def add_one(a):
        a[:] += 1

    specialization = add_one.specialize(nm.float64[:]).compile()

    assert specialization.compiled is True
    assert specialization.loaded is False
    assert specialization.source_path is not None
    assert specialization.object_path is not None
    assert specialization.library_path is not None
    assert specialization.wrapper_path is not None
    assert specialization.source_path.read_text() == specialization.source

    def fail_compile(*args, **kwargs):
        raise AssertionError("precompiled specialization should be reused")

    monkeypatch.setattr(NumetaCompiledFunction, "compile", fail_compile)
    monkeypatch.setattr(PyCExtension, "compile", fail_compile)

    array = np.zeros(3, dtype=np.float64)
    add_one(array)

    np.testing.assert_array_equal(array, np.ones(3))
    assert specialization.loaded is True
    assert len(add_one.specializations) == 1


def test_specialization_compile_failure_does_not_report_compiled(tmp_path, backend, monkeypatch):
    @nm.jit(backend=backend, directory=tmp_path)
    def add_one(a):
        a[:] += 1

    specialization = add_one.specialize(nm.float64[:])

    def fail_compile(*args, **kwargs):
        raise CompilationError(
            command=["broken-compiler", "source"],
            cwd=tmp_path,
            stderr="deliberate failure",
        )

    monkeypatch.setattr(NumetaCompiledFunction, "compile", fail_compile)

    with pytest.raises(CompilationError) as exc_info:
        specialization.compile()

    assert exc_info.value.command == ("broken-compiler", "source")
    assert specialization.compiled is False
    assert specialization.object_path is None
    assert specialization.library_path is None
    assert specialization.wrapper_path is None


def test_specialization_inspection_survives_bundle_reload(tmp_path, backend):
    name = f"specialization_bundle_{backend}"
    bundle_dir = tmp_path / "bundles"
    code_dir = tmp_path / "code"
    library = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def fill(a):
        a[:] = 4

    original = fill.specialize(nm.float64[:]).compile()
    original_source = original.source
    library.save(bundle_dir, "")

    loaded = nm.NumetaLibrary.load(name, bundle_dir)
    specialization = loaded.fill.specializations[0]

    assert specialization.signature_id == original.signature_id
    assert specialization.source == original_source
    assert specialization.compiled is True
    assert specialization.loaded is False
    for path in (
        specialization.source_path,
        specialization.object_path,
        specialization.library_path,
        specialization.wrapper_path,
    ):
        assert isinstance(path, Path)
        assert path.exists()
        assert path.is_relative_to(bundle_dir)

    loaded.write_code(code_dir)
    source_suffix = "_src.f90" if backend == "fortran" else "_src.c"
    written_source = code_dir / f"{specialization.symbol}{source_suffix}"
    assert written_source.read_text() == specialization.source

    array = np.zeros(2, dtype=np.float64)
    loaded.fill(array)
    np.testing.assert_array_equal(array, np.full(2, 4.0))
    assert specialization.loaded is True

    with pytest.raises(RuntimeError, match="reattach=True"):
        loaded.fill.specialize(nm.float64[:, :])


def test_get_specialization_rejects_unknown_or_foreign_handle(tmp_path, backend):
    @nm.jit(backend=backend, directory=tmp_path / "first")
    def first(a):
        a[:] = 1

    @nm.jit(backend=backend, directory=tmp_path / "second")
    def second(a):
        a[:] = 2

    first_specialization = first.specialize(nm.int64[:])
    second.specialize(nm.int64[:])

    with pytest.raises(KeyError, match="no specialization with id"):
        first.get_specialization("sig-v1-not-present")
    with pytest.raises(ValueError, match="different Numeta function"):
        second.get_specialization(first_specialization)


def test_specialization_compile_invokes_configured_compiler(tmp_path, backend, monkeypatch):
    @nm.jit(backend=backend, directory=tmp_path)
    def add_one(a):
        a[:] += 1

    specialization = add_one.specialize(nm.float64[:])
    commands = []
    original_run_command = Compiler.run_command

    def record_command(self, command, cwd):
        commands.append(tuple(command))
        return original_run_command(self, command, cwd)

    monkeypatch.setattr(Compiler, "run_command", record_command)
    specialization.compile()

    assert commands
    expected = nm.settings.fortran_compiler if backend == "fortran" else nm.settings.c_compiler
    assert any(command[0].endswith(Path(expected).name) for command in commands)
