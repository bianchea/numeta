from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import numeta as nm
from numeta.compiler import Compiler
from numeta.native_name_registry import native_name_registry


def test_batch_build_creates_callable_bundle_without_execution(tmp_path, backend):
    name = f"batch_aot_{backend}"
    library = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def fill(count: nm.comptime, values, *, value):
        for i in range(count):
            values[i] = value

    events = []
    report = library.build(
        tmp_path,
        specializations={
            "fill": [
                {"args": (2, nm.float64[:]), "kwargs": {"value": float}},
                {"args": [3, nm.float64[:]], "kwargs": {"value": float}},
            ]
        },
        compile_flags="-O1",
        timing_callback=events.append,
    )

    bundle = (tmp_path / f"{name}.numeta").absolute()
    assert isinstance(report, nm.NumetaBuildReport)
    assert report.library_name == name
    assert report.bundle_path == bundle
    assert report.core_library_path.exists()
    assert report.wrapper_path.exists()
    assert report.function_names == ("fill",)
    assert report.specialization_count == 2
    assert report.created_count == 2
    assert report.reused_count == 0
    assert report.for_function("fill") == report.specializations
    with pytest.raises(KeyError, match="no requested specializations"):
        report.for_function("missing")

    assert fill._fast_call == {}
    assert all(specialization.compiled for specialization in report.specializations)
    assert all(not specialization.loaded for specialization in report.specializations)
    assert {event["phase"] for event in events} >= {
        "build.validate",
        "build.specialize",
        "build.total",
        "save.link",
        "save.wrapper",
        "save.total",
    }

    loaded = nm.NumetaLibrary.load(name, tmp_path)
    first = np.zeros(4, dtype=np.float64)
    second = np.zeros(4, dtype=np.float64)
    loaded.fill(2, first, value=1.5)
    loaded.fill(3, second, value=-2.0)
    np.testing.assert_array_equal(first, np.array([1.5, 1.5, 0.0, 0.0]))
    np.testing.assert_array_equal(second, np.array([-2.0, -2.0, -2.0, 0.0]))


def test_batch_build_deduplicates_and_reports_reused_specializations(tmp_path, backend):
    name = f"batch_reuse_{backend}"
    library = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def add_one(values):
        values[:] += 1

    existing = add_one.specialize(nm.float64[:])
    report = library.build(
        tmp_path,
        {"add_one": [(nm.float64[:],), (nm.float64[:],)]},
    )

    assert report.specialization_count == 1
    assert report.created_count == 0
    assert report.reused_count == 1
    assert report.specializations[0].signature_id == existing.signature_id
    assert report.reused_specializations == report.specializations

    loaded = nm.NumetaLibrary.load(name, tmp_path)
    values = np.zeros(3, dtype=np.float64)
    loaded.add_one(values)
    np.testing.assert_array_equal(values, np.ones(3))


def test_batch_build_keeps_unrequested_existing_specializations_callable(tmp_path, backend):
    name = f"batch_existing_roots_{backend}"
    library = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def add_one(values):
        values[:] += 1

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def double(values):
        values[:] *= 2

    add_one.specialize(nm.float64[:])
    report = library.build(tmp_path, {"double": [(nm.float64[:],)]})

    assert report.function_names == ("double",)
    assert add_one.specializations[0].wrapper_path == report.wrapper_path

    loaded = nm.NumetaLibrary.load(name, tmp_path)
    values = np.ones(3, dtype=np.float64)
    loaded.add_one(values)
    loaded.double(values)
    np.testing.assert_array_equal(values, np.full(3, 4.0))


def test_batch_build_compiles_shared_dependency_once(tmp_path, backend, monkeypatch):
    name = f"batch_dependency_{backend}"
    library = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def set_zero(values):
        values[:] = 0

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def reset_and_add(values):
        set_zero(values)
        values[:] += 2

    compiled_names = []
    original_compile_to_obj = Compiler.compile_to_obj

    def record_compile(self, *args, **kwargs):
        compiled_names.append(kwargs["name"])
        return original_compile_to_obj(self, *args, **kwargs)

    monkeypatch.setattr(Compiler, "compile_to_obj", record_compile)
    report = library.build(tmp_path, {"reset_and_add": [(nm.float64[:],)]})

    requested = report.specializations[0]
    dependency = set_zero.specializations[0]
    counts = Counter(compiled_names)
    assert requested.dependency_symbols == (dependency.symbol,)
    assert counts[requested.symbol] == 1
    assert counts[dependency.symbol] == 1

    loaded = nm.NumetaLibrary.load(name, tmp_path)
    values = np.full(3, 7.0)
    loaded.reset_and_add(values)
    np.testing.assert_array_equal(values, np.full(3, 2.0))


def test_batch_build_validates_all_requests_before_constructing(tmp_path, backend):
    name = f"batch_validate_{backend}"
    library = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def fill(values):
        values[:] = 1

    with pytest.raises(KeyError, match="Unknown library function"):
        library.build(
            tmp_path,
            {
                "fill": [(nm.float64[:],)],
                "missing": [(nm.float64[:],)],
            },
        )
    assert fill.specializations == ()

    with pytest.raises(ValueError, match="not supported"):
        library.build(
            tmp_path,
            {"fill": [(nm.float64[:],), (object(),)]},
        )
    assert fill.specializations == ()
    assert not (tmp_path / f"{name}.numeta").exists()


def test_batch_build_rolls_back_new_specializations_on_save_failure(tmp_path, backend, monkeypatch):
    name = f"batch_rollback_{backend}"
    library = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def fill(values):
        values[:] = 1

    reserved_before = native_name_registry.reserved_names.copy()

    def fail_link(*args, **kwargs):
        raise RuntimeError("deliberate link failure")

    monkeypatch.setattr(Compiler, "compile_to_library", fail_link)
    with pytest.raises(RuntimeError, match="deliberate link failure"):
        library.build(tmp_path, {"fill": [(nm.float64[:],)]})

    assert fill.specializations == ()
    assert fill.return_signatures == {}
    assert native_name_registry.reserved_names == reserved_before
    assert not (tmp_path / f"{name}.numeta").exists()
    assert not list(Path(tmp_path).glob(f".{name}.numeta.stage.*"))


def test_batch_build_rolls_back_after_symbolic_construction_failure(tmp_path, backend):
    name = f"batch_symbolic_rollback_{backend}"
    library = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def valid(values):
        values[:] = 1

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def broken(values):
        raise RuntimeError("deliberate symbolic failure")

    reserved_before = native_name_registry.reserved_names.copy()
    with pytest.raises(RuntimeError, match="deliberate symbolic failure"):
        library.build(
            tmp_path,
            {
                "valid": [(nm.float64[:],)],
                "broken": [(nm.float64[:],)],
            },
        )

    assert valid.specializations == ()
    assert broken.specializations == ()
    assert valid.return_signatures == {}
    assert broken.return_signatures == {}
    assert native_name_registry.reserved_names == reserved_before
    assert not (tmp_path / f"{name}.numeta").exists()


def test_batch_build_rejects_invalid_call_schema(tmp_path, backend):
    library = nm.NumetaLibrary(f"batch_schema_{backend}")

    @nm.jit(backend=backend, library=library)
    def fill(values):
        values[:] = 1

    with pytest.raises(TypeError, match="positional argument tuple/list"):
        library.build(tmp_path, {"fill": (nm.float64[:],)})
    with pytest.raises(TypeError, match="must be a tuple or list"):
        library.build(tmp_path, {"fill": [{"args": nm.float64[:]}]})
    with pytest.raises(ValueError, match="unsupported keys"):
        library.build(tmp_path, {"fill": [{"parameters": (nm.float64[:],)}]})
    with pytest.raises(ValueError, match="must not be empty"):
        library.build(tmp_path, {"fill": []})
    assert fill.specializations == ()


def test_batch_build_persists_c_pointer_specialization(tmp_path):
    name = "batch_c_pointer"
    library = nm.NumetaLibrary(name)

    @nm.jit(backend="c", directory=tmp_path / "jit", library=library)
    def set_first(values, value):
        values[0] = value

    report = library.build(
        tmp_path,
        {
            "set_first": [
                (nm.ptr(nm.f8, restrict=True), float),
            ]
        },
    )

    specialization = report.specializations[0]
    assert specialization.signature_id.startswith("sig-v1-")
    assert "restrict" in specialization.source

    loaded = nm.NumetaLibrary.load(name, tmp_path)
    restored = loaded.set_first.get_specialization(specialization.signature_id)
    assert restored.signature == specialization.signature
    assert restored.source == specialization.source
