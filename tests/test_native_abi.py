import json
from pathlib import Path

import numpy as np
import pytest

import numeta as nm
from numeta.exceptions import CorruptLibraryError, IncompatibleLibraryError
from numeta.native_abi import codegen_abi_settings, numpy_c_abi_version
from numeta.pyc_extension import PyCExtension, WRAPPER_CACHE_FORMAT_VERSION


@pytest.fixture
def restore_codegen_abi_settings():
    original = codegen_abi_settings()
    yield
    if original["add_shape_descriptors"]:
        nm.settings.set_add_shape_descriptors()
    else:
        nm.settings.unset_add_shape_descriptors()
    nm.settings.ignore_fixed_shape_in_nested_calls = original["ignore_fixed_shape_in_nested_calls"]
    if original["use_numpy_allocator"]:
        nm.settings.set_numpy_allocator()
    else:
        nm.settings.unset_numpy_allocator()
    if original["reorder_kwargs"]:
        nm.settings.set_reorder_kwargs()
    else:
        nm.settings.unset_reorder_kwargs()


def _set_codegen_setting(name, value):
    if name == "add_shape_descriptors":
        setter = (
            nm.settings.set_add_shape_descriptors
            if value
            else nm.settings.unset_add_shape_descriptors
        )
        setter()
    elif name == "ignore_fixed_shape_in_nested_calls":
        nm.settings.ignore_fixed_shape_in_nested_calls = value
    elif name == "use_numpy_allocator":
        setter = nm.settings.set_numpy_allocator if value else nm.settings.unset_numpy_allocator
        setter()
    elif name == "reorder_kwargs":
        setter = nm.settings.set_reorder_kwargs if value else nm.settings.unset_reorder_kwargs
        setter()
    else:
        raise AssertionError(f"Unknown test setting {name!r}")


def _bundle_manifest(directory, name):
    return json.loads((Path(directory) / f"{name}.numeta" / "manifest.json").read_text())


def test_wrapper_cache_records_abi_and_build_provenance():
    wrapper = PyCExtension("cache_metadata", functions=[], do_checks=False)
    compiler = {"executable": "/toolchain/gcc", "version": "gcc test"}

    info = wrapper.build_cache_info(
        "-O2 -fno-strict-aliasing",
        backend="c",
        compiler_identity=compiler,
        simd_arch="avx2",
        simd_features=("fma",),
    )

    assert info["format_version"] == WRAPPER_CACHE_FORMAT_VERSION == 2
    assert info["numpy_c_abi"] == numpy_c_abi_version()
    assert info["compiler"] == compiler
    assert info["compile_flags"] == ["-O2", "-fno-strict-aliasing"]
    assert info["backend"] == "c"
    assert info["simd_arch"] == "avx2"
    assert info["simd_features"] == ["fma"]
    assert info["do_checks"] is False
    assert info["function_checks"] == {}
    assert info["codegen_settings"] == codegen_abi_settings()


def test_wrapper_cache_reports_specific_codegen_mismatch(restore_codegen_abi_settings):
    wrapper = PyCExtension("cache_mismatch", functions=[])
    compiler = {"executable": "/toolchain/gcc", "version": "gcc test"}
    wrapper.cache_info = wrapper.build_cache_info(
        "-O2",
        backend="c",
        compiler_identity=compiler,
    )
    original = nm.settings.add_shape_descriptors
    _set_codegen_setting("add_shape_descriptors", not original)

    mismatches = wrapper.cache_mismatches(
        "-O2",
        backend="c",
        compiler_identity=compiler,
    )

    field = "codegen_settings.add_shape_descriptors"
    assert mismatches[field] == (original, not original)
    assert field in wrapper.format_cache_mismatches(
        "-O2",
        backend="c",
        compiler_identity=compiler,
    )
    assert (
        wrapper.cache_matches(
            "-O2",
            backend="c",
            compiler_identity=compiler,
        )
        is False
    )


def test_target_compile_rejects_settings_changed_after_specialization(
    tmp_path, backend, restore_codegen_abi_settings
):
    @nm.jit(backend=backend, directory=tmp_path)
    def fill(values):
        values[:] = 1

    specialization = fill.specialize(nm.float64[:])
    original = nm.settings.add_shape_descriptors
    _set_codegen_setting("add_shape_descriptors", not original)

    with pytest.raises(
        IncompatibleLibraryError,
        match="codegen_settings|add_shape_descriptors",
    ):
        specialization.compile()

    assert specialization.compiled is False
    assert specialization.object_path is None
    assert specialization.wrapper_path is None


@pytest.mark.parametrize(
    "setting_name",
    [
        "add_shape_descriptors",
        "ignore_fixed_shape_in_nested_calls",
        "use_numpy_allocator",
        "reorder_kwargs",
    ],
)
def test_bundle_load_rejects_codegen_setting_mismatch(
    tmp_path, backend, setting_name, restore_codegen_abi_settings
):
    name = f"abi_setting_{setting_name}_{backend}"
    library = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def fill(values):
        values[:] = 1

    fill.specialize(nm.float64[:])
    library.save(tmp_path, "-O1")
    original = codegen_abi_settings()[setting_name]
    _set_codegen_setting(setting_name, not original)

    with pytest.raises(
        IncompatibleLibraryError,
        match=rf"codegen_settings\.{setting_name}",
    ):
        nm.NumetaLibrary.load(name, tmp_path)


def test_bundle_records_target_and_wrapper_provenance(tmp_path, backend):
    name = f"abi_provenance_{backend}"
    library = nm.NumetaLibrary(name)

    @nm.jit(
        backend=backend,
        directory=tmp_path / "jit",
        library=library,
        compile_flags="-O2",
        do_checks=False,
        simd_arch="scalar",
    )
    def fill(values):
        values[:] = 3

    specialization = fill.specialize(nm.float64[:])
    library.save(tmp_path, "-O1")
    manifest = _bundle_manifest(tmp_path, name)
    target = manifest["targets"][specialization.symbol]
    wrapper = manifest["wrapper_cache_info"]

    assert manifest["format_version"] == 2
    assert manifest["abi"]["codegen_settings"] == codegen_abi_settings()
    assert target["backend"] == backend
    assert target["compile_flags"] == ["-O2"]
    assert target["do_checks"] is False
    assert target["simd_arch"] == "scalar"
    assert target["simd_features"] == []
    assert target["compiler"] == manifest["toolchains"][backend]
    assert target["codegen_settings"] == codegen_abi_settings()
    assert wrapper["backend"] == backend
    assert wrapper["compile_flags"] == ["-O1"]
    assert wrapper["compiler"] == manifest["toolchains"]["c"]
    assert wrapper["do_checks"] is False
    assert wrapper["function_checks"] == {specialization.symbol: False}
    assert wrapper["numpy_c_abi"] == numpy_c_abi_version()

    loaded = nm.NumetaLibrary.load(name, tmp_path)
    assert loaded.fill._library_pyc_extension.do_checks is False
    values = np.zeros(2, dtype=np.float64)
    loaded.fill(values)
    np.testing.assert_array_equal(values, np.full(2, 3.0))


@pytest.mark.parametrize("metadata_field", ["codegen_settings", "do_checks"])
def test_bundle_rejects_inconsistent_target_metadata(tmp_path, backend, metadata_field):
    name = f"inconsistent_target_{metadata_field}_{backend}"
    library = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, library=library)
    def fill(values):
        values[:] = 1

    specialization = fill.specialize(nm.float64[:])
    library.save(tmp_path, "-O1")
    manifest_path = Path(tmp_path) / f"{name}.numeta" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    target = manifest["targets"][specialization.symbol]
    if metadata_field == "codegen_settings":
        current = target[metadata_field]["add_shape_descriptors"]
        target[metadata_field]["add_shape_descriptors"] = not current
        expected_message = "Native target ABI metadata mismatch"
    else:
        target[metadata_field] = not target[metadata_field]
        expected_message = "Native target check policy"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(CorruptLibraryError, match=expected_message):
        nm.NumetaLibrary.load(name, tmp_path)


def test_bundle_preserves_mixed_wrapper_check_policies(tmp_path, backend):
    library = nm.NumetaLibrary(f"mixed_checks_{backend}")

    @nm.jit(backend=backend, library=library, do_checks=True)
    def checked(checked_values):
        checked_values[:] = 1

    @nm.jit(backend=backend, library=library, do_checks=False)
    def unchecked(unchecked_values):
        unchecked_values[:] = 2

    checked_specialization = checked.specialize(nm.float64[:])
    unchecked_specialization = unchecked.specialize(nm.float64[:])
    library.save(tmp_path, "-O1")
    manifest = _bundle_manifest(tmp_path, library.name)
    function_checks = manifest["wrapper_cache_info"]["function_checks"]

    assert function_checks == {
        checked_specialization.symbol: True,
        unchecked_specialization.symbol: False,
    }
    wrapper_source = (
        Path(tmp_path)
        / f"{library.name}.numeta"
        / "libraries"
        / f"{library.name}{PyCExtension.SUFFIX}.c"
    ).read_text()
    assert "Input array 'checked_values'" in wrapper_source
    assert "Input array 'unchecked_values'" not in wrapper_source

    loaded = nm.NumetaLibrary.load(library.name, tmp_path)
    assert loaded.checked._library_pyc_extension.function_checks == function_checks
