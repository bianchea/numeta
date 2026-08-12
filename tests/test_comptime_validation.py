import numpy as np
import pytest

import numeta as nm
from numeta.library_signature import signature_id
from numeta.signature import (
    Signature,
    _c_signature_backend_available,
    get_signature_and_runtime_args,
    get_signature_and_runtime_args_py,
    parse_function_parameters,
    validate_comptime_value,
)


class HashableCustomValue:
    def __hash__(self):
        return 1


class TupleSubclass(tuple):
    pass


def _parse(func, args, *, use_c):
    params, fixed_indices, fixed_count, var_positional_name = parse_function_parameters(func)
    parser = get_signature_and_runtime_args if use_c else get_signature_and_runtime_args_py
    return parser(
        args,
        {},
        params=params,
        fixed_param_indices=fixed_indices,
        n_positional_or_default_args=fixed_count,
        catch_var_positional_name=var_positional_name,
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        3,
        1.5,
        1 + 2j,
        "left",
        np.int64(3),
        np.dtype("float64"),
        np.float64,
        nm.float64,
        nm.ptr(nm.float64, const=True),
        slice(1, 5, 2),
        (3, (np.int32(2), "nested", slice(None))),
    ],
)
def test_supported_comptime_values_are_stable_keys(value):
    validate_comptime_value(value, "config")

    signature = Signature((value,))
    hash(signature)
    assert signature_id(signature).startswith("sig-v1-")


@pytest.mark.parametrize("use_c", [False, True], ids=["python", "c"])
@pytest.mark.parametrize(
    ("value", "type_name", "path"),
    [
        ([], "builtins.list", ""),
        ({"mode": 1}, "builtins.dict", ""),
        (np.zeros(1), "numpy.ndarray", ""),
        (HashableCustomValue(), "HashableCustomValue", ""),
        ((1, [2]), "builtins.list", "\\[1\\]"),
        (TupleSubclass((1,)), "TupleSubclass", ""),
    ],
)
def test_signature_parsers_reject_unstable_comptime_values(use_c, value, type_name, path):
    if use_c and not _c_signature_backend_available:
        pytest.skip("C extension not available")

    def kernel(config: nm.comptime, out):
        pass

    with pytest.raises(TypeError, match=rf"config.*{path}.*{type_name}"):
        _parse(kernel, (value, np.zeros(1)), use_c=use_c)


@pytest.mark.parametrize("value", [float("inf"), complex(1, float("nan")), np.float64(np.nan)])
def test_non_finite_comptime_values_are_rejected(value):
    with pytest.raises(ValueError, match="config.*finite"):
        validate_comptime_value(value, "config")


def test_unhashable_numpy_scalar_has_clear_error():
    with pytest.raises(TypeError, match="config.*hashable"):
        validate_comptime_value(np.timedelta64(1), "config")


def test_jit_rejects_invalid_comptime_before_specializing(tmp_path, backend):
    @nm.jit(backend=backend, directory=tmp_path)
    def fill(config: nm.comptime, out):
        out[:] = 1

    out = np.zeros(1)
    with pytest.raises(TypeError, match="config.*builtins.list"):
        fill([], out)

    assert fill.specializations == ()


def test_explicit_signature_references_validate_comptime_values(tmp_path, backend):
    @nm.jit(backend=backend, directory=tmp_path)
    def fill(config: nm.comptime, out):
        out[:] = config

    valid_signature = fill.get_signature(2, np.zeros(1))
    invalid_signature = ([], *valid_signature[1:])

    with pytest.raises(TypeError, match="config.*builtins.list"):
        fill.get_specialization(invalid_signature)
    with pytest.raises(TypeError, match="config.*builtins.list"):
        fill.clear_generated_state([invalid_signature])


def test_batch_build_rejects_invalid_comptime_before_compilation(tmp_path, backend):
    library = nm.NumetaLibrary(f"invalid_comptime_{backend}")

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def fill(config: nm.comptime, out):
        out[:] = 1

    with pytest.raises(TypeError, match="config.*builtins.dict"):
        library.build(tmp_path, {"fill": [({"mode": 1}, nm.float64[:])]})

    assert fill.specializations == ()


def test_aot_bundle_preserves_nested_comptime_value(tmp_path, backend):
    name = f"nested_comptime_{backend}"
    library = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def fill(config: nm.comptime, out):
        out[:] = config[0]

    config = (2, (np.int32(3), "nested", slice(0, 2)))
    fill.specialize(config, nm.float64[:])
    library.save(tmp_path, "-O1")

    loaded = nm.NumetaLibrary.load(name, tmp_path)
    out = np.zeros(2)
    loaded.fill(config, out)

    np.testing.assert_array_equal(out, np.full(2, 2.0))
