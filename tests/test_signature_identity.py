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
)


def _parse(func, args, kwargs, *, use_c):
    params, fixed_indices, fixed_count, var_positional_name = parse_function_parameters(func)
    parser = get_signature_and_runtime_args if use_c else get_signature_and_runtime_args_py
    return parser(
        args,
        kwargs,
        params=params,
        fixed_param_indices=fixed_indices,
        n_positional_or_default_args=fixed_count,
        catch_var_positional_name=var_positional_name,
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (True, 1),
        (np.int64(1), 1),
        (np.float32(1.0), np.float64(1.0)),
        ((np.int64(1),), (1,)),
    ],
)
def test_python_comptime_signatures_use_exact_value_types(left, right):
    def kernel(value: nm.comptime, out):
        pass

    out = np.zeros(1, dtype=np.float64)
    left_signature = _parse(kernel, (left, out), {}, use_c=False)[1]
    right_signature = _parse(kernel, (right, out), {}, use_c=False)[1]

    assert isinstance(left_signature, Signature)
    assert type(left_signature[0]) is type(left)
    assert type(right_signature[0]) is type(right)
    assert left_signature != right_signature
    assert len({left_signature: "left", right_signature: "right"}) == 2
    assert signature_id(left_signature) != signature_id(right_signature)
    assert {left_signature: "left"}[tuple(left_signature)] == "left"


@pytest.mark.skipif(not _c_signature_backend_available, reason="C extension not available")
@pytest.mark.parametrize("value", [True, 1, np.int64(1), np.float32(1.0), np.float64(1.0)])
def test_c_and_python_parsers_share_type_aware_identity(value):
    def kernel(item: nm.comptime, out):
        pass

    out = np.zeros(1, dtype=np.float64)
    c_result = _parse(kernel, (value, out), {}, use_c=True)
    python_result = _parse(kernel, (value, out), {}, use_c=False)

    assert isinstance(c_result[1], Signature)
    assert c_result == python_result


def test_default_and_explicit_comptime_types_are_distinct():
    def kernel(out, value: nm.comptime = 1):
        pass

    out = np.zeros(1, dtype=np.float64)
    default_signature = _parse(kernel, (out,), {}, use_c=False)[1]
    explicit_signature = _parse(kernel, (out, np.int64(1)), {}, use_c=False)[1]

    assert default_signature != explicit_signature
    assert len({default_signature, explicit_signature}) == 2


def test_python_signature_parser_does_not_mutate_kwargs():
    def kernel(first, second):
        pass

    first = np.zeros(1, dtype=np.float64)
    second = np.ones(1, dtype=np.float64)
    kwargs = {"second": second}

    _parse(kernel, (first,), kwargs, use_c=False)

    assert list(kwargs) == ["second"]
    assert kwargs["second"] is second


def test_runtime_only_signatures_remain_plain_tuples():
    def kernel(value, out):
        pass

    out = np.zeros(1, dtype=np.float64)
    python_signature = _parse(kernel, (1, out), {}, use_c=False)[1]

    assert type(python_signature) is tuple
    if _c_signature_backend_available:
        c_signature = _parse(kernel, (1, out), {}, use_c=True)[1]
        assert type(c_signature) is tuple


def test_jit_dispatch_distinguishes_equal_comptime_types(tmp_path, backend):
    @nm.jit(backend=backend, directory=tmp_path)
    def classify(value: nm.comptime, out):
        if type(value) is bool:
            out[:] = 3
        elif isinstance(value, np.integer):
            out[:] = 2
        else:
            out[:] = 1

    cases = [(1, 1.0), (np.int64(1), 2.0), (True, 3.0)]
    signatures = []
    for value, expected in cases:
        out = np.zeros(2, dtype=np.float64)
        classify(value, out)
        np.testing.assert_array_equal(out, np.full(2, expected))
        signatures.append(classify.get_signature(value, out))

    assert len(classify.specializations) == 3
    assert len(set(signatures)) == 3
    assert len({specialization.signature_id for specialization in classify.specializations}) == 3


def test_aot_bundle_preserves_equal_comptime_types(tmp_path, backend):
    name = f"typed_comptime_{backend}"
    library = nm.NumetaLibrary(name)

    @nm.jit(backend=backend, directory=tmp_path / "jit", library=library)
    def classify(value: nm.comptime, out):
        if isinstance(value, np.integer):
            out[:] = 2
        else:
            out[:] = 1

    template = np.zeros(2, dtype=np.float64)
    python_specialization = classify.specialize(1, template)
    numpy_specialization = classify.specialize(np.int64(1), template)

    assert python_specialization.signature != numpy_specialization.signature
    assert python_specialization.signature_id != numpy_specialization.signature_id

    library.save(tmp_path, "-O1")
    loaded = nm.NumetaLibrary.load(name, tmp_path)

    for value, expected in ((1, 1.0), (np.int64(1), 2.0)):
        out = np.zeros(2, dtype=np.float64)
        loaded.classify(value, out)
        np.testing.assert_array_equal(out, np.full(2, expected))

    assert len(loaded.classify.specializations) == 2
