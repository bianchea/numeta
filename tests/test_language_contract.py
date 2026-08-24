import numeta as nm
import numpy as np
import pytest


def test_scalar_value_first_and_whole_storage(backend, tmp_path):
    @nm.jit(backend=backend, directory=tmp_path)
    def kernel(out):
        inferred = nm.scalar(3)
        explicit = nm.scalar(2.5, dtype=nm.f4)
        uninitialized = nm.scalar(dtype=nm.f8)
        legacy = nm.scalar(nm.f8, 4.0)
        uninitialized[:] = inferred + explicit
        uninitialized[:] += legacy
        out[0] = uninitialized[:]

    out = np.zeros(1)
    kernel(out)
    np.testing.assert_allclose(out, [9.5])


def test_scalar_index_is_an_actionable_trace_error(backend, tmp_path):
    @nm.jit(backend=backend, directory=tmp_path)
    def kernel(out):
        value = nm.scalar(1)
        out[0] = value[0]

    with pytest.raises(nm.NumetaTypeError, match=r"Scalar storage.*scalar\[:\]"):
        kernel(np.zeros(1, dtype=np.int64))


@pytest.mark.parametrize(
    "left,right",
    [(-7, 3), (7, -3), (-7, -3), (7, 3)],
)
def test_python_and_truncating_integer_divmod(left, right, backend, tmp_path):
    @nm.jit(backend=backend, directory=tmp_path)
    def kernel(a, b, out):
        out[0] = a // b
        out[1] = a % b
        out[2] = nm.trunc_div(a, b)
        out[3] = nm.trunc_mod(a, b)

    out = np.zeros(4, dtype=np.int64)
    kernel(left, right, out)
    expected_trunc = int(left / right)
    np.testing.assert_array_equal(
        out,
        [left // right, left % right, expected_trunc, left - expected_trunc * right],
    )


def test_integer_bitwise_shift_and_round(backend, tmp_path):
    @nm.jit(backend=backend, directory=tmp_path)
    def kernel(a, out):
        out[0] = a << 2
        out[1] = a >> 1
        out[2] = a ^ 3
        out[3] = ~a
        out[4] = round(nm.scalar(2.5))
        out[5] = round(nm.scalar(3.5))

    out = np.zeros(6, dtype=np.int64)
    kernel(6, out)
    np.testing.assert_array_equal(out, [24, 3, 5, -7, 2, 4])


def test_complex_constructor_keeps_imaginary_component(backend, tmp_path):
    @nm.jit(backend=backend, directory=tmp_path)
    def kernel(out):
        out[0] = nm.complex64(1.25, -2.5)
        out[1] = nm.complex128(3.5, 4.25)

    out = np.zeros(2, dtype=np.complex128)
    kernel(out)
    np.testing.assert_allclose(out, [1.25 - 2.5j, 3.5 + 4.25j])


def test_removed_cond_has_conversion_guidance():
    with pytest.raises(RuntimeError, match=r"with nm.If"):
        nm.cond(True)


def test_source_inspection_and_specialization_lock(backend, tmp_path):
    @nm.jit(
        backend=backend,
        directory=tmp_path,
        allow_new_specializations=False,
    )
    def increment(value):
        return value + 1

    with pytest.raises(RuntimeError, match="specialize"):
        increment(np.float64(1))

    specialization = increment.specialize(nm.f8)
    assert specialization.signature_id in increment.sources
    assert increment.source == specialization.source
    assert increment(np.float64(1)) == 2


def test_source_before_specialization_has_guidance(tmp_path):
    @nm.jit(backend="c", directory=tmp_path)
    def identity(value):
        return value

    with pytest.raises(RuntimeError, match=r"specialize\(\.\.\.\)"):
        _ = identity.source


def test_cache_and_directory_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="both cache=True and directory"):
        nm.jit(cache=True, directory=tmp_path)(lambda value: value)


def test_generic_where_and_astype_are_elementwise(backend, tmp_path):
    @nm.jit(backend=backend, directory=tmp_path)
    def kernel(values, out):
        out[:] = nm.astype(nm.where(values > 0, values, -values), nm.f4)

    values = np.array([-2.0, 0.0, 3.0], dtype=np.float64)
    out = np.zeros(3, dtype=np.float32)
    kernel(values, out)
    np.testing.assert_allclose(out, [2.0, 0.0, 3.0])


def test_symbolic_python_protocol_errors_have_guidance(tmp_path):
    @nm.jit(backend="c", directory=tmp_path)
    def bad_print(value):
        print(f"value={value}")

    with pytest.raises(nm.NumetaTypeError, match=r"f-strings.*nm.Print"):
        bad_print(1.0)
