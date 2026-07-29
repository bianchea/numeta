import numeta as nm
import numpy as np
import pytest


NUMERIC_DTYPES = [
    np.float64,
    np.float32,
    np.int64,
    np.int32,
    np.complex64,
    np.complex128,
    nm.float64,
    nm.float32,
    nm.int64,
    nm.int32,
    nm.complex64,
    nm.complex128,
]
if hasattr(np, "float128"):
    NUMERIC_DTYPES.extend([np.float128, nm.float128])


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_cast(dtype, backend):

    @nm.jit(backend=backend)
    def set_nine(a):
        a_int = nm.view(a, dtype)
        a_int[:] = 9.0

    # should contain everything
    a = np.ones(16, dtype=np.bool_)
    set_nine(a)

    np_dtype = nm.get_datatype(dtype).get_numpy()
    if np.issubdtype(np_dtype, np.integer):
        np.testing.assert_allclose(a.view(np_dtype)[0], np.array(9, dtype=np_dtype), atol=0)
    else:
        np.testing.assert_allclose(
            a.view(np_dtype)[0], np.array(9, dtype=np_dtype), rtol=10e2 * np.finfo(np_dtype).eps
        )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_cast_getitem(dtype, backend):

    @nm.jit(backend=backend)
    def set_nine(a):
        a_int = nm.view(a[0], dtype)
        a_int[:] = 9.0

    # should contain everything
    a = np.ones(16, dtype=np.bool_)
    set_nine(a)

    np_dtype = nm.get_datatype(dtype).get_numpy()
    if np.issubdtype(np_dtype, np.integer):
        np.testing.assert_allclose(a.view(np_dtype)[0], np.array(9, dtype=np_dtype), atol=0)
    else:
        np.testing.assert_allclose(
            a.view(np_dtype)[0], np.array(9, dtype=np_dtype), rtol=10e2 * np.finfo(np_dtype).eps
        )


def test_cast_struct(backend):

    dtype = np.dtype([("a", np.float64), ("b", np.int64)])

    @nm.jit(backend=backend)
    def set_nine(a):
        a_int = nm.view(a, dtype)
        a_int["a"][:] = 9.0
        a_int["b"][:] = 9

    # should contain everything
    a = np.ones(16, dtype=np.bool_)
    set_nine(a)

    check = np.empty((), dtype)
    check["a"] = 9.0
    check["b"] = 9

    # cannot use rtol
    np.testing.assert_allclose(a.view(dtype)["a"][0], check["a"], atol=0)
    np.testing.assert_allclose(a.view(dtype)["b"][0], check["b"], atol=0)


def test_cast_array_infers_shape_when_itemsize_matches(backend):

    @nm.jit(backend=backend)
    def set_all(a):
        a_view = nm.view(a, np.float64)
        a_view[:] = 3.5

    a = np.zeros((3, 4), dtype=np.int64)
    set_all(a)
    np.testing.assert_allclose(a.view(np.float64), np.full((3, 4), 3.5, dtype=np.float64), atol=0)


def test_cast_array_with_explicit_shape(backend):

    @nm.jit(backend=backend)
    def set_two(a):
        a_view = nm.view(a, np.float64, shape=(2,))
        a_view[:] = 11.0

    a = np.zeros(16, dtype=np.bool_)
    set_two(a)
    np.testing.assert_allclose(a.view(np.float64), np.array([11.0, 11.0], dtype=np.float64), atol=0)


def test_cast_array_type_dtype_with_shape_inference(backend):

    @nm.jit(backend=backend)
    def set_all(a):
        a_view = nm.view(a, nm.float32[:, :])
        a_view[:] = 2.25

    a = np.zeros((2, 3), dtype=np.int32)
    set_all(a)
    np.testing.assert_allclose(a.view(np.float32), np.full((2, 3), 2.25, dtype=np.float32), atol=0)
