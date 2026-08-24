import numpy as np
import numeta as nm
from numeta.settings import settings


def test_call_array_scalar(backend):

    @nm.jit(backend=backend)
    def callee(n, a):
        a[:] = n

    @nm.jit(backend=backend)
    def caller(n, a):
        callee(n, a)

    a = np.zeros((), dtype=np.int64)
    caller(1, a)

    expected = np.zeros((), dtype=np.int64)
    expected[...] = 1
    np.testing.assert_equal(a, expected)


def test_call_array(backend):

    @nm.jit(backend=backend)
    def callee(n, a):
        a[:, 2] = n

    @nm.jit(backend=backend)
    def caller(n, a):
        callee(n, a)

    a = np.zeros((5, 10), dtype=np.int64)
    caller(1, a)

    expected = np.zeros((5, 10), dtype=np.int64)
    expected[:, 2] = 1
    np.testing.assert_equal(a, expected)


def test_call_getitem_scalar(backend):

    @nm.jit(backend=backend)
    def callee(n, a):
        a[:] = n

    @nm.jit(backend=backend)
    def caller(n, a):
        callee(n, a[3, 7])

    a = np.zeros((5, 10), dtype=np.int64)
    caller(1, a)

    expected = np.zeros((5, 10), dtype=np.int64)
    expected[3, 7] = 1
    np.testing.assert_equal(a, expected)


def test_call_getitem_slice(backend):

    @nm.jit(backend=backend)
    def callee(n, a):
        a[:] = n

    @nm.jit(backend=backend)
    def caller(n, a):
        callee(50, a[:])
        callee(n, a[:2, :3])
        callee(n + 1, a[3:4, 8:])
        callee(n + 2, a[:, 7])

    a = np.zeros((5, 10), dtype=np.int64)
    caller(1, a)

    expected = np.zeros((5, 10), dtype=np.int64)
    expected[:] = 50
    expected[:2, :3] = 1
    expected[3:4, 8:] = 2
    expected[:, 7] = 3
    np.testing.assert_equal(a, expected)


def test_call_getitem_slice_runtime_dep(backend):

    @nm.jit(backend=backend)
    def callee(n, a):
        a[:] = n

    @nm.jit(backend=backend)
    def caller(n, m, a):
        callee(50, a[:])
        callee(n, a[:n, : m // 2])
        callee(2, a[2 : n + 1, m - 2 :])
        callee(3, a[:, n])
        callee(1, a[:, n : m - n])

    n = 4
    m = 9
    a = np.zeros((5, 10), dtype=np.int64)
    caller(n, m, a)

    expected = np.zeros((5, 10), dtype=np.int64)
    expected[:] = 50
    expected[:n, : m // 2] = n
    expected[2 : n + 1, m - 2 :] = 2
    expected[:, n] = 3
    expected[:, n : m - n] = 1
    np.testing.assert_equal(a, expected)


def test_call_getattr_scalar(backend):

    @nm.jit(backend=backend)
    def callee(n, a):
        a[:] = n

    @nm.jit(backend=backend)
    def caller(n, a):
        callee(n, a["x"])

    dtype = np.dtype([("x", np.int64), ("y", np.float64, (2, 2))], align=True)

    a = np.zeros((), dtype=dtype)
    caller(1, a)

    expected = np.zeros((), dtype=dtype)
    expected["x"] = 1
    np.testing.assert_equal(a, expected)


def test_call_getattr_array(backend):

    @nm.jit(backend=backend)
    def callee(n, a):
        a[:] = n

    @nm.jit(backend=backend)
    def caller(n, a):
        callee(n, a["y"])

    dtype = np.dtype([("x", np.int64), ("y", np.float64, (2, 2))], align=True)

    a = np.zeros((), dtype=dtype)
    caller(2.0, a)

    expected = np.zeros((), dtype=dtype)
    expected["y"] = 2.0
    np.testing.assert_equal(a, expected)


def test_call_matmul(backend):

    @nm.jit(backend=backend)
    def callee(a, d):
        a[:] = d

    @nm.jit(backend=backend)
    def caller(a, b, c):
        callee(c, nm.Matmul(a, b))

    n = 10
    a = np.random.random((n, n))
    b = np.random.random((n, n))
    c = np.zeros((10, 10))
    caller(a, b, c)

    expected = a @ b
    np.testing.assert_allclose(c, expected)


def test_call_matmul_fortran_order(backend):

    @nm.jit(backend=backend)
    def callee(a, d):
        a[:] = d

    @nm.jit(backend=backend)
    def caller(a, b, c):
        callee(c, nm.Matmul(a, b))

    n = 10
    a = np.random.random((n, n)).astype(np.float64, order="F")
    b = np.random.random((n, n)).astype(np.float64, order="F")
    c = np.zeros((10, 10), order="F")
    caller(a, b, c)

    expected = a @ b
    np.testing.assert_allclose(c, expected)


def test_intrinsics_dot_matmul_transpose(backend):
    @nm.jit(backend=backend)
    def dot(a, b):
        return nm.Dotproduct(a, b)

    @nm.jit(backend=backend)
    def matmul_ab(a, b, out):
        out[:] = nm.Matmul(a, b)

    @nm.jit(backend=backend)
    def trans(a, out):
        out[:] = nm.Transpose(a)

    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    np.testing.assert_allclose(dot(a, b), np.dot(a, b))

    m = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order="F")
    n = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64, order="F")
    out = np.zeros((2, 2), dtype=np.float64, order="F")
    matmul_ab(m, n, out)
    np.testing.assert_allclose(out, m @ n)

    out_t = np.zeros((2, 2), dtype=np.float64, order="F")
    trans(m, out_t)
    np.testing.assert_allclose(out_t, m.T)


def test_nested_calls(backend):
    @nm.jit(backend=backend)
    def inner(n, arr):
        arr[0] = n

    @nm.jit(backend=backend)
    def middle(n, arr):
        inner(n, arr)
        arr[1] = n + 1

    @nm.jit(backend=backend)
    def caller(n, arr):
        middle(n, arr)

    arr = np.zeros(2, dtype=np.int64)
    caller(5, arr)

    expected = np.array([5, 6], dtype=np.int64)
    np.testing.assert_equal(arr, expected)


def test_nested_calls_with_literals(backend):
    @nm.jit(backend=backend)
    def inner(n, arr):
        arr[0] = n

    @nm.jit(backend=backend)
    def middle(n, arr):
        inner(2, arr)
        arr[1] = n + 1

    @nm.jit(backend=backend)
    def caller(arr):
        middle(3, arr)

    arr = np.zeros(2, dtype=np.int64)
    caller(arr)

    expected = np.array([2, 4], dtype=np.int64)
    np.testing.assert_equal(arr, expected)


def test_nested_calls_arithmetic(backend):
    @nm.jit(backend=backend)
    def inner(a, b):
        return a + b

    @nm.jit(backend=backend)
    def outer(a, b, c):
        return inner(a, b) * c

    np.testing.assert_equal(outer(2, 4, 3), 18)


def test_nested_struct_field_call_with_c_signature_intent_regression():
    dtype = np.dtype([("x", np.int32), ("y", np.float64, (2,))], align=True)

    original_use_c_dispatch = settings.use_c_dispatch
    original_use_c_signature_parser = settings.use_c_signature_parser

    try:
        for use_c_dispatch in (True, False):
            settings.use_c_dispatch = use_c_dispatch
            for use_c_signature_parser in (True, False):
                settings.use_c_signature_parser = use_c_signature_parser

                @nm.jit(backend="fortran")
                def inner(v):
                    v[0] = 3.5

                @nm.jit(backend="fortran")
                def outer(arr):
                    inner(arr[0]["y"])

                arr = np.zeros(1, dtype=dtype)
                outer(arr)

                expected = np.zeros(1, dtype=dtype)
                expected[0]["y"][0] = 3.5
                np.testing.assert_equal(arr, expected)
    finally:
        settings.use_c_dispatch = original_use_c_dispatch
        settings.use_c_signature_parser = original_use_c_signature_parser
