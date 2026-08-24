import numpy as np
import numeta as nm


def assert_inline_dependency(caller, caller_signature, *, expected):
    dependencies = caller._compiled_functions[caller_signature].symbolic_function.get_dependencies()
    assert (len(dependencies) > 0) is expected


def test_inline_array_scalar(backend):

    @nm.jit(backend=backend, inline=True)
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


def test_inline_array(backend):

    @nm.jit(backend=backend, inline=True)
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


def test_inline_getitem_scalar(backend):

    @nm.jit(backend=backend, inline=True)
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


def test_inline_getitem_slice(backend):

    @nm.jit(backend=backend, inline=True)
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


def test_inline_getitem_slice_runtime_dep(backend):

    @nm.jit(backend=backend, inline=True)
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


def test_inline_getattr_scalar(backend):

    @nm.jit(backend=backend, inline=True)
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


def test_inline_getattr_array(backend):

    @nm.jit(backend=backend, inline=True)
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


def test_inline_matmul(backend):

    @nm.jit(backend=backend, inline=True)
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


def test_inline_nested_calls(backend):
    @nm.jit(backend=backend, inline=True)
    def inner(n, arr):
        arr[0] = n

    @nm.jit(backend=backend, inline=True)
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


def test_inline_nested_calls_dependencies(backend):
    """Test to check that inline nested calls propagate dependencies correctly."""

    @nm.jit(backend=backend)
    def inner(n, arr):
        arr[0] = n

    @nm.jit(backend=backend, inline=True)
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


def test_inline_matmul_fortran_order(backend):

    @nm.jit(backend=backend, inline=True)
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


def test_inline_loop_matmul(backend):
    @nm.jit(backend=backend, inline=True)
    def callee(n, a, b, c):
        for i in nm.range(n):
            for k in nm.range(n):
                c[i, :] += a[i, k] * b[k, :]

    @nm.jit(backend=backend)
    def caller(n, a, b, c):
        callee(n, a, b, c)

    n = 4
    a = np.random.random((n, n))
    b = np.random.random((n, n))
    c = np.zeros((n, n))
    caller(n, a, b, c)

    expected = a @ b
    np.testing.assert_allclose(c, expected)


def test_inline_loop_if(backend):
    @nm.jit(backend=backend, inline=True)
    def callee(n, arr):
        for i in nm.range(n):
            with nm.If(i < 2):
                arr[i] = i
            with nm.ElseIf(i < n // 2):
                arr[i] = i + 1
            with nm.Else():
                arr[i] = -i

    @nm.jit(backend=backend)
    def caller(n, arr):
        callee(n, arr)

    n = 10
    arr = np.zeros(n, dtype=np.int64)
    caller(n, arr)

    expected = np.array([0, 1, 3, 4, 5, -5, -6, -7, -8, -9], dtype=np.int64)
    np.testing.assert_equal(arr, expected)


def test_inline_jit_threshold_inline(backend):
    @nm.jit(backend=backend, inline=2)
    def callee(a):
        a[:] = 1
        a[:] = 2

    @nm.jit(backend=backend)
    def caller(a):
        callee(a)

    arr = np.zeros(5, dtype=np.int64)
    caller(arr)

    expected = np.full(5, 2, dtype=np.int64)
    np.testing.assert_equal(arr, expected)

    signature = caller.get_signature(arr)
    assert_inline_dependency(caller, signature, expected=False)


def test_inline_jit_threshold_call(backend):
    @nm.jit(backend=backend, inline=1)
    def callee(a):
        a[:] = 1
        a[:] = 2

    @nm.jit(backend=backend)
    def caller(a):
        callee(a)

    arr = np.zeros(5, dtype=np.int64)
    caller(arr)

    expected = np.full(5, 2, dtype=np.int64)
    np.testing.assert_equal(arr, expected)

    signature = caller.get_signature(arr)
    assert_inline_dependency(caller, signature, expected=True)


def test_inline_jit_threshold_loop_inline(backend):
    @nm.jit(backend=backend, inline=3)
    def callee(n, a):
        for i in nm.range(n):
            for j in nm.range(n):
                a[i, j] += 1

    @nm.jit(backend=backend)
    def caller(n, a):
        callee(n, a)

    arr = np.zeros((3, 3), dtype=np.int64)
    caller(3, arr)

    expected = np.ones((3, 3), dtype=np.int64)
    np.testing.assert_equal(arr, expected)

    signature = caller.get_signature(3, arr)
    assert_inline_dependency(caller, signature, expected=False)


def test_inline_jit_threshold_loop_call(backend):
    @nm.jit(backend=backend, inline=2)
    def callee(n, a):
        for i in nm.range(n):
            for j in nm.range(n):
                a[i, j] += 1

    @nm.jit(backend=backend)
    def caller(n, a):
        callee(n, a)

    arr = np.zeros((3, 3), dtype=np.int64)
    caller(3, arr)

    expected = np.ones((3, 3), dtype=np.int64)
    np.testing.assert_equal(arr, expected)

    signature = caller.get_signature(3, arr)
    assert_inline_dependency(caller, signature, expected=True)


def test_inline_jit_threshold_if_inline(backend):
    @nm.jit(backend=backend, inline=5)
    def callee(n, a):
        for i in nm.range(n):
            with nm.If(i < 2):
                a[i] = i
            with nm.Else():
                a[i] = -i

    @nm.jit(backend=backend)
    def caller(n, a):
        callee(n, a)

    arr = np.zeros(5, dtype=np.int64)
    caller(5, arr)

    expected = np.array([0, 1, -2, -3, -4], dtype=np.int64)
    np.testing.assert_equal(arr, expected)

    signature = caller.get_signature(5, arr)
    callee_signature = callee.get_signature(5, arr)
    assert_inline_dependency(caller, signature, expected=False)


def test_inline_jit_threshold_if_call(backend):
    @nm.jit(backend=backend, inline=4)
    def callee(n, a):
        for i in nm.range(n):
            with nm.If(i < 2):
                a[i] = i
            with nm.Else():
                a[i] = -i

    @nm.jit(backend=backend)
    def caller(n, a):
        callee(n, a)

    arr = np.zeros(5, dtype=np.int64)
    caller(5, arr)

    expected = np.array([0, 1, -2, -3, -4], dtype=np.int64)
    np.testing.assert_equal(arr, expected)

    signature = caller.get_signature(5, arr)
    callee_signature = callee.get_signature(5, arr)
    assert_inline_dependency(caller, signature, expected=True)


def test_inline_tmp_scalar(backend):

    @nm.jit(backend=backend, inline=True)
    def callee(n, a):
        i = nm.int64(5)
        a[:] = n + i

    @nm.jit(backend=backend)
    def caller(a):
        n = nm.int64(1, name="n")
        callee(n, a)

    a = np.zeros((), dtype=np.int64)
    caller(a)

    expected = np.zeros((), dtype=np.int64)
    expected[...] = 6
    np.testing.assert_equal(a, expected)


def test_inline_name_mangling(backend):
    """
    To test if the local variables are properly renamed.
    """

    @nm.jit(backend=backend, inline=True)
    def callee(n, a):
        f = nm.float64(5.0)
        a[n] = f

    @nm.jit(backend=backend, directory="build")
    def caller(a):
        n = nm.int32(6)
        callee(n, a)

    a = np.zeros(10, dtype=np.int64)
    caller(a)

    expected = np.zeros(10, dtype=np.int64)
    expected[6] = 5.0
    np.testing.assert_equal(a, expected)


def test_inline_slice_composition(backend):

    @nm.jit(backend=backend, inline=True)
    def callee(arr):
        arr[:4] = 1.0
        arr[4:6] = 2.0
        arr[6:] = 3.0
        arr[2] = 4.0

    @nm.jit(backend=backend)
    def caller(a, b, c, d):
        callee(a[3:10])
        callee(b)
        callee(c[5:])
        callee(d[:10])

    a = np.zeros(12)
    b = np.zeros(12)
    c = np.zeros(12)
    d = np.zeros(12)
    caller(a, b, c, d)

    expected_a = np.zeros(12)
    expected_a[3:7] = 1.0
    expected_a[7:9] = 2.0
    expected_a[9:10] = 3.0
    expected_a[5] = 4.0
    np.testing.assert_allclose(a, expected_a)

    expected_b = np.zeros(12)
    expected_b[:7] = 1.0
    expected_b[4:6] = 2.0
    expected_b[6:] = 3.0
    expected_b[2] = 4.0
    np.testing.assert_allclose(b, expected_b)

    expected_c = np.zeros(12)
    expected_c[5:12] = 1.0
    expected_c[9:11] = 2.0
    expected_c[11:] = 3.0
    expected_c[7] = 4.0
    np.testing.assert_allclose(c, expected_c)
    expected_c = np.zeros(12)
    expected_c[5:12] = 1.0
    expected_c[9:11] = 2.0
    expected_c[11:] = 3.0
    expected_c[7] = 4.0
    np.testing.assert_allclose(c, expected_c)

    expected_d = np.zeros(12)
    expected_d[:7] = 1.0
    expected_d[4:6] = 2.0
    expected_d[6:10] = 3.0
    expected_d[2] = 4.0
    np.testing.assert_allclose(d, expected_d)
