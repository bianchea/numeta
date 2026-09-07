import operator

import numpy as np
import pytest

import numeta as nm
from numeta.ast import Variable
from numeta.ast.expressions import LiteralNode
from numeta.type_rules import resolve_binary


DTYPES = [np.bool_, np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]
for extended in (np.longdouble, np.clongdouble):
    if np.dtype(extended) not in [np.dtype(dtype) for dtype in DTYPES]:
        DTYPES.append(extended)

OPS = [
    ("+", np.add),
    ("-", np.subtract),
    ("*", np.multiply),
    ("/", np.true_divide),
    (".eq.", np.equal),
    (".lt.", np.less),
]


@pytest.mark.parametrize("left", DTYPES)
@pytest.mark.parametrize("right", DTYPES)
@pytest.mark.parametrize("op,ufunc", OPS)
def test_dtype_matrix(left, right, op, ufunc):
    a = Variable("a", dtype=nm.get_datatype(left))
    b = Variable("b", dtype=nm.get_datatype(right))
    if (op == "-" and left is right is np.bool_) or (
        op == ".lt." and any(np.dtype(t).kind == "c" for t in (left, right))
    ):
        with pytest.raises(TypeError):
            resolve_binary(a, b, op)
        return
    expected = ufunc.resolve_dtypes((np.dtype(left), np.dtype(right), None))
    inputs, result = resolve_binary(a, b, op)
    assert tuple(np.dtype(t.get_numpy()) for t in (*inputs, result)) == expected


@pytest.mark.parametrize(
    "left,right",
    [
        (np.int32, np.float32),
        (np.float32, np.float64),
        (np.int64, np.complex64),
        (np.bool_, np.float32),
        (np.bool_, np.bool_),
        (np.complex64, np.complex128),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("array", [False, True])
def test_mixed_values(backend, left, right, reverse, array):
    @nm.jit(backend=backend)
    def kernel(a, b, out, selected, equal):
        x, y = (b, a) if reverse else (a, b)
        out[:] = x + y
        selected[:] = nm.where(x != 0, x, y)
        equal[:] = x == y

    a, b = left(1), right(0.25 if np.dtype(right).kind in "fc" else 1)
    if array:
        a, b = np.full(2, a), np.full(2, b)
    dtype = np.result_type(np.dtype(left), np.dtype(right))
    out, selected, equal = (
        np.empty(2, dtype=dtype),
        np.empty(2, dtype=dtype),
        np.empty(2, dtype=bool),
    )
    kernel(a, b, out, selected, equal)
    x, y = (b, a) if reverse else (a, b)
    np.testing.assert_equal(out, np.broadcast_to(x + y, (2,)))
    np.testing.assert_equal(selected, np.broadcast_to(np.where(x != 0, x, y), (2,)))
    np.testing.assert_equal(equal, np.broadcast_to(x == y, (2,)))


def test_precision_before_store(backend):
    @nm.jit(backend=backend)
    def kernel(a, b, out):
        out[:] = a + b
        return a + b

    out = np.zeros(1, np.float32)
    assert kernel(np.float32(2**24), np.float64(1), out) == 2**24 + 1
    assert out[0] == 2**24
    source = kernel.source
    assert ("real(" in source) if backend == "fortran" else ("npy_float64" in source)


def test_integer_true_division(backend):
    @nm.jit(backend=backend)
    def kernel(a, b):
        return a / b

    assert kernel(3, 2) == 1.5


@pytest.mark.parametrize("value,expected", [(1.0, nm.f4), (np.float64(1), nm.f8)])
def test_weak_literal_provenance(value, expected):
    a = Variable("a", dtype=nm.f4)
    for expr in (a + value, value + a):
        assert expr.dtype is expected
        assert expr.get_with_updated_variables({}).dtype is expected


@pytest.mark.parametrize("value", [2**40, 1e40])
def test_literal_overflow(value):
    dtype = nm.i4 if isinstance(value, int) else nm.f4
    with pytest.raises(OverflowError, match="wider explicitly typed operand"):
        _ = (Variable("a", dtype=dtype) + value).dtype


def test_invalid_where():
    with pytest.raises(TypeError, match="explicit comparison"):
        _ = nm.where(LiteralNode(1), 2, 3).dtype


def test_typed_indices_bounds_and_dimensions(backend):
    @nm.jit(backend=backend)
    def kernel(a, n):
        i = nm.astype(n, nm.i8)
        tmp = nm.empty((i + 1,), dtype=nm.f8)
        tmp[:] = 0
        tmp[i % 3] = a[i % 3]
        a[:i] = tmp[:i]
        return a[i % 3]

    a = np.arange(5, dtype=float)
    assert kernel(a, 2.0) == 2


def test_numeric_boolean_conversions(backend):
    @nm.jit(backend=backend)
    def kernel(a, b):
        return nm.astype(nm.astype(a, nm.bool8), nm.f8) + b

    assert kernel(2.0, 0.5) == 1.5


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.mul,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.eq,
        operator.ne,
    ],
)
def test_boolean_operations_nested(backend, op):
    @nm.jit(backend=backend)
    def kernel(a, b):
        return nm.astype(op(a, b), nm.i8)

    assert kernel(np.bool_(True), np.bool_(True)) == int(op(np.bool_(True), np.bool_(True)))


def test_large_weak_integer_selected_real_type(backend):
    @nm.jit(backend=backend)
    def kernel(a):
        return a + 2**70

    assert kernel(np.float64(0)) == float(2**70)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, operator.truediv])
def test_arithmetic_values_and_return_dtype(backend, dtype, op):
    @nm.jit(backend=backend)
    def kernel(a, b):
        return op(a, b)

    a, b = np.int32(3), dtype(2.5)
    result = kernel(a, b)
    expected = op(a, b)
    assert np.asarray(result).dtype == np.asarray(expected).dtype
    np.testing.assert_allclose(result, expected)


def test_literal_explicit_conversions(backend):
    @nm.jit(backend=backend)
    def kernel():
        return nm.astype(2**70, nm.f8), nm.astype(3.5, nm.i8), nm.scalar(2, dtype=nm.bool8)

    large, integer, boolean = kernel()
    assert large == float(2**70)
    assert integer == 3
    assert boolean == True
