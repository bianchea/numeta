import pytest

import numeta as nm
from numeta.array_shape import SCALAR
from numeta.ast import Variable


def test_vector_binary_dtype():
    vec = nm.Vector[nm.f8, 8]
    a = Variable("a", dtype=vec)
    b = Variable("b", dtype=vec)

    expr = a + b

    assert expr.dtype is vec
    assert expr._shape is SCALAR


def test_vector_scalar_binary_dtype():
    vec = nm.Vector[nm.f8, 8]
    v = Variable("v", dtype=vec)
    s = Variable("s", dtype=nm.f8)

    expr = v + s

    assert expr.dtype is vec
    assert expr._shape is SCALAR


def test_vector_binary_rejects_mismatched_lanes():
    a = Variable("a", dtype=nm.Vector[nm.f8, 4])
    b = Variable("b", dtype=nm.Vector[nm.f8, 8])

    with pytest.raises(TypeError):
        _ = (a + b).dtype


def test_broadcast_vload_reduce_dtypes():
    x = Variable("x", dtype=nm.f8, shape=(None,))
    i = Variable("i", dtype=nm.i8)
    av = nm.broadcast(1.0, lanes=4)
    xv = nm.vload(x, i, lanes=4)

    assert av.dtype is nm.Vector[nm.f8, 4]
    assert xv.dtype is nm.Vector[nm.f8, 4]
    assert nm.reduce_sum(xv).dtype is nm.f8
