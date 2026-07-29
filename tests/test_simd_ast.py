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


@pytest.mark.parametrize(
    "operation",
    [nm.unpack_low, nm.unpack_high, nm.pairwise_halves_sum],
)
def test_lane_permutation_intrinsics_preserve_vector_dtype(operation):
    vec = nm.Vector[nm.f8, 4]
    a = Variable("a", dtype=vec)
    b = Variable("b", dtype=vec)

    expr = operation(a, b)

    assert expr.dtype is vec
    assert expr._shape is SCALAR


def test_lane_permutation_intrinsics_reject_mismatched_lanes():
    a = Variable("a", dtype=nm.Vector[nm.f8, 4])
    b = Variable("b", dtype=nm.Vector[nm.f8, 8])

    with pytest.raises(TypeError):
        _ = nm.unpack_low(a, b).dtype


def test_radial_simd_conversion_dtypes_and_validation():
    f64x4 = Variable("f64x4", dtype=nm.Vector[nm.f8, 4])
    f32x4 = nm.astype(f64x4, nm.f4)
    i32x4 = nm.astype(f64x4, nm.i4)

    assert f32x4.dtype is nm.Vector[nm.f4, 4]
    assert nm.astype(f32x4, nm.f8).dtype is nm.Vector[nm.f8, 4]
    assert i32x4.dtype is nm.Vector[nm.i4, 4]
    assert nm.astype(i32x4, nm.f8).dtype is nm.Vector[nm.f8, 4]
    assert nm.extract_lane(i32x4, 3).dtype is nm.i4

    with pytest.raises(ValueError, match="extraction lane"):
        nm.extract_lane(i32x4, 4)
    with pytest.raises(ValueError, match="comparison"):
        nm.compare(f64x4, f64x4, "unsupported")
