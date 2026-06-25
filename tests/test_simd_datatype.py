import pytest

import numeta as nm


def test_vector_type_identity():
    assert nm.Vector[nm.f8, 4] is nm.Vector[nm.f8, 4]


def test_vector_type_metadata():
    vec = nm.Vector[nm.f8, 8]
    assert vec.base_dtype() is nm.f8
    assert vec.lanes() == 8
    assert vec.get_nbytes() == 64
    assert vec.get_cnumpy() == "nm_vec_f64_8"


def test_vector_rejects_invalid_lanes():
    with pytest.raises(ValueError):
        nm.Vector[nm.f8, 0]


def test_vector_array_type_metadata():
    array_type = nm.Vector[nm.f8, 4][10]

    assert array_type.dtype is nm.Vector[nm.f8, 4]
    assert array_type.shape.as_tuple() == (10,)
