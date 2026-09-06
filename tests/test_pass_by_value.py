import pytest

from numeta.ast import Variable
import numeta as nm


def test_pass_by_value_scalar_allowed():
    v = Variable("x", dtype=nm.f8, pass_by_value=True)
    assert v.pass_by_value is True


def test_pass_by_value_array_raises():
    with pytest.raises(ValueError, match="scalar variables"):
        Variable("arr", dtype=nm.f8, shape=(3,), pass_by_value=True)
