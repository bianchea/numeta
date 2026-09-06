import pytest

import numeta as nm


@pytest.mark.parametrize("comparison", [lambda a, b: a == b, lambda a, b: a != b])
def test_symbolic_equality_cannot_drive_python_if(backend, comparison):
    @nm.jit(backend=backend)
    def kernel(a, b):
        if comparison(a, b):
            return 1
        return 0

    with pytest.raises(nm.NumetaTypeError, match="with nm.If"):
        kernel(1, 1)


def test_boolean_composition_returns_logical_dtype(backend):
    @nm.jit(backend=backend)
    def kernel(a, b):
        return ~((a < b) | (a == b))

    assert kernel(3, 2)
    assert not kernel(2, 2)
    assert not kernel(1, 2)
