import pytest

import numeta as nm


@pytest.mark.parametrize("value", [-1, -2, -15, -(2**60), 2**60])
@pytest.mark.parametrize("shift", [0, 1, 7, 40])
def test_right_shift_is_signed(value, shift, backend):
    @nm.jit(backend=backend)
    def kernel(a, b):
        return a >> b

    assert kernel(value, shift) == value >> shift
