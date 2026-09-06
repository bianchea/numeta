import numpy as np
import pytest

import numeta as nm
from numeta.datatype import get_datatype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_round_large_values_and_halfway_cases(backend, dtype):
    @nm.jit(backend=backend)
    def rounded(value):
        return round(value)

    values = [
        0.0,
        0.5,
        1.5,
        2.5,
        3.5,
        2.25,
        2.75,
        3_000_000_000.25,
        3_000_000_000.5,
        3_000_000_001.5,
        2**40 + 0.5,
        2**40 + 1.5,
    ]
    if dtype == np.float64:
        values.append(np.nextafter(float(2**63), 0.0))
    for value in values:
        for sign in (1, -1):
            argument = dtype(sign * value)
            assert rounded(argument) == round(float(argument))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ndigits", [0, 1, -1])
def test_round_ndigits_preserves_large_floating_values(backend, dtype, ndigits):
    @nm.jit(backend=backend)
    def rounded(value, out):
        result = round(value, ndigits)
        assert result.dtype is get_datatype(dtype)
        out[0] = result

    out = np.empty(1, dtype=dtype)
    values = [0.0, 0.5, 1.5, 2.5, 3_000_000_000.5, 3_000_000_005.0, 1e20]
    if dtype == np.float64:
        values.append(1e100)
    for value in values:
        for sign in (1, -1):
            argument = dtype(sign * value)
            rounded(argument, out)
            assert out[0] == dtype(round(float(argument), ndigits))
            if out[0] == 0:
                assert np.signbit(out[0]) == np.signbit(argument)


@pytest.mark.parametrize("ndigits", [None, 0, 2])
def test_round_integer_does_not_lose_precision(backend, ndigits):
    @nm.jit(backend=backend)
    def rounded(value):
        return round(value) if ndigits is None else round(value, ndigits)

    for value in (2**60 + 1, -(2**60 + 1), 2**63 - 1, -(2**63)):
        assert rounded(np.int64(value)) == value


def test_round_ndigits_survives_inline_substitution(backend):
    @nm.jit(backend=backend, inline=True)
    def inner(value, out):
        out[0] = round(value, 1)

    @nm.jit(backend=backend)
    def outer(value, out):
        inner(value, out)

    out = np.zeros(1)
    outer(3_000_000_000.25, out)
    assert out[0] == round(3_000_000_000.25, 1)
