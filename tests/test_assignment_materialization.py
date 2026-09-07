import numeta as nm
import numpy as np
import pytest

CASES = [
    pytest.param("add", 10.0, 16.0, id="add"),
    pytest.param("sub", 6.0, 0.0, id="sub"),
    pytest.param("mul", 16.0, 128.0, id="mul"),
    pytest.param("div", 4.0, 0.5, id="div"),
]


def _make_scalar_loop_kernel(op: str, backend: str, *, materialized: bool):
    if op == "add":
        if materialized:

            @nm.jit(backend=backend)
            def kernel(n, out):
                value = nm.scalar(nm.f8, 8.0)
                for _ in nm.range(n):
                    value[:] += 2.0
                out[0] = value

        else:

            @nm.jit(backend=backend)
            def kernel(n, out):
                value = nm.scalar(nm.f8, 8.0)
                for _ in nm.range(n):
                    value += 2.0
                out[0] = value

        return kernel

    if op == "sub":
        if materialized:

            @nm.jit(backend=backend)
            def kernel(n, out):
                value = nm.scalar(nm.f8, 8.0)
                for _ in nm.range(n):
                    value[:] -= 2.0
                out[0] = value

        else:

            @nm.jit(backend=backend)
            def kernel(n, out):
                value = nm.scalar(nm.f8, 8.0)
                for _ in nm.range(n):
                    value -= 2.0
                out[0] = value

        return kernel

    if op == "mul":
        if materialized:

            @nm.jit(backend=backend)
            def kernel(n, out):
                value = nm.scalar(nm.f8, 8.0)
                for _ in nm.range(n):
                    value[:] *= 2.0
                out[0] = value

        else:

            @nm.jit(backend=backend)
            def kernel(n, out):
                value = nm.scalar(nm.f8, 8.0)
                for _ in nm.range(n):
                    value *= 2.0
                out[0] = value

        return kernel

    if op == "div":
        if materialized:

            @nm.jit(backend=backend)
            def kernel(n, out):
                value = nm.scalar(nm.f8, 8.0)
                for _ in nm.range(n):
                    value[:] /= 2.0
                out[0] = value

        else:

            @nm.jit(backend=backend)
            def kernel(n, out):
                value = nm.scalar(nm.f8, 8.0)
                for _ in nm.range(n):
                    value /= 2.0
                out[0] = value

        return kernel

    raise AssertionError(f"Unexpected op: {op}")


@pytest.mark.parametrize("op,rebound_expected,materialized_expected", CASES)
def test_scalar_loop_augassign_requires_explicit_slice(
    op, rebound_expected, materialized_expected, backend
):
    rebound_kernel = _make_scalar_loop_kernel(op, backend, materialized=False)
    materialized_kernel = _make_scalar_loop_kernel(op, backend, materialized=True)

    rebound_out = np.zeros(1, dtype=np.float64)
    materialized_out = np.zeros(1, dtype=np.float64)

    with pytest.raises(nm.NumetaTypeError, match=r"Bare storage.*runtime update") as exc_info:
        rebound_kernel(4, rebound_out)
    assert "x = x" in str(exc_info.value)
    assert "x[:]" in str(exc_info.value)
    materialized_kernel(4, materialized_out)

    np.testing.assert_allclose(
        materialized_out, np.array([materialized_expected], dtype=np.float64)
    )
