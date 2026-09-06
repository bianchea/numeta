"""Keep native matrix-layout regressions isolated from the pytest process."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import numeta as nm


def _exercise_rectangular_matmul(backend, left_order, right_order, debug):
    @nm.jit(backend=backend, debug=debug)
    def returned(a, b):
        return nm.matmul(a, b)

    @nm.jit(backend=backend, debug=debug)
    def assigned(a, b, out):
        out[:] = nm.matmul(a, b)

    @nm.jit(backend=backend, debug=debug)
    def copy_into(value, out):
        out[:] = value

    @nm.jit(backend=backend, debug=debug)
    def passed(a, b, out):
        copy_into(nm.matmul(a, b), out)

    for dtype in (np.float32, np.float64, np.int32, np.int64, np.complex64, np.complex128):
        a = np.array(np.arange(6).reshape(2, 3) - 2, dtype=dtype, order=left_order)
        b = np.array(np.arange(12).reshape(3, 4) - 3, dtype=dtype, order=right_order)
        if np.issubdtype(dtype, np.complexfloating):
            a += 2j
            b -= 1j
        expected = a @ b
        actual = returned(a, b)
        assert actual.shape == (2, 4)
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected)
        if backend == "fortran":
            native_left = "a" if left_order == "C" else "transpose(a)"
            native_right = "b" if right_order == "C" else "transpose(b)"
            source = "".join(returned.source.split()).replace("&", "")
            assert f"matmul({native_right},{native_left})" in source
        for order in ("C", "F"):
            for kernel in (assigned, passed):
                out = np.full((2, 4), -999, dtype=dtype, order=order)
                kernel(a, b, out)
                np.testing.assert_allclose(out, expected)

    if backend == "fortran":
        # Fortran also supports vector/matrix products; vectors have no order reversal.
        vector = np.array([1 + 2j, 3 - 1j, -2 + 1j])
        for left, right in ((vector, vector), (vector, b), (a, vector)):
            np.testing.assert_allclose(returned(left, right), left @ right)


@pytest.mark.parametrize("left_order,right_order", [("C", "C"), ("C", "F"), ("F", "C"), ("F", "F")])
@pytest.mark.parametrize("debug", [False, True])
def test_rectangular_matmul_layouts(backend, left_order, right_order, debug, tmp_path):
    script = (
        "import runpy; "
        f"namespace = runpy.run_path({str(Path(__file__).resolve())!r}); "
        f"namespace['_exercise_rectangular_matmul']({backend!r}, {left_order!r}, "
        f"{right_order!r}, {debug!r})"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Native matmul subprocess exited with {result.returncode}\n"
        f"{result.stdout}\n{result.stderr}"
    )


@pytest.mark.parametrize("shape", [(), (2, 3, 4)])
def test_matmul_rejects_unsupported_rank_before_compilation(backend, shape):
    @nm.jit(backend=backend)
    def kernel(a, b):
        return nm.matmul(a, b)

    with pytest.raises(nm.NumetaTypeError, match="nm.matmul requires rank-1 or rank-2"):
        kernel.specialize(np.ones(shape), np.ones((4, 2)))
