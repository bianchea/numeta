import numpy as np
import pytest

import numeta as nm


def test_bare_prange_infers_arrays_descriptors_and_local_storage(backend):
    @nm.jit(backend=backend)
    def kernel(a, b, scale):
        for i in nm.prange(a.shape[0], num_threads=2, schedule="dynamic", chunk=3):
            local = nm.scalar(a[i] * scale)
            b[i] = local + a.shape[0]

    a = np.arange(20, dtype=np.float64)
    b = np.zeros_like(a)
    kernel(a, b, 2.0)
    np.testing.assert_array_equal(b, a * 2 + 20)
    source = kernel.source
    assert "default(none)" in source
    assert "schedule(dynamic, 3)" in source
    assert "num_threads(2)" in source
    assert "private(" in source
    assert "shared(" in source
    assert ("shape_a" if backend == "fortran" else "a_dims") in source


def test_parallel_shared_scalar_write_is_rejected(backend):
    @nm.jit(backend=backend)
    def kernel(a):
        total = nm.scalar(0.0)
        for i in nm.prange(a.shape[0]):
            total[:] += a[i]
        return total

    with pytest.raises(nm.NumetaError, match="captured scalar.*private"):
        kernel(np.ones(20))


@pytest.mark.parametrize(
    "options",
    [
        {"schedule": "invalid"},
        {"num_threads": 0},
        {"chunk": -1},
        {"chunk": 2, "schedule": "runtime"},
        {"chunk": 2, "schedule": "auto"},
    ],
)
def test_parallel_invalid_options_fail_before_compilation(backend, options):
    @nm.jit(backend=backend)
    def kernel(a):
        for i in nm.prange(a.shape[0], **options):
            a[i] = 1.0

    with pytest.raises(ValueError, match="OpenMP"):
        kernel(np.zeros(2))


def test_parallel_overlapping_sharing_is_rejected(backend):
    @nm.jit(backend=backend)
    def kernel(a):
        for i in nm.prange(a.shape[0], shared=[a], private=[a]):
            a[i] = 1.0

    with pytest.raises(ValueError, match="overlap"):
        kernel(np.zeros(2))
