import numeta as nm
import numpy as np
import time


def test_omp(backend):
    @nm.jit(backend=backend)
    def test_mul(a, b, c):
        n_threads = nm.scalar(nm.i8, nm.omp.omp_get_max_threads())
        i_thread = nm.scalar(nm.i8, 0)

        for j in nm.prange(
            b.shape[1],
            private=[i_thread],
            shared=[a, b, c, a.shape[0].variable, b.shape[0].variable],
            schedule="static",
        ):
            for k in nm.range(b.shape[0]):
                for i in nm.range(a.shape[0]):
                    i_thread[:] = nm.omp.omp_get_thread_num()

                    nm.omp.atomic_update_add(c[i, j].real, a[i, k].real * b[k, j].real)
                    nm.omp.atomic_update_sub(c[i, j].real, a[i, k].imag * b[k, j].imag)
                    nm.omp.atomic_update_add(c[i, j].imag, a[i, k].real * b[k, j].imag)
                    nm.omp.atomic_update_add(c[i, j].imag, a[i, k].imag * b[k, j].real)

    n = 50

    a = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    b = np.random.rand(n, n) + 1j * np.random.rand(n, n)

    c = np.zeros((n, n), dtype=np.complex128)

    test_mul(a, b, c)

    np.testing.assert_allclose(c, a.dot(b))


def test_time_wall_clock(backend):
    @nm.jit(backend=backend)
    def capture_time(out):
        out[0] = nm.time()

    t0 = np.zeros(1, dtype=np.float64)
    t1 = np.zeros(1, dtype=np.float64)

    capture_time(t0)
    time.sleep(0.01)
    capture_time(t1)

    assert t1[0] >= t0[0]


def test_time_elapsed_scalar_capture(backend):
    @nm.jit(backend=backend)
    def timed_accumulate(n, out):
        t0 = nm.scalar(nm.f8, nm.time())
        acc = nm.scalar(nm.f8, 0.0)
        for _ in nm.range(n):
            acc[:] = acc + 1.0
        t1 = nm.scalar(nm.f8, nm.time())
        out[0] = t1 - t0
        out[1] = acc

    out = np.zeros(2, dtype=np.float64)
    timed_accumulate(100_000, out)

    assert out[0] >= 0.0
    assert out[1] == 100_000.0
