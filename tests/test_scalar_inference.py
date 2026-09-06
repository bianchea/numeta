import numpy as np

import numeta as nm


def test_scalar_infers_numpy_scalar_and_symbolic_initializer(backend):
    @nm.jit(backend=backend)
    def kernel(out):
        value = nm.scalar(np.float32(1.5))
        snapshot = nm.scalar(value + np.float32(2.5))
        legacy_keyword = nm.scalar(dtype=nm.f4, value=4.0)
        out[0] = value
        out[1] = snapshot
        out[2] = legacy_keyword

    out = np.zeros(3, dtype=np.float32)
    kernel(out)
    np.testing.assert_array_equal(out, [1.5, 4.0, 4.0])
