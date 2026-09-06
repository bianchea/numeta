import numpy as np
import pytest

import numeta as nm


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("lanes", [2, 4, 8, 16])
@pytest.mark.parametrize("arch", ["scalar", "avx2"])
def test_extract_lane_preserves_element_type_and_value(dtype, lanes, arch):
    @nm.jit(backend="c", simd_arch=arch)
    def kernel(a, out):
        value = nm.vload(a, 0, lanes=lanes)
        for lane in range(lanes):
            out[lane] = nm.extract_lane(value, lane)

    a = np.arange(lanes, dtype=dtype) + dtype(0.25 if np.issubdtype(dtype, np.floating) else 100)
    out = np.zeros_like(a)
    kernel(a, out)
    np.testing.assert_array_equal(out, a)
    assert "_mm_extract_epi32" not in kernel.source
