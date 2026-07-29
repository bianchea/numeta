"""Value conversion for scalar and SIMD expressions."""

from numeta.datatype import float32, float64, get_datatype, int32
from numeta.type_rules import is_vector_dtype

from .scalar import scalar
from .simd import VCvtF32ToF64, VCvtF64ToF32, VCvtF64ToI32, VCvtI32ToF64


def astype(value, dtype):
    """Numerically convert a scalar or SIMD value to ``dtype``."""
    dtype = get_datatype(dtype)
    source_dtype = value.dtype
    if is_vector_dtype(source_dtype):
        conversion = {
            (float64, float32): VCvtF64ToF32,
            (float32, float64): VCvtF32ToF64,
            (float64, int32): VCvtF64ToI32,
            (int32, float64): VCvtI32ToF64,
        }.get((source_dtype.base_dtype(), dtype))
        if conversion is None:
            raise NotImplementedError(
                f"SIMD astype from {source_dtype.base_dtype()} to {dtype} is not implemented"
            )
        return conversion(value)

    result = scalar(dtype)
    result[:] = value
    return result
