from __future__ import annotations

from numeta.array_shape import SCALAR

from .intrinsic_functions import IntrinsicFunction


class Broadcast(IntrinsicFunction):
    token = "simd_broadcast"

    def __init__(self, value, lanes):
        super().__init__(value)
        self._lanes = int(lanes)

    @property
    def dtype(self):
        from numeta.datatype import make_vector_type

        return make_vector_type(self.arguments[0].dtype, self._lanes)

    @property
    def _shape(self):
        return SCALAR

    def get_with_updated_variables(self, variables_couples):
        value = self.arguments[0].get_with_updated_variables(variables_couples)
        return type(self)(value, self._lanes)


class VLoad(IntrinsicFunction):
    token = "simd_vload"

    def __init__(self, array, index, lanes, aligned=False):
        super().__init__(array, index)
        self._lanes = int(lanes)
        self.aligned = bool(aligned)

    @property
    def dtype(self):
        from numeta.datatype import make_vector_type

        return make_vector_type(self.arguments[0].dtype, self._lanes)

    @property
    def _shape(self):
        return SCALAR

    def get_with_updated_variables(self, variables_couples):
        array = self.arguments[0].get_with_updated_variables(variables_couples)
        index = self.arguments[1].get_with_updated_variables(variables_couples)
        return type(self)(array, index, self._lanes, aligned=self.aligned)


class VGather(IntrinsicFunction):
    token = "simd_vgather"

    def __init__(self, array, indices, lanes, offset=0):
        super().__init__(array, indices, offset)
        self._lanes = int(lanes)

    @property
    def dtype(self):
        from numeta.datatype import make_vector_type

        return make_vector_type(self.arguments[0].dtype, self._lanes)

    @property
    def _shape(self):
        return SCALAR

    def get_with_updated_variables(self, variables_couples):
        array = self.arguments[0].get_with_updated_variables(variables_couples)
        indices = self.arguments[1].get_with_updated_variables(variables_couples)
        offset = self.arguments[2].get_with_updated_variables(variables_couples)
        return type(self)(array, indices, self._lanes, offset=offset)


class Fma(IntrinsicFunction):
    token = "simd_fma"

    def __init__(self, a, b, c):
        super().__init__(a, b, c)

    @property
    def dtype(self):
        from numeta.type_rules import vector_common_dtype

        dtype = vector_common_dtype(*(arg.dtype for arg in self.arguments))
        return dtype if dtype is not None else self.arguments[0].dtype

    @property
    def _shape(self):
        return SCALAR


class Fnma(Fma):
    """Fused negative multiply-add: ``c - a*b``."""

    token = "simd_fnma"


class _VectorUnary(IntrinsicFunction):
    @property
    def dtype(self):
        return self.arguments[0].dtype

    @property
    def _shape(self):
        return SCALAR


class _VectorConversion(_VectorUnary):
    base_dtype = None

    @property
    def dtype(self):
        from numeta.datatype import make_vector_type

        return make_vector_type(self.base_dtype, self.arguments[0].dtype.lanes())


class VCvtF64ToF32(_VectorConversion):
    token = "simd_cvt_f64_f32"

    @property
    def base_dtype(self):
        from numeta import f4

        return f4


class VCvtF32ToF64(_VectorConversion):
    token = "simd_cvt_f32_f64"

    @property
    def base_dtype(self):
        from numeta import f8

        return f8


class VCvtF64ToI32(_VectorConversion):
    token = "simd_cvt_f64_i32"

    @property
    def base_dtype(self):
        from numeta import i4

        return i4


class VCvtI32ToF64(_VectorConversion):
    token = "simd_cvt_i32_f64"

    @property
    def base_dtype(self):
        from numeta import f8

        return f8


class VFloor(_VectorUnary):
    token = "simd_floor"


class VRcp(_VectorUnary):
    token = "simd_rcp"


class VRsqrt(_VectorUnary):
    token = "simd_rsqrt"


class VCompare(IntrinsicFunction):
    token = "simd_compare"
    predicates = frozenset({"lt", "le", "gt", "ge", "eq", "neq"})

    def __init__(self, a, b, predicate):
        predicate = str(predicate).lower()
        if predicate not in self.predicates:
            raise ValueError(f"Unsupported SIMD comparison {predicate!r}")
        super().__init__(a, b)
        self.predicate = predicate

    @property
    def dtype(self):
        from numeta.type_rules import vector_common_dtype

        return vector_common_dtype(*(arg.dtype for arg in self.arguments))

    @property
    def _shape(self):
        return SCALAR

    def get_with_updated_variables(self, variables_couples):
        args = [arg.get_with_updated_variables(variables_couples) for arg in self.arguments]
        return type(self)(*args, predicate=self.predicate)


class VBlend(IntrinsicFunction):
    token = "simd_blend"

    @property
    def dtype(self):
        from numeta.type_rules import vector_common_dtype

        return vector_common_dtype(self.arguments[0].dtype, self.arguments[1].dtype)

    @property
    def _shape(self):
        return SCALAR


class VMovemask(IntrinsicFunction):
    token = "simd_movemask"

    @property
    def dtype(self):
        from numeta import i4

        return i4

    @property
    def _shape(self):
        return SCALAR


class VExtractI32(IntrinsicFunction):
    token = "simd_extract_i32"

    def __init__(self, value, lane):
        super().__init__(value)
        self.lane = int(lane)
        lanes = value.dtype.lanes()
        if not 0 <= self.lane < lanes:
            raise ValueError(f"SIMD extraction lane must be in [0, {lanes})")

    @property
    def dtype(self):
        from numeta import i4

        return i4

    @property
    def _shape(self):
        return SCALAR

    def get_with_updated_variables(self, variables_couples):
        value = self.arguments[0].get_with_updated_variables(variables_couples)
        return type(self)(value, self.lane)


class VSet4F64(IntrinsicFunction):
    token = "simd_set4_f64"

    @property
    def dtype(self):
        from numeta import f8
        from numeta.datatype import make_vector_type

        return make_vector_type(f8, 4)

    @property
    def _shape(self):
        return SCALAR


class VExp2NegI32(VSet4F64):
    """Construct four exact normal binary64 values ``2**(-k)``."""

    token = "simd_exp2_neg_i32"


class ReduceSum(IntrinsicFunction):
    token = "simd_reduce_sum"

    def __init__(self, value):
        super().__init__(value)

    @property
    def dtype(self):
        from numeta.type_rules import is_vector_dtype

        dtype = self.arguments[0].dtype
        if not is_vector_dtype(dtype):
            raise TypeError("reduce_sum requires a vector value")
        return dtype.base_dtype()

    @property
    def _shape(self):
        return SCALAR


class UnpackLow(IntrinsicFunction):
    token = "simd_unpack_low"

    @property
    def dtype(self):
        from numeta.type_rules import vector_common_dtype

        return vector_common_dtype(*(arg.dtype for arg in self.arguments))

    @property
    def _shape(self):
        return SCALAR


class UnpackHigh(UnpackLow):
    token = "simd_unpack_high"


class PairwiseHalvesSum(UnpackLow):
    token = "simd_pairwise_halves_sum"
