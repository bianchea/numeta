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
