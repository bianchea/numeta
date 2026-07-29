from __future__ import annotations

from numeta.ast.expressions.simd_intrinsics import (
    Broadcast,
    Fma,
    Fnma,
    PairwiseHalvesSum,
    ReduceSum,
    UnpackHigh,
    UnpackLow,
    VGather,
    VBlend,
    VCompare,
    VCvtF32ToF64,
    VCvtF64ToF32,
    VCvtF64ToI32,
    VCvtI32ToF64,
    VExp2NegI32,
    VExtractI32,
    VFloor,
    VLoad,
    VMovemask,
    VRcp,
    VRsqrt,
    VSet4F64,
)
from numeta.ast.statements.simd_store import VStore
from numeta.datatype import make_vector_type


def simd_lanes(dtype, arch=None):
    from numeta.settings import settings
    from numeta.simd.target import make_simd_target, native_lanes

    target = make_simd_target(arch or settings.default_simd_arch)
    return native_lanes(dtype, target)


def vector(dtype, lanes, value=None, name=None):
    from numeta.wrappers.scalar import scalar

    return scalar(make_vector_type(dtype, lanes), value=value, name=name)


def broadcast(value, lanes):
    return Broadcast(value, lanes)


def vload(array, index, lanes, aligned=False):
    return VLoad(array, index, lanes, aligned=aligned)


def vgather(array, indices, lanes, offset=0):
    return VGather(array, indices, lanes, offset=offset)


def vstore(array, index, value, aligned=False):
    return VStore(array, index, value, aligned=aligned)


def fma(a, b, c):
    return Fma(a, b, c)


def fnma(a, b, c):
    return Fnma(a, b, c)


def rcp_approx(value):
    return VRcp(value)


def rsqrt_approx(value):
    return VRsqrt(value)


def compare(a, b, predicate):
    return VCompare(a, b, predicate)


def where(mask, when_true, when_false):
    return VBlend(when_false, when_true, mask)


def mask_bits(mask):
    return VMovemask(mask)


def extract_lane(value, lane):
    return VExtractI32(value, lane)


def vector_from_values(a0, a1, a2, a3):
    return VSet4F64(a0, a1, a2, a3)


def exp2_neg(value):
    return VExp2NegI32(value)


def reduce_sum(value):
    return ReduceSum(value)


def unpack_low(a, b):
    return UnpackLow(a, b)


def unpack_high(a, b):
    return UnpackHigh(a, b)


def pairwise_halves_sum(a, b):
    return PairwiseHalvesSum(a, b)
