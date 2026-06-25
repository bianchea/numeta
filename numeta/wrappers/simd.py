from __future__ import annotations

from numeta.ast.expressions.simd_intrinsics import Broadcast, Fma, ReduceSum, VGather, VLoad
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


def reduce_sum(value):
    return ReduceSum(value)
