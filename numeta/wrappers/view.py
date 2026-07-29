from __future__ import annotations
from typing import Any

from numeta.array_shape import ArrayShape
from numeta.ast.expressions import various as expr_various
from numeta.builder_helper import BuilderHelper
from numeta.datatype import ArrayType, get_datatype
from numeta.fortran.external_modules.iso_c_binding import iso_c
from numeta.ast.variable import Variable


def _normalize_shape_argument(shape_arg: Any, fortran_order: bool) -> ArrayShape:
    from numeta.datatype import size_t

    if isinstance(shape_arg, ArrayShape):
        return shape_arg

    if isinstance(shape_arg, (tuple, list)):
        return ArrayShape(tuple(shape_arg), fortran_order=fortran_order)

    shape_meta = getattr(shape_arg, "_shape", None)
    if (
        isinstance(shape_meta, ArrayShape)
        and not shape_meta.is_scalar
        and not shape_meta.is_unknown
    ):
        if shape_meta.rank != 1:
            raise ValueError("Shape array argument must be rank-1.")
        if shape_meta.rank == 0 or not isinstance(shape_meta.dim(0), int):
            raise ValueError("Shape array argument must have compile-time known length.")

        rank = shape_meta.dim(0)
        if isinstance(shape_arg, Variable):
            return ArrayShape.from_shape_vector(shape_arg, rank, fortran_order=fortran_order)

        tmp_shape = BuilderHelper.generate_local_variables(
            "nm_shape", dtype=size_t, shape=ArrayShape((rank,))
        )
        tmp_shape[:] = shape_arg
        return ArrayShape.from_shape_vector(tmp_shape, rank, fortran_order=fortran_order)

    return ArrayShape((shape_arg,), fortran_order=fortran_order)


def _shape_has_unknown_dims(shape: ArrayShape) -> bool:
    if shape.is_unknown:
        return True
    if shape.is_shape_vector:
        return False
    return any(dim is None for dim in shape.iter_dims())


def _fill_unknown_dims(shape: ArrayShape, source_shape: ArrayShape | None) -> ArrayShape:
    if source_shape is None:
        return shape
    if shape.is_unknown:
        return source_shape
    if source_shape.is_unknown:
        return shape
    if shape.rank != source_shape.rank:
        return shape
    if shape.is_shape_vector:
        return shape

    dims = []
    for i in range(shape.rank):
        dim = shape.dim(i)
        if dim is None:
            dims.append(source_shape.dim(i))
        else:
            dims.append(dim)
    return ArrayShape(tuple(dims), fortran_order=shape.fortran_order)


def _infer_default_array_shape(variable: Any, src_dtype: Any, dst_dtype: Any) -> ArrayShape | None:
    source_shape = getattr(variable, "_shape", None)
    if not isinstance(source_shape, ArrayShape):
        return None
    if source_shape.is_scalar or source_shape.is_unknown:
        return None

    src_bytes = src_dtype.get_nbytes()
    dst_bytes = dst_dtype.get_nbytes()

    if src_bytes == dst_bytes:
        return source_shape

    if source_shape.is_shape_vector:
        dims = tuple(source_shape.iter_dims())
    else:
        dims = source_shape.as_tuple()
        if any(dim is None for dim in dims):
            return None

    total_elements = 1
    for dim in dims:
        total_elements = total_elements * dim

    total_bytes = total_elements * src_bytes
    if not isinstance(total_bytes, int):
        return None
    return ArrayShape((total_bytes // dst_bytes,), fortran_order=False)


def view(variable: Any, dtype: Any, shape: Any = None) -> Variable:
    requested_shape = None
    if isinstance(dtype, ArrayType):
        requested_shape = dtype.shape
        dtype = dtype.dtype

    dtype = get_datatype(dtype)

    source_shape = getattr(variable, "_shape", None)
    source_fortran_order = (
        source_shape.fortran_order if isinstance(source_shape, ArrayShape) else False
    )

    if shape is not None:
        target_shape = _normalize_shape_argument(shape, fortran_order=source_fortran_order)
    elif requested_shape is not None:
        target_shape = _fill_unknown_dims(requested_shape, source_shape)
    else:
        src_dtype = getattr(variable, "dtype", None)
        if src_dtype is not None:
            target_shape = _infer_default_array_shape(variable, src_dtype, dtype)
        else:
            target_shape = None

    if isinstance(target_shape, ArrayShape):
        target_shape = BuilderHelper.normalize_allocation_shape(target_shape)
        if _shape_has_unknown_dims(target_shape):
            raise ValueError(
                "view() array shape has unresolved dimensions. "
                "Provide an explicit shape with known dimensions."
            )

    builder = BuilderHelper.get_current_builder()
    if target_shape is None:
        pointer = builder.generate_local_variables(
            "fc_v",
            dtype=dtype,
            pointer=True,
        )
    else:
        pointer = builder.generate_local_variables(
            "fc_v",
            dtype=dtype,
            shape=target_shape,
            pointer=True,
        )

    variable.target = True
    if target_shape is None:
        iso_c.c_f_pointer(iso_c.c_loc(variable), pointer)
    else:
        iso_c.c_f_pointer(
            iso_c.c_loc(variable),
            pointer,
            expr_various.ArrayConstructor(*tuple(target_shape.iter_dims())),
        )

    return pointer
