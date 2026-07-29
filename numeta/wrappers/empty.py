from __future__ import annotations
from typing import Any

from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType, float64, get_datatype, size_t
from numeta.array_shape import ArrayShape, SCALAR, UNKNOWN
from numeta.ast.variable import Variable
from numeta.settings import settings


def empty(
    shape: tuple[Any, ...] | list[Any] | int | ArrayShape | Variable,
    dtype: Any = float64,
    order: str = "C",
    name: str | None = None,
    force_dynamic_allocation: bool = False,
    allocation: str = "auto",
) -> Variable:
    def normalize_shape_argument(shape_arg):
        if isinstance(shape_arg, ArrayShape):
            return shape_arg

        if isinstance(shape_arg, (tuple, list)):
            return tuple(shape_arg)

        shape_meta = getattr(shape_arg, "_shape", None)
        if isinstance(shape_meta, ArrayShape) and shape_meta not in (SCALAR, UNKNOWN):
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

        return (shape_arg,)

    if order not in ["C", "F"]:
        raise ValueError(f"Invalid order: {order}, must be 'C' or 'F'")
    if allocation not in {"auto", "stack", "heap"}:
        raise ValueError("allocation must be 'auto', 'stack', or 'heap'")
    if force_dynamic_allocation:
        if allocation != "auto":
            raise ValueError(
                "force_dynamic_allocation=True cannot be combined with allocation != 'auto'"
            )
        allocation = "heap"

    fortran_order = order == "F"
    normalized_shape_arg = normalize_shape_argument(shape)
    if isinstance(normalized_shape_arg, ArrayShape):
        shape = normalized_shape_arg
    else:
        shape = ArrayShape(tuple(normalized_shape_arg), fortran_order=fortran_order)

    shape = BuilderHelper.normalize_allocation_shape(shape)

    dtype = get_datatype(dtype)

    if allocation == "stack" and shape.has_comptime_undefined_dims():
        try:
            builder = BuilderHelper.get_current_builder()
            backend = builder.numeta_function.backend
        except Exception:
            backend = None
        if backend == "c" and not settings.c_allow_vla_temporaries:
            raise ValueError(
                "allocation='stack' with runtime dimensions requires "
                "nm.settings.c_allow_vla_temporaries=True for the C backend"
            )

    if allocation == "heap":
        allocate = True
    elif allocation == "stack":
        allocate = False
    else:
        allocate = shape.has_comptime_undefined_dims()
    array = BuilderHelper.generate_local_variables(
        "fc_a",
        name=name,
        dtype=dtype,
        shape=shape,
        allocate=allocate,
    )

    return array
