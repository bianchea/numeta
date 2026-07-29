from __future__ import annotations
from typing import Any

import numpy as np
from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType, get_datatype
from numeta.array_shape import ArrayShape
from numeta.ast.variable import Variable


def constant(
    value: Any,
    dtype: Any = None,
    order: str = "C",
    name: str | None = None,
    *,
    static: bool = False,
) -> Variable:
    """Create a compile-time constant.

    ``static=True`` gives fixed C arrays static storage duration.  This keeps
    descriptor tables out of the stack frame and avoids reinitializing them on
    every generated-kernel call.  It is intentionally limited to C lowering;
    callers should not request it from a Fortran kernel.
    """
    if order not in ["C", "F"]:
        raise ValueError(f"Invalid order: {order}, must be 'C' or 'F'")

    builder = BuilderHelper.get_current_builder()
    if static and builder.numeta_function.backend != "c":
        raise ValueError("nm.constant(static=True) is supported only by the C backend")

    # Determine value shape and numpy representation
    if isinstance(value, np.ndarray):
        np_value = value
    elif isinstance(value, (list, tuple)):
        np_value = np.array(value)
    else:
        np_value = None

    if np_value is not None:
        shape = np_value.shape
        fortran_order = (
            np_value.flags.f_contiguous if isinstance(value, np.ndarray) else order == "F"
        )
        # Determine datatype if not provided
        if dtype is None:
            dtype = DataType.from_np_dtype(np_value.dtype.type)
        assign_value = np_value
    else:
        shape = None
        fortran_order = True
        if dtype is None:
            np_dtype = np.array(value).dtype.type
            dtype = DataType.from_np_dtype(np_dtype)
        assign_value = value

    dtype = get_datatype(dtype)

    if name is None:
        name = "fc_c"

    if shape is None or shape == ():
        return builder.generate_local_variables(
            name,
            dtype=dtype,
            parameter=static,
            c_static=static,
            # TODO
            # parameter=True, # parameter is not supported yet, so not really constant.
            # parameter=True,
            assign=assign_value,
        )

    array_shape = ArrayShape(shape, fortran_order=fortran_order)
    return builder.generate_local_variables(
        name,
        dtype=dtype,
        shape=array_shape,
        parameter=static,
        c_static=static,
        # TODO
        # parameter=True, # parameter is not supported yet, so not really constant.
        # parameter=True,
        assign=assign_value,
    )
