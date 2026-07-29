from __future__ import annotations
from typing import Any

from numeta.datatype import DataType, float64
from numeta.fortran.fortran_type import FortranType
from numeta.ast.variable import Variable
from .empty import empty


def zeros(
    shape: Any,
    dtype: Any = float64,
    order: str = "C",
    name: str | None = None,
    force_dynamic_allocation: bool = False,
    allocation: str = "auto",
) -> Variable:
    array = empty(
        shape,
        dtype=dtype,
        order=order,
        name=name,
        force_dynamic_allocation=force_dynamic_allocation,
        allocation=allocation,
    )
    array[:] = 0.0
    return array
