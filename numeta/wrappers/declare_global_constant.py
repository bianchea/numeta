from __future__ import annotations
from typing import Any, TYPE_CHECKING

import numpy as np
from numeta.ast import Variable as AstVariable, Namespace
from numeta.datatype import DataType, float64, get_datatype
from numeta.array_shape import ArrayShape, SCALAR, UNKNOWN
from numeta.numeta_function import NumetaCompiledFunction
from numeta.settings import settings

if TYPE_CHECKING:
    from numeta.numeta_library import NumetaLibrary

_n_global_constant: int = 0
_n_global_library: int = 0


def _next_global_library_name(name: str, backend: str) -> str:
    global _n_global_library
    library_name = f"{name}_namespace_{backend}_{_n_global_library}"
    _n_global_library += 1
    return library_name


def _normalize_global_constant_shape(shape, order: str) -> ArrayShape:
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

            if isinstance(shape_arg, AstVariable):
                assigned = getattr(shape_arg, "assign", None)
                if assigned is not None:
                    assigned_array = np.asarray(assigned).reshape(-1)
                    if assigned_array.size >= rank and np.issubdtype(
                        assigned_array.dtype, np.integer
                    ):
                        return tuple(int(assigned_array[i]) for i in range(rank))
                raise ValueError(
                    "declare_global_constant shape vectors must have compile-time integer values. "
                    "Dynamic shape vectors are not supported for global constants."
                )

            raise ValueError(
                "declare_global_constant shape vectors must have compile-time integer values. "
                "Dynamic shape vectors are not supported for global constants."
            )

        return (shape_arg,)

    normalized_shape_arg = normalize_shape_argument(shape)
    if isinstance(normalized_shape_arg, ArrayShape):
        return normalized_shape_arg

    return ArrayShape(tuple(normalized_shape_arg), fortran_order=order == "F")


def make_global_constant_target(
    shape: tuple[Any, ...] | list[Any] | int | ArrayShape | AstVariable,
    dtype: Any = float64,
    order: str = "C",
    name: str | None = None,
    value: Any = None,
    directory: str | None = None,
    backend: str | None = None,
) -> tuple[AstVariable, NumetaCompiledFunction]:
    if backend is None:
        backend = settings.default_backend
    if order not in ["C", "F"]:
        raise ValueError(f"Invalid order: {order}, must be 'C' or 'F'")

    normalized_shape = _normalize_global_constant_shape(shape, order)
    dtype_arg = get_datatype(dtype)

    if name is None:
        global _n_global_constant
        name = f"global_constant_{_n_global_constant}"
        _n_global_constant += 1

    # Lets create a namespace to host the global_constant variable.
    global_constant_namespace = Namespace(f"{name}_namespace")

    var = AstVariable(
        name=name,
        dtype=dtype_arg,
        shape=normalized_shape,
        assign=value,
        # TODO
        # parameter=True, # parameter is not supported yet, so not really constant.
        parent=global_constant_namespace,
    )

    # We have to compile the namespace when needed
    namespace_library = NumetaCompiledFunction(
        f"{name}_namespace",
        global_constant_namespace,
        library_name=_next_global_library_name(name, backend),
        path=directory,
        backend=backend,
    )
    global_constant_namespace.parent = namespace_library

    return var, namespace_library


def declare_global_constant(
    shape: tuple[Any, ...] | list[Any] | int | ArrayShape | AstVariable,
    dtype: Any = float64,
    order: str = "C",
    name: str | None = None,
    value: Any = None,
    directory: str | None = None,
    backend: str | None = None,
    library: NumetaLibrary | None = None,
) -> AstVariable:
    var, namespace_library = make_global_constant_target(
        shape,
        dtype=dtype,
        order=order,
        name=name,
        value=value,
        directory=directory,
        backend=backend,
    )

    if library is not None:
        library._nm_add_global(namespace_library)

    return var
