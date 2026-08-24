from __future__ import annotations
from typing import Any

from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType, get_datatype
from numeta.ast.tools import check_node
from numeta.array_shape import ArrayShape
from numeta.ast.variable import Variable
from numeta.exceptions import NumetaTypeError, raise_with_source

_MISSING = object()


def _is_dtype(value: Any) -> bool:
    try:
        get_datatype(value)
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def scalar(
    value_or_dtype: Any = _MISSING,
    value: Any = _MISSING,
    name: str | None = None,
    *,
    dtype: Any = None,
) -> Variable:
    """Create scalar storage, optionally initialized at the current trace location.

    Both the value-first API (``scalar(value, dtype=...)``) and the historical
    dtype-first API (``scalar(dtype, value)``) are supported.
    """
    if value is not _MISSING:
        if value_or_dtype is _MISSING or not _is_dtype(value_or_dtype):
            raise TypeError("scalar(dtype, value) requires a dtype as its first argument")
        if dtype is not None:
            raise TypeError("scalar() cannot specify dtype both positionally and by keyword")
        dtype = value_or_dtype
        # Historically ``None`` meant "uninitialized" in the dtype-first API.
        initializer = _MISSING if value is None else value
    elif _is_dtype(value_or_dtype) and dtype is None:
        # Backwards-compatible uninitialized scalar(dtype).
        dtype = value_or_dtype
        initializer = _MISSING
    else:
        initializer = value_or_dtype

    if initializer is _MISSING:
        if dtype is None:
            raise TypeError("scalar() requires a value or dtype")
    else:
        initializer_node = check_node(initializer)
        initializer_shape = getattr(initializer_node, "_shape", None)
        if not isinstance(initializer_shape, ArrayShape) or not initializer_shape.is_scalar:
            raise_with_source(
                NumetaTypeError,
                "nm.scalar(...) requires a scalar-shaped initializer. For an array, use "
                "nm.empty(shape, dtype=...) followed by a sliced assignment.",
                source_node=initializer_node,
            )
        if dtype is None:
            dtype = initializer_node.dtype

    dtype = get_datatype(dtype)

    var = BuilderHelper.generate_local_variables("fc_s", dtype=dtype, name=name)
    if initializer is not _MISSING:
        var[:] = initializer
    return var
