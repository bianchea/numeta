from __future__ import annotations


_ARITHMETIC_OPS = {"+", "-", "*", "/"}
_COMPARISON_OPS = {".eq.", ".ne.", ".lt.", ".le.", ".gt.", ".ge."}
_LOGICAL_OPS = {".and.", ".or."}


def is_vector_dtype(dtype) -> bool:
    return bool(getattr(dtype, "_is_vector", False))


def _same_dtype(left, right) -> bool:
    return left is right


def vector_common_dtype(*dtypes):
    vector_dtype = None
    for dtype in dtypes:
        if dtype is None:
            continue
        if not is_vector_dtype(dtype):
            continue
        if vector_dtype is None:
            vector_dtype = dtype
            continue
        if dtype is not vector_dtype:
            if dtype.base_dtype() is not vector_dtype.base_dtype():
                raise TypeError("Vector operands must have the same base dtype")
            if dtype.lanes() != vector_dtype.lanes():
                raise TypeError("Vector operands must have the same lane count")
    if vector_dtype is None:
        return None

    base_dtype = vector_dtype.base_dtype()
    for dtype in dtypes:
        if dtype is None or is_vector_dtype(dtype):
            continue
        if not _same_dtype(dtype, base_dtype):
            raise TypeError(
                "Vector-scalar operations require the scalar to match the vector base dtype"
            )
    return vector_dtype


def binary_result_dtype(left_dtype, right_dtype, op):
    if not (is_vector_dtype(left_dtype) or is_vector_dtype(right_dtype)):
        if op in _COMPARISON_OPS or op in _LOGICAL_OPS:
            from .datatype import bool8

            return bool8
        return left_dtype
    if op in _COMPARISON_OPS or op in _LOGICAL_OPS:
        raise TypeError("Vector comparisons and logical operations are not supported yet")
    if op not in _ARITHMETIC_OPS:
        raise TypeError(f"Vector operation {op!r} is not supported yet")
    return vector_common_dtype(left_dtype, right_dtype)


_UFUNCS = {
    "+": "add",
    "-": "subtract",
    "*": "multiply",
    "/": "true_divide",
    ".eq.": "equal",
    ".ne.": "not_equal",
    ".lt.": "less",
    ".le.": "less_equal",
    ".gt.": "greater",
    ".ge.": "greater_equal",
}


def validate_literal(value, dtype):
    """Check range without rejecting ordinary floating-point rounding."""
    import numpy as np
    from numeta.ast.expressions import LiteralNode
    from numeta.exceptions import raise_with_source

    if not isinstance(value, LiteralNode):
        return
    target = numpy_dtype(dtype)
    number = value.value
    if target.kind not in "biufc" or isinstance(number, str):
        return
    overflow = False
    if target.kind in "iu":
        limit = np.iinfo(target)
        try:
            number = int(number.real if isinstance(number, complex) else number)
            overflow = not limit.min <= number <= limit.max
        except (OverflowError, ValueError):
            overflow = True
    elif target.kind in "fc":
        limit = np.finfo(target).max
        if np.finfo(target).bits <= 64:
            limit = float(limit)
        parts = (number.real, number.imag) if np.iscomplexobj(number) else (number,)
        overflow = any(
            (isinstance(part, int) or np.isfinite(part)) and abs(part) > limit for part in parts
        )
    if overflow:
        raise_with_source(
            OverflowError,
            f"Literal overflows {target.name}. Expected a representable {target.name} value; "
            "received an out-of-range literal. Use a wider explicitly typed operand.",
            source_node=value,
        )


def _operand_specs(operands):
    import numpy as np
    from numeta.ast.expressions import LiteralNode

    all_literals = all(
        isinstance(value, LiteralNode) and type(value.value) in (int, float, complex, bool)
        for value in operands
    )
    return tuple(
        (
            type(value.value)
            if not all_literals
            and isinstance(value, LiteralNode)
            and type(value.value) in (int, float, complex)
            else numpy_dtype(value.dtype)
        )
        for value in operands
    )


def resolve_binary(left, right, op):
    """Return computation input dtypes and result dtype, preserving weak literals."""
    import numpy as np
    from numeta.datatype import get_datatype
    from numeta.exceptions import raise_with_source

    if is_vector_dtype(left.dtype) or is_vector_dtype(right.dtype) or op not in _UFUNCS:
        return (left.dtype, right.dtype), binary_result_dtype(left.dtype, right.dtype, op)
    try:
        specs = _operand_specs((left, right))
        if op in {".lt.", ".le.", ".gt.", ".ge."} and any(
            spec is complex or isinstance(spec, np.dtype) and spec.kind == "c" for spec in specs
        ):
            raise TypeError(
                "Ordered complex comparisons are unsupported; compare real parts or magnitudes"
            )
        resolved = getattr(np, _UFUNCS[op]).resolve_dtypes((*specs, None))
        dtypes = tuple(get_datatype(dtype) for dtype in resolved)
    except (TypeError, ValueError, KeyError) as error:
        raise_with_source(
            TypeError,
            f"Unsupported numeric operation {op!r}: {error}. Expected compatible numeric operands; "
            f"received {left.dtype._name} and {right.dtype._name}. Use nm.astype to convert explicitly.",
            source_node=left,
        )
    for value, dtype in zip((left, right), dtypes):
        validate_literal(value, dtype)
    return dtypes[:2], dtypes[2]


def resolve_selection(condition, left, right):
    import numpy as np
    from numeta.datatype import bool8, get_datatype
    from numeta.exceptions import raise_with_source

    if condition.dtype is not bool8:
        raise_with_source(
            TypeError,
            f"nm.where requires a Boolean condition. Expected bool8; received {condition.dtype._name}. "
            "Use an explicit comparison, such as condition != 0.",
            source_node=condition,
        )
    # The input loop of equal supplies common numeric promotion, including Boolean values.
    specs = _operand_specs((left, right))
    resolved = np.equal.resolve_dtypes((*specs, None))
    dtype = get_datatype(resolved[0])
    for value in (left, right):
        validate_literal(value, dtype)
    return dtype


def numpy_dtype(dtype):
    import numpy as np
    from numeta.datatype import size_t

    if dtype is size_t:
        return np.dtype(np.intp)
    if dtype.get_numpy() is None:
        raise TypeError(f"{dtype._name} is not a numeric dtype")
    return np.dtype(dtype.get_numpy())
