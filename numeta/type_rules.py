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
        return left_dtype
    if op in _COMPARISON_OPS or op in _LOGICAL_OPS:
        raise TypeError("Vector comparisons and logical operations are not supported yet")
    if op not in _ARITHMETIC_OPS:
        raise TypeError(f"Vector operation {op!r} is not supported yet")
    return vector_common_dtype(left_dtype, right_dtype)
