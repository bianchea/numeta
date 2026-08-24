from numeta.ast.expressions import BinaryOperationNode


def trunc_div(a, b):
    """Division with the native C/Fortran truncation-toward-zero rule."""
    return BinaryOperationNode(a, "truncdiv", b)


def trunc_mod(a, b):
    """Remainder paired with :func:`trunc_div`."""
    return BinaryOperationNode(a, "truncmod", b)
