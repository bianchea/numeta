"""Validate typed numeric IR before a backend consumes it."""

from dataclasses import fields

from numeta.exceptions import NumetaTypeError, raise_with_source
from .nodes import (
    IRNode,
    IRExpr,
    IRVarRef,
    IRGetItem,
    IRGetAttr,
    IRIntrinsic,
    IRAssign,
    IRCall,
    IRCallExpr,
    IROpaqueExpr,
    IROpaqueStmt,
    IRAllocate,
    IRSlice,
    IRBinary,
)


def fail(node, reason):
    raise_with_source(NumetaTypeError, reason, source_node=node.source)


def validate_index(expr, *, scalar=False):
    dtype = expr.vtype.dtype.datatype if expr.vtype else None
    name = getattr(dtype, "_name", "")
    if not (name == "size_t" or name.startswith(("int", "uint"))):
        fail(
            expr,
            f"Invalid index. Expected an integer; received {name or 'untyped value'}. "
            "Use nm.astype(index, nm.i8) for an explicit integer conversion.",
        )
    if scalar and expr.vtype.shape is not None:
        fail(
            expr,
            "Invalid bound. Expected an integer scalar; received an array. Select one element.",
        )


def _assignable(target):
    if isinstance(target, IRVarRef):
        return not target.metadata.get("procedure_reference")
    if isinstance(target, (IRGetItem, IRGetAttr)):
        return _assignable(target.base)
    if isinstance(target, IRIntrinsic) and target.name in {"real", "aimag"}:
        return _assignable(target.args[0])
    return False


def validate_numeric_ir(procedure):
    seen = set()

    def visit(node, procedure_reference=False):
        if not isinstance(node, IRNode):
            return
        if isinstance(node, IRVarRef) and node.metadata.get("procedure_reference"):
            if not procedure_reference:
                fail(
                    node,
                    "Expected a numeric value; received a procedure. Call the procedure first.",
                )
            return
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, IROpaqueStmt):
            return
        if isinstance(node, IROpaqueExpr):
            fail(
                node,
                "Unsupported value. Expected a typed numeric expression. Use a Numeta operation.",
            )
        if isinstance(node, IRExpr) and (node.vtype is None or node.vtype.dtype.datatype is None):
            fail(
                node,
                "Incomplete expression type. Expected a supported dtype. Use nm.astype explicitly.",
            )
        if isinstance(node, IRAssign):
            target = node.target
            if not _assignable(target):
                fail(
                    node,
                    "Assignment requires storage. Use nm.scalar or nm.empty and a sliced assignment.",
                )
        if isinstance(node, IRGetItem):
            for index in node.indices:
                if isinstance(index, IRSlice):
                    for bound in (index.start, index.stop, index.step):
                        if bound is not None:
                            validate_index(bound, scalar=True)
                else:
                    validate_index(index)
        if isinstance(node, IRAllocate):
            for dim in node.dims:
                validate_index(dim, scalar=True)
        if isinstance(node, IRBinary) and node.op not in {
            "add",
            "sub",
            "mul",
            "div",
            "pow",
            "floordiv",
            "mod",
            "truncdiv",
            "truncmod",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "and",
            "or",
            "shl",
            "shr",
            "xor",
            "bitand",
            "bitor",
        }:
            fail(node, "Unsupported numeric operation. Use a supported Numeta arithmetic helper.")
        for field in fields(node):
            if field.name in {"source", "metadata", "vtype"}:
                continue
            value = getattr(node, field.name)
            if isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)
            else:
                visit(
                    value,
                    isinstance(node, (IRCall, IRCallExpr)) and field.name in {"func", "callee"},
                )

    visit(procedure)
