from __future__ import annotations

from typing import Any

import numpy as np

from numeta.array_shape import ArrayShape
from numeta.settings import settings
from numeta.indexing import to_fortran_index, to_fortran_slice_start, to_fortran_slice_stop
from numeta.exceptions import raise_with_source
from numeta.ast.statements.tools import print_block

syntax_settings = settings.syntax

from numeta.fortran.fortran_type import FortranType
from numeta.ast.nodes import NamedEntity
from numeta.ast.variable import Variable
from numeta.ast.expressions import (
    BinaryOperationNode,
    FunctionCall,
    GetAttr,
    GetItem,
    IntrinsicFunction,
    LiteralNode,
)
from numeta.ast.expressions.binary_operation_node import BinaryOperationNodeNoPar
from numeta.ast.expressions.various import ArrayConstructor, Im, Re
from numeta.ast.statements import (
    Allocate,
    Assignment,
    Break,
    Call,
    Case,
    Continue,
    Deallocate,
    Else,
    ElseIf,
    For,
    Halt,
    If,
    Import,
    InterfaceBlock,
    Return,
    Section,
    Switch,
    TypingPolicy,
    While,
)
from numeta.ast.statements.function_declaration import FunctionInterfaceDeclaration
from numeta.ast.statements.namespace_declaration import NamespaceDeclaration
from numeta.ast.statements.procedure_declaration import (
    ProcedureDeclaration,
    ProcedureInterfaceDeclaration,
)
from numeta.ast.statements.struct_type_declaration import StructTypeDeclaration
from numeta.ast.statements.variable_declaration import VariableDeclaration
from numeta.ast.statements.various import Comment, PointerAssignment, Print, SimpleStatement


def _cast_scalar_to_dtype(value: Any, dtype: Any) -> Any:
    np_type = dtype.get_numpy()
    if np_type is None:
        return value
    try:
        return np_type(value)
    except Exception:
        return value


def _render_real_literal(value: Any) -> str:
    if isinstance(value, np.generic):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _literal_blocks_from_dtype(
    value: Any, dtype: Any, source_node: object | None = None
) -> list[str]:
    from numeta.datatype import DataType

    if not (isinstance(dtype, type) and issubclass(dtype, DataType)):
        raise_with_source(
            TypeError,
            f"dtype must be a DataType subclass, got {dtype}",
            source_node=source_node,
        )
        raise AssertionError("unreachable")

    casted_value = _cast_scalar_to_dtype(value, dtype)
    ftype = dtype.get_fortran()
    kind = ftype.get_kind_str()
    if ftype.type == "type":
        return [f"{casted_value}"]
    if ftype.type == "integer":
        return [f"{int(casted_value)}_{kind}"]
    if ftype.type == "real":
        return [f"{_render_real_literal(casted_value)}_{kind}"]
    if ftype.type == "complex":
        real_part = getattr(casted_value, "real", None)
        imag_part = getattr(casted_value, "imag", None)
        if real_part is None or imag_part is None:
            complex_value = complex(casted_value)
            real_part = complex_value.real
            imag_part = complex_value.imag
        return [
            "(",
            f"{_render_real_literal(real_part)}_{kind}",
            ",",
            f"{_render_real_literal(imag_part)}_{kind}",
            ")",
        ]
    if ftype.type == "logical":
        return [f".true._{kind}" if bool(casted_value) else f".false._{kind}"]
    if ftype.type == "character":
        return [f'"{casted_value}"']

    raise_with_source(ValueError, f"Unknown type: {ftype.type}", source_node=source_node)
    raise AssertionError("unreachable")


def render_literal_blocks_from_dtype(
    value: Any, dtype: Any, source_node: object | None = None
) -> list[str]:
    return _literal_blocks_from_dtype(value, dtype, source_node=source_node)


def render_expr_blocks(expr: Any) -> list[str]:
    if expr is None:
        return [""]
    if isinstance(expr, LiteralNode):
        return _literal_blocks_from_dtype(expr.value, expr.dtype, source_node=expr)
    if isinstance(expr, (int, float, complex, bool, str, np.generic)):
        literal = LiteralNode(expr)
        return _literal_blocks_from_dtype(literal.value, literal.dtype, source_node=literal)
    if isinstance(expr, Variable):
        return [expr.name]
    if isinstance(expr, NamedEntity):
        return [expr.name]
    if isinstance(expr, BinaryOperationNodeNoPar):
        return [
            *render_expr_blocks(expr.left),
            expr.op,
            *render_expr_blocks(expr.right),
        ]
    if isinstance(expr, BinaryOperationNode):
        return [
            "(",
            *render_expr_blocks(expr.left),
            expr.op,
            *render_expr_blocks(expr.right),
            ")",
        ]
    if isinstance(expr, FunctionCall):
        result = [expr.function.name, "("]
        for arg in expr.arguments:
            result.extend(render_expr_blocks(arg))
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        result.append(")")
        return result
    if isinstance(expr, GetAttr):
        return [*render_expr_blocks(expr.variable), "%", expr.attr]
    if isinstance(expr, GetItem):
        return _render_getitem_blocks(expr)
    if isinstance(expr, IntrinsicFunction):
        result = [expr.token, "("]
        for arg in expr.arguments:
            result.extend(render_expr_blocks(arg))
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        result.append(")")
        return result
    if isinstance(expr, ArrayConstructor):
        result = ["["]
        for element in expr.elements:
            if element is None:
                result.append("None")
            else:
                result.extend(render_expr_blocks(element))
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        result.append("]")
        return result
    if isinstance(expr, Re):
        return [*render_expr_blocks(expr.variable), "%", "re"]
    if isinstance(expr, Im):
        return [*render_expr_blocks(expr.variable), "%", "im"]
    return [str(expr)]


def _render_index_expr(expr: Any) -> list[str]:
    if isinstance(expr, LiteralNode) and isinstance(expr.value, (int, np.integer)):
        return [str(int(expr.value))]
    if isinstance(expr, (int, np.integer)):
        return [str(int(expr))]
    return render_expr_blocks(expr)


def _render_getitem_blocks(expr: GetItem) -> list[str]:
    result = render_expr_blocks(expr.variable)

    def render_block(block: Any) -> list[str]:
        if isinstance(block, slice):
            return _render_slice_blocks(block)
        return _render_index_expr(to_fortran_index(block))

    result.append("(")
    if isinstance(expr.sliced, tuple):
        dims = []
        for element in expr.sliced:
            dims.append(render_block(element))
        if not expr.variable._shape.fortran_order:
            dims = dims[::-1]
        result += dims[0]
        for dim in dims[1:]:
            result += [",", " "] + dim
    else:
        result += render_block(expr.sliced)
    result.append(")")
    return result


def _render_slice_blocks(slice_: slice) -> list[str]:
    result: list[str] = []
    result += _render_index_expr(to_fortran_slice_start(slice_.start))
    result.append(":")
    if slice_.stop is not None:
        stop = to_fortran_slice_stop(slice_.stop)
        result += _render_index_expr(stop)
    if slice_.step is not None:
        result.append(":")
        result += _render_index_expr(slice_.step)
    return result


def render_stmt_lines(stmt: Any, indent: int = 0) -> list[str]:
    if stmt is None:
        return []
    if isinstance(stmt, str):
        return [print_block([stmt], indent=indent)]
    if isinstance(stmt, Comment):
        prefix = getattr(stmt, "prefix", "! ")
        return [print_block(stmt.comment, indent=indent, prefix=prefix)]
    blocks = _render_stmt_blocks(stmt)
    if blocks is not None:
        return [print_block(blocks, indent=indent)]

    if isinstance(stmt, (For, While, If, ElseIf, Else, Switch, Case, InterfaceBlock)):
        return _render_scoped_stmt_lines(stmt, indent)
    if isinstance(
        stmt,
        (
            NamespaceDeclaration,
            ProcedureDeclaration,
            ProcedureInterfaceDeclaration,
            FunctionInterfaceDeclaration,
            StructTypeDeclaration,
        ),
    ):
        return _render_scoped_stmt_lines(stmt, indent)
    return [print_block(["! unsupported statement"], indent=indent)]


def _render_stmt_blocks(stmt: Any) -> list[str] | None:
    if isinstance(stmt, Import):
        result = ["use", " ", stmt.module.name]
        if stmt.only is not None:
            result += [", ", "only", ": ", stmt.only.name]
        return result
    if isinstance(stmt, TypingPolicy):
        return ["implicit", " ", stmt.implicit_type]
    if isinstance(stmt, Assignment):
        return [*render_expr_blocks(stmt.target), "=", *render_expr_blocks(stmt.value)]
    if isinstance(stmt, Break):
        return ["exit"]
    if isinstance(stmt, Continue):
        return ["cycle"]
    if isinstance(stmt, Halt):
        return ["stop"]
    if isinstance(stmt, SimpleStatement):
        token = getattr(stmt.__class__, "token", "")
        return [token]
    if isinstance(stmt, Return):
        return ["return"]
    if isinstance(stmt, Print):
        result = ["print *, "]
        for child in stmt.children:
            if isinstance(child, str):
                result.append(f"\"{child.replace('"', '""')}\"")
            else:
                result += render_expr_blocks(child)
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        return result
    if isinstance(stmt, Call):
        if isinstance(stmt.function, str):
            result = ["call", " ", stmt.function]
        else:
            result = ["call", " ", stmt.function.name]
        result.append("(")
        for arg in stmt.arguments:
            result += render_expr_blocks(arg)
            result += [", "]
        if result[-1] == ", ":
            result.pop()
        result.append(")")
        return result
    if isinstance(stmt, Allocate):
        return _render_allocate_blocks(stmt)
    if isinstance(stmt, Deallocate):
        return ["deallocate", "(", *render_expr_blocks(stmt.array), ")"]
    if isinstance(stmt, Section):
        return ["contains"]
    if isinstance(stmt, PointerAssignment):
        return [
            *render_expr_blocks(stmt.pointer),
            *_render_shape_blocks(
                stmt.pointer_shape,
                fortran_order=True,
                source_node=stmt.pointer,
            ),
            "=>",
            *render_expr_blocks(stmt.target),
        ]
    if isinstance(stmt, VariableDeclaration):
        return _render_variable_declaration_blocks(stmt)
    return None


def _render_allocate_blocks(stmt: Allocate) -> list[str]:
    result = ["allocate", "("]
    result += render_expr_blocks(stmt.target)

    dims = []
    for argument in stmt.shape:
        dims.append([*render_expr_blocks(argument)])

    if not stmt.target._shape.fortran_order:
        dims = dims[::-1]

    result.append("(")
    result += dims[0]
    for dim in dims[1:]:
        result += [",", " "] + dim
    result.append(")")

    result.append(")")
    return result


def _render_scoped_stmt_lines(stmt: Any, indent: int) -> list[str]:
    start_blocks = _render_scoped_start_blocks(stmt)
    end_blocks = _render_scoped_end_blocks(stmt)
    result = [print_block(start_blocks, indent=indent)]
    for child in _render_scoped_statements(stmt):
        if isinstance(stmt, If) and isinstance(child, (ElseIf, Else)):
            result.extend(render_stmt_lines(child, indent=indent))
        else:
            result.extend(render_stmt_lines(child, indent=indent + 1))
    if end_blocks:
        result.append(print_block(end_blocks, indent=indent))
    return result


def _render_scoped_start_blocks(stmt: Any) -> list[str]:
    if isinstance(stmt, For):
        result = ["do", " "]
        result += render_expr_blocks(stmt.iterator)
        result += [" ", "=", " "]
        result += render_expr_blocks(stmt.start)
        result.append(", ")
        result += render_expr_blocks(stmt.end)
        if stmt.step is not None:
            result.append(", ")
            result += render_expr_blocks(stmt.step)
        return result
    if isinstance(stmt, While):
        return ["do while", " ", "(", *render_expr_blocks(stmt.condition), ")"]
    if isinstance(stmt, If):
        return ["if", "(", *render_expr_blocks(stmt.condition), ")", "then"]
    if isinstance(stmt, ElseIf):
        return ["elseif", "(", *render_expr_blocks(stmt.condition), ")", "then"]
    if isinstance(stmt, Else):
        return ["else"]
    if isinstance(stmt, Switch):
        return ["select", " ", "case", " ", "(", *render_expr_blocks(stmt.value), ")"]
    if isinstance(stmt, Case):
        return ["case", " ", "(", *render_expr_blocks(stmt.value), ")"]
    if isinstance(stmt, InterfaceBlock):
        return ["interface"]
    if isinstance(stmt, NamespaceDeclaration):
        return ["module", " ", stmt.namespace.name]
    if isinstance(stmt, ProcedureDeclaration):
        return _render_procedure_start_blocks(stmt.procedure)
    if isinstance(stmt, ProcedureInterfaceDeclaration):
        return _render_procedure_start_blocks(stmt.procedure)
    if isinstance(stmt, FunctionInterfaceDeclaration):
        return _render_function_interface_start_blocks(stmt.function)
    if isinstance(stmt, StructTypeDeclaration):
        if syntax_settings.struct_type_bind_c:
            return ["type", ", ", "bind(C)", " ", "::", " ", stmt.struct_type.name]
        return ["type", " ", "::", " ", stmt.struct_type.name]
    raise_with_source(
        NotImplementedError,
        f"Unsupported scoped statement: {type(stmt)}",
        source_node=stmt,
    )
    raise AssertionError("unreachable")


def _render_scoped_end_blocks(stmt: Any) -> list[str]:
    if isinstance(stmt, (For, While)):
        return ["end", " ", "do"]
    if isinstance(stmt, If):
        return ["end", " ", "if"]
    if isinstance(stmt, (ElseIf, Else)):
        return []
    if isinstance(stmt, Switch):
        return ["end", " ", "select"]
    if isinstance(stmt, Case):
        return []
    if isinstance(stmt, InterfaceBlock):
        return ["end", " ", "interface"]
    if isinstance(stmt, NamespaceDeclaration):
        return ["end", " ", "module", " ", stmt.namespace.name]
    if isinstance(stmt, ProcedureDeclaration):
        return ["end", " ", "subroutine", " ", stmt.procedure.name]
    if isinstance(stmt, ProcedureInterfaceDeclaration):
        return ["end", " ", "subroutine", " ", stmt.procedure.name]
    if isinstance(stmt, FunctionInterfaceDeclaration):
        return ["end", " ", "function", " ", stmt.function.name]
    if isinstance(stmt, StructTypeDeclaration):
        return ["end", " ", "type", " ", stmt.struct_type.name]
    raise_with_source(
        NotImplementedError,
        f"Unsupported scoped statement: {type(stmt)}",
        source_node=stmt,
    )
    raise AssertionError("unreachable")


def _render_scoped_statements(stmt: Any) -> list[Any]:
    if isinstance(stmt, If):
        result = list(stmt.scope.get_statements())
        for branch in stmt.orelse:
            result.append(branch)
        return result
    if isinstance(stmt, ElseIf):
        return list(stmt.scope.get_statements())
    if isinstance(stmt, Else):
        return list(stmt.scope.get_statements())
    if isinstance(stmt, (For, While, Switch, Case)):
        return list(stmt.scope.get_statements())
    if isinstance(stmt, InterfaceBlock):
        return [method.get_interface_declaration() for method in stmt.methods]
    if isinstance(stmt, NamespaceDeclaration):
        return list(stmt.get_statements())
    if isinstance(stmt, ProcedureDeclaration):
        return list(stmt.get_statements())
    if isinstance(stmt, ProcedureInterfaceDeclaration):
        return list(stmt.get_statements())
    if isinstance(stmt, FunctionInterfaceDeclaration):
        return list(stmt.get_statements())
    if isinstance(stmt, StructTypeDeclaration):
        return list(stmt.get_statements())
    return []


def _render_procedure_start_blocks(procedure: Any) -> list[str]:
    result: list[str] = []

    if getattr(procedure, "pure", False):
        result += ["pure", " "]
    if getattr(procedure, "elemental", False):
        result += ["elemental", " "]

    result.extend(["subroutine", " ", procedure.name, "("])

    from numeta.ast.namespace import ExternalNamespace

    is_external = isinstance(getattr(procedure, "parent", None), ExternalNamespace)
    for variable in procedure.arguments.values():
        if variable.intent is None and not is_external:
            continue
        result.extend(render_expr_blocks(variable))
        result.append(", ")

    if result[-1] == ", ":
        result.pop()
    result.append(")")

    if getattr(procedure, "bind_c", False):
        result.extend([" ", f"bind(C, name='{procedure.name}')"])

    return result


def _render_function_interface_start_blocks(function: Any) -> list[str]:
    result = ["function", " ", function.name, "("]

    for variable in function.arguments:
        result.extend(render_expr_blocks(variable))
        result.append(", ")

    if result[-1] == ", ":
        result.pop()
    result.append(")")

    result_variable = function.get_result_variable()
    result.extend([" ", "result", "(", result_variable.name, ")"])

    if getattr(function, "bind_c", False):
        result.extend([" ", f"bind(C, name='{function.name}')"])

    return result


def _render_variable_declaration_blocks(stmt: VariableDeclaration) -> list[str]:
    result = _render_type_blocks(stmt.variable.dtype, source_node=stmt.variable)

    if stmt.variable.allocatable:
        result += [", ", "allocatable", ", ", "dimension"]
        result += ["("] + [":", ","] * (stmt.variable._shape.rank - 1) + [":", ")"]
    elif stmt.variable.pointer:
        result += [", ", "pointer"]
        if not stmt.variable._shape.is_scalar:
            result += [", ", "dimension"]
            result += ["("] + [":", ","] * (stmt.variable._shape.rank - 1) + [":", ")"]
    elif stmt.variable._shape.is_unknown:
        result += [", ", "dimension", "(", "1", ":", "*", ")"]
    elif stmt.variable._shape.rank > 0:
        result += [", ", "dimension"]
        result += _render_shape_blocks(
            stmt.variable._shape.as_tuple(),
            fortran_order=stmt.variable._shape.fortran_order,
            source_node=stmt.variable,
        )

    if stmt.variable.intent is not None:
        result += [", ", "intent", "(", stmt.variable.intent, ")"]

    if syntax_settings.force_value:
        if stmt.variable._shape.is_scalar and stmt.variable.intent == "in":
            result += [", ", "value"]

    if stmt.variable.parameter:
        result += [", ", "parameter"]

    if stmt.variable.target:
        result += [", ", "contiguous" if stmt.variable.pointer else "target"]

    if stmt.variable.bind_c:
        result += [", ", "bind", "(", "C", ", ", "name=", "'", stmt.variable.name, "'", ")"]

    result += [" :: ", stmt.variable.name]

    if stmt.variable.assign is not None:
        if isinstance(stmt.variable.assign, (int, float, complex, bool, str, np.generic)):
            values = _literal_blocks_from_dtype(
                stmt.variable.assign,
                stmt.variable.dtype,
                source_node=stmt.variable,
            )
        elif isinstance(stmt.variable.assign, np.ndarray):
            values = []
            for v in stmt.variable.assign.ravel():
                values += _literal_blocks_from_dtype(
                    v, stmt.variable.dtype, source_node=stmt.variable
                )
                values.append(", ")
            if values:
                values.pop()
        else:
            raise_with_source(
                ValueError,
                "Can only assign scalars or numpy ndarrays",
                source_node=stmt.variable,
            )
            raise AssertionError("unreachable")

        if stmt.variable._shape.is_unknown:
            raise_with_source(
                ValueError,
                "Cannot assign to a variable with unknown shape. "
                "Please specify the shape of the variable.",
                source_node=stmt.variable,
            )
            raise AssertionError("unreachable")
        result += [";", " data ", stmt.variable.name, " / ", *values, " /"]

    return result


def _render_shape_blocks(
    shape, fortran_order: bool = True, source_node: object | None = None
) -> list[str]:
    lbound = 1
    result = ["("]

    def convert(element):
        if element is None:
            return [str(lbound), ":", "*"]
        if isinstance(element, int):
            return [str(lbound), ":", str(element)]
        if isinstance(element, slice):
            start = _render_index_expr(to_fortran_slice_start(element.start))
            stop = (
                [""]
                if element.stop is None
                else _render_index_expr(to_fortran_slice_stop(element.stop))
            )

            if element.step is not None:
                raise_with_source(
                    NotImplementedError,
                    "Step in array dimensions is not implemented yet",
                    source_node=source_node,
                )
                raise AssertionError("unreachable")
            return start + [":"] + stop
        if isinstance(element, GetItem):
            base = render_expr_blocks(element.variable)

            def render_idx(idx):
                if isinstance(idx, slice):
                    start_blocks = _render_index_expr(to_fortran_slice_start(idx.start))
                    stop_blocks = (
                        [""]
                        if idx.stop is None
                        else _render_index_expr(to_fortran_slice_stop(idx.stop))
                    )
                    return [*start_blocks, ":", *stop_blocks]
                return _render_index_expr(to_fortran_index(idx))

            if isinstance(element.sliced, tuple):
                dims = [render_idx(idx) for idx in element.sliced]
                if not element.variable._shape.fortran_order:
                    dims = dims[::-1]
                blocks = [*base, "(", *dims[0]]
                for dim in dims[1:]:
                    blocks += [",", " "] + dim
                blocks.append(")")
                return [str(lbound), ":", *blocks]
            return [str(lbound), ":", *base, "(", *render_idx(element.sliced), ")"]
        return [str(lbound), ":", *render_expr_blocks(element)]

    if isinstance(shape, tuple):
        dims = [convert(d) for d in shape]
        if not fortran_order:
            dims = dims[::-1]
        result += dims[0]
        for dim in dims[1:]:
            result += [",", " "] + dim
    else:
        result += convert(shape)

    result += [")"]
    return result


def _render_type_blocks(dtype, source_node: object | None = None) -> list[str]:
    from numeta.datatype import DataType

    if isinstance(dtype, type) and issubclass(dtype, DataType):
        # Convert DataType to Fortran type representation
        ftype = dtype.get_fortran()
        kind = ftype.get_kind_str()
        if ftype.kind is not None:
            return [ftype.type, "(", kind, ")"]
        return [ftype.type]

    raise_with_source(
        TypeError,
        f"dtype must be a DataType subclass, got {dtype}",
        source_node=source_node,
    )
    raise AssertionError("unreachable")
