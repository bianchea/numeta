from __future__ import annotations

from typing import Any

import numpy as np

from numeta.settings import settings
from numeta.ast.statements.tools import print_block

syntax_settings = settings.syntax

from .fortran_syntax import (
    render_expr_blocks,
    render_stmt_lines,
    render_literal_blocks_from_dtype,
)

from numeta.ir.nodes import (
    IRAllocate,
    IRAssign,
    IRBinary,
    IRCall,
    IRCallExpr,
    IRDeallocate,
    IRExpr,
    IRFor,
    IRGetAttr,
    IRGetItem,
    IRIf,
    IRIntrinsic,
    IRLiteral,
    IRProcedure,
    IRPrint,
    IRReturn,
    IRSlice,
    IRUnary,
    IRVarRef,
    IRWhile,
    IROpaqueExpr,
    IROpaqueStmt,
)

_FORTRAN_BINARY_OPS = {
    "eq": ".eq.",
    "ne": ".ne.",
    "lt": ".lt.",
    "le": ".le.",
    "gt": ".gt.",
    "ge": ".ge.",
    "and": ".and.",
    "or": ".or.",
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "pow": "**",
}

# Mapping for intrinsics that differ in Fortran
_FORTRAN_INTRINSIC_MAP = {
    "ceil": "ceiling",
    "copysign": "sign",
}


class FortranEmitter:
    def __init__(self) -> None:
        self._shape_descriptor_by_base: dict[str, Any] = {}

    def emit_procedure(self, proc: IRProcedure) -> str:
        self._shape_descriptor_by_base = {}
        args_by_name = {arg.name: arg for arg in proc.args}
        for arg in proc.args:
            if arg.name.startswith("shape_"):
                base = arg.name[len("shape_") :]
                if base in args_by_name:
                    self._shape_descriptor_by_base[base] = arg

        pure = bool(proc.metadata.get("fortran_pure", False))
        elemental = bool(proc.metadata.get("fortran_elemental", False))
        bind_c = bool(proc.metadata.get("fortran_bind_c", False))

        args = [arg.name for arg in proc.args]

        start_blocks: list[str] = []
        if pure:
            start_blocks += ["pure", " "]
        if elemental:
            start_blocks += ["elemental", " "]
        start_blocks += ["subroutine", " ", proc.name, "("]
        for name in args:
            start_blocks += [name, ", "]
        if args:
            start_blocks.pop()
        start_blocks += [")"]
        if bind_c:
            start_blocks += [" ", f"bind(C, name='{proc.name}')"]

        lines: list[str] = []
        lines.append(print_block(start_blocks, indent=0))

        prelude_lines = proc.metadata.get("fortran_prelude_lines", [])
        prelude_items = proc.metadata.get("fortran_prelude_items", [])
        if prelude_lines:
            lines.extend(prelude_lines)
        elif prelude_items:
            for stmt in prelude_items:
                lines.extend(render_stmt_lines(stmt, indent=1))
        else:
            lines.append(print_block(["implicit", " ", "none"], indent=1))

        for var in proc.args:
            lines.append(self._declare_variable(var, indent=1))
        for var in proc.locals:
            lines.append(self._declare_variable(var, indent=1))

        for stmt in proc.body:
            lines.extend(self._emit_stmt(stmt, indent=1))

        lines.append(print_block(["end", " ", "subroutine", " ", proc.name], indent=0))
        return "".join(lines)

    def _emit_stmt(self, stmt, *, indent: int) -> list[str]:
        stmt_type = type(stmt)
        if stmt_type is IRAssign:
            target_shape = stmt.target.vtype.shape if stmt.target.vtype else None
            value = (
                self._expr_blocks_in_order(stmt.value, target_shape.order)
                if target_shape is not None and target_shape.rank == 2
                else self._expr_blocks(stmt.value)
            )
            blocks = self._expr_blocks(stmt.target) + ["="] + value
            return [print_block(blocks, indent=indent)]
        if stmt_type is IRCall:
            blocks = ["call", " "] + self._expr_blocks(stmt.func) + ["("]
            blocks += self._join_args(stmt.args)
            blocks += [")"]
            return [print_block(blocks, indent=indent)]
        if stmt_type is IRIf:
            lines = [
                print_block(["if", "(", *self._expr_blocks(stmt.cond), ")", "then"], indent=indent)
            ]
            for child in stmt.then:
                lines.extend(self._emit_stmt(child, indent=indent + 1))

            if stmt.else_:
                for idx, branch in enumerate(stmt.else_):
                    if (
                        type(branch) is IRIf
                        and getattr(branch.source, "__class__", None) is not None
                    ):
                        if branch.source.__class__.__name__ == "ElseIf":
                            lines.append(
                                print_block(
                                    ["elseif", "(", *self._expr_blocks(branch.cond), ")", "then"],
                                    indent=indent,
                                )
                            )
                            for child in branch.then:
                                lines.extend(self._emit_stmt(child, indent=indent + 1))
                            continue
                    lines.append(print_block(["else"], indent=indent))
                    remaining = stmt.else_[idx:]
                    for child in remaining:
                        lines.extend(self._emit_stmt(child, indent=indent + 1))
                    break

            lines.append(print_block(["end", " ", "if"], indent=indent))
            return lines
        if stmt_type is IRFor:
            var = stmt.var
            var_name = var.name if var is not None else "<var>"
            blocks = ["do", " ", var_name, " ", "=", " "]
            blocks += self._expr_blocks(stmt.start)
            blocks.append(", ")
            blocks += self._expr_blocks(stmt.stop)
            if stmt.step is not None:
                blocks.append(", ")
                blocks += self._expr_blocks(stmt.step)
            lines = [print_block(blocks, indent=indent)]
            for child in stmt.body:
                lines.extend(self._emit_stmt(child, indent=indent + 1))
            lines.append(print_block(["end", " ", "do"], indent=indent))
            if "openmp" in stmt.metadata:
                from numeta.parallel import render_parallel

                # Separate continued clauses keep OpenMP lines within Fortran's limit.
                options = stmt.metadata["openmp"]
                clauses = render_parallel(options)
                words = clauses.split(" ")
                directive = "    " * indent + "!$omp parallel do "
                prefix = []
                for word in words:
                    if len(directive) + len(word) > 110:
                        prefix.append(directive + "&\n")
                        directive = "    " * indent + "!$omp& "
                    directive += word + " "
                prefix.append(directive.rstrip() + "\n")
                lines = prefix + lines
                lines.append("    " * indent + "!$omp end parallel do\n")
            return lines
        if stmt_type is IRWhile:
            blocks = ["do while", " ", "("] + self._expr_blocks(stmt.cond) + [")"]
            lines = [print_block(blocks, indent=indent)]
            for child in stmt.body:
                lines.extend(self._emit_stmt(child, indent=indent + 1))
            lines.append(print_block(["end", " ", "do"], indent=indent))
            return lines
        if stmt_type is IRAllocate:
            blocks = ["allocate", "("]
            var_blocks = self._expr_blocks(stmt.var)
            dims = self._format_allocate_dims(stmt.var, getattr(stmt, "dims", []))
            blocks += var_blocks + dims + [")"]
            return [print_block(blocks, indent=indent)]
        if stmt_type is IRDeallocate:
            blocks = ["deallocate", "("] + self._expr_blocks(stmt.var) + [")"]
            return [print_block(blocks, indent=indent)]
        if stmt_type is IRReturn:
            if stmt.value is None:
                return [print_block(["return"], indent=indent)]
            return [print_block(["return", " ", *self._expr_blocks(stmt.value)], indent=indent)]
        if stmt_type is IRPrint:
            blocks = ["print", " ", "*", ",", " "]
            blocks += self._join_args(stmt.values)
            return [print_block(blocks, indent=indent)]
        if stmt_type is IROpaqueStmt:
            if stmt.payload is not None:
                return render_stmt_lines(stmt.payload, indent=indent)
        return [print_block(["! unsupported statement"], indent=indent)]

    def _is_integer_expr(self, expr: IRExpr) -> bool:
        if isinstance(expr, IRLiteral) and isinstance(expr.value, int):
            return True
        if isinstance(expr, IRGetItem) and expr.base is not None:
            return self._is_integer_expr(expr.base)
        source_dtype = getattr(getattr(expr, "source", None), "dtype", None)
        source_name = getattr(source_dtype, "_name", "")
        if source_name == "size_t" or source_name.startswith(("int", "uint")):
            return True
        if expr.vtype and expr.vtype.dtype.name == "integer":
            return True
        if (
            isinstance(expr, IRVarRef)
            and expr.var
            and expr.var.vtype
            and expr.var.vtype.dtype.name == "integer"
        ):
            return True
        return False

    def _expr_blocks_in_order(self, expr: IRExpr, order: str) -> list[str]:
        blocks = self._expr_blocks(expr)
        shape = expr.vtype.shape if expr.vtype else None
        if shape is not None and shape.rank == 2 and shape.order != order:
            return ["transpose(", *blocks, ")"]
        return blocks

    def _expr_blocks(self, expr: IRExpr | None) -> list[str]:
        if expr is None:
            return [""]
        expr_type = type(expr)
        if expr_type is IRLiteral:
            source: Any = expr.source
            if source is not None:
                return render_expr_blocks(source)
            if isinstance(expr.value, str):
                return [f'"{expr.value}"']
            if isinstance(expr.value, bool):
                return [".true." if expr.value else ".false."]
            return [str(expr.value)]
        if expr_type is IRVarRef:
            var = expr.var
            if var is not None:
                return [var.name]
            return ["<var>"]
        if expr_type is IRBinary:
            left = self._expr_blocks(expr.left)
            right = self._expr_blocks(expr.right)
            if expr.op == "shl":
                return ["ishft(", *left, ",", *right, ")"]
            if expr.op == "shr":
                return ["shifta(", *left, ",", *right, ")"]
            if expr.op in {"xor", "bitand", "bitor"}:
                intrinsic = {"xor": "ieor", "bitand": "iand", "bitor": "ior"}[expr.op]
                return [intrinsic, "(", *left, ",", *right, ")"]
            if expr.op in {"mod", "truncmod"}:
                intrinsic = "modulo" if expr.op == "mod" else "mod"
                return [intrinsic, "(", *left, ",", *right, ")"]
            if expr.op in {"floordiv", "truncdiv"}:
                if expr.op == "truncdiv":
                    return ["(", *left, "/", *right, ")"]
                if self._is_integer_expr(expr) or (
                    self._is_integer_expr(expr.left) and self._is_integer_expr(expr.right)
                ):
                    return [
                        "((",
                        *left,
                        "/",
                        *right,
                        ")-merge(1,0,mod(",
                        *left,
                        ",",
                        *right,
                        ")/=0.and.((",
                        *left,
                        "<0).neqv.(",
                        *right,
                        "<0))))",
                    ]
                return ["real(floor(", *left, "/", *right, "),kind=kind(", *left, "))"]
            op = _FORTRAN_BINARY_OPS.get(expr.op, expr.op)
            return ["(", *left, op, *right, ")"]
        if expr_type is IRUnary:
            if expr.op == "neg":
                return ["-", "(", *self._expr_blocks(expr.operand), ")"]
            if expr.op == "not":
                return [".not.", "(", *self._expr_blocks(expr.operand), ")"]
            return [expr.op, "(", *self._expr_blocks(expr.operand), ")"]
        if expr_type is IRCallExpr:
            blocks = self._expr_blocks(expr.callee) + ["("]
            blocks += self._join_args(expr.args)
            blocks += [")"]
            return blocks
        if expr_type is IRGetAttr:
            return [*self._expr_blocks(expr.base), "%", expr.name]
        if expr_type is IRGetItem:
            base_blocks = self._expr_blocks(expr.base)
            indices = list(expr.indices)
            order = None
            if expr.base is not None and expr.base.vtype is not None:
                shape = expr.base.vtype.shape
                if shape is not None:
                    order = shape.order
            if order == "C" and len(indices) > 1:
                indices = list(reversed(indices))
            dim_blocks = []
            for index in indices:
                if isinstance(index, IRSlice):
                    dim_blocks.append(self._slice_blocks(index))
                else:
                    dim_blocks.append(self._index_blocks(index))
            blocks = base_blocks + ["("]
            if dim_blocks:
                blocks += dim_blocks[0]
                for dim in dim_blocks[1:]:
                    blocks += [",", " "] + dim
            blocks += [")"]
            return blocks
        if expr_type is IRIntrinsic:
            if expr.name == "array_constructor":
                backing_array = self._array_constructor_backing_array_blocks(expr.args)
                if backing_array is not None:
                    return backing_array
                blocks = ["["]
                blocks += self._join_args(expr.args)
                blocks += ["]"]
                return blocks

            if expr.name == "shape" and expr.args:
                dims = self._shape_dims_for_expr(expr.args[0])
                if dims is None:
                    raise NotImplementedError(
                        "Cannot lower shape() intrinsic in Fortran without concrete dimensions."
                    )
                backing_array = self._shape_backing_array_blocks(dims)
                if backing_array is not None:
                    return backing_array
                blocks = ["["]
                for i, dim in enumerate(dims):
                    if i > 0:
                        blocks += [",", " "]
                    if isinstance(dim, int):
                        blocks += [str(dim)]
                    elif isinstance(dim, IRExpr):
                        blocks += self._expr_blocks(dim)
                    else:
                        dim_rendered = render_expr_blocks(dim)
                        if dim_rendered is None:
                            blocks += [str(dim)]
                        else:
                            blocks += dim_rendered
                blocks += ["]"]
                return blocks

            # Map intrinsic name if necessary
            name = _FORTRAN_INTRINSIC_MAP.get(expr.name, expr.name)

            if expr.name == "copysign":
                return [
                    "sign(abs(",
                    *self._expr_blocks(expr.args[0]),
                    "),",
                    *self._expr_blocks(expr.args[1]),
                    ")",
                ]

            if expr.name == "round_even":
                value = self._expr_blocks(expr.args[0])
                kind = ["kind(", *value, ")"]
                one = ["real(1,kind=", *kind, ")"]
                two = ["real(2,kind=", *kind, ")"]
                half = ["real(0.5,kind=", *kind, ")"]
                scale = (
                    one
                    if len(expr.args) == 1
                    else ["(real(10,kind=", *kind, ")**", *self._expr_blocks(expr.args[1]), ")"]
                )
                scaled = ["(", *value, "*", *scale, ")"]
                magnitude = ["abs(", *scaled, ")"]
                # ANINT returns a real, avoiding default-integer overflow. It
                # rounds ties away from zero; correct the even-lower halfway
                # cases and restore the sign (including a rounded negative zero).
                nearest = ["anint(", *magnitude, ")"]
                rounded = [
                    "sign(merge(",
                    *nearest,
                    "-",
                    *one,
                    ",",
                    *nearest,
                    ",modulo(",
                    *magnitude,
                    ",",
                    *two,
                    ")==",
                    *half,
                    "),",
                    *scaled,
                    ")",
                ]
                if len(expr.args) == 1:
                    return ["int(", *rounded, ",kind=", str(expr.vtype.dtype.kind), ")"]
                return ["(", *rounded, "/", *scale, ")"]

            if expr.name == "astype":
                dtype = expr.vtype.dtype
                intrinsic = {
                    "integer": "int",
                    "real": "real",
                    "complex": "cmplx",
                    "logical": "logical",
                }.get(dtype.name)
                if intrinsic is None:
                    raise NotImplementedError(f"Cannot cast to Fortran type {dtype.name!r}")
                kind = dtype.kind
                blocks = [intrinsic, "(", *self._expr_blocks(expr.args[0])]
                if kind is not None:
                    blocks.extend([",kind=", str(kind)])
                blocks.append(")")
                return blocks

            if expr.name == "select":
                return [
                    "merge(",
                    *self._expr_blocks(expr.args[1]),
                    ",",
                    *self._expr_blocks(expr.args[2]),
                    ",",
                    *self._expr_blocks(expr.args[0]),
                    ")",
                ]

            if expr.name == "matmul":
                # C-ordered matrices are represented by their transposes in Fortran.
                # Normalize operands to the declared result order, then use
                # (A B)^T = B^T A^T for a C-ordered result, including mixed layouts.
                left, right = expr.args
                shape = expr.vtype.shape if expr.vtype else None
                if shape is None:
                    # Python matmul does not conjugate complex vectors.
                    return ["sum(", *self._expr_blocks(left), "*", *self._expr_blocks(right), ")"]
                order = shape.order if shape.rank == 2 else "F"
                native_args = (right, left) if order == "C" else (left, right)
                return [
                    "matmul(",
                    *self._expr_blocks_in_order(native_args[0], order),
                    ",",
                    *self._expr_blocks_in_order(native_args[1], order),
                    ")",
                ]

            # Workaround for LOG10(complex) which is not standard in Fortran
            if name == "log10" and expr.args:
                arg0 = expr.args[0]
                is_complex = False
                if arg0.vtype and arg0.vtype.dtype.name == "complex":
                    is_complex = True
                elif (
                    type(arg0) is IRVarRef
                    and arg0.var
                    and arg0.var.vtype
                    and arg0.var.vtype.dtype.name == "complex"
                ):
                    is_complex = True

                if is_complex:
                    arg_blocks = self._expr_blocks(arg0)
                    return [
                        "(",
                        "log",
                        "(",
                        *arg_blocks,
                        ")",
                        "/",
                        "log",
                        "(",
                        "10.0",
                        ")",
                        ")",
                    ]

            # Cast integer arguments to real for math functions that require it
            if name in {
                "exp",
                "log",
                "log10",
                "sqrt",
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "atan2",
                "sinh",
                "cosh",
                "tanh",
                "asinh",
                "acosh",
                "atanh",
                "erf",
                "erfc",
                "gamma",
                "lgamma",
                "hypot",
            }:
                args_blocks = []
                for arg in expr.args:
                    arg_blocks = self._expr_blocks(arg)
                    if self._is_integer_expr(arg):
                        args_blocks.append(["real(", *arg_blocks, ")"])
                    else:
                        args_blocks.append(arg_blocks)

                blocks = [name, "("]
                for i, arg_b in enumerate(args_blocks):
                    if i > 0:
                        blocks.append(", ")
                    blocks += arg_b
                blocks.append(")")
                return blocks

            blocks = [name, "("]
            blocks += self._join_args(expr.args)
            blocks += [")"]
            return blocks
        if expr_type is IROpaqueExpr:
            if expr.payload is not None:
                return render_expr_blocks(expr.payload)
            return ["<expr>"]
        return ["<expr>"]

    def _shape_dims_for_expr(self, expr: IRExpr) -> list | None:
        shape = expr.vtype.shape if expr.vtype is not None else None
        if shape is None:
            return None

        rank = shape.rank
        if rank is None:
            return None

        raw_dims_attr = getattr(shape, "dims", None)
        raw_dims = list(raw_dims_attr) if raw_dims_attr is not None else [None] * rank
        dims = []
        descriptor_var = None
        if isinstance(expr, IRVarRef) and expr.var is not None:
            descriptor_var = self._shape_descriptor_by_base.get(expr.var.name)

        for i in range(rank):
            dim = raw_dims[i] if i < len(raw_dims) else None
            if dim is None:
                if descriptor_var is None:
                    raise NotImplementedError(
                        "Cannot lower shape() in Fortran: unresolved runtime dimension without shape descriptor."
                    )
                dim = IRGetItem(base=IRVarRef(var=descriptor_var), indices=[IRLiteral(value=i)])
            dims.append(dim)

        return dims

    def _shape_backing_array_blocks(self, dims: list[IRExpr]) -> list[str] | None:
        if not dims:
            return None

        base_var = None
        indices: list[int] = []
        ir_getitem_match = True
        for dim in dims:
            if not isinstance(dim, IRGetItem):
                ir_getitem_match = False
                break
            if len(dim.indices) != 1:
                ir_getitem_match = False
                break
            idx = dim.indices[0]
            if not isinstance(idx, IRLiteral) or not isinstance(idx.value, int):
                ir_getitem_match = False
                break
            if not isinstance(dim.base, IRVarRef) or dim.base.var is None:
                ir_getitem_match = False
                break
            if base_var is None:
                base_var = dim.base.var
            elif base_var is not dim.base.var:
                ir_getitem_match = False
                break
            indices.append(int(idx.value))

        if not ir_getitem_match or base_var is None:
            parsed_base = None
            parsed_indices: list[int] = []
            for dim in dims:
                rendered = "".join(self._expr_blocks(dim)).replace(" ", "")
                if rendered.startswith("(") and rendered.endswith(")"):
                    rendered = rendered[1:-1]
                if "(" not in rendered or not rendered.endswith(")"):
                    return None
                base_name, idx_text = rendered.split("(", 1)
                idx_text = idx_text[:-1]
                if not base_name or not idx_text.isdigit():
                    return None
                if parsed_base is None:
                    parsed_base = base_name
                elif parsed_base != base_name:
                    return None
                parsed_indices.append(int(idx_text))

            if parsed_base is None:
                return None
            if parsed_indices:
                return [parsed_base]
            return None

        rank = len(indices)
        if rank:
            return [base_var.name]
        return None

    def _array_constructor_backing_array_blocks(self, args: list[IRExpr]) -> list[str] | None:
        if not args:
            return None

        base_var = None
        indices: list[int] = []
        for arg in args:
            if not isinstance(arg, IRGetItem):
                return None
            if len(arg.indices) != 1:
                return None
            idx = arg.indices[0]
            if not isinstance(idx, IRLiteral) or not isinstance(idx.value, int):
                return None
            if not isinstance(arg.base, IRVarRef) or arg.base.var is None:
                return None
            if base_var is None:
                base_var = arg.base.var
            elif base_var is not arg.base.var:
                return None
            indices.append(int(idx.value))

        if base_var is None:
            parsed_base = None
            parsed_indices: list[int] = []
            for arg in args:
                rendered = "".join(self._expr_blocks(arg)).replace(" ", "")
                if rendered.startswith("(") and rendered.endswith(")"):
                    rendered = rendered[1:-1]
                if "(" not in rendered or not rendered.endswith(")"):
                    return None
                base_name, idx_text = rendered.split("(", 1)
                idx_text = idx_text[:-1]
                if not base_name or not idx_text.isdigit():
                    return None
                if parsed_base is None:
                    parsed_base = base_name
                elif parsed_base != base_name:
                    return None
                parsed_indices.append(int(idx_text))

            if parsed_base is None:
                return None
            rank = len(parsed_indices)
            if parsed_indices == list(range(rank)) or parsed_indices == list(range(1, rank + 1)):
                return [parsed_base]
            return None

        rank = len(indices)
        if indices == list(range(rank)) or indices == list(range(1, rank + 1)):
            return [base_var.name]
        return None

    def _join_args(self, args: list[IRExpr]) -> list[str]:
        blocks: list[str] = []
        for arg in args:
            blocks += self._expr_blocks(arg)
            blocks += [", "]
        if blocks:
            blocks.pop()
        return blocks

    def _index_blocks(self, expr: IRExpr) -> list[str]:
        shift = 1
        return self._shift_expr_blocks(expr, shift)

    def _slice_blocks(self, slice_: IRSlice) -> list[str]:
        lbound = 1

        if slice_.start is None:
            start_blocks = [str(lbound)]
        else:
            start_blocks = self._shift_expr_blocks(slice_.start, lbound)

        stop_blocks: list[str] = []
        if slice_.stop is not None:
            stop_blocks = self._shift_expr_blocks(slice_.stop, lbound - 1)

        result = [*start_blocks, ":"]
        if stop_blocks:
            result += stop_blocks
        if slice_.step is not None:
            result.append(":")
            result += self._expr_blocks(slice_.step)
        return result

    def _shift_expr_blocks(self, expr: IRExpr, shift: int) -> list[str]:
        if shift == 0:
            return self._expr_blocks(expr)
        if isinstance(expr, IRLiteral) and isinstance(expr.value, (int, float)):
            return [str(expr.value + shift)]
        op = "+" if shift > 0 else "-"
        return ["(", *self._expr_blocks(expr), op, str(abs(shift)), ")"]

    def _format_allocate_dims(self, var_expr: IRExpr | None, dims: list[IRExpr]) -> list[str]:
        if not dims:
            return []
        order = None
        if var_expr is not None and var_expr.vtype is not None:
            shape = var_expr.vtype.shape
            if shape is not None:
                order = shape.order

        dim_blocks = [self._expr_blocks(dim) for dim in dims]

        if order == "C" and len(dim_blocks) > 1:
            dim_blocks = list(reversed(dim_blocks))

        blocks = ["("] + dim_blocks[0]
        for dim in dim_blocks[1:]:
            blocks += [",", " "] + dim
        blocks.append(")")
        return blocks

    def _declare_variable(self, var, *, indent: int) -> str:
        if var.vtype is None:
            return print_block(["! undeclared :: ", var.name], indent=indent)

        dtype = var.vtype.dtype
        if dtype.kind is not None:
            type_block = [dtype.name, "(", str(dtype.kind), ")"]
        else:
            type_block = [dtype.name]

        blocks = list(type_block)
        shape = var.vtype.shape
        if var.allocatable:
            rank = 1
            if shape is not None and shape.rank:
                rank = shape.rank
            blocks += [", ", "allocatable", ", ", "dimension", "("]
            blocks += [":", ","] * (rank - 1)
            blocks += [":", ")"]
        elif var.pointer:
            blocks += [", ", "pointer"]
            if shape is not None and shape.rank:
                blocks += [", ", "dimension", "("]
                blocks += [":", ","] * (shape.rank - 1)
                blocks += [":", ")"]
        elif shape is not None and getattr(shape, "dims", None) is None:
            blocks += [
                ", ",
                "dimension",
                "(",
                "1",
                ":",
                "*",
                ")",
            ]
        elif shape is not None and shape.rank:
            blocks += [", ", "dimension"]
            blocks += self._shape_blocks(shape)

        if var.intent is not None:
            blocks += [", ", "intent", "(", var.intent, ")"]

        if var.pass_by_value:
            blocks += [", ", "value"]

        if var.parameter:
            blocks += [", ", "parameter"]

        if var.target:
            if var.pointer:
                blocks += [", ", "contiguous"]
            else:
                blocks += [", ", "target"]

        if var.bind_c:
            blocks += [", ", "bind", "(", "C", ", ", "name=", "'", var.name, "'", ")"]

        blocks += [" :: ", var.name]

        if var.assign is not None:
            values = []
            source_dtype = getattr(var.source, "dtype", None)
            if source_dtype is not None:
                if isinstance(var.assign, (int, float, complex, bool, str, np.generic)):
                    values = render_literal_blocks_from_dtype(var.assign, source_dtype)
                elif isinstance(var.assign, np.ndarray):
                    for v in var.assign.ravel():
                        values += render_literal_blocks_from_dtype(v, source_dtype)
                        values.append(", ")
                    if values:
                        values.pop()
            else:
                if isinstance(var.assign, (int, float, complex, bool, str, np.generic)):
                    values = self._literal_blocks(var.assign)
                elif isinstance(var.assign, np.ndarray):
                    for v in var.assign.ravel():
                        values += self._literal_blocks(v)
                        values.append(", ")
                    if values:
                        values.pop()
            if values:
                blocks += [";", " data ", var.name, " / ", *values, " /"]

        return print_block(blocks, indent=indent)

    def _shape_blocks(self, shape) -> list[str]:
        lbound = 1
        dims = list(getattr(shape, "dims", None) or [])
        if shape.order == "C" and len(dims) > 1:
            dims = list(reversed(dims))

        def dim_blocks(dim) -> list[str]:
            if dim is None:
                return [str(lbound), ":", "*"]
            if isinstance(dim, int):
                upper = dim + (lbound - 1)
                return [str(lbound), ":", str(upper)]
            if isinstance(dim, IRExpr):
                blocks = self._expr_blocks(dim)
            else:
                blocks = render_expr_blocks(dim)
                if blocks is None:
                    blocks = [str(dim)]
            return [str(lbound), ":", *blocks]

        result = ["("]
        for i, dim in enumerate(dims):
            if i:
                result += [",", " "]
            result += dim_blocks(dim)
        result.append(")")
        return result

    def _literal_blocks(self, value) -> list[str]:
        if isinstance(value, str):
            return [f'"{value}"']
        if isinstance(value, bool):
            return [".true." if value else ".false."]
        if isinstance(value, complex):
            return ["(", str(value.real), ",", str(value.imag), ")"]
        return [str(value)]
