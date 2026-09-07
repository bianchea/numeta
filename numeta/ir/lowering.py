from __future__ import annotations

from typing import Any, cast

from numeta.array_shape import ArrayShape
from numeta.ast import Variable
from numeta.settings import settings
from numeta.exceptions import raise_with_source
from numeta.ast.expressions import (
    BinaryOperationNode,
    FunctionCall,
    GetAttr,
    GetItem,
    WholeStorage,
    IntrinsicFunction,
    LiteralNode,
)
from numeta.ast.expressions.various import ArrayConstructor
from numeta.ast.statements import (
    Allocate,
    Assignment,
    Call,
    Case,
    Deallocate,
    Else,
    ElseIf,
    For,
    If,
    Print,
    Return,
    Switch,
    While,
)
from numeta.ast.statements.variable_declaration import VariableDeclaration
from numeta.ast.procedure import Procedure
from numeta.ast.function import Function

from .nodes import (
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
    IRNode,
    IRProcedure,
    IRPrint,
    IRReturn,
    IRShape,
    IRSimdStore,
    IRSlice,
    IRType,
    IRUnary,
    IRValueType,
    IRVar,
    IRVarRef,
    IRWhile,
    IROpaqueExpr,
    IROpaqueStmt,
)

_BINARY_OPS: dict[str, str] = {
    ".eq.": "eq",
    ".ne.": "ne",
    ".lt.": "lt",
    ".le.": "le",
    ".gt.": "gt",
    ".ge.": "ge",
    ".and.": "and",
    ".or.": "or",
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "**": "pow",
    "//": "floordiv",
    "%": "mod",
    "<<": "shl",
    ">>": "shr",
    "^": "xor",
    "bitand": "bitand",
    "bitor": "bitor",
}


def _lower_value_type_from_dtype(dtype, shape, lower_expr) -> IRValueType:
    # Used for C backend to avoid FortranType dependency
    if getattr(dtype, "_is_vector", False):
        base_dtype = dtype.base_dtype()
        base_type = IRType(name=base_dtype._name, kind=None, datatype=base_dtype)
        ir_type = IRType(
            name=dtype._name,
            datatype=dtype,
            kind=None,
            bitwidth=dtype.get_nbytes() * 8,
            is_vector=True,
            vector_lanes=dtype.lanes(),
            vector_base=base_type,
        )
        return IRValueType(dtype=ir_type, shape=_lower_ir_shape(shape, settings.syntax, lower_expr))
    ir_type = IRType(name=dtype._name, kind=None, datatype=dtype)
    return IRValueType(dtype=ir_type, shape=_lower_ir_shape(shape, settings.syntax, lower_expr))


def _is_scalar_shape(shape) -> bool:
    return isinstance(shape, ArrayShape) and shape.is_scalar


def _is_unknown_rank_shape(shape) -> bool:
    return isinstance(shape, ArrayShape) and shape.is_unknown


def _lower_ir_shape(shape, syntax_settings, lower_expr) -> IRShape | None:
    if _is_scalar_shape(shape):
        return None
    if _is_unknown_rank_shape(shape):
        return IRShape(rank=None, dims=None, order="C")
    dims = _lower_shape_dims(shape.as_tuple(), syntax_settings, lower_expr)
    order = "F" if getattr(shape, "fortran_order", False) else "C"
    return IRShape(rank=len(dims), dims=dims, order=order)


def lower_procedure(procedure: Procedure, backend: str = "fortran") -> IRProcedure:
    from .validation import validate_numeric_ir

    result = LoweringContext(procedure, backend).lower()
    validate_numeric_ir(result)
    return result


class LoweringContext:
    """Own one procedure's recursive value, index and type conversion."""

    def __init__(self, procedure, backend):
        self.procedure = procedure
        self.backend = backend

    def lower(self):
        procedure = self.procedure
        self.syntax_settings = settings.syntax
        self.iso_c_mode = settings.iso_C
        self.backend_is_c = self.backend == "c"
        self.arg_names = set(self.procedure.arguments)
        self.var_cache: dict[int, IRVar] = {}
        self.vtype_by_dtype_shape: dict[tuple[int, int], IRValueType] = {}
        self.vtype_by_ftype_shape: dict[tuple[int, int], IRValueType] = {}
        self.ftype_cache: dict[int, Any] = {}
        self.ir_type_cache: dict[int, IRType] = {}

        args = [self.lower_var(var, is_arg=True) for var in procedure.arguments.values()]
        locals_ = [
            self.lower_var(var, is_arg=False) for var in procedure.get_local_variables().values()
        ]
        body = [self.lower_stmt(stmt) for stmt in procedure.scope.get_statements()]

        decl = procedure.get_declaration()
        scope_ids = {id(stmt) for stmt in procedure.scope.get_statements()}
        prelude_items: list[Any] = []
        for stmt in decl.get_statements():
            if id(stmt) in scope_ids:
                continue
            if isinstance(stmt, VariableDeclaration):
                continue
            prelude_items.append(stmt)

        return IRProcedure(
            name=procedure.name,
            args=args,
            locals=locals_,
            body=body,
            result=None,
            source=procedure,
            metadata={
                "syntax_procedure": procedure,
                "fortran_prelude_items": prelude_items,
                "fortran_pure": procedure.pure,
                "fortran_elemental": procedure.elemental,
                "fortran_bind_c": procedure.bind_c,
                "c_attributes": tuple(getattr(procedure, "c_attributes", ())),
                "c_linkage": getattr(procedure, "c_linkage", None),
                "emit_mode": getattr(procedure, "emit_mode", None),
            },
        )

    def _get_vtype(self, expr, shape=None, dtype=None):
        if shape is None:
            shape = _safe_shape(expr)
        if dtype is None:
            dtype = getattr(expr, "dtype", None)
        if dtype is None:
            raise_with_source(
                ValueError,
                "Cannot determine expression dtype. Use a supported numeric value.",
                source_node=expr,
            )
        dtype_shape_key = (id(dtype), id(shape))
        lowered = self.vtype_by_dtype_shape.get(dtype_shape_key)
        if lowered is None:
            if self.backend_is_c:
                lowered = _lower_value_type_from_dtype(dtype, shape, self.lower_expr)
            else:
                dtype_key = id(dtype)
                ftype = self.ftype_cache.get(dtype_key)
                if ftype is None:
                    ftype = cast(Any, dtype).get_fortran(bind_c=self.iso_c_mode)
                    self.ftype_cache[dtype_key] = ftype
                ftype_shape_key = (id(dtype), id(shape))
                lowered = self.vtype_by_ftype_shape.get(ftype_shape_key)
                if lowered is None:
                    ftype_id = id(dtype)
                    lowered_type = self.ir_type_cache.get(ftype_id)
                    if lowered_type is None:
                        lowered_type = _lower_type(ftype, dtype)
                        self.ir_type_cache[ftype_id] = lowered_type
                    lowered = IRValueType(
                        dtype=lowered_type,
                        shape=_lower_ir_shape(shape, self.syntax_settings, self.lower_expr),
                    )
                    self.vtype_by_ftype_shape[ftype_shape_key] = lowered
            self.vtype_by_dtype_shape[dtype_shape_key] = lowered
        return lowered

    def lower_var(self, var: Variable, *, is_arg: bool) -> IRVar:
        key = id(var)
        if key in self.var_cache:
            return self.var_cache[key]
        var_dtype = var.dtype
        if var_dtype is None:
            raise_with_source(
                ValueError,
                f"Variable {var.name} has no dtype",
                source_node=var,
            )
        vtype = self._get_vtype(var)
        storage = "value"
        if getattr(var, "allocatable", False):
            storage = "allocatable"
        elif getattr(var, "pointer", False):
            storage = "pointer"
        ir_var = IRVar(
            name=var.name,
            vtype=vtype,
            intent=var.intent,
            storage=storage,
            is_const=var.intent == "in",
            is_arg=is_arg,
            allocatable=getattr(var, "allocatable", False),
            pointer=getattr(var, "pointer", False),
            target=getattr(var, "target", False),
            parameter=getattr(var, "parameter", False),
            c_static=getattr(var, "c_static", False),
            bind_c=getattr(var, "bind_c", False),
            assign=getattr(var, "assign", None),
            pass_by_value=var.pass_by_value,
            source=var,
        )
        self.var_cache[key] = ir_var
        return ir_var

    def lower_expr(self, expr) -> IRExpr:
        if isinstance(expr, IRExpr):
            return expr
        from numeta.ast.tools import check_node

        expr = check_node(expr)
        expr_type = type(expr)
        if isinstance(expr, (Procedure, Function)):
            return IRVarRef(
                var=IRVar(name=expr.name, source=expr),
                source=expr,
                metadata={"procedure_reference": True},
            )
        if expr_type is LiteralNode:
            return IRLiteral(
                value=expr.value,
                vtype=self._get_vtype(expr),
                source=expr,
            )
        if expr_type is Variable:
            ir_var = self.lower_var(expr, is_arg=expr.name in self.arg_names)
            return IRVarRef(
                var=ir_var,
                vtype=ir_var.vtype,
                source=expr,
            )
        if expr_type is WholeStorage:
            return self.lower_expr(expr.variable)
        if expr_type is BinaryOperationNode or isinstance(expr, BinaryOperationNode):
            op = _map_binary_op(expr.op)
            return IRBinary(
                op=op,
                left=self.lower_expr(expr.left),
                right=self.lower_expr(expr.right),
                vtype=self._get_vtype(expr),
                source=expr,
            )
        if expr_type is FunctionCall:
            return IRCallExpr(
                callee=self.lower_expr(expr.function),
                args=[self.lower_expr(arg) for arg in expr.arguments],
                vtype=self._get_vtype(expr),
                source=expr,
            )
        if expr_type is GetItem:
            indices = _lower_indices(expr.sliced, self.syntax_settings, self.lower_expr)
            shape = _safe_shape(expr)
            base = self.lower_expr(expr.variable)
            vtype = self._get_vtype(expr, shape)

            def indexed(value):
                indexed_type = IRValueType(dtype=value.vtype.dtype, shape=vtype.shape)
                if isinstance(value, IRIntrinsic) and value.name == "astype":
                    return IRIntrinsic(
                        name="astype",
                        args=[indexed(value.args[0])],
                        vtype=indexed_type,
                        source=expr,
                    )
                return IRGetItem(base=value, indices=indices, vtype=indexed_type, source=expr)

            return indexed(base)
        if expr_type is GetAttr:
            return IRGetAttr(
                base=self.lower_expr(expr.variable),
                name=expr.attr,
                vtype=self._get_vtype(expr),
                source=expr,
            )
        if expr_type is ArrayConstructor:
            return IRIntrinsic(
                name="array_constructor",
                args=[self.lower_expr(arg) for arg in expr.elements],
                vtype=self._get_vtype(expr),
                source=expr,
            )
        if expr_type is IntrinsicFunction or isinstance(expr, IntrinsicFunction):
            token = getattr(expr, "token", "")
            args = [self.lower_expr(arg) for arg in expr.arguments]
            if (
                token == "round_even"
                and getattr(expr.arguments[0].dtype, "_name", "").startswith("int")
                and (expr.ndigits is None or expr.ndigits >= 0)
            ):
                # Already-integral values must not lose precision through a real
                # intermediate, especially above the exact float64 integer range.
                return IRIntrinsic(
                    name="astype", args=args[:1], vtype=self._get_vtype(expr), source=expr
                )
            if token == "-" and len(args) == 1:
                return IRUnary(
                    op="neg",
                    operand=args[0],
                    vtype=self._get_vtype(expr),
                    source=expr,
                )
            if token == ".not." and len(args) == 1:
                return IRUnary(
                    op="not",
                    operand=args[0],
                    vtype=self._get_vtype(expr),
                    source=expr,
                )
            metadata = {}
            if token == "simd_vload":
                metadata["aligned"] = bool(getattr(expr, "aligned", False))
            elif token == "simd_compare":
                metadata["predicate"] = expr.predicate
            elif token == "simd_extract_i32":
                metadata["lane"] = expr.lane
            return IRIntrinsic(
                name=token,
                args=args,
                vtype=self._get_vtype(expr),
                source=expr,
                metadata=metadata,
            )
        from numeta.ast.expressions.various import Re, Im

        if isinstance(expr, (Re, Im)):
            return IRIntrinsic(
                name="real" if isinstance(expr, Re) else "aimag",
                args=[self.lower_expr(expr.variable)],
                vtype=self._get_vtype(expr),
                source=expr,
            )
        return IROpaqueExpr(payload=expr, source=expr)

    def lower_stmt(self, stmt) -> IRNode:
        if isinstance(stmt, Assignment):
            return IRAssign(
                target=self.lower_expr(stmt.target),
                value=self.lower_expr(stmt.value),
                source=stmt,
            )
        if isinstance(stmt, Call):
            return IRCall(
                func=self.lower_expr(stmt.function),
                args=[self.lower_expr(arg) for arg in stmt.arguments],
                source=stmt,
            )
        if isinstance(stmt, If):
            then_body = [self.lower_stmt(s) for s in stmt.scope.get_statements()]
            else_body: list[IRNode] = []
            for branch in stmt.orelse:
                if isinstance(branch, ElseIf):
                    nested = IRIf(
                        cond=self.lower_expr(branch.condition),
                        then=[self.lower_stmt(s) for s in branch.scope.get_statements()],
                        else_=[],
                        source=branch,
                    )
                    else_body.append(nested)
                elif isinstance(branch, Else):
                    else_body.extend([self.lower_stmt(s) for s in branch.scope.get_statements()])
                else:
                    else_body.append(self.lower_stmt(branch))
            return IRIf(
                cond=self.lower_expr(stmt.condition),
                then=then_body,
                else_=else_body,
                source=stmt,
            )
        if isinstance(stmt, For):
            iterator = stmt.iterator
            if isinstance(iterator, Variable):
                loop_var = self.lower_var(iterator, is_arg=False)
            else:
                loop_var = IRVar(name=str(iterator), source=iterator)
            return IRFor(
                var=loop_var,
                start=self.lower_expr(stmt.start),
                stop=self.lower_expr(stmt.end),
                step=self.lower_expr(stmt.step) if stmt.step is not None else None,
                body=[self.lower_stmt(s) for s in stmt.scope.get_statements()],
                source=stmt,
                metadata=(
                    {
                        "openmp": {
                            key: (
                                [v.name for v in value] if key in {"shared", "private"} else value
                            )
                            for key, value in stmt.openmp.items()
                        }
                    }
                    if hasattr(stmt, "openmp")
                    else {}
                ),
            )
        if isinstance(stmt, While):
            return IRWhile(
                cond=self.lower_expr(stmt.condition),
                body=[self.lower_stmt(s) for s in stmt.scope.get_statements()],
                source=stmt,
            )
        if isinstance(stmt, Switch):
            cases = [s for s in stmt.scope.get_statements() if isinstance(s, Case)]
            if not cases:
                return IROpaqueStmt(payload=stmt, source=stmt)

            def build_case(case_stmt):
                comparison = BinaryOperationNode(stmt.value, ".eq.", case_stmt.value)
                comparison._source_location = getattr(case_stmt, "source_location", None)
                cond = self.lower_expr(comparison)
                return IRIf(
                    cond=cond,
                    then=[self.lower_stmt(s) for s in case_stmt.scope.get_statements()],
                    else_=[],
                    source=case_stmt,
                )

            root = build_case(cases[0])
            current = root
            for case_stmt in cases[1:]:
                next_if = build_case(case_stmt)
                current.else_.append(next_if)
                current = next_if
            return root
        if isinstance(stmt, Return):
            return IRReturn(value=None, source=stmt)
        if isinstance(stmt, Print):
            return IRPrint(values=[self.lower_expr(value) for value in stmt.to_print], source=stmt)
        if isinstance(stmt, Allocate):
            return IRAllocate(
                var=self.lower_expr(stmt.target),
                dims=[self.lower_expr(dim) for dim in stmt.shape],
                source=stmt,
            )
        if isinstance(stmt, Deallocate):
            return IRDeallocate(var=self.lower_expr(stmt.array), source=stmt)
        if stmt.__class__.__name__ == "VStore":
            return IRSimdStore(
                array=self.lower_expr(stmt.array),
                index=self.lower_expr(stmt.index),
                value=self.lower_expr(stmt.value),
                aligned=bool(getattr(stmt, "aligned", False)),
                source=stmt,
            )
        return IROpaqueStmt(payload=stmt, source=stmt)


def _lower_shape_dims(dims, syntax_settings, lower_expr) -> tuple:
    lowered = []
    for dim in dims:
        if isinstance(dim, int):
            lowered.append(dim)
            continue
        lowered_dim = lower_expr(dim) if dim is not None else None
        lowered.append(lowered_dim if lowered_dim is not None else dim)
    return tuple(lowered)


def _lower_type(ftype, dtype) -> IRType:
    name = getattr(ftype, "type", str(ftype))
    kind = None
    if hasattr(ftype, "get_kind_str"):
        kind = ftype.get_kind_str()
    return IRType(name=name, kind=kind, datatype=dtype)


def _safe_shape(expr):
    return expr._shape


def _lower_indices(slice_, syntax_settings, lower_expr) -> list[IRExpr | IRSlice]:
    if isinstance(slice_, tuple):
        return [_lower_single_index(item, syntax_settings, lower_expr) for item in slice_]
    return [_lower_single_index(slice_, syntax_settings, lower_expr)]


def _lower_single_index(item, syntax_settings, lower_expr) -> IRExpr | IRSlice:
    if isinstance(item, slice):
        return _normalize_slice(item, syntax_settings, lower_expr)
    expr = lower_expr(item)
    return _shift_expr(expr, -syntax_settings.array_lower_bound)


def _normalize_slice(slice_: slice, syntax_settings, lower_expr) -> IRSlice:
    lbound = syntax_settings.array_lower_bound
    c_like = syntax_settings.c_like_bounds

    if slice_.start is None:
        start = lower_expr(0)
    else:
        start = lower_expr(slice_.start)
        start = _shift_expr(start, -lbound)

    stop = None
    if slice_.stop is not None:
        stop_expr = lower_expr(slice_.stop)
        shift = (0 if c_like else 1) - lbound
        stop = _shift_expr(stop_expr, shift)

    step = None
    if slice_.step is not None:
        step = lower_expr(slice_.step)

    return IRSlice(start=start, stop=stop, step=step)


def _shift_expr(expr: IRExpr, delta: int) -> IRExpr:
    if delta == 0:
        return expr
    if isinstance(expr, IRLiteral) and isinstance(expr.value, (int, float)):
        return IRLiteral(value=expr.value + delta, vtype=expr.vtype, source=expr.source)
    op = "add" if delta > 0 else "sub"
    return IRBinary(
        op=op,
        left=expr,
        right=IRLiteral(value=abs(delta), vtype=expr.vtype),
        vtype=expr.vtype,
        source=expr.source,
    )


def _map_binary_op(op: str) -> str:
    return _BINARY_OPS.get(op, op)
