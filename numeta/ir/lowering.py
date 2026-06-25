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
}


def _lower_value_type_from_dtype(dtype, shape) -> IRValueType:
    # Used for C backend to avoid FortranType dependency
    if getattr(dtype, "_is_vector", False):
        base_dtype = dtype.base_dtype()
        base_type = IRType(name=getattr(base_dtype, "_name", str(base_dtype)), kind=None)
        ir_type = IRType(
            name=dtype.name,
            kind=None,
            bitwidth=dtype.get_nbytes() * 8,
            is_vector=True,
            vector_lanes=dtype.lanes(),
            vector_base=base_type,
        )
        return IRValueType(dtype=ir_type, shape=_lower_ir_shape(shape, settings.syntax))
    ir_type = IRType(name=dtype.name, kind=None)
    return IRValueType(dtype=ir_type, shape=_lower_ir_shape(shape, settings.syntax))


def _is_scalar_shape(shape) -> bool:
    return isinstance(shape, ArrayShape) and shape.is_scalar


def _is_unknown_rank_shape(shape) -> bool:
    return isinstance(shape, ArrayShape) and shape.is_unknown


def _lower_ir_shape(shape, syntax_settings) -> IRShape | None:
    if _is_scalar_shape(shape):
        return None
    if _is_unknown_rank_shape(shape):
        return IRShape(rank=None, dims=None, order="C")
    dims = _lower_shape_dims(shape.as_tuple(), syntax_settings)
    order = "F" if getattr(shape, "fortran_order", False) else "C"
    return IRShape(rank=len(dims), dims=dims, order=order)


def lower_procedure(procedure: Procedure, backend: str = "fortran") -> IRProcedure:
    syntax_settings = settings.syntax
    iso_c_mode = settings.iso_C
    backend_is_c = backend == "c"
    arg_names = set(procedure.arguments)
    var_cache: dict[int, IRVar] = {}
    vtype_by_dtype_shape: dict[tuple[int, int], IRValueType] = {}
    vtype_by_ftype_shape: dict[tuple[int, int], IRValueType] = {}
    ftype_cache: dict[int, Any] = {}
    ir_type_cache: dict[int, IRType] = {}

    literal_type = LiteralNode
    variable_type = Variable
    binary_type = BinaryOperationNode
    function_call_type = FunctionCall
    getitem_type = GetItem
    getattr_type = GetAttr
    array_constructor_type = ArrayConstructor
    intrinsic_type = IntrinsicFunction

    def _get_vtype(expr, shape=None):
        if shape is None:
            shape = _safe_shape(expr)
        dtype = getattr(expr, "dtype", None)
        if dtype is None:
            raise_with_source(
                ValueError,
                f"Cannot determine dtype for expression: {expr}",
                source_node=expr,
            )
        dtype_shape_key = (id(dtype), id(shape))
        lowered = vtype_by_dtype_shape.get(dtype_shape_key)
        if lowered is None:
            if backend_is_c:
                lowered = _lower_value_type_from_dtype(dtype, shape)
            else:
                dtype_key = id(dtype)
                ftype = ftype_cache.get(dtype_key)
                if ftype is None:
                    ftype = cast(Any, dtype).get_fortran(bind_c=iso_c_mode)
                    ftype_cache[dtype_key] = ftype
                ftype_shape_key = (id(ftype), id(shape))
                lowered = vtype_by_ftype_shape.get(ftype_shape_key)
                if lowered is None:
                    ftype_id = id(ftype)
                    lowered_type = ir_type_cache.get(ftype_id)
                    if lowered_type is None:
                        lowered_type = _lower_type(ftype)
                        ir_type_cache[ftype_id] = lowered_type
                    lowered = IRValueType(
                        dtype=lowered_type,
                        shape=_lower_ir_shape(shape, syntax_settings),
                    )
                    vtype_by_ftype_shape[ftype_shape_key] = lowered
            vtype_by_dtype_shape[dtype_shape_key] = lowered
        return lowered

    def lower_var(var: Variable, *, is_arg: bool) -> IRVar:
        key = id(var)
        if key in var_cache:
            return var_cache[key]
        var_dtype = var.dtype
        if var_dtype is None:
            raise_with_source(
                ValueError,
                f"Variable {var.name} has no dtype",
                source_node=var,
            )
        shape = var._shape
        dtype_shape_key = (id(var_dtype), id(shape))
        vtype = vtype_by_dtype_shape.get(dtype_shape_key)
        if vtype is None:
            if backend_is_c:
                vtype = _lower_value_type_from_dtype(var_dtype, shape)
            else:
                dtype_key = id(var_dtype)
                ftype = ftype_cache.get(dtype_key)
                if ftype is None:
                    ftype = cast(Any, var_dtype).get_fortran(bind_c=iso_c_mode)
                    ftype_cache[dtype_key] = ftype
                ftype_shape_key = (id(ftype), id(shape))
                vtype = vtype_by_ftype_shape.get(ftype_shape_key)
                if vtype is None:
                    ftype_id = id(ftype)
                    lowered_type = ir_type_cache.get(ftype_id)
                    if lowered_type is None:
                        lowered_type = _lower_type(ftype)
                        ir_type_cache[ftype_id] = lowered_type
                    vtype = IRValueType(
                        dtype=lowered_type,
                        shape=_lower_ir_shape(shape, syntax_settings),
                    )
                    vtype_by_ftype_shape[ftype_shape_key] = vtype
            vtype_by_dtype_shape[dtype_shape_key] = vtype
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
            bind_c=getattr(var, "bind_c", False),
            assign=getattr(var, "assign", None),
            pass_by_value=var.pass_by_value,
            source=var,
        )
        var_cache[key] = ir_var
        return ir_var

    def lower_expr(expr) -> IRExpr:
        if isinstance(expr, IRExpr):
            return expr
        expr_type = type(expr)
        if expr_type is Procedure or expr_type is Function:
            return IRVarRef(var=IRVar(name=expr.name, source=expr), source=expr)
        if expr_type is literal_type:
            return IRLiteral(
                value=expr.value,
                vtype=_get_vtype(expr),
                source=expr,
            )
        if expr_type is variable_type:
            ir_var = lower_var(expr, is_arg=expr.name in arg_names)
            return IRVarRef(
                var=ir_var,
                vtype=ir_var.vtype,
                source=expr,
            )
        if expr_type is binary_type or isinstance(expr, binary_type):
            op = _map_binary_op(expr.op)
            return IRBinary(
                op=op,
                left=lower_expr(expr.left),
                right=lower_expr(expr.right),
                vtype=_get_vtype(expr),
                source=expr,
            )
        if expr_type is function_call_type:
            return IRCallExpr(
                callee=lower_expr(expr.function),
                args=[lower_expr(arg) for arg in expr.arguments],
                vtype=_get_vtype(expr),
                source=expr,
            )
        if expr_type is getitem_type:
            indices = _lower_indices(expr.sliced, syntax_settings)
            shape = _safe_shape(expr)
            return IRGetItem(
                base=lower_expr(expr.variable),
                indices=indices,
                vtype=_get_vtype(expr, shape),
                source=expr,
            )
        if expr_type is getattr_type:
            return IRGetAttr(
                base=lower_expr(expr.variable),
                name=expr.attr,
                vtype=_get_vtype(expr),
                source=expr,
            )
        if expr_type is array_constructor_type:
            return IRIntrinsic(
                name="array_constructor",
                args=[lower_expr(arg) for arg in expr.elements],
                vtype=_get_vtype(expr),
                source=expr,
            )
        if expr_type is intrinsic_type or isinstance(expr, intrinsic_type):
            token = getattr(expr, "token", "")
            args = [lower_expr(arg) for arg in expr.arguments]
            if token == "-" and len(args) == 1:
                return IRUnary(
                    op="neg",
                    operand=args[0],
                    vtype=_get_vtype(expr),
                    source=expr,
                )
            if token == ".not." and len(args) == 1:
                return IRUnary(
                    op="not",
                    operand=args[0],
                    vtype=_get_vtype(expr),
                    source=expr,
                )
            metadata = {}
            if token == "simd_vload":
                metadata["aligned"] = bool(getattr(expr, "aligned", False))
            return IRIntrinsic(
                name=token,
                args=args,
                vtype=_get_vtype(expr),
                source=expr,
                metadata=metadata,
            )
        return IROpaqueExpr(payload=expr, source=expr)

    def lower_stmt(stmt) -> IRNode:
        if isinstance(stmt, Assignment):
            return IRAssign(
                target=lower_expr(stmt.target),
                value=lower_expr(stmt.value),
                source=stmt,
            )
        if isinstance(stmt, Call):
            return IRCall(
                func=lower_expr(stmt.function),
                args=[lower_expr(arg) for arg in stmt.arguments],
                source=stmt,
            )
        if isinstance(stmt, If):
            then_body = [lower_stmt(s) for s in stmt.scope.get_statements()]
            else_body: list[IRNode] = []
            for branch in stmt.orelse:
                if isinstance(branch, ElseIf):
                    nested = IRIf(
                        cond=lower_expr(branch.condition),
                        then=[lower_stmt(s) for s in branch.scope.get_statements()],
                        else_=[],
                        source=branch,
                    )
                    else_body.append(nested)
                elif isinstance(branch, Else):
                    else_body.extend([lower_stmt(s) for s in branch.scope.get_statements()])
                else:
                    else_body.append(lower_stmt(branch))
            return IRIf(
                cond=lower_expr(stmt.condition),
                then=then_body,
                else_=else_body,
                source=stmt,
            )
        if isinstance(stmt, For):
            iterator = stmt.iterator
            if isinstance(iterator, Variable):
                loop_var = lower_var(iterator, is_arg=False)
            else:
                loop_var = IRVar(name=str(iterator), source=iterator)
            return IRFor(
                var=loop_var,
                start=lower_expr(stmt.start),
                stop=lower_expr(stmt.end),
                step=lower_expr(stmt.step) if stmt.step is not None else None,
                body=[lower_stmt(s) for s in stmt.scope.get_statements()],
                source=stmt,
            )
        if isinstance(stmt, While):
            return IRWhile(
                cond=lower_expr(stmt.condition),
                body=[lower_stmt(s) for s in stmt.scope.get_statements()],
                source=stmt,
            )
        if isinstance(stmt, Switch):
            cases = [s for s in stmt.scope.get_statements() if isinstance(s, Case)]
            if not cases:
                return IROpaqueStmt(payload=stmt, source=stmt)

            def build_case(case_stmt):
                cond = IRBinary(
                    op="eq",
                    left=lower_expr(stmt.value),
                    right=lower_expr(case_stmt.value),
                )
                return IRIf(
                    cond=cond,
                    then=[lower_stmt(s) for s in case_stmt.scope.get_statements()],
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
            return IRPrint(values=[lower_expr(value) for value in stmt.to_print], source=stmt)
        if isinstance(stmt, Allocate):
            return IRAllocate(
                var=lower_expr(stmt.target),
                dims=[lower_expr(dim) for dim in stmt.shape],
                source=stmt,
            )
        if isinstance(stmt, Deallocate):
            return IRDeallocate(var=lower_expr(stmt.array), source=stmt)
        if stmt.__class__.__name__ == "VStore":
            return IRSimdStore(
                array=lower_expr(stmt.array),
                index=lower_expr(stmt.index),
                value=lower_expr(stmt.value),
                aligned=bool(getattr(stmt, "aligned", False)),
                source=stmt,
            )
        return IROpaqueStmt(payload=stmt, source=stmt)

    args = [lower_var(var, is_arg=True) for var in procedure.arguments.values()]
    locals_ = [lower_var(var, is_arg=False) for var in procedure.get_local_variables().values()]
    body = [lower_stmt(stmt) for stmt in procedure.scope.get_statements()]

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
        },
    )


def _lower_value_type(ftype, shape) -> IRValueType:
    dtype = _lower_type(ftype)
    return IRValueType(dtype=dtype, shape=_lower_ir_shape(shape, settings.syntax))


def _lower_shape_dims(dims, syntax_settings) -> tuple:
    lowered = []
    for dim in dims:
        if isinstance(dim, int):
            lowered.append(dim)
            continue
        lowered_dim = _lower_index_value(dim, syntax_settings)
        lowered.append(lowered_dim if lowered_dim is not None else dim)
    return tuple(lowered)


def _lower_type(ftype) -> IRType:
    name = getattr(ftype, "type", str(ftype))
    kind = None
    if hasattr(ftype, "get_kind_str"):
        kind = ftype.get_kind_str()
    return IRType(name=name, kind=kind)


def _safe_shape(expr):
    return expr._shape


def _lower_indices(slice_, syntax_settings) -> list[IRExpr | IRSlice]:
    if isinstance(slice_, tuple):
        return [_lower_single_index(item, syntax_settings) for item in slice_]
    return [_lower_single_index(slice_, syntax_settings)]


def _lower_single_index(item, syntax_settings) -> IRExpr | IRSlice:
    if isinstance(item, slice):
        return _normalize_slice(item, syntax_settings)
    expr = _lower_index_value(item, syntax_settings) or IROpaqueExpr(payload=item, source=item)
    return _shift_expr(expr, -syntax_settings.array_lower_bound)


def _lower_index_value(value, syntax_settings) -> IRExpr | None:
    if value is None:
        return None
    value_type = type(value)
    if value_type is int or value_type is float or value_type is bool or value_type is str:
        return IRLiteral(value=value)
    if value_type is LiteralNode:
        return IRLiteral(value=value.value)
    if value_type is Variable:
        return IRVarRef(var=IRVar(name=value.name, source=value), source=value)
    if value_type is BinaryOperationNode or isinstance(value, BinaryOperationNode):
        op = _map_binary_op(value.op)
        left = _lower_index_value(value.left, syntax_settings) or IROpaqueExpr(
            payload=value.left, source=value.left
        )
        right = _lower_index_value(value.right, syntax_settings) or IROpaqueExpr(
            payload=value.right, source=value.right
        )
        return IRBinary(op=op, left=left, right=right)
    if value_type is GetItem:
        base = _lower_index_value(value.variable, syntax_settings) or IROpaqueExpr(
            payload=value.variable, source=value.variable
        )
        return IRGetItem(base=base, indices=_lower_indices(value.sliced, syntax_settings))
    if value_type is GetAttr:
        base = _lower_index_value(value.variable, syntax_settings) or IROpaqueExpr(
            payload=value.variable, source=value.variable
        )
        return IRGetAttr(base=base, name=value.attr)
    if value_type is FunctionCall:
        callee = _lower_index_value(value.function, syntax_settings) or IROpaqueExpr(
            payload=value.function, source=value.function
        )
        args = []
        for argument in value.arguments:
            lowered_arg = _lower_index_value(argument, syntax_settings)
            if lowered_arg is not None:
                args.append(lowered_arg)
        return IRCallExpr(
            callee=callee,
            args=args,
        )
    if value_type is IntrinsicFunction or isinstance(value, IntrinsicFunction):
        args = []
        for argument in value.arguments:
            lowered_arg = _lower_index_value(argument, syntax_settings)
            if lowered_arg is not None:
                args.append(lowered_arg)
        return IRIntrinsic(
            name=getattr(value, "token", ""),
            args=args,
        )
    return IROpaqueExpr(payload=value, source=value)


def _normalize_slice(slice_: slice, syntax_settings) -> IRSlice:
    lbound = syntax_settings.array_lower_bound
    c_like = syntax_settings.c_like_bounds

    if slice_.start is None:
        start = IRLiteral(value=0)
    else:
        start = _lower_index_value(slice_.start, syntax_settings) or IROpaqueExpr(
            payload=slice_.start, source=slice_.start
        )
        start = _shift_expr(start, -lbound)

    stop = None
    if slice_.stop is not None:
        stop_expr = _lower_index_value(slice_.stop, syntax_settings) or IROpaqueExpr(
            payload=slice_.stop, source=slice_.stop
        )
        shift = (0 if c_like else 1) - lbound
        stop = _shift_expr(stop_expr, shift)

    step = None
    if slice_.step is not None:
        step = _lower_index_value(slice_.step, syntax_settings) or IROpaqueExpr(
            payload=slice_.step, source=slice_.step
        )

    return IRSlice(start=start, stop=stop, step=step)


def _shift_expr(expr: IRExpr, delta: int) -> IRExpr:
    if delta == 0:
        return expr
    if isinstance(expr, IRLiteral) and isinstance(expr.value, (int, float)):
        return IRLiteral(value=expr.value + delta)
    op = "add" if delta > 0 else "sub"
    return IRBinary(op=op, left=expr, right=IRLiteral(value=abs(delta)))


def _map_binary_op(op: str) -> str:
    return _BINARY_OPS.get(op, op)
