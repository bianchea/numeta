"""Normalization passes over backend-neutral Numeta IR."""

from __future__ import annotations

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
    IRPrint,
    IRProcedure,
    IRReduce,
    IRReturn,
    IRSimdStore,
    IRSlice,
    IRUnary,
    IRVar,
    IRVarRef,
    IRWhile,
)

_REDUCTIONS = {"sum", "maxval", "minval", "all"}


def normalize_reductions(proc: IRProcedure) -> IRProcedure:
    """Extract nested reductions into explicit scalar IR statements."""
    if proc.metadata.get("reductions_normalized"):
        return proc

    used_names = {variable.name for variable in (*proc.args, *proc.locals)}
    counter = 0

    def new_temp(expr: IRIntrinsic) -> IRVar:
        nonlocal counter
        while True:
            counter += 1
            name = f"_nm_reduce_{counter}"
            if name not in used_names:
                used_names.add(name)
                break
        variable = IRVar(name=name, vtype=expr.vtype, source=expr.source)
        proc.locals.append(variable)
        return variable

    def rewrite_expr(expr: IRExpr | None):
        if expr is None:
            return [], None
        prelude = []

        def rewrite_child(child):
            child_prelude, rewritten = rewrite_expr(child)
            prelude.extend(child_prelude)
            return rewritten

        if isinstance(expr, IRBinary):
            expr.left = rewrite_child(expr.left)
            expr.right = rewrite_child(expr.right)
        elif isinstance(expr, IRUnary):
            expr.operand = rewrite_child(expr.operand)
        elif isinstance(expr, IRCallExpr):
            expr.callee = rewrite_child(expr.callee)
            expr.args = [rewrite_child(argument) for argument in expr.args]
        elif isinstance(expr, IRGetItem):
            expr.base = rewrite_child(expr.base)
            indices = []
            for index in expr.indices:
                if isinstance(index, IRSlice):
                    index.start = rewrite_child(index.start)
                    index.stop = rewrite_child(index.stop)
                    index.step = rewrite_child(index.step)
                    indices.append(index)
                else:
                    indices.append(rewrite_child(index))
            expr.indices = indices
        elif isinstance(expr, IRGetAttr):
            expr.base = rewrite_child(expr.base)
        elif isinstance(expr, IRIntrinsic):
            expr.args = [rewrite_child(argument) for argument in expr.args]
            if expr.name in _REDUCTIONS and expr.args:
                temporary = new_temp(expr)
                target = IRVarRef(var=temporary, vtype=expr.vtype, source=expr.source)
                prelude.append(
                    IRReduce(
                        op=expr.name,
                        target=target,
                        value=expr.args[0],
                        source=expr.source,
                    )
                )
                return prelude, target
        return prelude, expr

    def rewrite_block(statements):
        rewritten = []
        for statement in statements:
            prelude = []

            def rewrite(child):
                child_prelude, value = rewrite_expr(child)
                prelude.extend(child_prelude)
                return value

            if isinstance(statement, IRAssign):
                statement.target = rewrite(statement.target)
                statement.value = rewrite(statement.value)
            elif isinstance(statement, IRCall):
                statement.func = rewrite(statement.func)
                statement.args = [rewrite(argument) for argument in statement.args]
            elif isinstance(statement, IRIf):
                statement.cond = rewrite(statement.cond)
                statement.then = rewrite_block(statement.then)
                statement.else_ = rewrite_block(statement.else_)
            elif isinstance(statement, IRFor):
                statement.start = rewrite(statement.start)
                statement.stop = rewrite(statement.stop)
                statement.step = rewrite(statement.step)
                statement.body = rewrite_block(statement.body)
            elif isinstance(statement, IRWhile):
                statement.cond = rewrite(statement.cond)
                statement.body = rewrite_block(statement.body)
            elif isinstance(statement, IRReturn):
                statement.value = rewrite(statement.value)
            elif isinstance(statement, IRPrint):
                statement.values = [rewrite(value) for value in statement.values]
            elif isinstance(statement, IRAllocate):
                statement.var = rewrite(statement.var)
                statement.dims = [rewrite(dimension) for dimension in statement.dims]
            elif isinstance(statement, IRDeallocate):
                statement.var = rewrite(statement.var)
            elif isinstance(statement, IRSimdStore):
                statement.array = rewrite(statement.array)
                statement.index = rewrite(statement.index)
                statement.value = rewrite(statement.value)
            rewritten.extend(prelude)
            rewritten.append(statement)
        return rewritten

    proc.body = rewrite_block(proc.body)
    proc.metadata["reductions_normalized"] = True
    return proc
