"""Backend-neutral IR analysis for runtime array-shape contracts."""

from __future__ import annotations

from .ir.lowering import lower_procedure
from .ir.nodes import IRAssign, IRBinary, IRGetItem, IRIf, IRLiteral, IRUnary, IRVarRef
from .wrapper_spec import ShapeEquality


def infer_shape_equalities(symbolic_function, argument_specs) -> tuple[ShapeEquality, ...]:
    """Infer equal-shape requirements for whole-array elementwise assignments."""
    procedure = lower_procedure(symbolic_function, backend="c")
    array_arguments = {
        argument.name
        for argument in argument_specs
        if not argument.is_comptime and isinstance(argument.rank, int) and argument.rank > 0
    }
    equalities: set[ShapeEquality] = set()

    def elementwise_sources(expression):
        if isinstance(expression, IRVarRef) and expression.var is not None:
            if expression.var.name in array_arguments and expression.vtype.shape is not None:
                return {expression.var.name}
            return set()
        if isinstance(expression, IRLiteral):
            return set()
        if isinstance(expression, IRBinary):
            left = elementwise_sources(expression.left)
            right = elementwise_sources(expression.right)
            if left is None or right is None:
                return None
            return left | right
        if isinstance(expression, IRUnary):
            return elementwise_sources(expression.operand)
        if isinstance(expression, IRGetItem):
            return set() if expression.vtype.shape is None else None
        return None

    def visit(statements):
        for statement in statements:
            if (
                isinstance(statement, IRAssign)
                and isinstance(statement.target, IRVarRef)
                and statement.target.var is not None
                and statement.target.var.name in array_arguments
            ):
                sources = elementwise_sources(statement.value)
                if sources is not None:
                    for source in sources:
                        if source != statement.target.var.name:
                            equalities.add(ShapeEquality(statement.target.var.name, source))

            if isinstance(statement, IRIf):
                visit(statement.then)
                visit(statement.else_)
            body = getattr(statement, "body", None)
            if isinstance(body, list):
                visit(body)

    visit(procedure.body)
    return tuple(sorted(equalities))
