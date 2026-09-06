"""Read-only checks on the completed trace; never schedule or materialize expressions."""

from collections import Counter
from numbers import Integral
import warnings

from .ast.expressions import (
    BinaryOperationNode,
    ExpressionNode,
    FunctionCall,
    GetAttr,
    GetItem,
    IntrinsicFunction,
    LiteralNode,
    WholeStorage,
)
from .ast.expressions.various import ArrayConstructor, Re, Im
from .ast.statements import Assignment, Break, Continue, For, While
from .ast.variable import Variable
from .exceptions import (
    NumetaError,
    NumetaTypeError,
    NumetaPerformanceWarning,
    format_source_location,
    raise_with_source,
)


def expression_children(node):
    if isinstance(node, BinaryOperationNode):
        return (node.left, node.right)
    if isinstance(node, (FunctionCall, IntrinsicFunction)):
        return node.arguments
    if isinstance(node, (GetAttr, WholeStorage, Re, Im)):
        return (node.variable,)
    if isinstance(node, GetItem):
        return (node.variable, node.sliced)
    if isinstance(node, ArrayConstructor):
        return node.elements
    if isinstance(node, (tuple, list)):
        return node
    if isinstance(node, slice):
        return (node.start, node.stop, node.step)
    return ()


def walk_expression(node):
    yield node
    for child in expression_children(node):
        yield from walk_expression(child)


def storage_base(node):
    while isinstance(node, (GetItem, GetAttr, WholeStorage)):
        node = node.variable
    if isinstance(node, (Re, Im)):
        return storage_base(node.variable)
    return node if isinstance(node, Variable) else None


def constant_array_element(node):
    """Identify a full constant-index access, without assuming runtime aliasing."""
    if not isinstance(node, GetItem):
        return None
    base = node.variable
    while isinstance(base, WholeStorage):
        base = base.variable
    if not isinstance(base, Variable) or base._shape.is_scalar or base._shape.is_unknown:
        return None
    indices = node.sliced if isinstance(node.sliced, tuple) else (node.sliced,)
    if len(indices) != base._shape.rank:
        return None
    indices = tuple(index.value if isinstance(index, LiteralNode) else index for index in indices)
    if not all(isinstance(index, Integral) for index in indices):
        return None
    return id(base), tuple(int(index) for index in indices)


def array_write_key(target):
    while isinstance(target, WholeStorage):
        target = target.variable
    if isinstance(target, Variable) and not target._shape.is_scalar:
        return id(target), None  # A whole-array overwrite overlaps every element.
    return constant_array_element(target)


def validate_trace(builder):
    uses = Counter()
    count = 0

    def consume(expression, statement, writes, target=False):
        for node in walk_expression(expression):
            if isinstance(node, FunctionCall):
                uses[id(node)] += 1
            if not isinstance(node, ExpressionNode):
                continue
            if node.dtype is None or node._shape is None:
                raise_with_source(
                    NumetaTypeError, "Cannot determine expression dtype or shape.", node
                )
            if target or isinstance(node, (Variable, LiteralNode)):
                continue
            dependencies = []
            for dependency in walk_expression(node):
                if not isinstance(dependency, Variable) or not dependency._shape.is_scalar:
                    continue
                dependencies.append(id(dependency))
            element = constant_array_element(node)
            if element is not None:
                dependencies.extend((element, (element[0], None)))
            for dependency in dependencies:
                write = writes.get(dependency)
                if write is not None and node._trace_sequence < write._trace_sequence:
                    write_location = format_source_location(write) or "location unavailable"
                    raise_with_source(
                        NumetaError,
                        "An expression created before a write is used afterward. "
                        "Use nm.scalar(expr) when creating the expression to snapshot its value.\n"
                        f"Intervening write:\n{write_location}",
                        node,
                    )

    def visit(statements, writes, loop_depth=0):
        nonlocal count
        for statement in statements:
            count += 1
            if isinstance(statement, (Break, Continue)) and loop_depth == 0:
                raise_with_source(
                    NumetaError, "nm.Break()/nm.Continue() require a Numeta loop.", statement
                )
            if isinstance(statement, Assignment):
                base = storage_base(statement.target)
                if base is None:
                    raise_with_source(
                        NumetaTypeError,
                        "Assignment requires storage. Use nm.scalar or nm.empty.",
                        statement,
                    )
                consume(statement.target, statement, writes, target=True)
                # A storage target is not a read, but its index expressions are.
                for node in walk_expression(statement.target):
                    if isinstance(node, GetItem):
                        consume(node.sliced, statement, writes)
                consume(statement.value, statement, writes)
                target_shape = statement.target._shape
                value_shape = statement.value._shape
                if target_shape.is_scalar and not value_shape.is_scalar:
                    raise_with_source(
                        NumetaTypeError,
                        "Cannot store an array in a scalar; use nm.empty and a sliced assignment.",
                        statement,
                    )
                if base._shape.is_scalar:
                    writes[id(base)] = statement
                else:
                    key = array_write_key(statement.target)
                    if key is not None:
                        writes[key] = statement
            else:
                for child in statement.children:
                    consume(child, statement, writes)
            if hasattr(statement, "scope"):
                visit(
                    statement.scope.body,
                    writes.copy(),
                    loop_depth + isinstance(statement, (For, While)),
                )
                for branch in getattr(statement, "orelse", ()):
                    for child in branch.children:
                        consume(child, branch, writes)
                    visit(branch.scope.body, writes.copy(), loop_depth)

    visit(builder.symbolic_function.scope.body, {})
    for call in builder.effectful_calls:
        use_count = uses[id(call)]
        if use_count != 1:
            raise_with_source(
                NumetaError,
                f"An effectful expression was {'discarded' if use_count == 0 else 'used more than once'}. "
                "Materialize it once with nm.scalar(...), for example nm.scalar(nm.time()). "
                "Declare an external function pure=True only when it has no observable effects.",
                call,
            )
    from .settings import settings

    threshold = getattr(settings, "performance_warning_threshold", 5000)
    if threshold is not None and count > threshold:
        warnings.warn(
            f"Trace contains {count} statements. Use nm.range(...) to avoid Python loop unrolling.",
            NumetaPerformanceWarning,
            stacklevel=3,
        )
