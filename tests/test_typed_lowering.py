from dataclasses import fields

import numpy as np
import pytest

import numeta as nm
from numeta.array_shape import ArrayShape
from numeta.ast import Variable
from numeta.ast.procedure import Procedure
from numeta.ast.scope import Scope
from numeta.ast.statements import Assignment
from numeta.ir import lower_procedure
from numeta.ir.nodes import IRBinary, IRIntrinsic, IRGetItem, IRVarRef, IRNode, IRExpr
from numeta.ir.validation import validate_numeric_ir


def walk(node):
    if not isinstance(node, IRNode):
        return
    yield node
    for field in fields(node):
        if field.name in {"source", "metadata", "vtype", "var"}:
            continue
        value = getattr(node, field.name)
        for child in value if isinstance(value, (list, tuple)) else (value,):
            yield from walk(child)


def test_casts_precede_arithmetic_and_indices_are_typed(backend):
    proc = Procedure("typed")
    a = Variable("a", dtype=nm.f4, shape=ArrayShape((4,)))
    b = Variable("b", dtype=nm.f8)
    i = Variable("i", dtype=nm.f8)
    out = Variable("out", dtype=nm.f8)
    proc.add_variable(a, b, i, out)
    old = Scope.current_scope
    Scope.current_scope = proc.scope
    try:
        Assignment(out, a[nm.astype(i, nm.i8) % 4] + b)
    finally:
        Scope.current_scope = old
    ir = lower_procedure(proc, backend)
    add = next(node for node in walk(ir) if isinstance(node, IRBinary) and node.op == "add")
    assert isinstance(add.left, IRIntrinsic) and add.left.name == "astype"
    assert add.left.vtype.dtype.datatype is nm.f8
    assert add.left.args[0].vtype.dtype.datatype is nm.f4
    item = add.left.args[0]
    assert isinstance(item, IRGetItem)
    index = item.indices[0]
    assert index.vtype.dtype.datatype is nm.i8
    assert index.vtype.shape is None
    assert index.source is not None
    for expr in walk(ir):
        if isinstance(expr, IRExpr):
            assert expr.vtype is not None


@pytest.mark.parametrize("bound", [0.5, True])
def test_reject_invalid_indices(backend, bound):
    @nm.jit(backend=backend)
    def kernel(a):
        return a[bound]

    with pytest.raises(nm.NumetaTypeError, match="Expected an integer"):
        kernel(np.arange(4.0))


def test_array_cast_index_and_slice(backend):
    @nm.jit(backend=backend)
    def kernel(a, out):
        converted = nm.astype(a, nm.f8)
        out[:] = converted[1:]
        return converted[0]

    out = np.empty(2)
    assert kernel(np.array([1, 2, 3], dtype=np.float32), out) == 1
    np.testing.assert_equal(out, [2, 3])


def test_typed_literal_rounds_before_operation(backend):
    @nm.jit(backend=backend)
    def kernel(a):
        return (a + np.float32(1)) - a

    assert kernel(np.float32(2**24)) == 0


@pytest.mark.parametrize("inline", [False, True])
def test_promoted_call_arguments_and_inlining(backend, inline):
    @nm.jit(backend=backend, inline=inline)
    def helper(a):
        return a + np.float64(1)

    @nm.jit(backend=backend)
    def kernel(a):
        return helper(a + 1.0)

    # The weak addition rounds in float32, then the strongly typed literal promotes.
    assert kernel(np.float32(2**24)) == 2**24 + 1


def test_validator_rejects_opaque_values_and_procedure_values():
    from numeta.ir.nodes import IROpaqueExpr, IRProcedure, IRAssign, IRVar

    for value in (
        IROpaqueExpr(payload=object()),
        IRVarRef(var=IRVar(name="function"), metadata={"procedure_reference": True}),
    ):
        with pytest.raises(nm.NumetaTypeError, match="Expected a typed numeric|Expected a numeric"):
            validate_numeric_ir(IRProcedure(body=[value]))


def test_shift_preserves_type_and_source():
    from numeta.ir.lowering import _shift_expr
    from numeta.ir.nodes import IRLiteral, IRValueType, IRType

    source = object()
    vtype = IRValueType(IRType("int64", datatype=nm.i8), None)
    for expr in (
        IRLiteral(value=3, vtype=vtype, source=source),
        IRVarRef(vtype=vtype, source=source),
    ):
        shifted = _shift_expr(expr, 1)
        assert shifted.vtype is vtype
        assert shifted.source is source


def test_shape_checks_survive_promoted_array_casts(backend):
    @nm.jit(backend=backend)
    def kernel(a, b, out):
        out[:] = a + b

    with pytest.raises(ValueError, match="shape|Shape"):
        kernel(np.ones(2, np.float32), np.ones(3, np.float64), np.zeros(2))


def test_emitter_uses_canonical_literal_type(backend):
    from numeta.c.emitter import CEmitter
    from numeta.fortran.emitter import FortranEmitter
    from numeta.ir.nodes import IRLiteral, IRType, IRValueType

    emitter = CEmitter() if backend == "c" else FortranEmitter()
    literal = IRLiteral(value=1, vtype=IRValueType(IRType("real", datatype=nm.f8), None))
    assert not emitter._is_integer_expr(literal)
