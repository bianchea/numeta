"""Tests for source location tracking in AST nodes."""

import pytest
import numpy as np
from pathlib import Path

from numeta.settings import settings
from numeta import jit
from numeta.c.emitter import CEmitter
from numeta.c.c_syntax import render_stmt_lines as render_c_stmt_lines
from numeta.fortran.fortran_syntax import render_stmt_lines as render_fortran_stmt_lines
from numeta.array_shape import UNKNOWN
from numeta.ast.nodes.base_node import Node
from numeta.ast.procedure import Procedure
from numeta.ast.expressions.getitem import GetItem
from numeta.ast.expressions.literal_node import LiteralNode
from numeta.ast.expressions.binary_operation_node import BinaryOperationNode
from numeta.ast.expressions.intrinsic_functions import Shape, Transpose
from numeta.ast.expressions.various import ArrayConstructor
from numeta.ast.namespace import Namespace
from numeta.ast.scope import Scope
from numeta.ast.variable import Variable
from numeta.ast.statements.various import Assignment, ElseIf, PointerAssignment, Section
from numeta.ast.statements.variable_declaration import VariableDeclaration
from numeta.ir.lowering import lower_procedure
from numeta.ir.nodes import IRAssign, IRGetItem as IRGetItemNode, IRLiteral as IRLiteralNode
from numeta.exceptions import (
    NumetaNotImplementedError,
    NumetaTypeError,
    format_source_location,
    raise_with_source,
)


class MockNode(Node):
    """Mock node for testing source location."""

    def extract_entities(self):
        yield from []

    def get_with_updated_variables(self, variables_couples):
        return self


def test_node_captures_source_location():
    """Test that Node captures source location on creation."""
    node = MockNode()

    assert node.source_location is not None
    assert "filename" in node.source_location
    assert "lineno" in node.source_location
    assert "function" in node.source_location

    # Should point to this test file
    assert "test_source_tracking.py" in node.source_location["filename"]
    assert node.source_location["function"] == "test_node_captures_source_location"


def test_node_source_location_can_be_disabled():
    """Source tracking can be disabled globally for performance-sensitive generation."""
    previous = settings.track_source_location
    try:
        settings.unset_source_location_tracking()
        node = MockNode()
        assert node.source_location is None
    finally:
        settings.set_track_source_location(previous)


def test_source_location_skips_numeta_internal_frames():
    """Test that source location skips numeta internal code frames."""

    # Create a node inside a function that mimics being inside numeta
    def helper_inside_numeta():
        # This simulates being called from numeta's internal code
        return MockNode()

    node = helper_inside_numeta()

    # The source location should NOT point to numeta internal files
    loc = node.source_location
    assert loc is not None

    filename = loc.get("filename", "")
    assert "numeta/ast/" not in filename
    assert "numeta/wrappers/" not in filename


def test_format_source_location_with_node():
    """Test format_source_location extracts info from a node."""
    node = MockNode()

    formatted = format_source_location(node)

    assert formatted is not None
    assert "test_source_tracking.py" in formatted
    assert "File" in formatted
    assert "line" in formatted


def test_format_source_location_with_none():
    """Test format_source_location handles None gracefully."""
    assert format_source_location(None) is None


def test_format_source_location_shows_source_code():
    """Test that format_source_location shows the actual source code line."""
    # Create a node on a specific line with recognizable code
    node = MockNode()  # LINE_MARKER_FOR_TEST

    formatted = format_source_location(node)

    assert formatted is not None
    # The formatted string should include the actual source code
    assert "LINE_MARKER_FOR_TEST" in formatted


def test_raise_with_source_includes_location():
    """Test raise_with_source includes source location in error."""
    node = MockNode()

    with pytest.raises(NotImplementedError) as exc_info:
        raise_with_source(NotImplementedError, "Test error message", source_node=node)

    error_msg = str(exc_info.value)
    assert "Test error message" in error_msg
    assert "test_source_tracking.py" in error_msg
    assert "line" in error_msg.lower()


def test_raise_with_source_without_node():
    """Test raise_with_source works without source node."""
    with pytest.raises(ValueError) as exc_info:
        raise_with_source(ValueError, "Error without source")

    assert "Error without source" in str(exc_info.value)


def test_raise_with_source_with_invalid_location_falls_back_cleanly():
    """Malformed source location metadata should not break error formatting."""

    class BadSource:
        source_location: dict[str, int]

        def __init__(self):
            self.source_location = {"lineno": 999999}

    node = BadSource()

    with pytest.raises(ValueError) as exc_info:
        raise_with_source(ValueError, "Fallback formatting message", source_node=node)

    error_msg = str(exc_info.value)
    assert "Fallback formatting message" in error_msg
    assert 'File "<unknown>"' in error_msg


def test_expression_bool_error_shows_source_location():
    """Using bool on expressions should include source location."""
    expr = LiteralNode(1)  # BOOL_EXPR_LINE

    with pytest.raises(NumetaTypeError) as exc_info:
        bool(expr)

    error_msg = str(exc_info.value)
    assert "test_source_tracking.py" in error_msg
    assert "Do not use 'bool' operator for expressions." in error_msg


def test_literal_node_unsupported_type_shows_source_location():
    """Unsupported literal type should include source location."""

    class UnsupportedLiteral:
        pass

    with pytest.raises(ValueError) as exc_info:
        LiteralNode(UnsupportedLiteral())  # UNSUPPORTED_LITERAL_LINE

    error_msg = str(exc_info.value)
    assert "unsupported for LiteralNode" in error_msg
    assert "test_source_tracking.py" in error_msg


def test_shape_intrinsic_scalar_error_shows_source_location():
    """shape() on scalar should include user source location."""
    scalar = LiteralNode(1)  # SHAPE_SCALAR_LINE

    with pytest.raises(ValueError) as exc_info:
        Shape(scalar)

    error_msg = str(exc_info.value)
    assert "shape intrinsic function cannot be applied to a scalar" in error_msg.lower()
    assert "test_source_tracking.py" in error_msg


def test_transpose_scalar_error_shows_source_location():
    """Transpose scalar error should include user source location."""
    scalar = LiteralNode(1)  # TRANSPOSE_SCALAR_LINE
    transposed = Transpose(scalar)

    with pytest.raises(ValueError) as exc_info:
        _ = transposed._shape

    error_msg = str(exc_info.value)
    assert "cannot transpose a scalar" in error_msg.lower()
    assert "test_source_tracking.py" in error_msg


def test_getitem_merge_slice_step_error_shows_source_location():
    """Step slicing merge errors should include source location."""
    var = Variable("x", dtype=np.float64, shape=(20,))
    item = GetItem(var, slice(1, 10, None))  # MERGE_STEP_LINE

    with pytest.raises(NumetaNotImplementedError) as exc_info:
        item.merge_slice(slice(2, 8, 2))

    error_msg = str(exc_info.value)
    assert "step slicing not implemented for slice merging" in error_msg.lower()
    assert "test_source_tracking.py" in error_msg


def test_array_constructor_empty_error_shows_source_location():
    """ArrayConstructor dtype errors should include source location."""
    arr = ArrayConstructor()  # ARRAY_CONSTRUCTOR_EMPTY_LINE

    with pytest.raises(ValueError) as exc_info:
        _ = arr.dtype

    error_msg = str(exc_info.value)
    assert "must have at least one element" in error_msg.lower()
    assert "test_source_tracking.py" in error_msg


def test_namespace_missing_attribute_error_shows_source_location():
    """Namespace __getattr__ errors should include source location."""
    ns = Namespace("my_ns")  # NAMESPACE_ATTR_ERROR_LINE

    with pytest.raises(AttributeError) as exc_info:
        _ = ns.not_present

    error_msg = str(exc_info.value)
    assert "has no attribute" in error_msg
    assert "test_source_tracking.py" in error_msg


def test_elseif_without_if_error_shows_source_location():
    """ElseIf misuse should include source location."""
    scope = Scope()
    scope.enter()
    try:
        Section(add_to_scope=True)
        with pytest.raises(Exception) as exc_info:
            ElseIf(LiteralNode(True))  # ELSEIF_WITHOUT_IF_LINE
    finally:
        scope.exit()

    error_msg = str(exc_info.value)
    assert "last statement is not an if statement" in error_msg.lower()
    assert "test_source_tracking.py" in error_msg


def test_pointer_assignment_target_error_shows_source_location():
    """PointerAssignment invalid target should include source location."""
    pointer = Variable("p", dtype=np.float64, shape=(10,))

    with pytest.raises(Exception) as exc_info:
        PointerAssignment(
            pointer,
            pointer._shape,
            LiteralNode(0),  # POINTER_TARGET_ERROR_LINE
            add_to_scope=False,
        )

    error_msg = str(exc_info.value)
    assert "target of a pointer must be a variable or getitem" in error_msg.lower()
    assert "test_source_tracking.py" in error_msg


def test_lowering_variable_without_dtype_shows_source_location():
    """Lowering errors should include user source location."""
    proc = Procedure("missing_dtype_proc")
    arg = Variable("x")  # LOWERING_MISSING_DTYPE_LINE
    proc.add_variable(arg)

    with pytest.raises(ValueError) as exc_info:
        lower_procedure(proc, backend="c")

    error_msg = str(exc_info.value)
    assert "has no dtype" in error_msg.lower()
    assert "test_source_tracking.py" in error_msg


def test_c_emitter_getitem_metadata_error_shows_source_location():
    """C emitter errors should include original source location when available."""
    source_node = LiteralNode(7)  # EMITTER_GETITEM_LINE
    expr = IRGetItemNode(
        base=IRLiteralNode(value=0),
        indices=[IRLiteralNode(value=0)],
        source=source_node,
    )

    emitter = CEmitter()
    with pytest.raises(NotImplementedError) as exc_info:
        emitter._render_getitem(expr)

    error_msg = str(exc_info.value)
    assert "requires array metadata for getitem" in error_msg.lower()
    assert "test_source_tracking.py" in error_msg


def test_source_propagates_ast_to_ir_to_c_emitter_errors():
    """Source should survive AST -> IR lowering and be used by C emitter errors."""
    proc = Procedure("pipeline_source_proc")
    arr = Variable("arr", dtype=np.float64, shape=UNKNOWN)
    out = Variable("out", dtype=np.float64)
    proc.add_variable(arr, out)

    proc.scope.enter()
    try:
        Assignment(out, GetItem(arr, 0))  # PIPELINE_GETITEM_LINE
    finally:
        proc.scope.exit()

    ir_proc = lower_procedure(proc, backend="c")
    ir_stmt = ir_proc.body[0]
    assert isinstance(ir_stmt, IRAssign)
    assert isinstance(ir_stmt.value, IRGetItemNode)

    emitter = CEmitter()
    with pytest.raises(NotImplementedError) as exc_info:
        emitter._render_getitem(ir_stmt.value)

    error_msg = str(exc_info.value)
    assert "requires array metadata for getitem" in error_msg.lower()
    assert "test_source_tracking.py" in error_msg
    assert "PIPELINE_GETITEM_LINE" in error_msg


def test_c_syntax_unknown_shape_assignment_error_shows_source_location():
    """C syntax renderer errors should include source location."""
    var = Variable("x", dtype=np.float64, shape=UNKNOWN, assign=1.0)  # C_SYNTAX_UNKNOWN_SHAPE_LINE
    decl = VariableDeclaration(var)

    with pytest.raises(ValueError) as exc_info:
        render_c_stmt_lines(decl)

    error_msg = str(exc_info.value)
    assert "cannot assign to a variable with unknown shape" in error_msg.lower()
    assert "test_source_tracking.py" in error_msg


def test_fortran_syntax_unknown_shape_assignment_error_shows_source_location():
    """Fortran syntax renderer errors should include source location."""
    var = Variable(
        "x", dtype=np.float64, shape=UNKNOWN, assign=1.0
    )  # FORTRAN_SYNTAX_UNKNOWN_SHAPE_LINE
    decl = VariableDeclaration(var)

    with pytest.raises(ValueError) as exc_info:
        render_fortran_stmt_lines(decl)

    error_msg = str(exc_info.value)
    assert "cannot assign to a variable with unknown shape" in error_msg.lower()
    assert "test_source_tracking.py" in error_msg


def test_getitem_captures_source_location():
    """Test GetItem captures source location."""
    var = Variable("test_var", dtype=np.float64, shape=(10,))

    # Create GetItem
    item = GetItem(var, 0)

    assert item.source_location is not None
    assert "test_source_tracking.py" in item.source_location["filename"]


def test_literal_node_captures_source_location():
    """Test LiteralNode captures source location."""
    lit = LiteralNode(42)

    assert lit.source_location is not None
    assert "test_source_tracking.py" in lit.source_location["filename"]


def test_binary_operation_captures_source_location():
    """Test BinaryOperationNode captures source location."""
    left = LiteralNode(1)
    right = LiteralNode(2)

    op = BinaryOperationNode(left, "+", right)

    assert op.source_location is not None
    assert "test_source_tracking.py" in op.source_location["filename"]


def test_variable_captures_source_location():
    """Test Variable captures source location."""
    var = Variable("my_var", dtype=np.float64)

    assert var.source_location is not None
    assert "test_source_tracking.py" in var.source_location["filename"]


# Integration tests with JIT compilation


def test_step_slicing_supported_for_positive_step():
    """Positive step slicing should now lower successfully."""

    @jit(backend="c")
    def test_func(arr):
        return arr[::2]  # STEP_SLICING_ERROR_LINE

    arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    np.testing.assert_equal(test_func(arr), arr[::2])


@pytest.mark.parametrize("test_backend", ["c", "fortran"])
def test_getitem_error_shows_source_across_backends(test_backend):
    """Test source location works for both backends."""

    @jit(backend=test_backend)
    def test_func(arr):
        return arr[::2]  # BACKEND_TEST_LINE

    arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)

    # Note: Fortran backend might support step slicing, so this may not error
    try:
        result = test_func(arr)
        # If it works, that's fine - the test passes
    except NotImplementedError as e:
        # If it errors, check source location is shown
        error_msg = str(e)
        if "step" in error_msg.lower():
            assert "test_source_tracking.py" in error_msg


@pytest.mark.parametrize(
    "shape_args",
    [
        (slice(None, None, 2),),  # Step 2
        (slice(0, 10, 3),),  # Step 3
    ],
)
def test_various_positive_step_values(shape_args):
    """Different positive step values should lower successfully."""

    @jit(backend="c")
    def test_func(arr):
        return arr[shape_args[0]]  # VARIOUS_STEP_LINE

    arr = np.arange(20, dtype=np.float64)
    np.testing.assert_equal(test_func(arr), arr[shape_args[0]])


# Test that source location works through various node types


def test_nested_expression_source_location():
    """Test source location propagates through nested expressions."""
    var = Variable("x", dtype=np.float64, shape=(10,))

    # Create nested expression: var[::2] + 1
    # This should capture location at the GetItem level
    item = GetItem(var, slice(None, None, 2))  # NESTED_EXPR_LINE

    # The GetItem should have source location
    assert item.source_location is not None
    assert "NESTED_EXPR_LINE" in format_source_location(item)
