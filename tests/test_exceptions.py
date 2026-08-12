import inspect
import types

import pytest

from numeta.ast.expressions import EqBinaryNode, GetItem, LiteralNode, NeBinaryNode
from numeta.ast.variable import Variable
from numeta.builder_helper import BuilderHelper
from numeta.compiler import Compiler
from numeta.exceptions import (
    CompilationError,
    NumetaError,
    NumetaNotImplementedError,
    NumetaTypeError,
)
from numeta.wrappers.cond import CondHelper


def test_builder_get_current_builder_raises_numeta_error():
    previous_builder = BuilderHelper.current_builder
    BuilderHelper.current_builder = None
    try:
        with pytest.raises(NumetaError, match="not initialized"):
            BuilderHelper.get_current_builder()
    finally:
        BuilderHelper.current_builder = previous_builder


def test_expression_bool_raises_numeta_type_error():
    expr = LiteralNode(1)
    with pytest.raises(NumetaTypeError, match="Do not use 'bool' operator"):
        bool(expr)


def test_eq_binary_bool_non_named_entity_raises_numeta_type_error():
    expr = EqBinaryNode(1, 2)
    with pytest.raises(NumetaTypeError, match="Do not use '==' operator"):
        bool(expr)


def test_ne_binary_bool_non_named_entity_raises_numeta_type_error():
    expr = NeBinaryNode(1, 2)
    with pytest.raises(NumetaTypeError, match="Do not use '!=' operator"):
        bool(expr)


def test_getitem_merge_slice_step_raises_numeta_not_implemented_error():
    var = Variable("a", dtype=int, shape=(10,))
    expr = GetItem(var, slice(1, 6, None))

    with pytest.raises(NumetaNotImplementedError, match="Step slicing not implemented"):
        expr.merge_slice(slice(1, 3, 2))


def test_getitem_merge_slice_invalid_combination_raises_value_error():
    var = Variable("a", dtype=int, shape=(10,))
    expr = GetItem(var, 2)

    with pytest.raises(ValueError, match="Cannot merge old slice with new one"):
        expr.merge_slice(3)


def test_compiler_run_command_raises_compilation_error(monkeypatch, tmp_path):
    import numeta.compiler as compiler_module

    def fake_run(*args, **kwargs):
        return compiler_module.sp.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="stdout output\n",
            stderr="stderr output\n",
        )

    monkeypatch.setattr(compiler_module.sp, "run", fake_run)

    compiler = Compiler("gcc", "")
    with pytest.raises(CompilationError, match="Error while compiling") as exc_info:
        compiler.run_command(["gcc", "-v"], cwd=tmp_path)

    message = str(exc_info.value)
    assert "gcc -v" in message
    assert "stdout output" in message
    assert "stderr output" in message
    assert exc_info.value.command == ("gcc", "-v")
    assert exc_info.value.cwd == tmp_path
    assert exc_info.value.stdout == "stdout output\n"
    assert exc_info.value.stderr == "stderr output\n"


def test_compiler_reports_missing_toolchain():
    from numeta.exceptions import ToolchainNotFoundError

    with pytest.raises(ToolchainNotFoundError, match="NUMETA_CC") as exc_info:
        Compiler(
            "numeta-compiler-that-does-not-exist",
            "",
            setting="set_c_compiler",
            env_var="NUMETA_CC",
        )

    assert exc_info.value.compiler == "numeta-compiler-that-does-not-exist"


def test_cond_helper_wraps_generated_source_with_except_exception():
    helper = CondHelper()

    frame = types.SimpleNamespace(
        f_code=types.SimpleNamespace(co_filename="phase1_test.py"),
        f_lineno=1,
    )

    original_getsourcelines = inspect.getsourcelines
    inspect.getsourcelines = lambda _code: (["    pass\n"], 1)
    try:
        helper.if_stack.append(1)
        source_lines, *_ = helper.get_source_cache(frame)
    finally:
        inspect.getsourcelines = original_getsourcelines

    rendered_source = "".join(source_lines)

    assert "except Exception:" in rendered_source
    assert "RuntimeError('impossible to parse the code')" in rendered_source
