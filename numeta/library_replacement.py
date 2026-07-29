"""Validation and state-transfer helpers for library replacement."""

from .ast.namespace import Namespace
from .numeta_function import NumetaCompiledFunction, NumetaFunction


def validate_function_compatibility(
    old_func: NumetaFunction,
    new_func: NumetaFunction,
) -> None:
    if old_func.backend != new_func.backend:
        raise ValueError("Cannot replace function with different backend")
    if tuple(old_func.compile_flags) != tuple(new_func.compile_flags):
        raise ValueError(
            "Cannot replace function with different compile_flags in minimal incremental mode"
        )
    if old_func.do_checks != new_func.do_checks:
        raise ValueError(
            "Cannot replace function with different do_checks in minimal incremental mode"
        )
    if old_func.inline or new_func.inline:
        raise ValueError("replace() does not support inline functions yet")


def validate_specialization_compatibility(
    old_func: NumetaFunction,
    new_func: NumetaFunction,
    signature,
) -> None:
    old_spec = old_func._wrapper_specs.get(signature)
    if old_spec is None:
        old_spec = old_func.build_wrapper_spec(signature)
    new_spec = new_func._wrapper_specs.get(signature)
    if new_spec is None:
        new_spec = new_func.build_wrapper_spec(signature)

    old_name, old_args, old_returns = old_spec
    new_name, new_args, new_returns = new_spec
    if old_name != new_name:
        raise AssertionError("replacement did not preserve compiled symbol name")
    if old_args != new_args:
        raise ValueError(
            f"Replacement for {old_func.name!r} changed argument ABI for signature {signature!r}"
        )
    if old_returns != new_returns:
        raise ValueError(
            f"Replacement for {old_func.name!r} changed return ABI for signature {signature!r}"
        )


def single_global_variable(global_target: NumetaCompiledFunction):
    namespace = global_target.symbolic_function
    if not isinstance(namespace, Namespace):
        raise ValueError("Global target must be backed by a namespace")
    if len(namespace.variables) != 1:
        raise ValueError(f"Global namespace {namespace.name!r} must contain exactly one variable")
    return next(iter(namespace.variables.values()))


def validate_global_compatibility(
    old_target: NumetaCompiledFunction,
    new_target: NumetaCompiledFunction,
    *,
    allow_shape_change: bool,
) -> None:
    old_var = single_global_variable(old_target)
    new_var = single_global_variable(new_target)

    if old_target.func_name != new_target.func_name:
        raise ValueError(
            f"Replacement changed global namespace symbol from {old_target.func_name!r} "
            f"to {new_target.func_name!r}"
        )
    if old_var.name != new_var.name:
        raise ValueError(
            f"Replacement changed global variable name from {old_var.name!r} "
            f"to {new_var.name!r}"
        )
    if old_target.backend != new_target.backend:
        raise ValueError("Cannot replace global constant with different backend")
    if old_var.dtype is not new_var.dtype:
        raise ValueError(
            f"Replacement for global constant {old_var.name!r} changed dtype from "
            f"{old_var.dtype!r} to {new_var.dtype!r}"
        )

    old_shape = old_var._shape
    new_shape = new_var._shape
    if old_shape.fortran_order != new_shape.fortran_order:
        raise ValueError(f"Replacement for global constant {old_var.name!r} changed array order")
    if old_shape.rank != new_shape.rank:
        raise ValueError(
            f"Replacement for global constant {old_var.name!r} changed rank from "
            f"{old_shape.rank} to {new_shape.rank}"
        )
    if not allow_shape_change and old_shape.as_tuple() != new_shape.as_tuple():
        raise ValueError(
            f"Replacement for global constant {old_var.name!r} changed shape from "
            f"{old_shape.as_tuple()!r} to {new_shape.as_tuple()!r}. "
            "Pass allow_shape_change=True only if dependent generated code is known "
            "to remain compatible."
        )


def adopt_compiled_target_state(
    target: NumetaCompiledFunction,
    replacement: NumetaCompiledFunction,
) -> None:
    """Transfer compiled state while preserving dependency object identity."""
    target.adopt_compiled_state(replacement)
