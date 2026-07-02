import warnings

from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    overload,
)

from .numeta_function import NumetaFunction
from .numeta_library import NumetaLibrary
from .settings import settings


@overload
def jit(func):
    """@jit used with no arguments."""
    ...


@overload
def jit(
    *,
    directory: Optional[str] = None,
    do_checks: bool | None = None,
    compile_flags: str | Iterable[str] | None = None,
    namer: Optional[Callable[..., str]] = None,
    name: str | None = None,
    inline: bool | int = False,
    library: NumetaLibrary | None = None,
    reattach: bool = False,
    backend: str | None = None,
    simd_arch: str | None = None,
    simd_features: Iterable[str] | str | None = None,
    c_attributes: Iterable[str] | str | None = None,
    c_linkage: str | None = None,
    emit_mode: str | None = None,
):
    """@jit(...) used with arguments."""
    ...


def jit(
    func: Callable[..., Any] | None = None,
    *,
    directory: Optional[str] = None,
    do_checks: bool | None = None,
    compile_flags: str | Iterable[str] | None = None,
    namer: Optional[Callable[..., str]] = None,
    name: str | None = None,
    inline: bool | int = False,
    library: NumetaLibrary | None = None,
    reattach: bool = False,
    backend: str | None = None,
    simd_arch: str | None = None,
    simd_features: Iterable[str] | str | None = None,
    c_attributes: Iterable[str] | str | None = None,
    c_linkage: str | None = None,
    emit_mode: str | None = None,
):
    """
    Compile a function with the Numeta JIT, either directly or via parameters.

    Overload Resolution
    -------------------
    1.  **No-arg form**: `@jit`
        - Returns a `NumetaFunction` wrapping the target.
    2.  **With-arg form**: `@jit(directory=..., inline=2, ...)`
        - Returns a decorator that, when applied, produces a `NumetaFunction`.

    Parameters
    ----------
    func
        The function to compile when using `@jit` with no args.
    directory
        Target directory for compiled output (default: none → temp dir).
    do_checks
        Whether to enable compile-time argument validation. If None, uses settings default.
    compile_flags
        Flags for the compiler optimization step. If None, uses settings default.
    namer
        Optional callable to name the JIT-generated symbols.
    inline
        Controls inlining behavior (bool or max-stmts int).
    library
        Optional library container used to group jitted functions.
    reattach
        Allow rebinding source code onto a loaded library entry so it can compile
        new signatures in the current process.
    backend
        Backend to use for code generation ("fortran" or "c"). If None, uses settings default.

    Returns
    -------
    NumetaFunction
    """
    if backend is None:
        backend = settings.default_backend
    if do_checks is None:
        do_checks = settings.default_do_checks
    if compile_flags is None:
        compile_flags = settings.default_compile_flags
    if simd_arch is None:
        simd_arch = settings.default_simd_arch
    if simd_features is None:
        simd_features = settings.default_simd_features
    elif isinstance(simd_features, str):
        simd_features = (simd_features,)
    simd_features_tuple = tuple(str(feature).lower() for feature in simd_features)
    if isinstance(c_attributes, str):
        c_attributes_tuple = (c_attributes,)
    else:
        c_attributes_tuple = tuple(c_attributes or ())
    compile_flags = settings._normalize_compile_flags(compile_flags)
    compile_flags_list = list(compile_flags)
    if func is None:

        def decorator_wrapper(f) -> NumetaFunction:
            python_name = f.__name__
            if python_name.startswith("_nm"):
                raise ValueError("Cannot create functions that startwith '_nm'")
            if library is not None and python_name in library:
                nm_func = library[python_name]
                if nm_func._func is None:
                    if not reattach:
                        raise ValueError(
                            f"Function '{python_name}' was loaded from a NumetaLibrary without source code. "
                            "Pass reattach=True to @nm.jit(...) to rebind the Python body "
                            "and allow compiling new signatures."
                        )
                    nm_func._func = f
                    if namer is not None:
                        nm_func.namer = namer
                if nm_func.do_checks != do_checks:
                    warnings.warn(
                        f"function {python_name} has been loaded with different do_checks value: {nm_func.do_checks}",
                        stacklevel=2,
                    )
                if tuple(nm_func.compile_flags) != tuple(compile_flags_list):
                    warnings.warn(
                        f"function {python_name} has been loaded with different compile_flags value: {nm_func.compile_flags}",
                        stacklevel=2,
                    )
                if getattr(nm_func, "simd_arch", settings.default_simd_arch) != simd_arch:
                    warnings.warn(
                        f"function {python_name} has been loaded with different simd_arch value: {getattr(nm_func, 'simd_arch', settings.default_simd_arch)}",
                        stacklevel=2,
                    )
            else:
                nm_func = NumetaFunction(
                    f,
                    directory=directory,
                    do_checks=do_checks,
                    compile_flags=compile_flags_list,
                    namer=namer,
                    inline=inline,
                    backend=backend,
                    simd_arch=simd_arch,
                    simd_features=simd_features_tuple,
                    c_name=name,
                    c_attributes=c_attributes_tuple,
                    c_linkage=c_linkage,
                    emit_mode=emit_mode,
                )
                if library is not None:
                    library.register(nm_func)
            return nm_func

        return decorator_wrapper
    else:
        nm_func = NumetaFunction(
            func,
            directory=directory,
            do_checks=do_checks,
            compile_flags=compile_flags_list,
            namer=namer,
            inline=inline,
            backend=backend,
            simd_arch=simd_arch,
            simd_features=simd_features_tuple,
            c_name=name,
            c_attributes=c_attributes_tuple,
            c_linkage=c_linkage,
            emit_mode=emit_mode,
        )
        return nm_func


def c_function_attributes(*attributes: str, **options):
    """Attach C function metadata to a Python function before applying ``@jit``."""

    attrs = list(attributes)
    for key, value in options.items():
        if value is None or value is False:
            continue
        if key == "aligned":
            attrs.append(f"aligned({int(value)})")
        elif key == "visibility":
            attrs.append(f'visibility("{value}")')
        elif key == "section":
            attrs.append(f'section("{value}")')
        elif key == "inline":
            if value in {"always", "force", True}:
                attrs.append("always_inline")
            elif value in {"never", "no", False}:
                attrs.append("noinline")
        elif value is True:
            attrs.append(key)
        else:
            attrs.append(f"{key}({value})")

    def decorator(func):
        if hasattr(func, "c_attributes"):
            func.c_attributes = tuple(getattr(func, "c_attributes", ())) + tuple(attrs)
            for compiled in getattr(func, "_compiled_functions", {}).values():
                compiled.c_attributes = tuple(getattr(compiled, "c_attributes", ())) + tuple(attrs)
                compiled.symbolic_function.c_attributes = tuple(
                    getattr(compiled.symbolic_function, "c_attributes", ())
                ) + tuple(attrs)
            return func
        existing = tuple(getattr(func, "_numeta_c_attributes", ()))
        setattr(func, "_numeta_c_attributes", existing + tuple(attrs))
        return func

    return decorator
