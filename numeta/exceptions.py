class NumetaError(Exception):
    """Base exception for all numeta errors."""

    pass


class NumetaPerformanceWarning(UserWarning):
    """A valid trace may be expensive to generate or compile."""


class CompilationError(NumetaError):
    """Error during compilation."""

    def __init__(
        self,
        message=None,
        *,
        command=None,
        cwd=None,
        stdout="",
        stderr="",
    ):
        self.command = tuple(command or ())
        self.cwd = cwd
        self.stdout = stdout
        self.stderr = stderr

        if message is None:
            import shlex

            rendered_command = shlex.join(self.command)
            message = f"Error while compiling in {cwd}:\n  {rendered_command}"
            if stdout:
                message += f"\nCompiler stdout:\n{stdout.rstrip()}"
            if stderr:
                message += f"\nCompiler stderr:\n{stderr.rstrip()}"
        super().__init__(message)


class ToolchainNotFoundError(NumetaError):
    """A configured native compiler cannot be found or executed."""

    def __init__(self, compiler, *, setting=None, env_var=None):
        self.compiler = compiler
        self.setting = setting
        self.env_var = env_var
        guidance = []
        if setting:
            guidance.append(f"settings.{setting}(...)")
        if env_var:
            guidance.append(env_var)
        suffix = f" Configure it with {' or '.join(guidance)}." if guidance else ""
        super().__init__(
            f"Native compiler {compiler!r} was not found or is not executable.{suffix}"
        )


class LibraryFormatError(NumetaError):
    """Base error for persisted Numeta library bundles."""


class CorruptLibraryError(LibraryFormatError):
    """A persisted library bundle is incomplete or has invalid checksums."""


class IncompatibleLibraryError(LibraryFormatError):
    """A native library bundle is incompatible with the current runtime."""


class LegacyLibraryFormatError(LibraryFormatError):
    """A legacy pickle-based Numeta library was found."""


class NumetaTypeError(NumetaError):
    """Type error in numeta code."""

    pass


class NumetaNotImplementedError(NumetaError):
    """Feature not yet implemented."""

    pass


def format_source_location(node):
    """Format source location info from a node for error messages."""
    if node is None:
        return None

    loc = getattr(node, "source_location", None)
    if loc is None:
        return None

    filename = loc.get("filename", "<unknown>")
    lineno = loc.get("lineno", 0)

    import linecache

    source_line = linecache.getline(filename, lineno).rstrip()
    if source_line:
        return f'  File "{filename}", line {lineno}\n    {source_line}'

    return f'  File "{filename}", line {lineno}'


def raise_with_source(
    exception_class,
    message,
    source_node=None,
    *,
    expected=None,
    received=None,
    repair=None,
    use_site=False,
):
    """Raise an exception with source location information.

    Args:
        exception_class: The exception class to raise (e.g., NotImplementedError)
        message: The error message
        source_node: The AST/IR node that caused the error (should have source_location)
    """
    if expected is not None:
        message += f"\nExpected: {expected}."
    if received is not None:
        message += f"\nReceived: {received}."
    if repair is not None:
        message += f"\nRepair: {repair}"
    if use_site:
        from types import SimpleNamespace
        from numeta.ast.nodes.base_node import capture_source_location

        source_node = SimpleNamespace(source_location=capture_source_location())
    loc_info = format_source_location(source_node)
    if loc_info:
        full_message = f"{message}\n\n{loc_info}"
    else:
        full_message = message

    raise exception_class(full_message)


def describe_value(value):
    """Describe symbolic values without invoking guarded Python protocols."""
    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "_shape", None)
    name = getattr(dtype, "_name", "untyped")
    if shape is None:
        return name + " value"
    if shape.is_scalar:
        return name + " scalar (rank 0)"
    return name + f" array (rank {shape.rank if not shape.is_unknown else 'unknown'})"
