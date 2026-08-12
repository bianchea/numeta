"""Public inspection and ahead-of-time compilation for one JIT signature."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .library_signature import signature_id

if TYPE_CHECKING:
    from .numeta_function import NumetaFunction


def _existing_path(value) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.exists() else None


class NumetaSpecialization:
    """Read-only view of one specialized Numeta function signature."""

    __slots__ = ("_function", "_signature")

    def __init__(self, function: "NumetaFunction", signature) -> None:
        self._function = function
        self._signature = signature

    @property
    def function(self) -> "NumetaFunction":
        return self._function

    @property
    def signature(self):
        return self._signature

    @property
    def signature_id(self) -> str:
        return signature_id(self._signature)

    @property
    def _target(self):
        try:
            return self._function._compiled_functions[self._signature]
        except KeyError as exc:
            raise RuntimeError(
                f"Specialization {self.signature_id} is no longer registered on "
                f"function {self._function.name!r}"
            ) from exc

    @property
    def symbol(self) -> str:
        return self._target.func_name

    @property
    def backend(self) -> str:
        return self._target.backend

    @property
    def compile_flags(self) -> tuple[str, ...]:
        return tuple(self._target.compile_flags)

    @property
    def return_signature(self) -> tuple:
        return tuple(self._function.return_signatures.get(self._signature, ()))

    @property
    def dependency_symbols(self) -> tuple[str, ...]:
        dependencies = self._target.symbolic_function.get_dependencies().values()
        return tuple(
            dict.fromkeys(
                dependency.func_name
                for dependency in dependencies
                if hasattr(dependency, "func_name")
            )
        )

    @property
    def source(self) -> str:
        return self._target.render_source()

    @property
    def source_path(self) -> Path | None:
        return self._target.source_path

    @property
    def object_path(self) -> Path | None:
        return _existing_path(self._target._obj_files)

    @property
    def library_path(self) -> Path | None:
        return self._target.library_path

    @property
    def wrapper_path(self) -> Path | None:
        wrapper = self._function._existing_pyc_extension(self._signature)
        return _existing_path(wrapper.lib_path) if wrapper is not None else None

    @property
    def compiled(self) -> bool:
        return self.library_path is not None and self.wrapper_path is not None

    @property
    def loaded(self) -> bool:
        return self._signature in self._function._fast_call

    def compile(self) -> "NumetaSpecialization":
        """Compile native and CPython wrapper libraries without executing the kernel."""
        self._function._compile_signature(self._signature)
        return self

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(function={self._function.name!r}, "
            f"signature_id={self.signature_id!r}, backend={self.backend!r}, "
            f"compiled={self.compiled!r})"
        )
