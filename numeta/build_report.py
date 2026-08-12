"""Structured results returned by batch ahead-of-time builds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .specialization import NumetaSpecialization


@dataclass(frozen=True)
class NumetaBuildReport:
    """Immutable summary of a completed :class:`NumetaLibrary` build."""

    library_name: str
    bundle_path: Path
    core_library_path: Path
    wrapper_path: Path
    specializations: tuple[NumetaSpecialization, ...]
    created_specializations: tuple[NumetaSpecialization, ...]
    reused_specializations: tuple[NumetaSpecialization, ...]
    elapsed_s: float

    @property
    def function_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(spec.function.name for spec in self.specializations))

    @property
    def specialization_count(self) -> int:
        return len(self.specializations)

    @property
    def created_count(self) -> int:
        return len(self.created_specializations)

    @property
    def reused_count(self) -> int:
        return len(self.reused_specializations)

    def for_function(self, name: str) -> tuple[NumetaSpecialization, ...]:
        """Return requested specializations for one registered function."""
        matches = tuple(spec for spec in self.specializations if spec.function.name == name)
        if not matches:
            raise KeyError(f"Build report has no requested specializations for {name!r}")
        return matches
