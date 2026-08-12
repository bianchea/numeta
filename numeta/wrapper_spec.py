"""Typed metadata for generated CPython wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class ShapeEquality:
    """Require two runtime array arguments to have identical shapes."""

    left: str
    right: str


@dataclass(frozen=True)
class WrapperSpec:
    """Complete ABI and validation metadata for one generated wrapper."""

    name: str
    arguments: tuple
    returns: tuple
    shape_equalities: tuple[ShapeEquality, ...] = ()

    @classmethod
    def create(
        cls,
        name,
        arguments,
        returns,
        shape_equalities: Iterable[ShapeEquality | tuple[str, str]] = (),
    ) -> "WrapperSpec":
        equalities = tuple(
            equality if isinstance(equality, ShapeEquality) else ShapeEquality(*equality)
            for equality in shape_equalities
        )
        return cls(name, tuple(arguments), tuple(returns), equalities)

    @classmethod
    def coerce(cls, value) -> "WrapperSpec":
        """Normalize wrapper metadata from current or pre-dataclass callers."""
        if isinstance(value, cls):
            return value
        if len(value) == 3:
            name, arguments, returns = value
            equalities = ()
        elif len(value) == 4:
            name, arguments, returns, equalities = value
        else:
            raise TypeError("wrapper specification must contain three or four fields")
        return cls.create(name, arguments, returns, equalities)
