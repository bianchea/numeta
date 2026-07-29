from dataclasses import dataclass
from typing import Any, Iterator, Optional, Tuple, cast


@dataclass(frozen=True)
class ArrayShape:
    # None      ⇒ unknown shape
    # ()        ⇒ scalar
    # (3,4)     ⇒ fixed 2-D So it is know at compile time
    # (None, 4) ⇒ 2-D with an undefined dimension at compile time
    _shape: Optional[Tuple | Any]
    fortran_order: bool = False

    @classmethod
    def from_shape_vector(cls, vector: object, rank: int, fortran_order: bool = False):
        vector_shape = getattr(vector, "_shape", None)
        if isinstance(vector_shape, ArrayShape) and not vector_shape.is_unknown:
            if vector_shape.rank != 1:
                raise ValueError("Shape vector must be rank-1.")
            vector_len = vector_shape.dim(0)
            if isinstance(vector_len, int) and vector_len != rank:
                raise ValueError("Shape vector length does not match provided rank.")
        return cls(vector, fortran_order=fortran_order)

    @property
    def shape(self):
        return self._shape

    @property
    def is_unknown(self) -> bool:
        return self._shape is None

    @property
    def is_scalar(self) -> bool:
        return isinstance(self._shape, tuple) and len(self._shape) == 0

    @property
    def is_shape_vector(self) -> bool:
        return self._shape is not None and not isinstance(self._shape, tuple)

    def _shape_vector_rank(self) -> int:
        if self._shape is None or isinstance(self._shape, tuple):
            raise ValueError("ArrayShape does not hold a shape vector.")
        vector_shape = getattr(self._shape, "_shape", None)
        if not isinstance(vector_shape, ArrayShape):
            raise ValueError("Shape vector must have an ArrayShape.")
        if vector_shape.is_unknown:
            raise ValueError("Shape vector rank is unknown.")
        if vector_shape.rank != 1:
            raise ValueError("Shape vector must be rank-1.")
        length = vector_shape.dim(0)
        if not isinstance(length, int):
            raise ValueError("Shape vector length must be known at compile time.")
        return length

    def dim(self, index: int):
        if self._shape is None:
            raise ValueError("Cannot access dimensions for unknown shape.")
        if isinstance(self._shape, tuple):
            return self._shape[index]
        return cast(Any, self._shape)[index]

    def iter_dims(self) -> Iterator:
        for i in range(self.rank):
            yield self.dim(i)

    def as_tuple(self) -> Tuple:
        return tuple(self.iter_dims())

    def __getattr__(self, name: str):
        if name == "dims":
            return self.as_tuple()
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    def __repr__(self) -> str:
        if self._shape is None:
            return "ArrayShape<unknown>"
        elif self.is_scalar:
            return "ArrayShape<scalar>"
        else:
            inner = ", ".join(map(str, self.as_tuple()))
            return f"ArrayShape[{inner}]"

    @property
    def rank(self) -> int:
        """
        Returns the rank of the array shape.
        """
        if self._shape is None:
            raise ValueError("Cannot get rank for unknown shape.")
        if isinstance(self._shape, tuple):
            return len(self._shape)
        return self._shape_vector_rank()

    @property
    def comptime_undefined_dims(self):
        """
        Returns the indices of the dimensions that are undefined at compile time.
        """
        return [i for i, dim in enumerate(self.iter_dims()) if not self._is_comptime_dim(dim)]

    def has_comptime_undefined_dims(self):
        """
        Checks if the argument has undefined dimensions at compile time.
        """
        if self._shape is None:
            # The dimensions are unknown
            return False
        if self.is_shape_vector:
            return True
        for dim in self.iter_dims():
            if not self._is_comptime_dim(dim):
                return True
        return False

    @staticmethod
    def _is_comptime_dim(dim):
        if isinstance(dim, int):
            return True

        # Shape dimensions are often normalized to LiteralNode instances.
        try:
            from numeta.ast.expressions import LiteralNode

            if isinstance(dim, LiteralNode):
                return isinstance(dim.value, int)
        except Exception:
            pass

        return False


# sentinels
UNKNOWN = ArrayShape(None)
SCALAR = ArrayShape(())
