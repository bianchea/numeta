from abc import abstractmethod

from numeta.ast.nodes import Node
from numeta.exceptions import NumetaTypeError, raise_with_source

BinaryOperationNode = None
EqBinaryNode = None
NeBinaryNode = None
GetItem = None
GetAttr = None
Neg = None
Abs = None
Transpose = None
Re = None
Im = None
Assignment = None


class ExpressionNode(Node):
    __slots__ = []

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def dtype(self):
        """Return the DataType of the expression."""

    @property
    @abstractmethod
    def _shape(self):
        """Return the shape of the expression if any."""

    def _get_shape_descriptor(self):
        from .various import ArrayConstructor

        return ArrayConstructor(*self._shape.as_tuple())

    @abstractmethod
    def extract_entities(self):
        """Extract the nested entities of the expression."""

    def get_with_updated_variables(self, variables_couples):
        return self

    def __bool__(self) -> bool:
        raise_with_source(
            NumetaTypeError,
            "Do not use 'bool' operator for expressions. Python and/or/not, chained "
            "comparisons and ternary expressions cannot trace runtime conditions. Use "
            "parenthesized comparisons with &, |, ~, nm.minimum/nm.maximum, or with nm.If/Else.",
            source_node=self,
        )

    def _reject_python_protocol(self, message):
        raise_with_source(NumetaTypeError, message, source_node=self)

    def __iter__(self):
        self._reject_python_protocol(
            "Symbolic expressions cannot be iterated by Python. Use nm.range(...) for "
            "runtime loops; enumerate(symbolic) is not supported."
        )

    def __index__(self):
        self._reject_python_protocol(
            "A symbolic value is not a Python integer. Use nm.range(...) instead of "
            "Python range(...) for runtime loop bounds."
        )

    def __float__(self):
        self._reject_python_protocol(
            "A symbolic value cannot be converted by Python float() or math.*. Use "
            "Numeta's numeric intrinsics instead."
        )

    def __str__(self):
        self._reject_python_protocol(
            "Python print() cannot display a runtime symbolic value. Use nm.Print(...) instead."
        )

    def __format__(self, format_spec):
        self._reject_python_protocol(
            "Python formatting and f-strings cannot format runtime symbolic values. Use "
            "nm.Print(...) instead."
        )

    def __array__(self, dtype=None, copy=None):
        self._reject_python_protocol(
            "NumPy cannot materialize a symbolic expression during tracing. Use nm.empty(...) "
            "for storage and a sliced assignment instead of np.zeros/np.asarray."
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self._reject_python_protocol(
            f"NumPy ufunc {ufunc.__name__!r} is not supported on symbolic values. Use the "
            "corresponding Numeta intrinsic."
        )

    def __array_function__(self, func, types, args, kwargs):
        self._reject_python_protocol(
            f"NumPy function {func.__name__!r} is not supported on symbolic values. Use "
            "Numeta storage constructors and intrinsics."
        )

    def __rshift__(self, other):
        return BinaryOperationNode(self, ">>", other)

    def __rrshift__(self, other):
        return BinaryOperationNode(other, ">>", self)

    def __lshift__(self, other):
        return BinaryOperationNode(self, "<<", other)

    def __rlshift__(self, other):
        return BinaryOperationNode(other, "<<", self)

    def __neg__(self):
        return Neg(self)

    def __abs__(self):
        return Abs(self)

    def __round__(self, ndigits=None):
        from .intrinsic_functions import Round

        return Round(self, ndigits)

    def __add__(self, other):
        return BinaryOperationNode(self, "+", other)

    def __radd__(self, other):
        return BinaryOperationNode(other, "+", self)

    def __sub__(self, other):
        return BinaryOperationNode(self, "-", other)

    def __rsub__(self, other):
        return BinaryOperationNode(other, "-", self)

    def __mul__(self, other):
        return BinaryOperationNode(self, "*", other)

    def __rmul__(self, other):
        return BinaryOperationNode(other, "*", self)

    def __truediv__(self, other):
        return BinaryOperationNode(self, "/", other)

    def __rtruediv__(self, other):
        return BinaryOperationNode(other, "/", self)

    def __floordiv__(self, other):
        return BinaryOperationNode(self, "//", other)

    def __rfloordiv__(self, other):
        return BinaryOperationNode(other, "//", self)

    def __mod__(self, other):
        return BinaryOperationNode(self, "%", other)

    def __rmod__(self, other):
        return BinaryOperationNode(other, "%", self)

    def __pow__(self, other):
        return BinaryOperationNode(self, "**", other)

    def __rpow__(self, other):
        return BinaryOperationNode(other, "**", self)

    def __and__(self, other):
        from numeta.datatype import bool8

        return BinaryOperationNode(self, ".and." if self.dtype is bool8 else "bitand", other)

    def __rand__(self, other):
        from numeta.datatype import bool8

        return BinaryOperationNode(other, ".and." if self.dtype is bool8 else "bitand", self)

    def __or__(self, other):
        from numeta.datatype import bool8

        return BinaryOperationNode(self, ".or." if self.dtype is bool8 else "bitor", other)

    def __ror__(self, other):
        from numeta.datatype import bool8

        return BinaryOperationNode(other, ".or." if self.dtype is bool8 else "bitor", self)

    def __xor__(self, other):
        return BinaryOperationNode(self, "^", other)

    def __rxor__(self, other):
        return BinaryOperationNode(other, "^", self)

    def __invert__(self):
        from numeta.datatype import bool8
        from .intrinsic_functions import Not

        return Not(self) if self.dtype is bool8 else BinaryOperationNode(self, "^", -1)

    def __ne__(self, other):
        return NeBinaryNode(self, other)

    def __eq__(self, other):
        return EqBinaryNode(self, other)

    def __ge__(self, other):
        return BinaryOperationNode(self, ".ge.", other)

    def __gt__(self, other):
        return BinaryOperationNode(self, ".gt.", other)

    def __le__(self, other):
        return BinaryOperationNode(self, ".le.", other)

    def __lt__(self, other):
        return BinaryOperationNode(self, ".lt.", other)

    @property
    def real(self):
        return Re(self)

    @property
    def imag(self):
        return Im(self)

    @property
    def T(self):
        return Transpose(self)

    def __getitem__(self, key):
        if isinstance(key, slice) and key.start is None and key.stop is None and key.step is None:
            from .whole_storage import WholeStorage

            return WholeStorage(self)
        if isinstance(key, str):
            return GetAttr(self, key)
        if self._shape.is_scalar:
            raise_with_source(
                NumetaTypeError,
                "Scalar storage cannot be indexed. Use scalar[:] to read or write the whole scalar.",
                source_node=self,
            )
        return GetItem(self, key)


def _bind_expression_classes():
    global BinaryOperationNode, EqBinaryNode, NeBinaryNode
    global GetItem, GetAttr, Neg, Abs, Transpose, Re, Im

    from .binary_operation_node import BinaryOperationNode, EqBinaryNode, NeBinaryNode
    from .getattr import GetAttr
    from .getitem import GetItem
    from .intrinsic_functions import Abs, Neg, Transpose
    from .various import Im, Re


_bind_expression_classes()
