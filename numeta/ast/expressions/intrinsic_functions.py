from numeta.ast.tools import check_node
from numeta.array_shape import ArrayShape, SCALAR
from numeta.exceptions import raise_with_source
from .expression_node import ExpressionNode


class IntrinsicFunction(ExpressionNode):
    token = ""

    def __init__(self, *arguments):
        super().__init__()
        self.arguments = [check_node(arg) for arg in arguments]
        self._shape_cache = None

    def extract_entities(self):
        for arg in self.arguments:
            yield from arg.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        new_args = [arg.get_with_updated_variables(variables_couples) for arg in self.arguments]
        return type(self)(*new_args)

    @property
    def dtype(self):
        # default behavior for a lot of intrinsic functions
        return self.arguments[0].dtype

    @property
    def _shape(self):
        # default behavior for a lot of intrinsic functions
        cached_shape = self._shape_cache
        if cached_shape is not None:
            return cached_shape
        shape = self.arguments[0]._shape
        self._shape_cache = shape
        return shape


class UnaryIntrinsicFunction(IntrinsicFunction):

    def __init__(self, argument):
        super().__init__(check_node(argument))


class BinaryIntrinsicFunction(IntrinsicFunction):

    def __init__(self, argument1, argument2):
        super().__init__(
            check_node(argument1),
            check_node(argument2),
        )


class MathIntrinsic(IntrinsicFunction):
    """
    Base class for mathematical intrinsic functions.
    If the argument is integer, it promotes the return type to real (float).
    If the argument is complex, it returns complex.
    If the argument is real, it returns real.
    """

    @property
    def dtype(self):
        from numeta.datatype import int32, int64
        from numeta.settings import settings

        # We look at the first argument to determine the output type base
        arg_dtype = self.arguments[0].dtype

        if arg_dtype in (int32, int64):
            return settings.syntax.DEFAULT_FLOAT

        # For real or complex, return the same type
        return arg_dtype


class UnaryMathIntrinsic(MathIntrinsic, UnaryIntrinsicFunction):
    pass


class Abs(UnaryIntrinsicFunction):
    token = "abs"

    @property
    def dtype(self):
        from numeta.datatype import float32, float64, float128, complex64, complex128, complex256

        arg_dtype = self.arguments[0].dtype

        if arg_dtype == complex64:
            return float32
        elif arg_dtype == complex128:
            return float64
        elif arg_dtype == complex256:
            return float128

        return arg_dtype


class Neg(UnaryIntrinsicFunction):
    token = "-"


class Not(UnaryIntrinsicFunction):
    token = ".not."


class Allocated(UnaryIntrinsicFunction):
    token = "allocated"

    @property
    def dtype(self):
        from numeta.datatype import bool8

        return bool8

    @property
    def _shape(self):
        return SCALAR


class All(UnaryIntrinsicFunction):
    token = "all"

    @property
    def dtype(self):
        from numeta.datatype import bool8

        return bool8

    @property
    def _shape(self):
        return SCALAR


class Shape(UnaryIntrinsicFunction):
    token = "shape"

    def __init__(self, argument):
        if argument._shape.is_scalar:
            raise_with_source(
                ValueError,
                "The shape intrinsic function cannot be applied to a scalar.",
                source_node=argument,
            )
        super().__init__(argument)

    @property
    def dtype(self):
        from numeta.settings import settings

        return settings.syntax.DEFAULT_INTEGER

    @property
    def _shape(self):
        var_shape = self.arguments[0]._shape
        if var_shape.is_scalar or var_shape.is_unknown:
            raise_with_source(
                ValueError,
                "The shape intrinsic function can only be applied to variables with a defined shape.",
                source_node=self.arguments[0],
            )
        return ArrayShape((var_shape.rank,))


class Real(UnaryIntrinsicFunction):
    token = "real"

    @property
    def dtype(self):
        from numeta.settings import settings

        return settings.syntax.DEFAULT_FLOAT


class Imag(UnaryIntrinsicFunction):
    token = "aimag"

    @property
    def dtype(self):
        from numeta.settings import settings

        return settings.syntax.DEFAULT_FLOAT


class Conjugate(UnaryIntrinsicFunction):
    token = "conjg"

    @property
    def dtype(self):
        from numeta.settings import settings

        return settings.syntax.DEFAULT_COMPLEX

    @property
    def _shape(self):
        return self.arguments[0]._shape


class Complex(IntrinsicFunction):
    token = "cmplx"

    def __init__(self, real, imaginary, kind=None):
        from numeta.settings import settings

        if kind is None:
            kind = settings.syntax.DEFAULT_COMPLEX.get_fortran().kind
        super().__init__(real, imaginary, kind)

    @property
    def dtype(self):
        from numeta.settings import settings

        return settings.syntax.DEFAULT_COMPLEX

    @property
    def _shape(self):
        return self.arguments[0]._shape


class Transpose(UnaryIntrinsicFunction):
    token = "transpose"

    @property
    def _shape(self):
        arg_shape = self.arguments[0]._shape
        if arg_shape.is_scalar:
            raise_with_source(
                ValueError,
                "Cannot transpose a scalar.",
                source_node=self.arguments[0],
            )
        if arg_shape.is_unknown:
            raise_with_source(
                ValueError,
                "Cannot transpose a variable with unknown shape.",
                source_node=self.arguments[0],
            )
        if arg_shape.rank != 2:
            raise_with_source(
                ValueError,
                "Transpose can only be applied to 2-D arrays.",
                source_node=self.arguments[0],
            )
        return ArrayShape(
            tuple(reversed(arg_shape.as_tuple())), fortran_order=arg_shape.fortran_order
        )

    @property
    def shape(self):
        if self.arguments[0]._shape.is_scalar:
            raise_with_source(
                ValueError,
                "Cannot transpose a scalar.",
                source_node=self.arguments[0],
            )
        elif self.arguments[0]._shape.is_unknown:
            raise_with_source(
                ValueError,
                "Cannot transpose a variable with unknown shape.",
                source_node=self.arguments[0],
            )
        elif self.arguments[0]._shape.rank != 2:
            raise_with_source(
                ValueError,
                "Transpose can only be applied to 2-D arrays.",
                source_node=self.arguments[0],
            )
        return ArrayShape(tuple(reversed(self.arguments[0]._shape.as_tuple())))


class Exp(UnaryMathIntrinsic):
    token = "exp"


class Log(UnaryMathIntrinsic):
    token = "log"


class Log10(UnaryMathIntrinsic):
    token = "log10"


class Sqrt(UnaryMathIntrinsic):
    token = "sqrt"


class Floor(UnaryMathIntrinsic):
    token = "floor"


class Ceil(UnaryMathIntrinsic):
    token = "ceil"


class Sin(UnaryMathIntrinsic):
    token = "sin"


class Cos(UnaryMathIntrinsic):
    token = "cos"


class Tan(UnaryMathIntrinsic):
    token = "tan"


class Sinh(UnaryMathIntrinsic):
    token = "sinh"


class Cosh(UnaryMathIntrinsic):
    token = "cosh"


class Tanh(UnaryMathIntrinsic):
    token = "tanh"


class Arcsin(UnaryMathIntrinsic):
    token = "asin"


class Arccos(UnaryMathIntrinsic):
    token = "acos"


class Arctan(UnaryMathIntrinsic):
    token = "atan"


class Arctan2(BinaryIntrinsicFunction, MathIntrinsic):
    token = "atan2"


class Arcsinh(UnaryMathIntrinsic):
    token = "asinh"


class Arccosh(UnaryMathIntrinsic):
    token = "acosh"


class Arctanh(UnaryMathIntrinsic):
    token = "atanh"


class Hypot(BinaryIntrinsicFunction, MathIntrinsic):
    token = "hypot"


class Copysign(BinaryIntrinsicFunction, MathIntrinsic):
    token = "copysign"


class Dotproduct(BinaryIntrinsicFunction):
    token = "dot_product"

    @property
    def dtype(self):
        return self.arguments[0].dtype

    @property
    def _shape(self):
        return SCALAR


class Rank(UnaryIntrinsicFunction):
    token = "rank"

    @property
    def dtype(self):
        from numeta.datatype import int64

        return int64

    @property
    def _shape(self):
        return SCALAR


class Size(BinaryIntrinsicFunction):
    token = "size"

    @property
    def dtype(self):
        from numeta.datatype import int64

        return int64

    @property
    def _shape(self):
        return SCALAR


class Max(IntrinsicFunction):
    token = "max"

    @property
    def dtype(self):
        return self.arguments[0].dtype

    @property
    def _shape(self):
        return SCALAR


class Maxval(UnaryIntrinsicFunction):
    token = "maxval"

    @property
    def dtype(self):
        return self.arguments[0].dtype

    @property
    def _shape(self):
        return SCALAR


class Min(IntrinsicFunction):
    token = "min"

    @property
    def dtype(self):
        return self.arguments[0].dtype

    @property
    def _shape(self):
        return SCALAR


class Minval(UnaryIntrinsicFunction):
    token = "minval"

    @property
    def dtype(self):
        return self.arguments[0].dtype

    @property
    def _shape(self):
        return SCALAR


class Iand(BinaryIntrinsicFunction):
    token = "iand"


class Ior(BinaryIntrinsicFunction):
    token = "ior"


class Xor(BinaryIntrinsicFunction):
    token = "xor"


class Ishft(BinaryIntrinsicFunction):
    token = "ishft"


class Ibset(BinaryIntrinsicFunction):
    token = "ibset"


class Ibclr(BinaryIntrinsicFunction):
    token = "ibclr"


class Popcnt(UnaryIntrinsicFunction):
    token = "popcnt"

    @property
    def dtype(self):
        from numeta.datatype import int64

        return int64

    @property
    def _shape(self):
        return SCALAR


class Trailz(UnaryIntrinsicFunction):
    token = "trailz"

    @property
    def dtype(self):
        from numeta.datatype import int64

        return int64

    @property
    def _shape(self):
        return SCALAR


class Sum(UnaryIntrinsicFunction):
    token = "sum"

    @property
    def dtype(self):
        return self.arguments[0].dtype

    @property
    def _shape(self):
        return SCALAR


class Matmul(BinaryIntrinsicFunction):
    token = "matmul"

    @property
    def dtype(self):
        return self.arguments[0].dtype

    @property
    def _shape(self):
        a_shape = self.arguments[0]._shape
        b_shape = self.arguments[1]._shape
        if a_shape.rank == 1:
            return ArrayShape((b_shape.dim(1),))
        if b_shape.rank == 1:
            return ArrayShape((a_shape.dim(0),))
        return ArrayShape((a_shape.dim(0), b_shape.dim(1)))


# Aliases to match numpy conventions
abs = Abs
negative = Neg
logical_not = Not
allocated = Allocated
all = All
shape = Shape
real = Real
imag = Imag
conjugate = Conjugate
conj = Conjugate
complex = Complex
transpose = Transpose
exp = Exp
log = Log
log10 = Log10
sqrt = Sqrt
floor = Floor
ceil = Ceil
sin = Sin
cos = Cos
tan = Tan
sinh = Sinh
cosh = Cosh
tanh = Tanh
arcsin = Arcsin
arccos = Arccos
arctan = Arctan
arctan2 = Arctan2
arcsinh = Arcsinh
arccosh = Arccosh
arctanh = Arctanh
hypot = Hypot
copysign = Copysign
dot = Dotproduct
ndim = Rank
size = Size
maximum = Max
max = Maxval
minimum = Min
min = Minval
bitwise_and = Iand
bitwise_or = Ior
bitwise_xor = Xor
ishft = Ishft
ibset = Ibset
ibclr = Ibclr
popcnt = Popcnt
trailz = Trailz
sum = Sum
matmul = Matmul
