from .datatype import (
    DataType,
    StructType,
    int32,
    int64,
    float32,
    float64,
    float128,
    complex64,
    complex128,
    complex256,
    bool8,
    char,
    size_t,
    c_ptr,
    Vector,
    VectorType,
    ArrayType,
    PointerType,
    ptr,
    make_vector_type,
    get_datatype,
)
from .types_hint import comptime

integer4 = int32
integer8 = int64
i4 = int32
i8 = int64

real4 = float32
real8 = float64
real16 = float128
f4 = float32
f8 = float64
f16 = float128
r4 = float32
r8 = float64
r16 = float128

complex8 = complex64
complex16 = complex128
complex32 = complex256
c8 = complex64
c16 = complex128
c32 = complex256

logical1 = bool8
b1 = bool8

from .fortran.external_modules import iso_c, omp

from .jit import c_function_attributes, jit
from .c.emitter import CTranslationUnit
from .numeta_library import NumetaLibrary
from .build_report import NumetaBuildReport
from .specialization import NumetaSpecialization
from .exceptions import (
    CompilationError,
    CorruptLibraryError,
    IncompatibleLibraryError,
    LegacyLibraryFormatError,
    LibraryFormatError,
    NumetaError,
    NumetaNotImplementedError,
    NumetaPerformanceWarning,
    NumetaTypeError,
    ToolchainNotFoundError,
)
from .settings import settings
from ._version import __version__
import math as _math

pi = _math.pi
e = _math.e
inf = _math.inf
nan = _math.nan

from . import ast as _ast
from . import wrappers as _wrappers

for _public_name in (*_ast.__all__, *_wrappers.__all__):
    globals()[_public_name] = getattr(
        _ast if _public_name in _ast.__all__ else _wrappers,
        _public_name,
    )

__all__ = [
    "Arg",
    "ArrayType",
    "CTranslationUnit",
    "CompilationError",
    "CorruptLibraryError",
    "DataType",
    "IncompatibleLibraryError",
    "LegacyLibraryFormatError",
    "LibraryFormatError",
    "NumetaBuildReport",
    "NumetaError",
    "NumetaLibrary",
    "NumetaNotImplementedError",
    "NumetaPerformanceWarning",
    "NumetaSpecialization",
    "NumetaTypeError",
    "PointerType",
    "StructType",
    "ToolchainNotFoundError",
    "Vector",
    "VectorType",
    "__version__",
    "b1",
    "bool8",
    "c16",
    "c32",
    "c8",
    "c_function_attributes",
    "c_ptr",
    "char",
    "complex128",
    "complex16",
    "complex256",
    "complex32",
    "complex64",
    "complex8",
    "comptime",
    "f16",
    "f4",
    "f8",
    "float128",
    "float32",
    "float64",
    "get_datatype",
    "i4",
    "i8",
    "int32",
    "int64",
    "integer4",
    "integer8",
    "iso_c",
    "jit",
    "logical1",
    "make_vector_type",
    "pi",
    "e",
    "inf",
    "nan",
    "omp",
    "ptr",
    "r16",
    "r4",
    "r8",
    "real16",
    "real4",
    "real8",
    "settings",
    "size_t",
    *_ast.__all__,
    *_wrappers.__all__,
]

__all__ = list(dict.fromkeys(__all__))

del _ast, _math, _public_name, _wrappers

settings.initialize_default_datatypes()
