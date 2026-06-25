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

from .jit import jit
from .numeta_library import NumetaLibrary
from .wrappers import *
from .ast import *
from .settings import settings

settings.initialize_default_datatypes()
