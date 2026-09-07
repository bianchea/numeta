"""C99 complex arithmetic with NumPy 1.x and 2.x array storage."""

COMPLEX_COMPAT_HEADER = """
#ifndef NUMETA_NUMPY_COMPLEX_COMPAT
#define NUMETA_NUMPY_COMPLEX_COMPAT
#include <numpy/npy_math.h>
#include <complex.h>
#if NPY_ABI_VERSION < 0x02000000
/* NumPy 1.x uses two-field structs, with the same memory layout as C99
 * complex. Generated kernels and their wrappers use the C99 scalar ABI.
 * NumPy declarations above retain their original types; scalar/array data
 * crosses the boundary through NumPy's void-pointer APIs. */
#define npy_complex64 float _Complex
#define npy_complex128 double _Complex
#define npy_clongdouble long double _Complex
#endif
#endif
"""
