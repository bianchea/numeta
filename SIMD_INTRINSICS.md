# Numeta SIMD intrinsics

Numeta exposes semantic SIMD operations through the top-level `numeta`
namespace. User code does not call compiler intrinsics such as
`_mm256_cmp_pd` directly; the C SIMD backend selects the native instruction.

## Enabling SIMD

Select an architecture and the required optional features on `nm.jit`:

```python
import numeta as nm

@nm.jit(
    backend="c",
    simd_arch="avx2",
    simd_features=("fma",),
    compile_flags="-O3 -mavx2 -mfma",
)
def kernel(...):
    ...
```

The radial conversion, comparison, mask, reciprocal, and exponent operations
currently have an AVX2 implementation. Unsupported targets raise a
source-located error instead of emitting invalid C.

## Loading, storing, and constructing vectors

```python
x = nm.vload(input_array, index, lanes=4)
one = nm.broadcast(1.0, lanes=4)
explicit = nm.vector_from_values(a0, a1, a2, a3)
nm.vstore(output_array, index, x)
```

## Arithmetic and reciprocal seeds

```python
y = nm.fma(a, b, c)       # a*b + c
z = nm.fnma(a, b, c)      # c - a*b

# AVX reciprocal instructions operate on float32 vectors. Convert an f64x4
# vector to f32x4 for the seed and convert the result back.
rcp_seed = nm.astype(nm.rcp_approx(nm.astype(x, nm.f4)), nm.f8)
rsqrt_seed = nm.astype(nm.rsqrt_approx(nm.astype(x, nm.f4)), nm.f8)
```

`rcp_approx` and `rsqrt_approx` provide hardware approximation seeds. Use a refinement
step when full floating-point accuracy is required.

## Comparisons, masks, and blending

```python
mask = nm.compare(x, threshold, "lt")
selected = nm.where(mask, when_true, when_false)
bits = nm.mask_bits(mask)
```

Supported predicates are:

```text
lt, le, gt, ge, eq, neq
```

`where(mask, b, a)` selects lanes from `b` where the comparison mask is true
and from `a` otherwise. `mask_bits` returns the lane mask as an integer.

## Conversion and extraction

```python
f32 = nm.astype(f64, nm.f4)
f64 = nm.astype(f32, nm.f8)

i32 = nm.astype(f64, nm.i4)
f64_again = nm.astype(i32, nm.f8)

lane0 = nm.extract_lane(i32, 0)
lane3 = nm.extract_lane(i32, 3)
```

The extraction lane must be a compile-time integer within the vector's lane
range.

## Floor and exact powers of two

```python
rounded = nm.floor(x)
indices = nm.vcvt_f64_i32(rounded)
powers = nm.exp2_neg(indices)
```

For four signed 32-bit integer exponents, `exp2_neg` constructs four
normal binary64 values representing `2**(-k)` through their exponent bits.
Callers must keep exponents in the normal binary64 exponent range.

## Complete example

```python
@nm.jit(
    backend="c",
    simd_arch="avx2",
    simd_features=("fma",),
    compile_flags="-O3 -mavx2 -mfma",
)
def reciprocal_below_limit(out, inp, limit):
    x = nm.vload(inp, 0, lanes=4)
    seed = nm.astype(nm.rcp_approx(nm.astype(x, nm.f4)), nm.f8)
    refined = nm.fma(seed, nm.fnma(x, seed, 1.0), seed)
    mask = nm.compare(x, nm.broadcast(limit, 4), "lt")
    result = nm.where(mask, refined, x)
    nm.vstore(out, 0, result)
```

