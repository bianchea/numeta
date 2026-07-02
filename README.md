# Numeta


Numeta is a small Python transpiler with a metaprogramming focus. It is inspired by
[Numba](https://github.com/numba/numba) but keeps the feature set intentionally
narrow so it stays easy to reason about. It targets numeric kernels and grew out of
my work on integral evaluations over Gaussian basis functions in computational chemistry.

The design favors simplicity: no AST or bytecode parsing, just type hints to
separate compile-time values from runtime arrays. Numeta translates a restricted
Python + NumPy-like subset into Fortran or C, then compiles and executes the
generated code.

The default backend is Fortran, because, well, [real programmers want to write FORTRAN
programs in any language](https://en.wikipedia.org/wiki/Real_Programmers_Don%27t_Use_Pascal).
More rationale is in [Why Fortran Backend](#why-fortran-backend).

## Table of Contents

- [Features](#features)
- [How it Works](#how-it-works)
- [Limitations](#limitations)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Defaults](#defaults)
  - [Backends](#backends)
  - [C Backend Shared-Kernel ABI](#c-backend-shared-kernel-abi)
  - [Type Hints](#type-hints)
  - [Parallelizing Loops](#parallelizing-loops)
  - [Cache Compiled Code](#cache-compiled-code)
- [Examples](#examples)
  - [First For Loop](#first-for-loop)
  - [Conditional Statements](#conditional-statements)
  - [How to Link an External Library](#how-to-link-an-external-library)
  - [Parallel Loop Example](#parallel-loop-example)
  - [Compile-Time Example](#compile-time-example)
  - [Custom Naming of Compiled Functions](#custom-naming-of-compiled-functions)
- [Why Fortran Backend?](#why-fortran-backend)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Metaprogramming Focus**: Leverages metaprogramming for flexible code generation.
- **Simple Compiler Pipeline**: Avoids heavy AST/bytecode parsing by working from type hints and runtime objects.
- **Type Annotations**: Uses Python's type hints to differentiate between compiled and compile-time variables.
- **Library-Based Persistence**: Organize jitted functions into reusable, saveable NumetaLibrary bundles.

## How it Works

1. **Annotate** compile-time values using `nm.comptime` and write kernels with `@nm.jit`.
2. **Transpile** the supported Python + NumPy-like subset into Fortran or C.
3. **Compile** the generated Fortran/C into a shared library.
4. **Execute** the compiled routine from Python.

## Limitations

Numeta is still experimental. Compiled functions can return scalars or NumPy arrays,
but not arbitrary Python objects. Only a subset of Python and NumPy is currently
supported. Backend coverage is evolving and some features may be better supported
in one backend than the other.

## Installation

To install numeta, use:

```bash
git clone https://gitlab.com/andrea_bianchi/numeta
cd numeta
pip install .
```

You will need a Fortran compiler (only `gfortran` is currently supported) available on your `PATH` for the Fortran backend, or a C compiler (`gcc`) for the C backend.

## Quick Start

Here's a quick example demonstrating how Numeta works:

```python
import numeta as nm

@nm.jit
def mixed_loops(n: nm.comptime, array) -> None:
    for i in range(n):
        for j in nm.range(n):
            array[j, i] = i + j
```

This runs as normal Python code. The first loop (`n`) is compile-time and will be
unrolled, while the second loop is compiled and executed as Fortran. The generated
Fortran code looks like this:

```fortran
subroutine mixed_loops(array) bind(C)
    use iso_c_binding, only: c_double
    use iso_c_binding, only: c_int64_t
    implicit none
    real(c_double), dimension(0:9, 0:9), intent(inout) :: array
    integer(c_int64_t) :: fc_i1
    integer(c_int64_t) :: fc_i2
    integer(c_int64_t) :: fc_i3
    do fc_i1 = 0_c_int64_t, 2_c_int64_t
        array(0, fc_i1) = (0_c_int64_t + fc_i1)
    end do
    do fc_i2 = 0_c_int64_t, 2_c_int64_t
        array(1, fc_i2) = (1_c_int64_t + fc_i2)
    end do
    do fc_i3 = 0_c_int64_t, 2_c_int64_t
        array(2, fc_i3) = (2_c_int64_t + fc_i3)
    end do
end subroutine mixed_loops
```

This is where one can appreciate the beauty of Fortran. Note that the indices are
reversed because Fortran arrays are column-major, meaning the first index is the
column and the second is the row.

## Usage

### Defaults

You can set global defaults once instead of passing them to every `@nm.jit` call:

```python
import numeta as nm

nm.settings.set_default_backend("c")
nm.settings.set_default_do_checks(False)
nm.settings.set_default_compile_flags("-O2 -march=native")
```

Passing `None` to `@nm.jit` parameters will also use these defaults.

### Backends

Numeta supports two code generation backends:

- `fortran` (default)
- `c`

Pick per function:

```python
@nm.jit(backend="c")
def add_one(a):
    a[0] += 1
```

Or set a global default:

```python
nm.settings.set_default_backend("c")
```

### C Backend Shared-Kernel ABI

The C backend has a lower-level code generation path for reusable helper kernels.
It is intended for generated code that needs stable C signatures, direct helper
calls, SIMD vector values at function boundaries, and explicit pointer qualifiers.
These APIs are most useful when Numeta is being used as a C code generator rather
than as a normal Python-callable JIT wrapper.

Use `nm.Vector[dtype, lanes]` when a helper should receive a SIMD value directly,
without storing lanes to a temporary array in the caller and reloading them in the
callee:

```python
import numeta as nm

v4 = nm.Vector[nm.f8, 4]

@nm.jit(
    backend="c",
    simd_arch="avx2",
    simd_features=("fma",),
    name="build_R_L8",
    c_attributes=["hot", "aligned(64)", 'visibility("hidden")'],
)
def build_R_L8(workspace, outer_p, outer_px, prefactor):
    scale = nm.broadcast(prefactor, lanes=4)
    value = nm.fma(outer_p, scale, outer_px)
    nm.vstore(workspace, 0, value)

build_R_L8(
    nm.ptr(nm.f8, restrict=True),
    v4,
    v4,
    nm.f8,
)
```

The C signature uses vector parameters by value:

```c
__attribute__((hot, aligned(64), visibility("hidden")))
void build_R_L8(npy_float64 * restrict workspace,
                nm_vec_f64_4 outer_p,
                nm_vec_f64_4 outer_px,
                npy_float64 prefactor);
```

Use `nm.ptr(...)` for raw C pointer arguments. Qualifiers are emitted in the C
signature:

```python
@nm.jit(backend="c")
def kernel(workspace, bra_batch_buffer, E):
    workspace[0] = bra_batch_buffer[0] + E[0]

kernel(
    nm.ptr(nm.f8, restrict=True),
    nm.ptr(nm.f8, const=True, restrict=True),
    nm.ptr(nm.f8, const=True, restrict=True),
)
```

This emits arguments such as:

```c
npy_float64 * restrict workspace,
const npy_float64 * restrict bra_batch_buffer,
const npy_float64 * restrict E
```

Wrappers can call declared C helpers with `nm.external_function(...)`. This keeps
the call as a normal C call and preserves vector arguments:

```python
build_R = nm.external_function(
    "build_R_L8",
    [
        nm.ptr(nm.f8, restrict=True),
        nm.Arg(v4, pass_by_value=True),
        nm.Arg(v4, pass_by_value=True),
        nm.Arg(nm.f8, pass_by_value=True),
    ],
    None,
)

@nm.jit(backend="c", simd_arch="avx2", simd_features=("fma",))
def wrapper(workspace, lanes):
    outer_p = nm.vload(lanes, 0, lanes=4)
    outer_px = nm.vload(lanes, 4, lanes=4)
    build_R(workspace, outer_p, outer_px, 1.0)

wrapper(nm.ptr(nm.f8, restrict=True), nm.ptr(nm.f8, const=True))
```

For same-translation-unit emission, add the helper and its wrappers in the order
you want them emitted:

```python
from numeta.c.emitter import CEmitter

emitter = CEmitter(simd_arch="avx2", simd_features=("fma",))
unit = emitter.create_translation_unit("tttcc_l4v2_P4_Q4_L8.c")

unit.add_function(build_R_L8, linkage="static")
unit.add_function(eval_2222)
unit.add_function(eval_1322)

source = unit.emit()
header = unit.emit_header(guard="TTTCC_L4V2_P4_Q4_L8_H")
```

Each jitted function added to a `CTranslationUnit` should have exactly one
compiled signature. If a function has several specializations, add the specific
compiled procedure's `symbolic_function` instead.

`emit_mode="static_inline"` emits a helper as `static inline`, and
`c_attributes=[...]` accepts raw GCC/Clang attribute fragments such as
`"always_inline"`, `"noinline"`, `"section(\".text.hot.phasedint\")"`, or
`'visibility("hidden")'`.

Vector arguments are code-generation signatures, not Python runtime SIMD objects.
Call the function with Numeta type descriptors such as `nm.Vector[nm.f8, 4]` and
`nm.ptr(nm.f8)` to build or emit the specialization; use regular NumPy arrays and
scalars for normal Python-callable JIT execution.

### Type Hints

Use type hints to separate compile-time values from runtime arrays. Variables with
the `nm.comptime` type hint are compile-time, while other annotated values are
treated as runtime variables. Runtime variables should be NumPy types; structured
arrays are supported.

### Parallelizing Loops

Use `nm.prange` to parallelize loops with an OpenMP-style model. See the Parallel
Loop Example section for shared variables and scheduling options.

### Cache Compiled Code

Assignment materialization in `@nm.jit` follows Python's indexed assignment flow:

- Numeta materializes writes only when Python executes `target[...] = value`
  or an augmented form such as `target[...] += value`, `target[...] -= value`,
  `target[...] *= value`, or `target[...] /= value`.
- Bare rebinding such as `a = b`, `a += value`, `a -= value`, `a *= value`, or
  `a /= value` only updates the Python local name and does not emit a generated
  assignment by itself.

Use explicit indexed or sliced targets when you want the operation to appear in
the generated code.

Use `NumetaLibrary` to group multiple jitted functions and save or load them as a
unit. It keeps compiled code, dependencies, and wrappers together. Functions can be
accessed as attributes or via indexing if the function name might conflict with a
method name. Loading a library restores compiled code so you can cache across
executions.

```python
import numeta as nm
import numpy as np

lib = nm.NumetaLibrary("demo")

@nm.jit(library=lib)
def add(a):
    a[:] += 1

array = np.zeros(4, dtype=np.int64)
lib.add(array)
lib["add"](array)

lib.save("./build")
```

Load the saved library in another process to reuse compiled code:

```python
import numeta as nm
import numpy as np

lib_loaded = nm.NumetaLibrary.load("demo", "./build")

array = np.zeros(4, dtype=np.int64)
lib_loaded.add(array)
lib_loaded["add"](array)
```

Loaded libraries can also incrementally replace already-compiled specializations.
The replacement keeps the old exported symbol names, recompiles only the live
specializations for that function, and relinks the saved library on disk:

```python
lib = nm.NumetaLibrary.load("demo", "./build")

@nm.jit
def add(a):
    a[:] += 2

lib.replace(add)
lib.save("./build")
```

Reload in a fresh process when you need guaranteed use of the relinked shared
library.

## Examples

### First For Loop

Below is a simple example of a for loop using Numeta:

```python
import numeta as nm

@nm.jit
def first_for_loop(n, array) -> None:
    for i in nm.range(n):
        array[i] = i * 2
```

In this example:

- `n` is the size of the array.
- `array` is a rank-1 numpy array.
- The loop runs using `nm.range(n)` to generate a compiled loop that performs the operation.
- Fortran implicit casting is used to convert `i` to the appropriate type for the array.

Alternatively, you can use:

```python
@nm.jit
def do_loop(n, array) -> None:
    i = nm.scalar(nm.i8)
    with nm.do(i, 0, n - 1):
        array[i] = i * 2
```

This approach uses `nm.do` to emulate the Fortran `do` loop style.

For mutable scalar state inside `@nm.jit`, prefer explicit indexed assignment:

```python
acc = nm.scalar(nm.f8, 0.0)
acc[:] += 1.0
acc[:] *= 2.0
```

The `[:]` is what materializes the write. Bare `acc += 1.0` only rebinds the
Python local name.

### Conditional Statements

Below is an example of how to use conditional statements with Numeta:

```python
import numeta as nm

@nm.jit
def conditional_example(n, array) -> None:
    for i in nm.range(n):
        if nm.cond(i < 1):
            array[i] = 0
        elif nm.cond(i < 2):
            array[i] = 1
        else:
            array[i] = 2
        nm.endif()
```

Note: You need to use `nm.endif()` at the end of the conditional block, though I'm working on improving this syntax to make it more intuitive.
It is currently difficult to maintain Python-like syntax for generated conditional code because some branches may never be taken, which complicates the code generation and obligates to read the AST.

Alternatively, you can use:

```python
with nm.If(i < 3):
    with nm.If(i < 1):
        array[i] = 0
    with nm.ElseIf(i < 2):
        array[i] = 1
    with nm.Else():
        array[i] = 2
```

This approach is safer, albeit less elegant, and will generate the same code as the previous example without needing to read the AST.

### How to Link an External Library

Below is an example of how to link an external library, specifically linking BLAS (very alpha):

```python
import numeta as nm

# Create an external library wrapper for BLAS
blas = nm.ExternalLibraryWrapper("blas")

# Add a method from LAPACK to the wrapper
blas.add_method(
    "dgemm",
    [
        nm.char,      # transa
        nm.char,      # transb
        nm.i8,        # m
        nm.i8,        # n
        nm.i8,        # k
        nm.f8,        # alpha
        nm.f8[:],     # a
        nm.i8,        # lda
        nm.f8[:],     # b
        nm.i8,        # ldb
        nm.f8,        # beta
        nm.f8[:],     # c
        nm.i8         # ldc
    ],
    None,
    bind_c=False
)

@nm.jit
def matmul(a, b, c):
    # Call the linked LAPACK dgemm method
    blas.dgemm("N",
               "N",
               b.shape[0],
               a.shape[1],
               c.shape[1],
               1.0,
               b,
               b.shape[0],
               a,
               a.shape[0],
               0.0,
               c,
               c.shape[0])
```

In this example:

- `blas = nm.ExternalLibraryWrapper("blas")` creates a wrapper for the LAPACK library.
- `blas.add_method()` adds the `dgemm` method for matrix multiplication.
- The method signature includes parameters such as matrix dimensions and scalars.
- The `matmul` function then uses `blas.dgemm` to perform matrix multiplication.

### Parallel Loop Example

Numeta provides the capability to parallelize loops using `nm.prange`. This closely
follows the OpenMP `parallel do` model, allowing for efficient parallel execution by
leveraging shared and private variables. Note that this feature is still in alpha,
and the syntax might change in the future.

Below is an example of how to parallelize a loop using `nm.prange`:

```python
import numeta as nm

@nm.jit
def pmul(a, b, c):
    for i in nm.prange(a.shape[0], default='private', shared=[a, b, c], schedule='static'):
        for k in nm.range(b.shape[0]):
            c[i, :] += a[i, k] * b[k, :]
```

In this example:

- `nm.prange(a.shape[0])` parallelizes the outer loop.
- `default='private'` specifies that loop variables are private by default.
- The `shared` list includes variables that are shared across threads.
- The `schedule='static'` controls the scheduling of loop iterations.

### Compile-Time Example

Compile-time variables are ordinary Python objects without type hints. They are
evaluated during compilation and can be used to specialize generated code.

```python
import numeta as nm
import numpy as np

@nm.jit
def sum_first_n(length: nm.comptime, a, result):
    result[:] = 0.0
    for i in range(length):
        result[:] += a[i]

array = np.random.random((10,))
result = np.zeros((1,), dtype=array.dtype)

sum_first_n(4, array, result)
```

When `sum_first_n` is compiled, the loop is unrolled because `length` is known at compile time.

### Custom Naming of Compiled Functions

The `@nm.jit` decorator accepts an optional `namer` parameter. It should be a
callable receiving the specification of the compile-time arguments. The return
value is used for the generated directories and symbols:

```python
@nm.jit(directory=tmp_path, namer=lambda spec: f"spec_{spec[0]}")
def fill(length: nm.comptime, a):
    for i in range(length):
        a[i] = i

arr = np.zeros(5, dtype=np.int64)
fill(3, arr)
```

The compiled library will be created under ``tmp_path/spec_3`` because the name
depends on the compile-time value ``length``.

**Note**: the type of the runtime variables is considered a compile-time variable.

## Why Fortran Backend?

I chose to use Fortran as the default backend for numeta because:

1. **Familiarity**: I have experience with Fortran, which made it easier to implement.
2. **Native Array Operations**: Fortran supports array operations natively, reducing the amount of code required to support them.
3. **Fast Compilation**: Fortran is relatively fast to compile, which is beneficial for JIT compilation.

While Fortran has some limitations, it allowed me to create a working prototype quickly. The C backend is also supported and can be selected when desired. I'm open to improving the generated code in both backends, so suggestions are welcome.

## Contributing

Contributions are welcome! If you'd like to contribute, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
