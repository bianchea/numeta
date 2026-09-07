# Predictable numerics verification

Implemented `next_plan.md` without changing version 0.6.0. Existing saved libraries
are not rewritten. Rebuild affected saved libraries from their kernel definitions:
mixed expressions can now infer wider return types, and integer `/` returns float64.

## Behavioral checks

- Arithmetic and comparison resolution cover all supported Boolean, integer, real,
  and complex dtypes, including extended precision where NumPy exposes it.
- Runtime checks cover reversed operands, scalar and equal-shape array operands,
  weak Python literals versus typed NumPy scalars, true division, selection,
  explicit narrowing, Boolean conversions, and compile-time literal overflow.
- IR checks cover casts before operations, casted/modulo indices, slice bounds,
  allocation dimensions, source/type preservation, call arguments, and inlining.
- Existing effect-use, snapshot, shape-descriptor, SIMD, and backend regressions
  remain in the full suites. Shape checks also propagate through promoted casts.
- Diagnostics cover offending use sites, declaration versus use locations,
  related stale-expression locations, cached notebook source, missing source,
  disabled tracking, and public dtype/rank descriptions.
- README conversion/snapshot examples execute on both backends.

For mixed float32 `a` and float64 `b`, computation now promotes before storage:

```c
/* Before: a mixed expression could infer a float32 return. */
/* After: the IR result is float64 and the input conversion is explicit. */
((npy_float64)(a)) + b
```

```fortran
! After: real(c_double) computation, even when the destination narrows.
real(a,kind=c_double) + b
```

## Environments and commands

Native compilers: GCC and GNU Fortran 15.3.1 (20260722).
Main environment: Python 3.14.6, NumPy 2.3.5.
Compatibility environment: Python 3.12.14, NumPy 1.26.4.

Full repository suites use tracked tests, including the new regression modules.
Unrelated untracked tests and scratch files were left untouched. An initial bare
`pytest` discovery included the pre-existing untracked `test_bug_reproducers.py`,
whose signature-collision assertion fails; that file is outside this change.

```bash
git ls-files -z 'tests/test_*.py' | OMP_NUM_THREADS=2 xargs -0 .venv/bin/pytest -q --backend=c
git ls-files -z 'tests/test_*.py' | OMP_NUM_THREADS=2 xargs -0 .venv/bin/pytest -q --backend=fortran
```

Final frozen-source results:

| Backend | Result | Time |
| --- | --- | --- |
| C | 1,815 passed, 4 warnings | 501.53 s |
| Fortran | 1,815 passed, 4 warnings | 440.60 s |

Warnings are the existing Python 3.14 warnings about `fork()` in a multithreaded
process in the compilation-runtime tests.

The pinned compatibility run used a temporary interpreter and virtual environment.
`LIBRARY_PATH` and `LD_LIBRARY_PATH` locate that interpreter's shared library;
these are test-process environment settings, not changes to Numeta defaults.

```bash
LIBRARY_PATH=/tmp/numeta-python/cpython-3.12.14-linux-x86_64-gnu/lib \
LD_LIBRARY_PATH=/tmp/numeta-python/cpython-3.12.14-linux-x86_64-gnu/lib \
OMP_NUM_THREADS=2 /tmp/numeta-numpy126/bin/python -m pytest -q \
  tests/test_numeric_promotion.py tests/test_typed_lowering.py \
  tests/test_focused_diagnostics.py tests/test_readme_contract_examples.py
```

Result: **637 passed in 51.07 seconds**. This includes compiled C and Fortran
execution. The run exposed NumPy 1.x's struct complex representation; generated
kernels and wrappers now share a C99-complex compatibility block while preserving
NumPy's array/scalar memory layout and the native scalar calling convention.

```bash
.venv/bin/pre-commit run --all-files
.venv/bin/python -m build --no-isolation
.venv/bin/python -m pip install --no-deps --target /tmp/numeta-wheel-smoke \
  dist/numeta-0.6.0-cp314-cp314-linux_x86_64.whl
```

Pre-commit passed. Source and wheel builds passed. The installed wheel was imported from `/tmp`,
including `_signature`, and compiled mixed real/complex addition and integer true
division on both backends successfully. Generated build artifacts are not committed.
Black and pre-commit needed execution outside the sandbox because Python's worker
startup was blocked inside it.

The existing CI matrix remains Python 3.12, 3.13, and 3.14 on C and Fortran, with
an added pinned NumPy 1.26.4/Python 3.12 job. Remote CI was not triggered; Python
3.13 and the full Python 3.12 matrix remain CI verification rather than local results.
