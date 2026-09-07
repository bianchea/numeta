# Numeta Language Contract

Numeta executes Python once to trace a specialization. It does not parse Python
source or bytecode, schedule expressions, perform CSE, or automatically save
intermediate values. Native compilers may optimize the emitted code.

## Aliases, expressions, and storage

`alias = x` creates a Python alias. `x = x + y` builds a lazy expression. Neither
emits a runtime store. Use `nm.scalar(expr)` to evaluate a scalar expression once
at that statement, `nm.empty(shape, dtype=...)` for uninitialized arrays, or
`nm.zeros(shape, dtype=...)` for zero-initialized arrays.

`nm.scalar(value, dtype=nm.f4)` selects a storage dtype. `nm.scalar(dtype=nm.f8)`
and `nm.scalar(nm.f8)` create uninitialized storage. The legacy
`nm.scalar(nm.f8, value)` form remains supported. `nm.empty((), dtype=nm.f8)` is the
shape-generic rank-zero equivalent. Array expressions cannot initialize a scalar;
allocate an array and assign through `[:]` instead.

| Expression | Meaning |
| --- | --- |
| `x[:] = x + y` | Overwrite storage at runtime |
| `x[:] += y` | Read, update, and overwrite storage |
| `a[i] = value` | Store an array element |
| `a[20:40] += y` | Update an array slice |
| `x += y` on bare storage | Error; choose an expression or an explicit store |
| `scalar[0]` | Error; scalar storage only supports `scalar[:]` |

An expression is not a snapshot of its inputs. A trace that constructs an
expression, overwrites one of its scalar dependencies, and then consumes that
expression is rejected when the intervening write provably dominates the use.
This also detects constant-index array-element reads followed by a write to that
same element or the whole array. Disjoint constant-index writes remain valid.
The diagnostic identifies both locations and suggests `nm.scalar(expr)`.
This bounded check does not prove overlap for runtime indices, partial slices,
different array arguments sharing memory, or general control-flow paths.

## Conversion and computation dtypes

```python
converted = nm.astype(expr, nm.f8)       # lazy conversion
snapshot = nm.scalar(expr, dtype=nm.f8)  # evaluate and store here
snapshot[:] = other                     # runtime overwrite
```

`nm.f8(expr)` also creates scalar storage, as does `nm.scalar(nm.f8, expr)`.
These supported shorthand forms do not mean lazy conversion. Scalar initializers
must have rank zero; choose an array element or allocate an array and store through
`[:]`. `nm.astype` preserves array shape without allocating or taking a snapshot.
It does not introduce additional SIMD conversion pairs.

Arithmetic `+`, `-`, `*`, `/`, numeric comparisons, and `nm.where` value branches
use NumPy-style promotion independent of operand order. NumPy scalars, storage,
arrays, and runtime kernel arguments are strongly typed. Ordinary Python literals
beside typed operands are weak: float32 plus `1.0` stays float32, but an explicit
float64 operand promotes it. Standalone literals and expressions containing only
untyped literals retain configured defaults. Integer `/` computes in float64;
`nm.trunc_div` retains native truncation. Storage dtypes remain authoritative:
computation is promoted before assignment conversion, including explicit narrowing.

`nm.where` requires a Boolean condition; compare numeric conditions explicitly,
for example `condition != 0`. Complex arithmetic, equality, and selection are
supported; ordered complex comparisons require explicit real parts or magnitudes.
Compile-time literals outside the selected dtype's range are rejected; use a wider
typed operand. Ordinary floating-point rounding is permitted. There are no new
runtime overflow checks or broadcasting rules.

Mixed expressions and integer division can now infer different return dtypes.
Rebuild affected saved libraries from their kernel definitions to adopt these rules;
existing saved bundles are not rewritten. The package version remains 0.6.0.

## Effects and output

Built-in math is pure. `nm.time()` and external functions are effectful by default.
An effectful expression cannot be discarded or reused; use
`tick = nm.scalar(nm.time())` before reusing `tick`. External declarations accept
`pure=True` only for functions whose repeated evaluation has no observable effects.

Use `nm.Print(...)` for runtime output. Python `print`, f-strings, `math.*`, NumPy
conversion, and iteration cannot consume symbolic values during tracing.

## Conditions and loops

Use `with nm.If(condition)`, `with nm.ElseIf(condition)`, and `with nm.Else()`.
Combine parenthesized comparisons using `&`, `|`, and `~`. Python `and`, `or`,
`not`, chained comparisons, and conditional expressions cannot express runtime
Numeta control flow. `nm.minimum` and `nm.maximum` cover common numeric selections.

`nm.range` and `nm.prange` emit loops. `with nm.While(condition)` emits a while loop.
`for case in nm.cases(value, range(n))` traces an explicit case selection.
Use `nm.Break()`, `nm.Continue()`, and void `nm.Return()` inside these constructs.
Abandoned range iterators (usually Python `break` or `return`) are rejected.
Bare Python `continue` and conditional early Python value returns cannot be
intercepted without parsing: they must not be used for runtime control flow.
Recursive Numeta calls are rejected.

Python `range` with compile-time bounds unrolls the trace. Traces exceeding 5,000
statements emit `nm.NumetaPerformanceWarning`; configure
`nm.settings.performance_warning_threshold` or set it to `None` to disable it.

Bare `nm.prange(n)` infers shared arrays, read-only scalars, and shape descriptors,
and makes iterators and loop-local storage private under OpenMP `default(none)`.
Captured scalar writes require explicit private storage or atomic updates.
Validated options include `shared`, `private`, `schedule`, `chunk`, and
`num_threads`. Chunks are supported for static, dynamic, and guided schedules.

## Arrays and numerical rules

For iterative updates, allocate two buffers outside the loop, compute the next
state without modifying the current state, then copy it back using a sliced
assignment. Python tuple/name swaps inside a runtime loop occur only once during
tracing; they do not emit runtime pointer swaps.

Integer `//` and `%` follow Python's floor-division relationship, including negative
operands. `nm.trunc_div` and `nm.trunc_mod` explicitly request native truncation.
`>>` is a signed right shift, not assignment. `round(x)` uses ties-to-even;
`round(x, ndigits)` requires compile-time digits.

Struct fields can be read or stored through `particles[i].mass` or
`particles[i]["mass"]`. Two-element `(name, dtype)` field declarations are scalar;
`(name, dtype, shape)` specifies an array field.

Optimized kernels use unchecked indexing: negative or out-of-range indices are
undefined behavior. Wrapper `do_checks` validates call arguments and supported
shape contracts; it is not runtime bounds checking.

## Inspection and specialization

Call the function or use `fn.specialize(...)` before accessing `fn.source` or
the read-only `fn.sources` mapping. `fn.source` follows the most recently selected
specialization, including cache hits. `allow_new_specializations=False` restricts
execution to registered or loaded signatures; explicit `specialize(...)` still
registers new ones. Numeta's package version remains 0.6.0 for this development work.
