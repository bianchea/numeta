# Repository Guidelines

## Project Structure & Module Organization

Numeta is a Python 3.12+ transpiler for numeric kernels. The package lives in
`numeta/`: `ast/` defines the symbolic model, `ir/` handles lowering, and `c/` and
`fortran/` contain backend emitters. SIMD support is under `simd/`, while user-facing
helpers are collected in `wrappers/`. The CPython extension source is
`numeta/_signature.c`. Tests are flat `tests/test_*.py` modules; shared backend
parameterization is in `tests/conftest.py`. Consult `README.md` for supported syntax
and `SIMD_INTRINSICS.md` for SIMD behavior.

Shared libraries, object/module files, transpiled sources, caches, and benchmark
output are generated artifacts. Do not commit them unless a fixture requires them.

## Build, Test, and Development Commands

- `python -m pip install -e .` installs Numeta in editable mode and builds the
  `_signature` extension.
- `pytest -v` runs the complete suite against both supported backends.
- `pytest -v --backend=c` or `pytest -v --backend=fortran` isolates one backend.
- `pytest tests/test_scalar.py -v` runs one focused module.
- `pre-commit run --all-files` applies the configured Black formatting check.
- `python -m build` creates source and wheel distributions in `dist/` (install the
  `build` package first).

Native execution requires `gcc`; Fortran tests also require `gfortran`. CI also
installs BLAS/LAPACK development libraries, NumPy, and pytest.

## Coding Style & Naming Conventions

Use four-space indentation and Black with the configured 100-character line length.
Follow existing Python conventions: `snake_case` for modules, functions, and
variables; `PascalCase` for classes; and `UPPER_CASE` for constants. Keep backend-
specific emission logic in its backend package and shared transformations in `ast/`
or `ir/`. Prefer explicit tests over comments for documenting edge cases.

## Testing Guidelines

Write pytest functions named `test_<behavior>` in `test_<area>.py`. Use the `backend`
fixture when behavior should match in C and Fortran; pin a backend only for genuinely
backend-specific features. Add regression tests for every bug fix and assert emitted
source when code generation is the contract. No coverage threshold is configured,
so prioritize meaningful branch and error-path coverage.

## Commit & Pull Request Guidelines

Recent history primarily uses scoped Conventional Commit subjects, for example
`fix(jit): avoid reusing loaded native symbols` and `feat(simd): add logical vector
C codegen`. Use an imperative, concise subject with an appropriate `feat`, `fix`,
`refactor`, `perf`, `test`, or `chore` prefix and a focused scope.

Pull requests should explain the behavioral change, identify affected backends, link
related issues, and list exact test commands run. Include compact before/after emitted
code for code-generation changes. Never commit credentials or local compiler output.
