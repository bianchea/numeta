"""Benchmark warmed runtime-only and comptime Numeta cache hits."""

import argparse
import json
import platform
import statistics
import tempfile
import timeit
from pathlib import Path

import numeta as nm
from numeta.numeta_function import _c_dispatch_base_available
from numeta.settings import settings

DISPATCH_MODES = (
    ("c", True, True),
    ("python-c-parser", False, True),
    ("python", False, False),
)


def _measure(call, *, number: int, repeat: int) -> dict[str, float]:
    samples = [
        elapsed * 1e9 / number for elapsed in timeit.repeat(call, number=number, repeat=repeat)
    ]
    return {
        "number": number,
        "median_ns": statistics.median(samples),
        "min_ns": min(samples),
        "max_ns": max(samples),
    }


def _measure_numba(*, number: int, literal_number: int, repeat: int, warmup: int) -> dict:
    try:
        import numba
    except ImportError as exc:
        raise RuntimeError("Install numba to use the --numba comparison") from exc

    @numba.njit
    def runtime(value):
        pass

    @numba.njit
    def mixed(config, value):
        numba.literally(config)

    runtime(1)
    mixed(3, 1)
    for _ in range(warmup):
        runtime(1)
    for _ in range(min(warmup, 10)):
        mixed(3, 1)

    runtime_signature = runtime.signatures[0]
    mixed_signature = mixed.signatures[0]
    runtime_native = runtime.overloads[runtime_signature].entry_point
    mixed_native = mixed.overloads[mixed_signature].entry_point
    runtime_result = _measure(lambda: runtime(1), number=number, repeat=repeat)
    mixed_result = _measure(lambda: mixed(3, 1), number=literal_number, repeat=repeat)
    runtime_native_result = _measure(lambda: runtime_native(1), number=number, repeat=repeat)
    mixed_native_result = _measure(lambda: mixed_native(3, 1), number=number, repeat=repeat)
    runtime_ns = runtime_result["median_ns"]
    mixed_ns = mixed_result["median_ns"]
    return {
        "mode": "numba-literal",
        "version": numba.__version__,
        "runtime_signature": str(runtime_signature),
        "comptime_signature": str(mixed_signature),
        "runtime": runtime_result,
        "comptime": mixed_result,
        "runtime_native": runtime_native_result,
        "comptime_native": mixed_native_result,
        "comptime_increment_ns": mixed_ns - runtime_ns,
        "comptime_ratio": mixed_ns / runtime_ns,
    }


def _build_kernels(directory: Path, backend: str):
    runtime_directory = directory / "runtime"
    mixed_directory = directory / "mixed"
    runtime_directory.mkdir(parents=True)
    mixed_directory.mkdir()

    @nm.jit(backend=backend, directory=runtime_directory)
    def runtime(value):
        pass

    @nm.jit(backend=backend, directory=mixed_directory)
    def mixed(config: nm.comptime, value):
        pass

    runtime_signature = runtime.get_signature(1)
    mixed_signature = mixed.get_signature(3, 1)
    runtime(1)
    mixed(3, 1)
    return (
        runtime,
        mixed,
        runtime._fast_call[runtime_signature],
        mixed._fast_call[mixed_signature],
    )


def run_benchmark(
    *,
    backend: str,
    number: int,
    repeat: int,
    warmup: int,
    include_numba: bool = False,
    numba_literal_number: int = 1_000,
) -> dict:
    if not _c_dispatch_base_available:
        raise RuntimeError("The _signature extension is required to compare C dispatch")

    original_settings = settings.use_c_dispatch, settings.use_c_signature_parser
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="numeta-dispatch-") as temporary:
            root = Path(temporary)
            for mode, use_c_dispatch, use_c_parser in DISPATCH_MODES:
                settings.use_c_dispatch = use_c_dispatch
                settings.use_c_signature_parser = use_c_parser
                runtime, mixed, runtime_native, mixed_native = _build_kernels(root / mode, backend)

                for _ in range(warmup):
                    runtime(1)
                    mixed(3, 1)

                runtime_result = _measure(lambda: runtime(1), number=number, repeat=repeat)
                mixed_result = _measure(lambda: mixed(3, 1), number=number, repeat=repeat)
                runtime_native_result = _measure(
                    lambda: runtime_native(1), number=number, repeat=repeat
                )
                mixed_native_result = _measure(
                    lambda: mixed_native(1), number=number, repeat=repeat
                )
                runtime_ns = runtime_result["median_ns"]
                mixed_ns = mixed_result["median_ns"]
                results.append(
                    {
                        "mode": mode,
                        "runtime": runtime_result,
                        "comptime": mixed_result,
                        "runtime_native": runtime_native_result,
                        "comptime_native": mixed_native_result,
                        "comptime_increment_ns": mixed_ns - runtime_ns,
                        "comptime_ratio": mixed_ns / runtime_ns,
                    }
                )

            if include_numba:
                results.append(
                    _measure_numba(
                        number=number,
                        literal_number=numba_literal_number,
                        repeat=repeat,
                        warmup=warmup,
                    )
                )
    finally:
        settings.use_c_dispatch, settings.use_c_signature_parser = original_settings

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numeta": nm.__version__,
        "backend": backend,
        "number": number,
        "repeat": repeat,
        "warmup": warmup,
        "results": results,
    }


def _print_table(report: dict) -> None:
    print(f"Python {report['python']} | Numeta {report['numeta']} | backend={report['backend']}")
    print("Times are median nanoseconds per warmed call; lower is better.")
    print(
        f"{'mode':<18} {'native-rt':>11} {'native-ct':>11} "
        f"{'runtime':>12} {'comptime':>12} {'increment':>12} {'ratio':>10}"
    )
    for result in report["results"]:
        print(
            f"{result['mode']:<18} "
            f"{result['runtime_native']['median_ns']:>11,.1f} "
            f"{result['comptime_native']['median_ns']:>11,.1f} "
            f"{result['runtime']['median_ns']:>12,.1f} "
            f"{result['comptime']['median_ns']:>12,.1f} "
            f"{result['comptime_increment_ns']:>12,.1f} "
            f"{result['comptime_ratio']:>9.2f}x"
        )
    numba_result = next(
        (result for result in report["results"] if result["mode"] == "numba-literal"), None
    )
    if numba_result is not None:
        print(
            "Numba comptime uses numba.literally() and compiled as "
            f"{numba_result['comptime_signature']}; "
            f"runtime samples={numba_result['runtime']['number']}, "
            f"literal samples={numba_result['comptime']['number']} per repeat."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("c", "fortran"), default="c")
    parser.add_argument("--number", type=int, default=200_000)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5_000)
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    parser.add_argument("--numba", action="store_true", help="include a Numba comparison")
    parser.add_argument(
        "--numba-literal-number",
        type=int,
        default=1_000,
        help="calls per repeat for the slower numba.literally() dispatcher",
    )
    args = parser.parse_args()
    if args.number <= 0 or args.repeat <= 0 or args.warmup < 0 or args.numba_literal_number <= 0:
        parser.error(
            "number, repeat, and numba-literal-number must be positive; "
            "warmup must be non-negative"
        )

    report = run_benchmark(
        backend=args.backend,
        number=args.number,
        repeat=args.repeat,
        warmup=args.warmup,
        include_numba=args.numba,
        numba_literal_number=args.numba_literal_number,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_table(report)


if __name__ == "__main__":
    main()
