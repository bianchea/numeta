import multiprocessing

import numeta as nm
import numpy as np
import pytest

from numeta.ir import IRAssign, IRIntrinsic, IRReduce, IRVarRef, lower_procedure
from numeta.ir.passes import normalize_reductions


def _compile_shared_directory_specialization(directory, value, barrier, results):
    @nm.jit(backend="c", directory=directory)
    def concurrent_fill(fill_value: nm.comptime, out):
        out[:] = fill_value

    barrier.wait()
    out = np.zeros(100_000, dtype=np.int64)
    concurrent_fill(value, out)
    results.put((value, bool(np.all(out == value))))


def _construct_shared_directory_specialization(directory, value, barrier, results):
    @nm.jit(backend="c", directory=directory)
    def concurrent_specialize(fill_value: nm.comptime, out):
        out[:] = fill_value

    barrier.wait()
    specialization = concurrent_specialize.specialize(value, nm.int64[:])
    results.put(specialization.symbol)


def test_shared_jit_directory_is_process_safe(tmp_path):
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    results = context.Queue()
    directory = tmp_path / "shared"
    processes = [
        context.Process(
            target=_compile_shared_directory_specialization,
            args=(directory, value, barrier, results),
        )
        for value in (1, 2)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(30)

    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(results.get(timeout=5) for _ in processes) == [(1, True), (2, True)]


def test_shared_directory_specialize_reserves_unique_symbols(tmp_path):
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    results = context.Queue()
    directory = tmp_path / "shared"
    processes = [
        context.Process(
            target=_construct_shared_directory_specialization,
            args=(directory, value, barrier, results),
        )
        for value in (1, 2)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(30)

    assert [process.exitcode for process in processes] == [0, 0]
    symbols = {results.get(timeout=5) for _ in processes}
    assert len(symbols) == 2


def test_whole_array_assignment_rejects_shape_mismatch(tmp_path, backend):
    @nm.jit(backend=backend, directory=tmp_path)
    def copy(source, target):
        target[:] = source[:]

    source = np.arange(64.0).reshape(8, 8)
    target = np.empty((8, 7))

    with pytest.raises(ValueError, match="shapes.*must match"):
        copy(source, target)


def test_aot_bundle_preserves_shape_checks(tmp_path):
    name = "shape_contract_bundle"
    library = nm.NumetaLibrary(name)

    @nm.jit(backend="c", directory=tmp_path / "jit", library=library)
    def copy(source, target):
        target[:] = source[:]

    source = np.arange(16.0).reshape(4, 4)
    copy(source, np.empty_like(source))
    library.save(tmp_path / "bundles")

    loaded = nm.NumetaLibrary.load(name, tmp_path / "bundles")
    with pytest.raises(ValueError, match="shapes.*must match"):
        loaded.copy(source, np.empty((4, 3)))


def test_c_reduction_accepts_array_expression(tmp_path):
    @nm.jit(backend="c", directory=tmp_path)
    def squared_sum(values):
        return nm.sum(values * values)

    values = np.arange(1.0, 5.0)

    np.testing.assert_allclose(squared_sum(values), np.sum(values * values))


def test_reduction_normalization_creates_explicit_ir_statement(tmp_path):
    @nm.jit(backend="c", directory=tmp_path)
    def squared_sum(values):
        return nm.sum(values * values)

    specialization = squared_sum.specialize(nm.float64[:])
    procedure = lower_procedure(specialization._target.symbolic_function, backend="c")

    normalize_reductions(procedure)

    reductions = [statement for statement in procedure.body if isinstance(statement, IRReduce)]
    assignments = [statement for statement in procedure.body if isinstance(statement, IRAssign)]
    assert len(reductions) == 1
    assert reductions[0].op == "sum"
    assert isinstance(assignments[-1].value, IRVarRef)
    assert not isinstance(assignments[-1].value, IRIntrinsic)


def test_postponed_comptime_annotation_is_resolved(tmp_path, backend):
    namespace = {"nm": nm}
    exec(
        "from __future__ import annotations\n"
        "def fill(value: nm.comptime, out):\n"
        "    out[:] = value\n",
        namespace,
    )
    fill = nm.jit(backend=backend, directory=tmp_path)(namespace["fill"])
    out = np.zeros(4, dtype=np.int64)

    fill(7, out)

    np.testing.assert_array_equal(out, 7)


def test_loaded_function_cache_miss_requests_reattach(tmp_path, backend):
    @nm.jit(backend=backend, directory=tmp_path)
    def increment(value):
        return value + 1

    assert increment(np.float64(1.0)) == 2.0
    increment._func = None

    with pytest.raises(RuntimeError, match="reattach=True"):
        increment(np.int64(1))


def test_jit_creates_missing_directory_parents(tmp_path, backend):
    directory = tmp_path / "native" / backend / "increment"

    @nm.jit(backend=backend, directory=directory)
    def increment(value):
        return value + 1

    assert increment(1) == 2
    assert directory.is_dir()
