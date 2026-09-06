import numpy as np
import pytest

import numeta as nm


@pytest.mark.parametrize("duplicate", [False, True])
def test_effectful_calls_need_exactly_one_use(backend, duplicate):
    @nm.jit(backend=backend)
    def kernel(out):
        tick = nm.time()
        if duplicate:
            out[0] = tick + tick

    with pytest.raises(nm.NumetaError, match="effectful expression.*"):
        kernel(np.zeros(1))


def test_materialized_time_can_be_reused(backend):
    @nm.jit(backend=backend)
    def kernel(out):
        tick = nm.scalar(nm.time())
        out[0] = tick - tick

    out = np.ones(1)
    kernel(out)
    assert out[0] == 0


def test_stale_expression_reports_creation_and_write(backend):
    @nm.jit(backend=backend)
    def kernel(out):
        value = nm.scalar(1.0)
        lazy = value + 1
        value[:] = 5.0
        out[0] = lazy

    with pytest.raises(nm.NumetaError, match="snapshot") as error:
        kernel(np.zeros(1))
    assert "Intervening write" in str(error.value)
    assert "lazy = value + 1" in str(error.value)
    assert "value[:] = 5.0" in str(error.value)


def test_snapshot_survives_later_write(backend):
    @nm.jit(backend=backend)
    def kernel(out):
        value = nm.scalar(1.0)
        snapshot = nm.scalar(value + 1)
        value[:] = 5.0
        out[0] = snapshot

    out = np.zeros(1)
    kernel(out)
    assert out[0] == 2


@pytest.mark.parametrize("parallel", [False, True])
def test_python_break_abandons_numeta_iterator(backend, parallel):
    loop = nm.prange if parallel else nm.range

    @nm.jit(backend=backend)
    def kernel(out):
        for i in loop(10):
            out[i] = 1
            break

    with pytest.raises(nm.NumetaError, match=r"nm.Break"):
        kernel(np.zeros(10))


def test_recursion_rejected_before_return_lowering(backend):
    @nm.jit(backend=backend)
    def recursive(value):
        return recursive(value)

    with pytest.raises(RuntimeError, match="Recursive Numeta call"):
        recursive(1)


def test_unrolled_trace_emits_configurable_performance_warning(backend, monkeypatch):
    monkeypatch.setattr(nm.settings, "performance_warning_threshold", 2)

    @nm.jit(backend=backend)
    def kernel(out):
        for i in range(5):
            out[i] = i

    out = np.zeros(5)
    with pytest.warns(nm.NumetaPerformanceWarning, match=r"nm.range") as captured:
        kernel(out)
    assert len(captured) == 1
    np.testing.assert_array_equal(out, np.arange(5))
