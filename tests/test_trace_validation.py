import ctypes.util
import os

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


def test_stale_array_element_swap_reports_both_locations(backend):
    @nm.jit(backend=backend)
    def kernel(a):
        saved = a[0]
        a[0] = a[1]
        a[1] = saved

    with pytest.raises(nm.NumetaError, match="snapshot") as error:
        kernel.specialize(np.array([5.0, 1.0]))
    assert "saved = a[0]" in str(error.value)
    assert "a[0] = a[1]" in str(error.value)


@pytest.mark.parametrize("whole", [False, True])
def test_stale_multidimensional_element_after_write(backend, whole):
    @nm.jit(backend=backend)
    def kernel(a, out):
        saved = a[0, 1] + 1
        if whole:
            a[:] = 7
        else:
            a[0, 1] = 7
        a[1, 0] = 8  # A later disjoint write must not hide the overlapping write.
        out[0] = saved

    with pytest.raises(nm.NumetaError, match="snapshot"):
        kernel.specialize(np.ones((2, 2)), np.zeros(1))


def test_array_element_snapshot_allows_runtime_swap(backend):
    @nm.jit(backend=backend)
    def kernel(a):
        saved = nm.scalar(a[0])
        a[0] = a[1]
        a[1] = saved

    a = np.array([5.0, 1.0])
    kernel(a)
    np.testing.assert_array_equal(a, [1.0, 5.0])


def test_disjoint_array_writes_do_not_invalidate_expression(backend):
    @nm.jit(backend=backend)
    def kernel(a, out):
        saved = a[0, 1] + 1
        a[1, 0] = 9
        out[0] = saved

    out = np.zeros(1)
    kernel(np.array([[1.0, 2.0], [3.0, 4.0]]), out)
    assert out[0] == 3


def test_fresh_array_read_after_write_is_valid(backend):
    @nm.jit(backend=backend)
    def kernel(a, out):
        a[0] = 9
        saved = a[0] + 1
        out[0] = saved

    out = np.zeros(1)
    kernel(np.ones(1), out)
    assert out[0] == 10


def test_array_write_in_sibling_branch_does_not_dominate_read(backend):
    @nm.jit(backend=backend)
    def kernel(a, flag, out):
        saved = a[0]
        with nm.If(flag):
            a[0] = 9
        with nm.Else():
            out[0] = saved

    out = np.zeros(1)
    kernel(np.array([5.0]), False, out)
    assert out[0] == 5


def test_stale_array_read_in_store_index_is_rejected(backend):
    @nm.jit(backend=backend)
    def kernel(indices, out):
        index = indices[0]
        indices[0] = 1
        out[index] = 4

    with pytest.raises(nm.NumetaError, match="snapshot"):
        kernel.specialize(np.array([0], dtype=np.int64), np.zeros(2))


def test_effectful_store_index_is_counted_once(backend):
    if ctypes.util.find_library("c") is None:
        pytest.skip("libc library not found")
    libc = nm.ExternalLibraryWrapper("c")
    libc.add_method("getpagesize", [], nm.int32)
    pagesize = os.sysconf("SC_PAGE_SIZE")

    @nm.jit(backend=backend)
    def kernel(out):
        out[libc.getpagesize() - pagesize] = 4

    out = np.zeros(1)
    kernel(out)
    assert out[0] == 4


def test_writing_a_different_array_does_not_invalidate_read(backend):
    @nm.jit(backend=backend)
    def kernel(a, b, out):
        saved = a[0]
        b[0] = 9
        out[0] = saved

    out = np.zeros(1)
    kernel(np.array([5.0]), np.array([1.0]), out)
    assert out[0] == 5


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
