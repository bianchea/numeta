import linecache
from types import SimpleNamespace

import numpy as np
import pytest
import numeta as nm
from numeta.ast import Variable
from numeta.ast.nodes.base_node import set_source_location_tracking
from numeta.exceptions import format_source_location, describe_value


def test_bare_update_reports_use_site():
    value = Variable("value", dtype=nm.f8)  # STORAGE_DECLARATION_MARKER
    with pytest.raises(nm.NumetaTypeError) as error:
        value += 1  # OFFENDING_UPDATE_MARKER
    message = str(error.value)
    assert "OFFENDING_UPDATE_MARKER" in message
    assert "STORAGE_DECLARATION_MARKER" not in message
    assert "Expected:" in message and "Received:" in message
    assert "x[:]" in message


def test_notebook_source_and_missing_source():
    filename = "<numeta-notebook-cell>"
    source = "value += 1  # NOTEBOOK_USE_MARKER\n"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    try:
        node = SimpleNamespace(source_location={"filename": filename, "lineno": 1})
        assert "NOTEBOOK_USE_MARKER" in format_source_location(node)
    finally:
        linecache.cache.pop(filename, None)
    assert format_source_location(node) == f'  File "{filename}", line 1'


def test_disabled_tracking_does_not_capture_use_site():
    value = Variable("value", dtype=nm.f8)
    set_source_location_tracking(False)
    try:
        with pytest.raises(nm.NumetaTypeError) as error:
            value += 1
        assert 'File "' not in str(error.value)
    finally:
        set_source_location_tracking(True)


def test_public_value_description_avoids_symbolic_printing():
    assert describe_value(Variable("value", dtype=nm.f4)) == "float32 scalar (rank 0)"


def test_scalar_initializer_diagnostic(backend):
    @nm.jit(backend=backend)
    def kernel(a):
        return nm.scalar(a, dtype=nm.f8)  # INITIALIZER_USE_MARKER

    with pytest.raises(nm.NumetaTypeError) as error:
        kernel(np.ones(3))
    message = str(error.value)
    assert "INITIALIZER_USE_MARKER" in message
    assert "Expected: a scalar initializer (rank 0)" in message
    assert "Received: float64 array (rank 1)" in message


def test_stale_diagnostic_preserves_related_locations(backend):
    @nm.jit(backend=backend)
    def kernel():
        value = nm.scalar(1.0)
        expr = value + 1  # EXPRESSION_CONSTRUCTION_MARKER
        value[:] = 2  # INTERVENING_WRITE_MARKER
        return expr

    with pytest.raises(nm.NumetaError) as error:
        kernel()
    message = str(error.value)
    assert "EXPRESSION_CONSTRUCTION_MARKER" in message
    assert "INTERVENING_WRITE_MARKER" in message
    assert "nm.scalar(expr)" in message
