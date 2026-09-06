import numpy as np
import pytest

import numeta as nm


@pytest.mark.parametrize("c_dispatch", [False, True])
@pytest.mark.parametrize("custom_parser", [False, True])
def test_source_tracks_existing_specialization_on_cache_hits(
    backend, c_dispatch, custom_parser, monkeypatch
):
    monkeypatch.setattr(nm.settings, "use_c_dispatch", c_dispatch)
    if not custom_parser:
        monkeypatch.setattr("numeta.signature.compile_custom_signature_parser", lambda *args: None)

    @nm.jit(backend=backend)
    def identity(value):
        return value

    with pytest.raises(RuntimeError, match="specialize"):
        _ = identity.sources
    assert identity(np.float64(1.5)) == 1.5
    floating_source = identity.source
    assert identity(np.int64(2)) == 2
    integer_source = identity.source
    assert floating_source != integer_source
    assert identity(np.float64(3.5)) == 3.5
    assert identity.source == floating_source
    assert identity(np.int64(4)) == 4
    assert identity.source == integer_source
    with pytest.raises(TypeError):
        identity.sources["invalid"] = "source"
