import numeta as nm
import pytest


def test_cond_is_a_compatibility_error():
    with pytest.raises(RuntimeError, match="removed") as exc_info:
        nm.cond(True)
    assert "with nm.If" in str(exc_info.value)


def test_endif_is_a_compatibility_error():
    with pytest.raises(RuntimeError, match="removed") as exc_info:
        nm.endif()
    assert "with nm.If" in str(exc_info.value)
